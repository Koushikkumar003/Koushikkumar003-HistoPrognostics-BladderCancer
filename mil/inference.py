import os
import pyvips
import torch
import numpy as np
from timeit import default_timer as timer
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import skimage
from skimage.filters import rank
from skimage.morphology import disk
from skimage.transform import resize
from PIL import Image
from scipy import ndimage
import pandas as pd


class MILInference:
    """Multiple Instance Learning inference for whole slide images (WSI)"""
    
    def __init__(self, dir_wsi, dir_out, backbone, network, dir_inference, dir_images_regions, 
                 input_shape=(3, 224, 224), ori_patch_size=512):
        # Directory paths
        self.dir_wsi = dir_wsi
        self.dir_results = dir_out
        self.dir_inference = dir_inference
        self.dir_images_regions = dir_images_regions
        
        # Create output directory if it doesn't exist
        os.makedirs(self.dir_results + self.dir_inference, exist_ok=True)
        
        # Model components
        self.backbone = backbone  # Feature extraction network
        self.network = network    # MIL aggregation network
        
        # Configuration
        self.input_shape = input_shape
        self.ori_patch_size = ori_patch_size

    def infer(self, current_wsi_subfolder):
        """Run inference on a single WSI"""
        # Extract WSI name and load region info
        self.current_wsi_name = current_wsi_subfolder.split('/')[-1]
        self.region_info_df = pd.read_csv(f"{self.dir_images_regions}{self.current_wsi_name}/region_info.csv")
        
        # Setup models on GPU
        self.backbone.cuda().eval()
        self.network.cuda().eval()
        
        output_dir = f"{self.dir_results}{self.dir_inference}{self.current_wsi_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load WSI thumbnail
        wsi_thumbnail = pyvips.Image.new_from_file(
            f"{self.dir_wsi}{self.current_wsi_name}.mrxs", autocrop=True, level=4).flatten()
        
        # Run patch classification if not already done
        patch_class_path = f"{output_dir}/patch_classification.npy"
        if not os.path.exists(patch_class_path):
            print(f'[Inference]: {self.current_wsi_name}')
            
            # Save thumbnail
            thumb_save = pyvips.Image.new_from_file(
                f"{self.dir_wsi}{self.current_wsi_name}.mrxs", autocrop=True, level=5).flatten()
            thumb_save.jpegsave(f"{output_dir}/thumbnail.jpeg", Q=100)
            
            # Extract features and classify patches
            patch_class, region_class, global_class = self._extract_features_and_classify(current_wsi_subfolder)
            
            # Save classifications
            np.save(f"{output_dir}/patch_classification.npy", patch_class)
            np.save(f"{output_dir}/region_classification.npy", region_class)
            np.save(f"{output_dir}/global_classification.npy", global_class)
        else:
            # Load existing classifications
            patch_class = np.load(f"{output_dir}/patch_classification.npy")
            region_class = np.load(f"{output_dir}/region_classification.npy")
            global_class = np.load(f"{output_dir}/global_classification.npy")
        
        # Normalize attention scores
        self._normalize_attention(patch_class, region_class, output_dir)
        
        # Generate prediction visualizations
        self._generate_visualizations(current_wsi_subfolder, patch_class, region_class, wsi_thumbnail, output_dir)

    def _extract_features_and_classify(self, wsi_subfolder):
        """Extract CNN features and perform MIL classification"""
        features, region_info = [], []
        patch_files = os.listdir(wsi_subfolder)
        
        # Extract features from each patch
        for i, patch_file in enumerate(patch_files):
            print(f"{i+1}/{len(patch_files)}", end='\r')
            
            # Load and preprocess patch
            patch = Image.open(os.path.join(wsi_subfolder, patch_file))
            patch = self._normalize_image(np.asarray(patch))
            patch = torch.tensor(patch[np.newaxis, ...]).cuda().float()
            
            # Extract features using backbone
            features.append(self.backbone(patch).squeeze().detach().cpu().numpy())
            
            # Get region info for this patch
            x_coor = int(patch_file.split('_')[0])
            y_coor = int(patch_file.split('_')[1].split('.')[0])
            region_val = self.region_info_df.loc[
                (self.region_info_df['X_coor'] == x_coor) & 
                (self.region_info_df['Y_coor'] == y_coor), 'Region'].item()
            region_info.append(region_val)
        
        print('CNN Features: done')
        
        # Convert to tensors
        features = torch.tensor(np.array(features)).cuda().float()
        region_info = torch.tensor(np.array(region_info)).cuda().float()
        
        # Perform MIL aggregation by regions
        instance_class, region_embeddings = [], []
        for region_id in range(int(torch.max(region_info).item()) + 1):
            region_features = features[region_info == region_id]
            
            # Calculate attention weights for this region
            attn_module = self.network.milAggregation.attentionModule
            A_V = attn_module.attention_V(region_features)
            A_U = attn_module.attention_U(region_features)
            w_logits = attn_module.attention_weights(A_V * A_U)
            instance_class.append(w_logits)
            
            # Get region embedding
            if len(region_features) > 1:
                embedding_reg, _ = self.network.milAggregation(region_features.squeeze())
                region_embeddings.append(embedding_reg)
            else:
                region_embeddings.append(region_features.squeeze(dim=0))
        
        # Global classification
        if len(region_embeddings) == 1:
            embedding = region_embeddings[0]
            patch_class = w_logits
        else:
            embedding, _ = self.network.milAggregation(torch.stack(region_embeddings).squeeze())
            patch_class = torch.cat(instance_class, dim=0)
            
            # Calculate region-level attention
            stacked_embeddings = torch.stack(region_embeddings).squeeze()
            attn_module = self.network.milAggregation.attentionModule
            A_V = attn_module.attention_V(stacked_embeddings)
            A_U = attn_module.attention_U(stacked_embeddings)
            w_logits = attn_module.attention_weights(A_V * A_U)
        
        global_class = self.network.classifier(embedding).detach().cpu().numpy()
        patch_class = patch_class.detach().cpu().numpy()
        
        # Map region classifications to patches
        region_class_aux = w_logits.detach().cpu().numpy()
        region_class = np.zeros(patch_class.shape)
        region_info_np = region_info.detach().cpu().numpy()
        
        for i, region_id in enumerate(region_info_np):
            region_class[i] = region_class_aux[int(region_id)]
        
        print('MIL classification: done')
        return patch_class, region_class, global_class

    def _normalize_attention(self, patch_class, region_class, output_dir):
        """Normalize attention scores using pre-computed min/max values"""
        min_max_att = np.load(f"{self.dir_results}{self.dir_inference}/min_max_att.npy")
        patch_class = (patch_class - min_max_att[0]) / (min_max_att[1] - min_max_att[0])
        region_class = (region_class - min_max_att[0]) / (min_max_att[1] - min_max_att[0])
        return patch_class, region_class

    def _generate_visualizations(self, wsi_subfolder, patch_class, region_class, wsi_thumbnail, output_dir):
        """Generate prediction colormaps and overlay visualizations"""
        # Load high-res WSI for colormap dimensions
        wsi_20x = pyvips.Image.new_from_file(
            f"{self.dir_wsi}{self.current_wsi_name}.mrxs", autocrop=True, level=2).flatten()
        colormap_h = wsi_20x.height // self.ori_patch_size
        colormap_w = wsi_20x.width // self.ori_patch_size
        
        colormap_path = f"{output_dir}/prediction_colormap.npy"
        if not os.path.exists(colormap_path):
            # Initialize colormaps
            pred_colormap = -np.ones((colormap_h, colormap_w), dtype=np.float32)
            pred_colormap_reg = -np.ones((colormap_h, colormap_w), dtype=np.float32)
            
            # Fill colormaps with attention scores
            patch_files = os.listdir(wsi_subfolder)
            for i, (patch_att, reg_att, patch_file) in enumerate(zip(patch_class, region_class, patch_files)):
                print(f"{i+1}/{len(patch_class)}", end='\r')
                
                x_coord = int(patch_file.split('_')[-1].split('.')[0])
                y_coord = int(patch_file.split('_')[-2])
                x_idx = x_coord // self.ori_patch_size
                y_idx = y_coord // self.ori_patch_size
                
                pred_colormap[x_idx, y_idx] = patch_att
                pred_colormap_reg[x_idx, y_idx] = reg_att
            
            # Save ROI masks
            self._save_roi_masks(pred_colormap, pred_colormap_reg, wsi_thumbnail, output_dir)
            
            # Clean up colormaps
            pred_colormap[pred_colormap == -1] = 0.
            pred_colormap_reg[pred_colormap_reg == -1] = 0.
            
            # Save colormaps
            np.save(colormap_path, pred_colormap)
            np.save(f"{output_dir}/prediction_colormap_region.npy", pred_colormap_reg)
            print('Prediction colormap: done')
        else:
            pred_colormap = np.load(colormap_path)
            pred_colormap_reg = np.load(f"{output_dir}/prediction_colormap_region.npy")
        
        # Generate final overlay images
        self._create_overlay_images(pred_colormap, pred_colormap_reg, wsi_thumbnail, output_dir)

    def _save_roi_masks(self, pred_colormap, pred_colormap_reg, wsi_thumbnail, output_dir):
        """Save ROI masks showing extracted patch regions"""
        roi_mask = np.ones_like(pred_colormap, dtype=np.float32)
        roi_mask[pred_colormap == -1] = 0
        roi_mask = resize(roi_mask, (wsi_thumbnail.height, wsi_thumbnail.width))
        Image.fromarray(np.uint8(roi_mask * 255)).save(f"{output_dir}/roi_patch_extraction.jpeg")
        
        roi_mask_reg = np.ones_like(pred_colormap_reg, dtype=np.float32)
        roi_mask_reg[pred_colormap_reg == -1] = 0
        roi_mask_reg = resize(roi_mask_reg, (wsi_thumbnail.height, wsi_thumbnail.width))
        Image.fromarray(np.uint8(roi_mask_reg * 255)).save(f"{output_dir}/roi_reg_extraction.jpeg")

    def _create_overlay_images(self, pred_colormap, pred_colormap_reg, wsi_thumbnail, output_dir):
        """Create final overlay images with attention heatmaps"""
        # Convert thumbnail to numpy array
        wsi_alpha = np.ndarray(
            buffer=wsi_thumbnail.write_to_memory(),
            dtype=np.uint8,
            shape=[wsi_thumbnail.height, wsi_thumbnail.width, wsi_thumbnail.bands])
        
        # Resize and smooth colormaps
        pred_colormap = resize(pred_colormap, (wsi_thumbnail.height, wsi_thumbnail.width))
        pred_colormap = pred_colormap ** 2  # Enhance contrast
        pred_colormap = rank.mean(pred_colormap, disk(21))  # Smooth
        
        pred_colormap_reg = resize(pred_colormap_reg, (wsi_thumbnail.height, wsi_thumbnail.width))
        pred_colormap_reg = pred_colormap_reg ** 2
        pred_colormap_reg = rank.mean(pred_colormap_reg, disk(21))
        
        # Create blended images
        alpha = 0.5
        wsi_blended = ((plt.cm.jet(pred_colormap)[:, :, :3] * 255) * alpha + 
                      wsi_alpha * (1 - alpha)).astype(np.uint8)
        wsi_blended_reg = ((plt.cm.jet(pred_colormap_reg)[:, :, :3] * 255) * alpha + 
                          wsi_alpha * (1 - alpha)).astype(np.uint8)
        
        # Remove background regions
        background_mask = self._create_background_mask(wsi_thumbnail)
        wsi_blended[background_mask == 0] = 255
        wsi_blended_reg[background_mask == 0] = 255
        
        # Save final images
        Image.fromarray(wsi_blended).save(f"{output_dir}/prediction.jpeg")
        Image.fromarray(wsi_blended_reg).save(f"{output_dir}/prediction_region.jpeg")
        print('Thumbnail: done')

    def _create_background_mask(self, wsi_thumbnail):
        """Create mask to exclude background regions"""
        full_slide = wsi_thumbnail.extract_area(0, 0, wsi_thumbnail.width, wsi_thumbnail.height)
        slide_numpy = np.frombuffer(full_slide.write_to_memory(), dtype=np.uint8).reshape(
            full_slide.height, full_slide.width, 3)
        
        # Create mask based on green channel threshold
        background_mask = np.ones((wsi_thumbnail.height, wsi_thumbnail.width))
        background_mask[slide_numpy[:, :, 1] > 240] = 0
        
        # Morphological operations to clean up mask
        background_mask = ndimage.binary_closing(background_mask, structure=np.ones((25, 25)))
        background_mask = ndimage.binary_opening(background_mask, structure=np.ones((25, 25)))
        background_mask = skimage.morphology.remove_small_objects(background_mask.astype(bool), min_size=5000)
        
        return background_mask

    def _normalize_image(self, x):
        """Normalize image for neural network input"""
        x = np.transpose(x, (2, 0, 1))  # Channel first
        x = x / 255.0  # Normalize to [0,1]
        return x.astype('float32')


# Main execution
if __name__ == "__main__":
    # Directory configuration
    dir_out = 'results/nmil/'
    dir_inference = 'inference/'
    dir_images = 'extracted_tiles_of_.../'
    dir_images_regions = 'extracted_tiles_of_..._regions/'
    dir_wsi = 'WSIs/'
    
    # Load test WSI list
    wsi_df = pd.read_excel('patient_set_split.xlsx')
    test_wsi = wsi_df.loc[wsi_df['Set'] == 'Test', 'WSI_OLD'].to_list()
    
    # Load pre-trained models
    backbone = torch.load('contrastive/models/backbone_....pth')
    network = torch.load(f'{dir_out}1_network_weights_best.pth')
    
    # Run inference on all test WSIs
    inference = MILInference(dir_wsi, dir_out, backbone, network, dir_inference, dir_images_regions)
    for wsi_name in reversed(test_wsi):
        inference.infer(current_wsi_subfolder=f"{dir_images}{wsi_name}")