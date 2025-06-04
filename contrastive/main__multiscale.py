"""
Multi-Scale Contrastive Learning for Medical Image Analysis
===========================

import os
import pandas as pd
import torch
import torchvision
import kornia
import numpy as np
from losses_con import SupConLoss
from dataset_multiscale import Dataset, Generator

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

def setup_environment():
    """Initialize CUDA and set random seeds for reproducibility."""
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(42)
    return torch.cuda.is_available()

def load_data_splits():
    """Load train/validation/test splits from Excel file."""
    wsi_df = pd.read_excel('patient_set_split.xlsx')
    
    train_list = wsi_df.loc[wsi_df['Set'] == 'Train', 'WSI_OLD'].to_list()
    val_list = wsi_df.loc[wsi_df['Set'] == 'Val', 'WSI_OLD'].to_list()
    test_list = wsi_df.loc[wsi_df['Set'] == 'Test', 'WSI_OLD'].to_list()
    
    return train_list, val_list, test_list

def get_config_info(config):
    """Return configuration details for different training modes."""
    config_map = {
        0: 'ImageNet backbone weights (no training)',
        1: 'Multi-task: Unsupervised Contrastive + Binary CrossEntropy',
        2: 'Supervised Classification with Binary CrossEntropy only',
        3: 'Supervised Contrastive Loss',
        4: 'Unsupervised Contrastive Loss',
        5: 'Supervised Contrastive Loss with BCG weak labels'
    }
    return config_map.get(config, 'Unknown configuration')

def get_output_paths(config, roi_choice):
    """Generate output directory and filename based on configuration."""
    path_map = {
        0: (f'data/embeddings_{roi_choice}_multi_imagenet/', f'report_{roi_choice}_multi_imagenet.txt'),
        1: (f'data/embeddings_{roi_choice}_multi_multi/', f'report_{roi_choice}_multi_multi.txt'),
        2: ('data/embeddings_ce/', 'report_ce.txt'),
        3: ('data/embeddings_supervised/', 'report_supervised.txt'),
        4: (f'data/embeddings_{roi_choice}_multi_unsupervised/', f'report_{roi_choice}_multi_unsupervised.txt'),
        5: (f'data/embeddings_{roi_choice}_multi_bcgweak/', f'report_{roi_choice}_multi_bcgweak.txt')
    }
    return path_map.get(config, path_map[4])  # Default to unsupervised

# =============================================================================
# MODEL ARCHITECTURE SETUP
# =============================================================================

def create_backbone_models():
    """Create three identical DenseNet121 backbones for different magnifications."""
    base_model = torchvision.models.densenet121(pretrained=True)
    modules = list(base_model.children())[:-1]
    
    # Create separate backbones for each magnification level
    backbone_400x = torch.nn.Sequential(*modules, torch.nn.AdaptiveMaxPool2d(1))
    backbone_100x = torch.nn.Sequential(*modules, torch.nn.AdaptiveMaxPool2d(1))
    backbone_25x = torch.nn.Sequential(*modules, torch.nn.AdaptiveMaxPool2d(1))
    
    return backbone_400x, backbone_100x, backbone_25x

def create_augmentation_pipeline():
    """Create data augmentation pipeline using Kornia."""
    return torch.nn.Sequential(
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.RandomRotation(degrees=45, p=0.5),
        kornia.augmentation.RandomAffine(degrees=0, scale=(0.95, 1.20), p=0.5),
        kornia.augmentation.RandomAffine(degrees=0, translate=(0.05, 0), p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0., p=0.5),
    )

def create_projection_head():
    """Create projection head for contrastive learning."""
    return torch.nn.Sequential(
        torch.nn.Linear(3072, 128),  # 3072 = 1024 * 3 (concatenated features from 3 scales)
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128)
    )

def create_classifier():
    """Create classifier head for supervised tasks."""
    return torch.nn.Sequential(
        torch.nn.Linear(3072, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 8)  # 8 classes
    )

def setup_models_on_gpu(models, use_gpu):
    """Move all models to GPU if available."""
    if use_gpu:
        for model in models:
            model.cuda()

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def extract_multiscale_features(backbones, inputs):
    """Extract features from all three magnification levels and concatenate."""
    backbone_400x, backbone_100x, backbone_25x = backbones
    X400, X100, X25 = inputs
    
    # Extract features from each scale
    F_400x = torch.squeeze(backbone_400x(X400))
    F_100x = torch.squeeze(backbone_100x(X100))
    F_25x = torch.squeeze(backbone_25x(X25))
    
    # Concatenate features from all scales
    return torch.cat((F_400x, F_100x, F_25x), 1)

def compute_contrastive_loss(features_orig, features_aug, projection, contrastive_loss_fn, labels, config):
    """Compute contrastive loss between original and augmented features."""
    # Project features to lower dimensional space
    proj_orig = projection(features_orig).unsqueeze(1)
    proj_aug = projection(features_aug).unsqueeze(1)
    
    # Normalize projected features
    proj_orig = torch.nn.functional.normalize(proj_orig, dim=-1)
    proj_aug = torch.nn.functional.normalize(proj_aug, dim=-1)
    
    # Combine original and augmented projections
    combined_features = torch.cat([proj_orig, proj_aug], 1)
    
    # Use labels for supervised contrastive learning (configs 3, 5)
    if config in [3, 5]:
        return contrastive_loss_fn(combined_features, labels=labels, mask=None)
    else:
        # Unsupervised contrastive learning
        return contrastive_loss_fn(combined_features, labels=None, mask=None)

def compute_classification_loss(features, labels, classifier, ce_loss_fn, alpha_ce):
    """Compute classification loss for labeled samples."""
    # Only use samples that have labels (not -1)
    labeled_mask = labels != -1
    
    if not labeled_mask.any():
        return torch.tensor(0.0).cuda()
    
    # Extract labeled samples
    labeled_features = features[labeled_mask]
    labeled_targets = labels[labeled_mask]
    
    # Forward pass through classifier
    predictions = classifier(labeled_features)
    if len(labeled_targets) == 1:
        predictions = predictions.unsqueeze(0)
    
    # Compute weighted classification loss
    label_proportion = len(labeled_targets) / len(labels)
    loss = alpha_ce * label_proportion * ce_loss_fn(predictions, labeled_targets.long())
    
    return loss

def train_single_epoch(train_generator, models, optimizers, loss_functions, config, epoch, total_epochs):
    """Train for one epoch."""
    backbone_400x, backbone_100x, backbone_25x, projection, classifier, transforms = models
    optimizer = optimizers
    contrastive_loss_fn, ce_loss_fn = loss_functions
    
    epoch_loss = 0.0
    alpha_ce = 0.5  # Weight for classification loss
    
    for i_iteration, (X400, X100, X25, Y) in enumerate(train_generator):
        # Set models to training mode
        for model in [backbone_400x, backbone_100x, backbone_25x, projection, classifier]:
            model.train()
        
        # Move data to GPU
        X400 = torch.tensor(X400).cuda().float()
        X100 = torch.tensor(X100).cuda().float()
        X25 = torch.tensor(X25).cuda().float()
        Y = torch.tensor(Y).cuda().float()
        
        # Extract multi-scale features
        features_orig = extract_multiscale_features((backbone_400x, backbone_100x, backbone_25x), (X400, X100, X25))
        
        total_loss = torch.tensor(0.0).cuda()
        
        # Contrastive learning (all configs except pure classification)
        if config != 2:
            # Apply augmentations
            X400_aug = transforms(X400.clone())
            X100_aug = transforms(X100.clone())
            X25_aug = transforms(X25.clone())
            
            # Extract features from augmented data
            features_aug = extract_multiscale_features((backbone_400x, backbone_100x, backbone_25x), (X400_aug, X100_aug, X25_aug))
            
            # Compute contrastive loss
            contrastive_loss = compute_contrastive_loss(features_orig, features_aug, projection, contrastive_loss_fn, Y, config)
            total_loss += contrastive_loss
        
        # Classification loss (for configs 1 and 2)
        if config in [1, 2]:
            classification_loss = compute_classification_loss(features_orig, Y, classifier, ce_loss_fn, alpha_ce)
            
            if config == 2:  # Pure classification
                total_loss = classification_loss
            else:  # Multi-task learning
                total_loss += classification_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update epoch loss
        epoch_loss += total_loss.cpu().detach().numpy() / len(train_generator)
        
        # Print progress
        print(f"[INFO] Epoch {epoch+1}/{total_epochs} -- Step {i_iteration+1}/{len(train_generator)}: Loss={total_loss.cpu().detach().numpy():.6f}", end='\r')
    
    print(f"[INFO] Epoch {epoch+1}/{total_epochs} completed: Average Loss={epoch_loss:.6f}")
    return epoch_loss

def save_model_weights(models, config, roi_choice, weights_directory):
    """Save trained model weights."""
    backbone_400x, backbone_100x, backbone_25x, projection, classifier = models
    
    # Define filename suffixes based on configuration
    suffix_map = {
        1: f'_{roi_choice}_multi_multi',
        2: f'_{roi_choice}_multi_ce',
        3: f'_{roi_choice}_multi_supervised',
        4: f'_{roi_choice}_multi_unsupervised',
        5: f'_{roi_choice}_multi_bcgweak'
    }
    
    suffix = suffix_map.get(config, f'_{roi_choice}_multi_unsupervised')
    
    # Save backbone models
    torch.save(backbone_400x, f'{weights_directory}backbone_400x_contrastive{suffix}.pth')
    torch.save(backbone_100x, f'{weights_directory}backbone_100x_contrastive{suffix}.pth')
    torch.save(backbone_25x, f'{weights_directory}backbone_25x_contrastive{suffix}.pth')
    
    # Save additional components based on configuration
    if config in [1, 3, 4, 5]:  # Configs that use projection head
        torch.save(projection, f'{weights_directory}projection_contrastive{suffix}.pth')
    
    if config in [1, 2]:  # Configs that use classifier
        torch.save(classifier, f'{weights_directory}classifier_contrastive{suffix}.pth')

# =============================================================================
# INFERENCE AND EMBEDDING GENERATION
# =============================================================================

def load_trained_models(config, roi_choice, weights_directory):
    """Load trained model weights for inference."""
    if config == 0:  # ImageNet weights
        backbone_400x = torch.load(f'{weights_directory}backbone_400x_imagenet.pth')
        backbone_100x = torch.load(f'{weights_directory}backbone_100x_imagenet.pth')
        backbone_25x = torch.load(f'{weights_directory}backbone_25x_imagenet.pth')
    else:
        # Load trained weights based on configuration
        suffix_map = {
            1: f'_{roi_choice}_multi_multi',
            2: f'_{roi_choice}_multi_ce',
            3: f'_{roi_choice}_multi_supervised',
            4: f'_{roi_choice}_multi_unsupervised',
            5: f'_{roi_choice}_multi_bcgweak'
        }
        suffix = suffix_map.get(config, f'_{roi_choice}_multi_unsupervised')
        
        backbone_400x = torch.load(f'{weights_directory}backbone_400x_contrastive{suffix}.pth')
        backbone_100x = torch.load(f'{weights_directory}backbone_100x_contrastive{suffix}.pth')
        backbone_25x = torch.load(f'{weights_directory}backbone_25x_contrastive{suffix}.pth')
    
    return backbone_400x, backbone_100x, backbone_25x

def generate_embeddings_for_wsi(wsi_folder, backbones, batch_size, input_shape, clinical_dataframe):
    """Generate embeddings for a single WSI."""
    backbone_400x, backbone_100x, backbone_25x = backbones
    
    # Set models to evaluation mode
    backbone_400x.eval()
    backbone_100x.eval()
    backbone_25x.eval()
    
    # Create dataset for this WSI
    dataset_wsi = Dataset(wsi_folder, clinical_dataframe, input_shape=input_shape, 
                         augmentation=False, preallocate=False, inference=True)
    wsi_generator = Generator(dataset_wsi, batch_size, shuffle=False, augmentation=False)
    
    embeddings_list = []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for (X400, X100, X25, Y) in wsi_generator:
            # Move data to GPU
            X400 = torch.tensor(X400).cuda().float()
            X100 = torch.tensor(X100).cuda().float()
            X25 = torch.tensor(X25).cuda().float()
            
            # Extract features
            features = extract_multiscale_features(backbones, (X400, X100, X25))
            
            # Handle single sample case
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            # Store embeddings
            embeddings_list.append(features.cpu().numpy())
    
    if embeddings_list:
        return np.vstack(embeddings_list)
    else:
        return None

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Main training and inference pipeline."""
    
    # Configuration
    ROI_CHOICE = 'front'  # Region of interest
    CONFIG_RUN_LIST = [0, 1, 2, 3, 4, 5]  # Which configurations to run
    GENERATE_EMBEDDINGS = True  # Whether to generate embeddings after training
    RETRAIN_BACKBONE = True  # Whether to train backbone or use pretrained
    
    # Hyperparameters
    INPUT_SHAPE = (3, 512, 512)
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    EPOCHS = 10
    
    # Data paths
    IMAGES_DIR = f'extracted_tiles_of_{ROI_CHOICE}_multiscale/'
    WEIGHTS_DIR = 'models/'
    
    # Setup
    use_gpu = setup_environment()
    clinical_dataframe = pd.read_excel('clinical_data.xlsx')
    train_list, val_list, test_list = load_data_splits()
    
    # Filter annotated slides if needed
    if ROI_CHOICE == "anno":
        available_wsi = os.listdir(IMAGES_DIR)
        train_list = [wsi for wsi in train_list if wsi in available_wsi]
        val_list = [wsi for wsi in val_list if wsi in available_wsi]
        test_list = [wsi for wsi in test_list if wsi in available_wsi]
    
    # Run each configuration
    for config in CONFIG_RUN_LIST:
        print('=' * 100)
        print(f'Running Configuration {config}: {get_config_info(config)}')
        print('=' * 100)
        
        # Get output paths
        output_dir, info_filename = get_output_paths(config, ROI_CHOICE)
        
        # Create models
        backbone_400x, backbone_100x, backbone_25x = create_backbone_models()
        transforms = create_augmentation_pipeline()
        projection = create_projection_head()
        classifier = create_classifier()
        
        # Setup GPU
        models = [backbone_400x, backbone_100x, backbone_25x, projection, transforms, classifier]
        setup_models_on_gpu(models, use_gpu)
        
        # Save initial ImageNet weights
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)
        torch.save(backbone_400x, f'{WEIGHTS_DIR}backbone_400x_imagenet.pth')
        torch.save(backbone_100x, f'{WEIGHTS_DIR}backbone_100x_imagenet.pth')
        torch.save(backbone_25x, f'{WEIGHTS_DIR}backbone_25x_imagenet.pth')
        
        # Training (skip for config 0 - ImageNet only)
        if RETRAIN_BACKBONE and config != 0:
            # Setup data loader
            dataset_train = Dataset(IMAGES_DIR, clinical_dataframe, input_shape=INPUT_SHAPE,
                                  augmentation=False, preallocate=False, inference=False,
                                  wsi_list=train_list, config=config)
            train_generator = Generator(dataset_train, BATCH_SIZE, shuffle=True, augmentation=False)
            
            # Setup optimizer
            if config in [1, 2]:  # Configs that use classifier
                trainable_params = (list(backbone_400x.parameters()) + list(backbone_100x.parameters()) + 
                                  list(backbone_25x.parameters()) + list(projection.parameters()) + 
                                  list(classifier.parameters()))
            else:
                trainable_params = (list(backbone_400x.parameters()) + list(backbone_100x.parameters()) + 
                                  list(backbone_25x.parameters()) + list(projection.parameters()))
            
            optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=trainable_params)
            
            # Setup loss functions
            contrastive_loss = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07).cuda()
            ce_loss = torch.nn.CrossEntropyLoss().cuda()
            
            # Training loop
            for epoch in range(EPOCHS):
                epoch_loss = train_single_epoch(
                    train_generator, 
                    (backbone_400x, backbone_100x, backbone_25x, projection, classifier, transforms),
                    optimizer,
                    (contrastive_loss, ce_loss),
                    config, epoch, EPOCHS
                )
                
                # Save training info
                with open(info_filename, 'w') as f:
                    f.write(f"Epoch {epoch+1}/{EPOCHS}: Loss={epoch_loss:.6f}")
            
            # Save trained weights
            save_model_weights((backbone_400x, backbone_100x, backbone_25x, projection, classifier), 
                             config, ROI_CHOICE, WEIGHTS_DIR)
        
        # Generate embeddings
        if GENERATE_EMBEDDINGS:
            print("Generating embeddings...")
            
            # Create output directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Load trained models
            inference_backbones = load_trained_models(config, ROI_CHOICE, WEIGHTS_DIR)
            setup_models_on_gpu(inference_backbones, use_gpu)
            
            # Process all WSIs
            all_wsi_list = train_list + val_list + test_list
            
            for wsi_name in all_wsi_list:
                print(f"Processing {wsi_name}...")
                
                # Skip if embeddings already exist
                output_path = os.path.join(output_dir, f'{wsi_name}.npy')
                if os.path.exists(output_path):
                    continue
                
                wsi_folder = os.path.join(IMAGES_DIR, wsi_name)
                
                # Skip if no tiles available
                if ROI_CHOICE == 'anno' and not os.path.isdir(wsi_folder):
                    continue
                if not os.path.exists(os.path.join(wsi_folder, '400x')) or len(os.listdir(os.path.join(wsi_folder, '400x'))) == 0:
                    continue
                
                # Generate embeddings
                embeddings = generate_embeddings_for_wsi(wsi_folder, inference_backbones, BATCH_SIZE, INPUT_SHAPE, clinical_dataframe)
                
                if embeddings is not None:
                    np.save(output_path, embeddings)
                    print(f"Saved embeddings for {wsi_name}: {embeddings.shape}")
        
        print(f"Configuration {config} completed!")

if __name__ == "__main__":
    main()