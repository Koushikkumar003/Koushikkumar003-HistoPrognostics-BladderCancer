# =============================
# Contrastive Learning Pipeline
# =============================

import os
import pandas as pd
import torch
import torchvision
import kornia
from losses_con import SupConLoss
from dataset_monoscale import *

# -----------------------------
# Settings & Hyperparameters
# -----------------------------
torch.cuda.empty_cache()
torch.manual_seed(42)

train_gpu = torch.cuda.is_available()
roi_choice = 'front'
mag_choice = '100x'
config_run_list = [0, 1, 2, 3, 4, 5]

# Modes
generate_embedding = True
supervised = True
multi_task_learning = False
retrain_backbone = True

# Hyperparameters
input_shape = (3, 512, 512)
bs = 16
lr = 0.0001
epochs = 10
alpha_ce = 0.5

# -----------------------------
# Load Datasets
# -----------------------------
dir_images = 'extracted_tiles_of_' + roi_choice + '_multiscale/'
clinical_dataframe = pd.read_excel('clinical_data.xlsx')
wsi_list = os.listdir(dir_images)

wsi_df = pd.read_excel('patient_set_split.xlsx')
train_list = wsi_df[wsi_df['Set'] == 'Train']['WSI_OLD'].tolist()
val_list = wsi_df[wsi_df['Set'] == 'Val']['WSI_OLD'].tolist()
test_list = wsi_df[wsi_df['Set'] == 'Test']['WSI_OLD'].tolist()

# Clean lists if only annotated ROIs are desired
if roi_choice == "anno":
    train_list = [x for x in train_list if x in wsi_list]
    val_list = [x for x in val_list if x in wsi_list]
    test_list = [x for x in test_list if x in wsi_list]

# -----------------------------
# Iterate Through Configurations
# -----------------------------
for config in config_run_list:
    print(f"\n{'*'*100}\nRunning Config: {config}\n{'*'*100}")

    weights_directory = 'models/'
    if config == 0:
        out_dir = f"data/embeddings_{roi_choice}_{mag_choice}_imagenet/"
        report_file = f"report_{roi_choice}_{mag_choice}_imagenet.txt"
        retrain_backbone = False
    elif config == 1:
        out_dir = f"data/embeddings_{roi_choice}_{mag_choice}_multi/"
        report_file = f"report_{roi_choice}_{mag_choice}_multi.txt"
    elif config == 2:
        out_dir = "data/embeddings_ce/"
        report_file = "report_ce.txt"
    elif config == 3:
        out_dir = "data/embeddings_supervised/"
        report_file = "report_supervised.txt"
    elif config == 5:
        out_dir = f"data/embeddings_{roi_choice}_{mag_choice}_bcgweak/"
        report_file = f"report_{roi_choice}_{mag_choice}_bcgweak.txt"
    else:
        out_dir = f"data/embeddings_{roi_choice}_{mag_choice}_unsupervised/"
        report_file = f"report_{roi_choice}_{mag_choice}_unsupervised.txt"

    # -------------------------
    # Initialize Model & Data
    # -------------------------
    if retrain_backbone:
        dataset_train = Dataset(dir_images, clinical_dataframe, input_shape, False, False, False, train_list, config)
        train_loader = Generator(dataset_train, bs, shuffle=True, augmentation=False)

    model = torchvision.models.densenet121(pretrained=True)
    modules = list(model.children())[:-1]
    backbone = torch.nn.Sequential(*modules, torch.nn.AdaptiveMaxPool2d(1))

    transforms = torch.nn.Sequential(
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.RandomRotation(degrees=45, p=0.5),
        kornia.augmentation.RandomAffine(degrees=0, scale=(0.95, 1.2), p=0.5),
        kornia.augmentation.RandomAffine(degrees=0, translate=(0.05, 0), p=0.5),
        kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0, p=0.5)
    )

    projection = torch.nn.Sequential(
        torch.nn.Linear(3072, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128)
    )

    classifier = torch.nn.Sequential(
        torch.nn.Linear(3072, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 8)
    )

    Lsimclr = SupConLoss(0.07, 'all', 0.07).cuda()
    Lce = torch.nn.CrossEntropyLoss().cuda()

    # Optimizer
    params = list(backbone.parameters()) + list(projection.parameters())
    if config in [1, 2]:
        params += list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    # Move to GPU
    if train_gpu:
        backbone.cuda()
        projection.cuda()
        classifier.cuda()
        transforms.cuda()

    torch.save(backbone, weights_directory + 'backbone_imagenet.pth')

    # -------------------------
    # Train Model
    # -------------------------
    if retrain_backbone and (config != 0):
        for epoch in range(epochs):
            L_epoch = 0
            for i, (X, Y) in enumerate(train_loader):
                model.train()
                X = torch.tensor(X).cuda().float()
                Y = torch.tensor(Y).cuda().float()

                F_o = torch.squeeze(backbone(X))
                Xa = transforms(X.clone())
                F_a = torch.squeeze(backbone(Xa))

                Fo_proj = torch.nn.functional.normalize(projection(F_o).unsqueeze(1), dim=-1)
                Fa_proj = torch.nn.functional.normalize(projection(F_a).unsqueeze(1), dim=-1)

                if config in [3, 5] and config != 1:
                    loss = Lsimclr(torch.cat([Fo_proj, Fa_proj], 1), labels=Y)
                else:
                    loss = Lsimclr(torch.cat([Fo_proj, Fa_proj], 1))

                if config in [1, 2] and (Y != -1).sum().item() > 0:
                    Xtt = F_o[Y != -1]
                    Ytt = Y[Y != -1].long()
                    label_prop = len(Ytt) / len(Y)
                    y_hat = classifier(Xtt)
                    if not multi_task_learning:
                        loss = alpha_ce * label_prop * Lce(y_hat, Ytt)
                    else:
                        loss += alpha_ce * label_prop * Lce(y_hat, Ytt)

                loss.backward()
                opt.step()
                opt.zero_grad()
                L_epoch += loss.item() / len(train_loader)

                print(f"[Epoch {epoch+1}/{epochs} | Step {i+1}/{len(train_loader)}]: Loss={loss.item():.6f}", end='\r')

            print(f"[Epoch {epoch+1}/{epochs}]: Average Loss={L_epoch:.6f}")
            with open(report_file, 'w') as f:
                f.write(f"Epoch {epoch+1} Loss: {L_epoch:.6f}\n")

            # Save model
            suffix = {
                1: '_multi',
                2: '_ce',
                3: '_supervised',
                5: '_bcgweak'
            }.get(config, '_unsupervised')

            torch.save(backbone, weights_directory + f'backbone_contrastive_{roi_choice}_{mag_choice}{suffix}.pth')
            if config != 2:
                torch.save(projection, weights_directory + f'projection_contrastive_{roi_choice}_{mag_choice}{suffix}.pth')
            if config in [1, 2]:
                torch.save(classifier, weights_directory + f'classifier_contrastive_{roi_choice}_{mag_choice}{suffix}.pth')

    # -------------------------
    # Generate Embeddings
    # -------------------------
    if generate_embedding:
        os.makedirs(out_dir, exist_ok=True)
        suffix = {
            0: '_imagenet', 1: '_multi', 2: '_ce', 3: '_supervised', 5: '_bcgweak'
        }.get(config, '_unsupervised')

        backbone_inference = torch.load(weights_directory + f'backbone_contrastive_{roi_choice}_{mag_choice}{suffix}.pth')
        backbone_inference.eval()

        for wsi in train_list + val_list + test_list:
            wsi_path = os.path.join(dir_images, wsi)
            out_path = os.path.join(out_dir, wsi + '.npy')

            if os.path.exists(out_path) or not os.path.isdir(wsi_path):
                continue

            np.save(out_path, np.zeros((1, 1)))  # Temporary flag
            dataset_wsi = Dataset(wsi_path, clinical_dataframe, input_shape, False, False, True)
            loader_wsi = Generator(dataset_wsi, bs, shuffle=False, augmentation=False)

            embeddings = []
            for X, _ in loader_wsi:
                X = torch.tensor(X).cuda().float()
                with torch.no_grad():
                    feats = backbone_inference(X)
                embeddings.append(feats.cpu().numpy())

            if embeddings:
                np.save(out_path, np.vstack(embeddings))
