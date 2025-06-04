import os
import torch
import numpy as np
import random
from PIL import Image
import pandas as pd
import math


class Dataset:
    def __init__(self, dir_dataset, data_frame, input_shape=(3, 512, 512),
                 augmentation=True, preallocate=True, inference=False,
                 wsi_list=[], config=0):

        self.dir_dataset = dir_dataset
        self.data_frame = data_frame
        self.input_shape = input_shape
        self.augmentation = augmentation
        self.preallocate = preallocate
        self.inference = inference
        self.wsi_list = wsi_list
        self.config = config

        self.images = []
        self.Y = []

        # Inference Mode: Just load images, no labels
        if self.inference:
            csv_files = [f for f in os.listdir(dir_dataset) if f.endswith('.csv')]
            if not csv_files:
                print(f'[WARN] No CSV file found in {dir_dataset}.')
            else:
                for name in os.listdir(dir_dataset):
                    self.images.append(os.path.join(dir_dataset, name))
                    self.Y.append(-1)

        else:  # Training Mode
            for wsi_count, path in enumerate(os.listdir(dir_dataset)):
                full_path = os.path.join(dir_dataset, path)
                print(f'{wsi_count + 1}/{len(self.wsi_list)}: {path}')

                csv_files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
                if not csv_files:
                    print(f'[WARN] No CSV file in {path}. Skipping.')
                    continue

                # Load tile information and labels
                tile_info_path = os.path.join(full_path, csv_files[0])
                tile_info_df = pd.read_csv(tile_info_path)

                try:
                    bcg_status = self.data_frame.loc[self.data_frame['SID'] == int(path), 'BCG_failure'].item()
                    bcg_label = 1 if bcg_status == 'Yes' else 0
                except Exception:
                    bcg_label = -1

                for name in os.listdir(full_path):
                    image_path = os.path.join(full_path, name)
                    self.images.append(image_path)

                    if self.config in [1, 2, 3]:
                        try:
                            x_coord, y_coord = map(int, name.split('_')[:2])
                            row = tile_info_df[(tile_info_df['X_coor'] == x_coord) &
                                               (tile_info_df['Y_coor'] == y_coord)]

                            grade = row['Grade'].item() if not row['Grade'].isna().all() else None
                            til = row['TILs'].item() if not row['TILs'].isna().all() else None

                            if pd.notna(grade):
                                self.Y.append(int(grade))
                            elif pd.notna(til):
                                self.Y.append(int(til))
                            else:
                                self.Y.append(-1)
                        except Exception:
                            self.Y.append(-1)

                    elif self.config == 4:
                        self.Y.append(-1)

                    elif self.config == 5:
                        self.Y.append(bcg_label)

        self.indexes = np.arange(len(self.images))
        self.Y = np.array(self.Y)

        # Preload images into RAM
        if self.preallocate:
            self.X = np.zeros((len(self.images), *self.input_shape), dtype=np.float32)
            print('[INFO]: Preloading images into RAM...')
            for i in self.indexes:
                try:
                    x = Image.open(self.images[i]).convert('RGB')
                    x = np.asarray(x)
                    x = np.transpose(x, (2, 0, 1)) / 255.0  # Normalize
                    self.X[i] = x
                except Exception as e:
                    print(f'[ERROR] Failed to load image {self.images[i]}: {e}')
            print('[INFO]: Preloading complete.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.preallocate:
            x = self.X[index]
        else:
            try:
                x = Image.open(self.images[index]).convert('RGB')
                x = np.asarray(x)
                x = np.transpose(x, (2, 0, 1)) / 255.0
            except Exception as e:
                print(f'[ERROR] Failed to load image {self.images[index]}: {e}')
                x = np.zeros(self.input_shape)

        y = self.Y[index]

        return torch.tensor(x, dtype=torch.float32).cuda(), torch.tensor(y, dtype=torch.float32).cuda()


class Generator:
    def __init__(self, dataset, batch_size, shuffle=True, augmentation=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = np.arange(len(self.dataset))
        self._idx = 0
        self._reset()

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.indexes):
            self._reset()
            raise StopIteration()

        end_idx = min(self._idx + self.batch_size, len(self.indexes))
        batch_indexes = self.indexes[self._idx:end_idx]
        self._idx = end_idx

        X, Y = [], []
        for idx in batch_indexes:
            x, y = self.dataset[idx]
            X.append(x.unsqueeze(0))
            Y.append(y.unsqueeze(0))

        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)

        # Optional: Apply dataset-level transforms
        if self.augmentation and hasattr(self.dataset, 'transforms'):
            X = self.dataset.transforms(X)

        return X, Y

    def _reset(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self._idx = 0
