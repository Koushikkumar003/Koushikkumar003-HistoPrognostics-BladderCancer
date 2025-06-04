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

        self.images_400x = []
        self.images_100x = []
        self.images_25x = []
        self.Y = []

        if inference:
            self._load_inference()
        else:
            self._load_training()

        self.indexes = np.arange(len(self.images_400x))
        self.Y = np.array(self.Y)

        if self.preallocate:
            self._preload_images()

    def _load_inference(self):
        """
        Load image paths for inference from the given dataset directory.
        Assumes subfolders 400x, 100x, and 25x with corresponding tiles.
        """
        base_csv = next((f for f in os.listdir(self.dir_dataset) if f.endswith('.csv')), None)
        if base_csv is None:
            print(f'[WARN] No CSV found in {self.dir_dataset}')
            return

        tile_info = pd.read_csv(os.path.join(self.dir_dataset, base_csv))
        folders = {res: os.path.join(self.dir_dataset, f"{res}") for res in ['400x', '100x', '25x']}
        filenames_400x = os.listdir(folders['400x'])

        for fname_400x in filenames_400x:
            try:
                x, y = map(int, fname_400x.replace('.jpeg', '').split('_'))
                row = tile_info[
                    (tile_info['400X_coor'] == x) & (tile_info['400Y_coor'] == y)
                ].iloc[0]

                coors = {
                    '400x': fname_400x,
                    '100x': f"{int(row['100X_coor'])}_{int(row['100Y_coor'])}.jpeg",
                    '25x': f"{int(row['25X_coor'])}_{int(row['25Y_coor'])}.jpeg"
                }

                paths = {
                    k: os.path.join(v, coors[k]) for k, v in folders.items()
                }

                for path in paths.values():
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"[ERROR] Missing file: {path}")

                self.images_400x.append(paths['400x'])
                self.images_100x.append(paths['100x'])
                self.images_25x.append(paths['25x'])
                self.Y.append(-1)

            except Exception as e:
                print(f"[WARN] Skipping tile {fname_400x}: {e}")

    def _load_training(self):
        """
        Load and label training data across all WSI folders.
        """
        for idx, wsi in enumerate(os.listdir(self.dir_dataset)):
            wsi_path = os.path.join(self.dir_dataset, wsi)
            print(f'{idx + 1}/{len(self.wsi_list)}: {wsi}')

            try:
                csv_file = next(f for f in os.listdir(wsi_path) if f.endswith('.csv'))
                tile_info = pd.read_csv(os.path.join(wsi_path, csv_file))
            except StopIteration:
                print(f'[WARN] No CSV in {wsi_path}, skipping.')
                continue

            folders = {
                res: os.path.join(wsi_path, f"{res}") for res in ['400x', '100x', '25x']
            }
            filenames_400x = os.listdir(folders['400x'])

            try:
                bcg_label = 1 if self.data_frame.loc[self.data_frame['SID'] == int(wsi), 'BCG_failure'].item() == 'Yes' else 0
            except:
                bcg_label = -1
