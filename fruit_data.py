from __future__ import print_function
from PIL import Image
import os
import numpy as np
import torch.utils.data as data

class Fruit(data.Dataset):
    
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.train = train

        if self.train:
            self.data = np.load(os.path.join(self.root_dir, "train_data.npy"))
            self.labels = np.load(os.path.join(self.root_dir, "train_labels.npy"))
        else:
            # Check if validation files exist
            val_data_path = os.path.join(self.root_dir, "validation_data.npy")
            val_labels_path = os.path.join(self.root_dir, "validation_labels.npy")
            if os.path.exists(val_data_path) and os.path.exists(val_labels_path):
                self.data = np.load(val_data_path)
                self.labels = np.load(val_labels_path)
            else:
                print("⚠️  Validation data not found! Using training data for testing.")
                self.data = np.load(os.path.join(self.root_dir, "train_data.npy"))
                self.labels = np.load(os.path.join(self.root_dir, "train_labels.npy"))

        # Fix shape if single image or already in correct format
        if len(self.data.shape) == 4:
            # multiple images: (N, C, H, W) -> (N, H, W, C)
            self.data = self.data.transpose((0, 2, 3, 1))
        elif len(self.data.shape) == 3:
            # single image: (H, W, C) -> (1, H, W, C)
            self.data = self.data[np.newaxis, :, :, :]
        elif len(self.data.shape) == 2:
            # grayscale single image: (H, W) -> (1, H, W, 1)
            self.data = self.data[np.newaxis, :, :, np.newaxis]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
