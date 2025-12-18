"""
Loads images from class-specific folders into a PyTorch Dataset with configurable
augmentations, normalization, and corrupted-image handling for training and validation.
"""

import os
from glob import glob

import PIL
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, root_dir, classes, img_size=(224, 224), augment=True):
        self.samples = []
        for cls in classes:
            files = glob(os.path.join(root_dir, cls, "*"))
            for f in files:
                self.samples.append((f, classes.index(cls)))
        self.img_size = img_size
        self.augment = augment
        # Lighter augmentation for histopathology (per help.md recommendations)
        self.transform_train = T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_val = T.Compose(
            [
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        try:
            img = Image.open(p).convert("RGB")
            img = self.transform_train(img) if self.augment else self.transform_val(img)
            return img, label
        except (OSError, IOError, PIL.UnidentifiedImageError) as e:
            # Skip corrupted or empty images by trying the next one
            print(f"Warning: Skipping corrupted image {p}: {e}")
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)
