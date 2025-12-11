"""
Defines a custom PyTorch Dataset class that loads image data and labels from
a directory structure where images are organized in class-specific folders.
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RajDataset(Dataset):
    """
    RajDataset loads images from a folder layout where root_dir contains class
    subfolders (e.g., root/classA/*.png, root/classB/*.png).
    Returns: (image_tensor, label_index, image_path)
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        img_size=224,
        ensure_exists=True,
    ):
        """
        Args:
            root_dir (str): root folder with subfolders per class (required)
            transform (torchvision.transforms): optional transform; if None a default transform is used
            img_size (int): image resizing size (ViT expects 224 usually)
            ensure_exists (bool): verify that image files exist
        """
        self.img_size = img_size
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.samples = []  # list of tuples (image_path, label_str)
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Walk folder structure and collect potential samples
        classes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        potential_samples = []
        for c in classes:
            cpath = os.path.join(root_dir, c)
            for fname in os.listdir(cpath):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                    potential_samples.append((os.path.join(cpath, fname), c))

        # Validate images and filter out corrupted ones
        corrupted = []
        for img_path, label in potential_samples:
            # Check if file exists
            if ensure_exists and not Path(img_path).exists():
                corrupted.append((img_path, "File not found"))
                continue

            # Try to open the image to verify it's valid
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify it's a valid image
                # Re-open after verify (verify makes the file object unusable)
                with Image.open(img_path) as img:
                    img.convert("RGB")  # Ensure it can be converted to RGB
                self.samples.append((img_path, label))
            except Exception as e:
                corrupted.append((img_path, str(e)))

        # Report any corrupted images
        if corrupted:
            print(f"Warning: Found {len(corrupted)} corrupted/invalid images. They will be skipped.")
            print("First few corrupted files:")
            for img_path, error in corrupted[:5]:
                print(f"  - {img_path}: {error}")

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root_dir}")

        # Build class index
        classes = sorted(list({lbl for _, lbl in self.samples}))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        label_idx = self.class_to_idx[label_str]
        return img_tensor, label_idx, img_path

    def num_classes(self):
        return len(self.class_to_idx)

    def class_names(self):
        return [self.idx_to_class[i] for i in range(self.num_classes())]


if __name__ == "__main__":
    # quick sanity check usage
    home = os.path.expanduser("~")
    ds = RajDataset(
        root_dir=f"{home}/local_data/train", img_size=224, ensure_exists=False
    )
    print("Found classes:", ds.class_names())
    print("Length:", len(ds))
    if len(ds) > 0:
        x, y, p = ds[0]
        print("Sample shape:", x.shape, "label:", y, "path:", p)
