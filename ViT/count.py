import os

from raj_dataset import RajDataset

home = os.path.expanduser("~")
ds = RajDataset(root_dir=f"{home}/local_data/train")
print(f"Total images: {len(ds)}")  # 72102
print(f"Classes: {ds.num_classes()}")  # 10

# find $HOME/local_data/train -name "*.png" | wc -l
