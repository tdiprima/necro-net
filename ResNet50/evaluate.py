# evaluation script
import json
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset


class SafeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        for cls in self.classes:
            cls_path = Path(root_dir) / cls
            if os.path.isdir(cls_path):
                files = glob(str(cls_path / "*"))
                for f in files:
                    self.samples.append((f, self.classes.index(cls)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        try:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except (OSError, IOError, PIL.UnidentifiedImageError) as e:
            print(f"Warning: Skipping corrupted image {p}: {e}")
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)


# ---- config ----
with open("../config.json", "r") as f:
    config = json.load(f)

VAL_DIR = Path(f"{config['test_root']}")
MODEL_PATH = Path("./models/DecaResNet_v2.pth")
IMG_SIZE = tuple(config["img_size"])
BATCH = config["batch_size"]

# ---- dataset ----
transform = T.Compose(
    [
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_ds = SafeImageDataset(VAL_DIR, transform=transform)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- rebuild model ----
n_classes = len(val_ds.classes)
# model = models.resnet34(pretrained=False)
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ---- run evaluation ----
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ---- metrics ----
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Generate classification report
class_report = classification_report(
    all_labels, all_preds, target_names=val_ds.classes, digits=3
)
print("📊 Classification Report:")
print(class_report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# ---- ROC Curves and AUC ----
# Binarize labels for multiclass ROC
y_bin = label_binarize(all_labels, classes=range(n_classes))
if n_classes == 2:
    y_bin = np.column_stack([1 - y_bin, y_bin])

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), all_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# ---- Save results to files ----
results_dir = "../outputs/evaluation_results"
Path(results_dir).mkdir(parents=True, exist_ok=True)

# Save classification report
with Path(results_dir) / "classification_report.txt".open("w") as f:
    f.write("Classification Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(class_report)
    f.write(f"\n\nModel: {MODEL_PATH}\n")
    f.write(f"Test Dataset: {VAL_DIR}\n")
    f.write(f"Number of classes: {n_classes}\n")
    f.write(f"Class names: {val_ds.classes}\n")

# Save AUC scores
with Path(results_dir) / "auc_scores.txt".open("w") as f:
    f.write("AUC Scores\n")
    f.write("=" * 30 + "\n\n")
    for i, class_name in enumerate(val_ds.classes):
        f.write(f"{class_name}: {roc_auc[i]:.4f}\n")
    f.write(f"\nMicro-average AUC: {roc_auc['micro']:.4f}\n")

# Save confusion matrix data
np.savetxt(Path(results_dir) / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
np.savetxt(
    Path(results_dir) / "confusion_matrix_normalized.csv",
    cm_norm,
    delimiter=",",
    fmt="%.4f",
)

# ---- Plot confusion matrix ----
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    xticklabels=val_ds.classes,
    yticklabels=val_ds.classes,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(Path(results_dir) / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- Plot ROC curves ----
plt.figure(figsize=(12, 8))
colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

# Plot ROC curve for each class
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label=f"{val_ds.classes[i]} (AUC = {roc_auc[i]:.3f})",
    )

# Plot micro-average ROC curve
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

# Plot random classifier line
plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Multi-class Classification")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(results_dir) / "roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- Plot AUC bar chart ----
plt.figure(figsize=(10, 6))
class_names = val_ds.classes
auc_scores = [roc_auc[i] for i in range(n_classes)]

bars = plt.bar(class_names, auc_scores, color=colors[: len(class_names)], alpha=0.7)
plt.xlabel("Classes")
plt.ylabel("AUC Score")
plt.title("AUC Scores by Class")
plt.ylim([0, 1])
plt.xticks(rotation=45, ha="right")

# Add value labels on bars
for bar, score in zip(bars, auc_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.3f}",
        ha="center",
        va="bottom",
    )

plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(Path(results_dir) / "auc_scores.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n📁 All results saved to '{results_dir}/' directory:")
print("  - classification_report.txt")
print("  - confusion_matrix.png")
print("  - confusion_matrix.csv")
print("  - confusion_matrix_normalized.csv")
print("  - roc_curves.png")
print("  - auc_scores.txt")
print("  - auc_scores.png")
print(f"\n🎯 Micro-average AUC: {roc_auc['micro']:.4f}")
