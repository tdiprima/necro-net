"""
PAAD Model Evaluation - Confusion Matrix & Per-Class Metrics
=============================================================
Generates detailed evaluation metrics for the trained PAAD ResNet50 model:
- Confusion matrix (saved as PNG)
- Per-class precision, recall, F1-score
- Overall accuracy and macro/weighted averages

Usage:
    python paad_confusion_matrix.py
"""

import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - adjust these if needed
MODEL_PATH = os.path.expanduser("~/local_data/output/paad_resnet50_best.pth")
TEST_DIR = os.path.expanduser("~/local_data/test")
OUTPUT_DIR = os.path.expanduser("~/local_data/output")

# Settings
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_model(model_path, num_classes, device):
    """Load the trained ResNet50 model."""
    # Create model architecture
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_features, num_classes))

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {model_path}")
    print(f"  Model accuracy at save: {checkpoint['accuracy']:.2f}%")
    print(f"  Saved at epoch: {checkpoint['epoch']}")

    return model, checkpoint.get("class_names", None)


# =============================================================================
# CORRUPT IMAGE HANDLING
# =============================================================================


class SafeImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder that gracefully handles corrupt images.
    Skips corrupt images during evaluation instead of crashing.
    """

    def __getitem__(self, index):
        """
        Override __getitem__ to catch and handle corrupt images.
        Returns None for corrupt images which are filtered out by collate_fn.
        """
        path, target = self.samples[index]
        try:
            # Load image
            sample = self.loader(path)
            if sample is None:
                return None

            # Apply transforms
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target

        except Exception:
            print(f"Warning: Skipping corrupt image: {path}")
            return None


def collate_fn_skip_corrupt(batch):
    """
    Custom collate function that filters out None values from corrupt images.
    """
    # Filter out None values (corrupt images)
    batch = [item for item in batch if item is not None]

    # If entire batch is corrupt, return empty tensors
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])

    # Use default collate for valid samples
    return torch.utils.data.dataloader.default_collate(batch)


# =============================================================================
# DATA LOADING
# =============================================================================


def get_test_loader(test_dir):
    """Create test data loader with corrupt image handling."""
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    test_dataset = SafeImageFolder(root=test_dir, transform=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn_skip_corrupt,
    )

    return test_loader, test_dataset.classes


# =============================================================================
# EVALUATION
# =============================================================================


def get_predictions(model, test_loader, device):
    """Get all predictions and true labels, skipping corrupt images."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            # Skip empty batches (all images were corrupt)
            if len(data) == 0:
                continue

            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(target.numpy())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred, class_names):
    """Compute per-class and overall metrics."""
    len(class_names)

    # Per-class metrics
    metrics = {}
    for i, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = np.sum((y_pred == i) & (y_true == i))
        fp = np.sum((y_pred == i) & (y_true != i))
        fn = np.sum((y_pred != i) & (y_true == i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = np.sum(y_true == i)

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    # Overall accuracy
    accuracy = np.sum(y_pred == y_true) / len(y_true)

    # Macro averages (unweighted mean across classes)
    macro_precision = np.mean([m["precision"] for m in metrics.values()])
    macro_recall = np.mean([m["recall"] for m in metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in metrics.values()])

    # Weighted averages (weighted by support)
    total_support = sum(m["support"] for m in metrics.values())
    weighted_precision = (
        sum(m["precision"] * m["support"] for m in metrics.values()) / total_support
    )
    weighted_recall = (
        sum(m["recall"] * m["support"] for m in metrics.values()) / total_support
    )
    weighted_f1 = sum(m["f1"] * m["support"] for m in metrics.values()) / total_support

    return metrics, {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def create_confusion_matrix(y_true, y_pred, num_classes):
    """Create confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Shorten class names for display (remove "400p-" prefix)
    short_names = [name.replace("400p-", "") for name in class_names]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short_names,
        yticklabels=short_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="PAAD Classification - Confusion Matrix",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")


def save_report(metrics, overall, class_names, cm, output_path):
    """Save detailed text report."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PAAD ResNet50 Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy:           {overall['accuracy']*100:.2f}%\n")
        f.write(f"  Macro Precision:    {overall['macro_precision']*100:.2f}%\n")
        f.write(f"  Macro Recall:       {overall['macro_recall']*100:.2f}%\n")
        f.write(f"  Macro F1-Score:     {overall['macro_f1']*100:.2f}%\n")
        f.write(f"  Weighted F1-Score:  {overall['weighted_f1']*100:.2f}%\n\n")

        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Class':<35} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n"
        )
        f.write("-" * 80 + "\n")

        for class_name in class_names:
            m = metrics[class_name]
            short_name = class_name.replace("400p-", "")
            f.write(
                f"{short_name:<35} {m['precision']*100:>9.2f}% {m['recall']*100:>9.2f}% "
                f"{m['f1']*100:>9.2f}% {m['support']:>10d}\n"
            )

        f.write("-" * 80 + "\n\n")

        # Confusion matrix as text
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write("(Rows = True Labels, Columns = Predicted Labels)\n\n")

        # Header
        short_names = [name.replace("400p-", "")[:8] for name in class_names]
        f.write(f"{'':>20} ")
        for name in short_names:
            f.write(f"{name:>8} ")
        f.write("\n")

        # Matrix rows
        for i, class_name in enumerate(class_names):
            short_name = class_name.replace("400p-", "")[:20]
            f.write(f"{short_name:>20} ")
            for j in range(len(class_names)):
                f.write(f"{cm[i, j]:>8d} ")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")

    print(f"Detailed report saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("PAAD Model Evaluation - Confusion Matrix & Metrics")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print(f"\nLoading test data from: {TEST_DIR}")
    test_loader, class_names = get_test_loader(TEST_DIR)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes, {len(test_loader.dataset)} test samples")

    # Load model
    print("\nLoading model...")
    model, saved_class_names = load_model(MODEL_PATH, num_classes, device)

    # Use saved class names if available
    if saved_class_names:
        class_names = saved_class_names

    # Get predictions
    print("\nRunning inference on test set...")
    y_pred, y_true = get_predictions(model, test_loader, device)
    print(f"Processed {len(y_true)} samples")

    # Compute metrics
    print("\nComputing metrics...")
    metrics, overall = compute_metrics(y_true, y_pred, class_names)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nOverall Accuracy: {overall['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:   {overall['macro_f1']*100:.2f}%")
    print("\nPer-Class Performance:")
    print("-" * 60)
    print(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)

    for class_name in class_names:
        m = metrics[class_name]
        short_name = class_name.replace("400p-", "")[:28]
        print(
            f"{short_name:<30} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% {m['f1']*100:>9.1f}%"
        )

    # Create confusion matrix
    cm = create_confusion_matrix(y_true, y_pred, num_classes)

    # Save outputs
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cm_image_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_image_path)

    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    save_report(metrics, overall, class_names, cm, report_path)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
