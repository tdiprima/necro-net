"""
Loads a pre-trained Vision Transformer (ViT) model for image classification,
performs inference on a dataset, and outputs predictions and optional evaluation
metrics (e.g., accuracy, confusion matrix, ROC curve) to a CSV file.
uv run eval_raj_vit.py --checkpoint models/vit_best.pth --output_csv preds.csv
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from rich_argparse import RichHelpFormatter
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from raj_dataset import RajDataset


def load_vit_model(num_classes, device, checkpoint_path):
    try:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
    except Exception:
        model = models.vit_b_16(pretrained=True)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)

    model = model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def main():
    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser(
        description="Evaluate ViT on Raj dataset",
        formatter_class=RichHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=f"{home}/local_data/test",
        help="root folder with class subfolders",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to model state (vit_best.pth or full ckpt)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_csv", type=str, default="preds.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    ds = RajDataset(
        root_dir=args.root_dir,
        img_size=args.img_size,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = load_vit_model(ds.num_classes(), device, args.checkpoint)
    print("Loaded model and dataset. Running inference...")

    all_preds = []
    all_labels = []
    all_paths = []
    all_probs = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, ncols=120):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)
            all_probs.extend(probs)

    # save CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "true_label",
                "pred_label",
                "true_label_str",
                "pred_label_str",
            ]
        )
        for p, t, pr in zip(all_paths, all_labels, all_preds):
            writer.writerow(
                [p, int(t), int(pr), ds.idx_to_class[int(t)], ds.idx_to_class[int(pr)]]
            )

    print(f"Saved predictions to {args.output_csv}")

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Print classification report
    print("\nClassification report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=ds.class_names(), digits=4
        )
    )

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion matrix:")
    print(cm)

    # Save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ds.class_names(),
        yticklabels=ds.class_names(),
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved confusion_matrix.png")

    # Calculate ROC curves and AUC scores for each class
    num_classes = ds.num_classes()
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Save ROC curves visualization
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{ds.class_names()[i]} (AUC = {roc_auc[i]:.4f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - One-vs-Rest")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=300)
    plt.close()
    print("Saved roc_curves.png")

    # Save AUC scores bar chart
    plt.figure(figsize=(12, 6))
    class_names = ds.class_names()
    auc_scores = [roc_auc[i] for i in range(num_classes)]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, num_classes))
    bars = plt.bar(range(num_classes), auc_scores, color=colors, edgecolor="black")
    plt.xlabel("Class")
    plt.ylabel("AUC Score")
    plt.title("AUC Scores by Class")
    plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
    plt.ylim([0, 1.0])
    plt.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("auc_scores.png", dpi=300)
    plt.close()
    print("Saved auc_scores.png")

    # Print average AUC
    avg_auc = np.mean(auc_scores)
    print(f"\nAverage AUC: {avg_auc:.4f}")


if __name__ == "__main__":
    main()
