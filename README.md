# Necro-Net

Deep learning pipeline for multi-class pathology image classification, featuring two model architectures — a fine-tuned ResNet50 and a Vision Transformer (ViT-B/16) — with full training, evaluation, and model export support.

## Background

Pathology image analysis is labor-intensive and requires significant domain expertise. When faced with a large corpus of whole-slide pathology images that needed to be classified across multiple tissue categories, I built this pipeline to automate that process reliably and reproducibly.

The result is a modular, production-ready training framework that handles the full ML lifecycle: data loading with augmentation, class-imbalance correction, two-phase fine-tuning (frozen backbone → full fine-tuning), early stopping, learning rate scheduling, comprehensive evaluation metrics, and model export to both ONNX and TorchScript for deployment.

## Project Structure

```
necro-net/
├── ResNet50/
│   └── raj_data_paad/
│       ├── train_classify.py        # ResNet50 training script
│       ├── paad_resnet50_trainer.py # Alternate ResNet50 trainer
│       ├── paad_confusion_matrix.py # Confusion matrix + metrics
│       ├── config.json              # Training configuration
│       ├── utils/
│       │   ├── data_loader.py
│       │   ├── dataset.py
│       │   └── training.py
│       └── tools/
│           ├── convert_to_onnx.py
│           └── convert_to_torchscript.py
├── ViT/
│   ├── train_raj_vit.py   # ViT-B/16 training script
│   ├── evaluation.py      # Unified evaluation for both architectures
│   ├── raj_dataset.py     # Dataset class
│   └── count.py
├── deps/
│   ├── ResNet50.toml
│   └── ViT.toml
├── pyproject_cpu.toml
├── pyproject_gpu.toml
└── sync_env.sh            # Environment setup helper
```

## Features

- **Two architectures**: ResNet50 (ImageNet pretrained) and ViT-B/16, both fine-tuned for multi-class pathology classification
- **Class-imbalance handling**: Automatically computes and applies class weights to the loss function
- **Two-phase fine-tuning**: Backbone frozen for initial epochs, then unfrozen with a reduced learning rate for full fine-tuning
- **Early stopping** with configurable patience
- **Learning rate scheduling** via `ReduceLROnPlateau` (ResNet50) or `StepLR` (ViT)
- **Model export**: Best ResNet50 model exported to both ONNX and TorchScript
- **Comprehensive evaluation**: Classification report, confusion matrix, per-class ROC curves, AUC scores (macro and micro-average), and prediction CSV
- **CPU optimization**: ViT trainer configurable for high-core-count CPU systems (default: 70 threads for 72-core machines)
- **Structured logging** via `loguru`

## Setup

### Prerequisites

- Python >= 3.9
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

### Install dependencies

For GPU:

```bash
./sync_env.sh gpu
```

For CPU-only:

```bash
./sync_env.sh cpu
```

This copies the appropriate `pyproject_gpu.toml` or `pyproject_cpu.toml` to `pyproject.toml` and runs `uv sync`.

## Usage

### Train ResNet50

1. Edit `ResNet50/raj_data_paad/config.json` to set your data paths and hyperparameters:

```json
{
  "train_root": "./local_data/train",
  "test_root":  "./local_data/test",
  "img_size":   [224, 224],
  "batch_size": 32,
  "epochs":     40,
  "lr":         0.0001,
  "unfreeze_epoch":       6,
  "early_stop_patience":  8
}
```

2. Run training from the `ResNet50/raj_data_paad/` directory:

```bash
cd ResNet50/raj_data_paad
uv run train_classify.py
```

Classes are auto-detected from subdirectory names in `train_root`. The best model is saved to `models/DecaResNet_v3.pth` and exported to `DecaResNet_v3.onnx` and `DecaResNet_v3.pt`.

---

### Train ViT

```bash
cd ViT
uv run train_raj_vit.py --epochs 20 --batch_size 32 --output_dir models
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--root_dir` | `~/local_data/train` | Dataset root (class subfolders) |
| `--epochs` | 10 | Max training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 3e-5 | Learning rate |
| `--val_split` | 0.1 | Fraction held out for validation |
| `--patience` | 5 | Early stopping patience |
| `--cpu_threads` | 70 | PyTorch CPU thread count |
| `--output_dir` | `models` | Where to save checkpoints |

Best model saved as `models/vit_best.pth`.

---

### Evaluate a Model

The unified evaluation script works with either architecture:

```bash
cd ViT

# Evaluate ResNet50
uv run evaluation.py --checkpoint ../ResNet50/raj_data_paad/models/DecaResNet_v3.pth --architecture resnet50

# Evaluate ViT
uv run evaluation.py --checkpoint models/vit_best.pth --architecture vit
```

Outputs:

- `preds.csv` — per-image predictions
- `confusion_matrix.png` — heatmap
- `roc_curves.png` — per-class ROC curves
- `roc_curves_with_micro.png` — ROC curves with micro-average
- `auc_scores.png` — AUC bar chart
- Console: full classification report with precision, recall, F1, macro/micro AUC

## Dataset Layout

Organize images in the standard PyTorch `ImageFolder` format:

```
local_data/
├── train/
│   ├── class_a/
│   │   ├── image1.png
│   │   └── image2.png
│   └── class_b/
│       └── image3.png
└── test/
    ├── class_a/
    └── class_b/
```

## License

See [LICENSE](LICENSE).

<br>
