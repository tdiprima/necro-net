"""
PAAD ResNet50 Trainer
=====================
Pancreatic Adenocarcinoma (PAAD) histopathology image classification using ResNet50.
Classifies 224x224 PNG images into 10 tissue/cancer types.

Combines best practices for PyTorch training including:
- Transfer learning with pretrained ResNet50
- GPU acceleration with CUDA support
- Pinned memory for faster data transfers
- Non-blocking async transfers
- Multi-worker data loading
- Data augmentation for medical images
- Corrupt image handling (automatically skips corrupted files)
- Saves both best model and final model

OUTPUT MODELS:
  - paad_resnet50_best.pth    <-- USE THIS FOR PREDICTIONS (best validation accuracy)
  - paad_resnet50_final.pth   (final epoch checkpoint, for reference only)
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Training configuration parameters."""

    # Data paths
    TRAIN_DIR = os.path.expanduser("~/local_data/train")
    TEST_DIR = os.path.expanduser("~/local_data/test")

    # Output paths
    OUTPUT_DIR = os.path.expanduser("~/local_data/output")
    LOG_FILE = os.path.expanduser("~/local_data/output/training_log.txt")
    BEST_MODEL_PATH = os.path.expanduser("~/local_data/output/paad_resnet50_best.pth")
    FINAL_MODEL_PATH = os.path.expanduser("~/local_data/output/paad_resnet50_final.pth")
    ROC_PLOT_PATH = os.path.expanduser("~/local_data/output/roc_curves.png")

    # Training hyperparameters
    BATCH_SIZE = 32  # Smaller batch for ResNet50 memory requirements
    LEARNING_RATE = 1e-4  # Lower LR for fine-tuning pretrained models
    NUM_EPOCHS = 65  # More epochs for medical imaging convergence
    WEIGHT_DECAY = 1e-4  # L2 regularization

    # Learning rate scheduler
    LR_PATIENCE = 3  # Reduce LR after 3 epochs without improvement
    LR_FACTOR = 0.5  # Reduce LR by half

    # Early stopping
    EARLY_STOPPING_PATIENCE = 7  # Stop if no improvement for 7 epochs

    # DataLoader settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    NON_BLOCKING = True

    # Model settings
    NUM_CLASSES = 10
    PRETRAINED = True
    FREEZE_BACKBONE_EPOCHS = 3  # Freeze backbone for first N epochs

    # Image settings (ResNet50 expects 224x224)
    IMAGE_SIZE = 224

    # ImageNet normalization (required for pretrained ResNet50)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# LOGGING
# =============================================================================


class Logger:
    """Logger that writes to both file and optionally stdout."""

    def __init__(self, log_file, also_print=False):
        self.log_file = log_file
        self.also_print = also_print
        # Ensure output directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # Clear/create log file
        with open(log_file, "w") as f:
            f.write(
                f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 70 + "\n\n")

    def log(self, message):
        """Write message to log file."""
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
        if self.also_print:
            print(message)

    def log_separator(self, char="=", width=70):
        """Write a separator line."""
        self.log(char * width)


# =============================================================================
# SYSTEM INFORMATION
# =============================================================================


def log_system_info(logger):
    """Display system and PyTorch configuration."""
    logger.log_separator()
    logger.log("PAAD RESNET50 TRAINER")
    logger.log("Pancreatic Adenocarcinoma Histopathology Classification")
    logger.log_separator()
    logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"PyTorch Version: {torch.__version__}")
    logger.log(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.log(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.log(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        logger.log(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.log("WARNING: Using CPU - Training will be significantly slower")

    logger.log(f"Training Device: {device}")
    logger.log_separator()
    return device


# =============================================================================
# CORRUPT IMAGE HANDLING
# =============================================================================


class SafeImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder that gracefully handles corrupt images.
    Skips corrupt images during training instead of crashing.
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
            # print(f"Warning: Skipping corrupt image: {path} ({str(e)})")
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


def get_data_transforms(config):
    """
    Define data transformations for PAAD histopathology images.
    Training: augmentation to improve generalization
    Testing: only normalization
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            # Data augmentation for medical images
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    return train_transform, test_transform


def load_datasets(config, train_transform, test_transform, logger):
    """Load PAAD training and test datasets using SafeImageFolder with corrupt image handling."""
    logger.log("\nLoading PAAD dataset...")
    logger.log(f"  Train directory: {config.TRAIN_DIR}")
    logger.log(f"  Test directory: {config.TEST_DIR}")

    train_dataset = SafeImageFolder(root=config.TRAIN_DIR, transform=train_transform)

    test_dataset = SafeImageFolder(root=config.TEST_DIR, transform=test_transform)

    # Log class information
    logger.log(f"\nFound {len(train_dataset.classes)} classes:")
    for idx, class_name in enumerate(train_dataset.classes):
        # Count samples per class
        class_count = sum(1 for _, label in train_dataset.samples if label == idx)
        logger.log(f"  [{idx}] {class_name}: {class_count} samples")

    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, config):
    """Create optimized DataLoaders with pinned memory and workers, handling corrupt images."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        drop_last=True,  # Drop incomplete batches for stable batch norm
        collate_fn=collate_fn_skip_corrupt,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn_skip_corrupt,
    )

    return train_loader, test_loader


def compute_class_weights(dataset, logger):
    """
    Compute class weights for imbalanced datasets using SOFT weighting.
    
    Uses square-root inverse frequency: weight = sqrt(max_count / class_count)
    This is gentler than full inverse frequency, balancing:
    - Better minority class recognition (important for rare tissue types)
    - Maintaining good overall accuracy (not over-penalizing majority classes)
    
    For medical imaging, this approach helps catch rare but clinically 
    important findings without sacrificing too much overall performance.
    """
    import math
    
    num_classes = len(dataset.classes)
    len(dataset)

    # Count samples per class
    class_counts = [0] * num_classes
    for _, label in dataset.samples:
        class_counts[label] += 1

    max_count = max(class_counts)
    
    # Compute weights using SOFT square-root formula
    # This gives Tumor ~1.0 and rare classes ~7x (vs 50x with full inverse)
    class_weights = []
    for count in class_counts:
        if count > 0:
            weight = math.sqrt(max_count / count)
        else:
            weight = 0.0
        class_weights.append(weight)

    # Convert to tensor
    class_weights = torch.FloatTensor(class_weights)

    # Log the weights
    logger.log("\nClass Weights (SOFT square-root weighting for medical imaging):")
    logger.log("  (Balances minority class detection with overall accuracy)")
    for idx, (class_name, count, weight) in enumerate(
        zip(dataset.classes, class_counts, class_weights)
    ):
        logger.log(f"  [{idx}] {class_name}: {count} samples, weight={weight:.4f}")

    return class_weights


# =============================================================================
# MODEL DEFINITION
# =============================================================================


def create_resnet50_model(num_classes, pretrained=True):
    """
    Create ResNet50 model for PAAD classification.
    Replaces final fully connected layer for 10-class output.
    """
    # Load pretrained ResNet50
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Replace the final fully connected layer
    # ResNet50's fc layer input features: 2048
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_features, num_classes))

    return model


def freeze_backbone(model):
    """Freeze all layers except the final classifier."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================


def train_epoch(
    model, device, train_loader, optimizer, criterion, epoch, logger, non_blocking=False
):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    num_batches = len(train_loader)
    log_interval = max(1, num_batches // 5)  # Log ~5 times per epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        # Skip empty batches (all images corrupt)
        if data.numel() == 0:
            continue

        # Transfer data to device with non-blocking option
        data = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Progress logging
        if (batch_idx + 1) % log_interval == 0:
            current_acc = 100.0 * correct / total
            logger.log(
                f"  Batch {batch_idx + 1}/{num_batches} | "
                f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion, non_blocking=False):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            # Skip empty batches (all images corrupt)
            if data.numel() == 0:
                continue

            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            probs = torch.softmax(output, dim=1)
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    import numpy as np
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return avg_loss, accuracy, auc


def save_model(model, path, config, epoch, accuracy, class_names, logger):
    """Save model checkpoint with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "accuracy": accuracy,
        "num_classes": config.NUM_CLASSES,
        "class_names": class_names,
        "image_size": config.IMAGE_SIZE,
        "imagenet_mean": config.IMAGENET_MEAN,
        "imagenet_std": config.IMAGENET_STD,
    }

    torch.save(checkpoint, path)
    logger.log(f"  Saved model to: {path}")


# =============================================================================
# ROC CURVE PLOTTING
# =============================================================================


def plot_roc_curves(model, device, test_loader, class_names, output_path, non_blocking=False):
    """Generate and save a per-class ROC curve PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            if data.numel() == 0:
                continue
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    n_classes = len(class_names)
    targets_bin = label_binarize(all_targets, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(targets_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — PAAD ResNet50")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def main():
    """Main training function."""
    config = Config()

    # Create output directory
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize logger (writes to file only)
    logger = Logger(config.LOG_FILE, also_print=False)

    # Also print a simple message to stdout
    print("PAAD ResNet50 Training Started")
    print(f"Log file: {config.LOG_FILE}")

    # Log system information and get device
    device = log_system_info(logger)

    # Print device info to stdout
    if torch.cuda.is_available():
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: CPU (WARNING: Training will be slow)")
    print("Training in progress...")

    # Adjust workers for CPU-only systems
    if not torch.cuda.is_available():
        config.NUM_WORKERS = 0
        config.PIN_MEMORY = False
        config.NON_BLOCKING = False

    # Log configuration
    logger.log("\nConfiguration:")
    logger.log(f"  Batch Size: {config.BATCH_SIZE}")
    logger.log(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.log(f"  Weight Decay: {config.WEIGHT_DECAY}")
    logger.log(f"  Epochs: {config.NUM_EPOCHS}")
    logger.log(f"  Freeze Backbone Epochs: {config.FREEZE_BACKBONE_EPOCHS}")
    logger.log(f"  LR Scheduler Patience: {config.LR_PATIENCE}")
    logger.log(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.log(f"  Workers: {config.NUM_WORKERS}")
    logger.log(f"  Pin Memory: {config.PIN_MEMORY}")
    logger.log(f"  Non-blocking Transfer: {config.NON_BLOCKING}")
    logger.log(f"  Pretrained: {config.PRETRAINED}")

    # Load data
    train_transform, test_transform = get_data_transforms(config)
    train_dataset, test_dataset = load_datasets(
        config, train_transform, test_transform, logger
    )
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, config)

    logger.log("\nDataset Summary:")
    logger.log(f"  Training samples: {len(train_dataset)}")
    logger.log(f"  Test samples: {len(test_dataset)}")
    logger.log(f"  Training batches: {len(train_loader)}")
    logger.log(f"  Test batches: {len(test_loader)}")

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(train_dataset, logger).to(device)

    # Initialize model
    logger.log("\nInitializing ResNet50 model...")
    model = create_resnet50_model(config.NUM_CLASSES, pretrained=config.PRETRAINED).to(device)

    # Initially freeze backbone for transfer learning warmup
    if config.FREEZE_BACKBONE_EPOCHS > 0:
        freeze_backbone(model)
        logger.log(
            f"  Backbone frozen for first {config.FREEZE_BACKBONE_EPOCHS} epochs"
        )

    # Loss function with class weights for imbalanced dataset
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer - only optimize trainable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",  # Maximize accuracy
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("\nModel Parameters:")
    logger.log(f"  Total: {total_params:,}")
    logger.log(f"  Trainable: {trainable_params:,}")

    # Training loop
    logger.log("\n")
    logger.log_separator()
    logger.log("TRAINING")
    logger.log_separator()

    start_time = time()
    best_accuracy = 0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time()

        # Unfreeze backbone after warmup period
        if epoch == config.FREEZE_BACKBONE_EPOCHS + 1:
            logger.log("\n>>> Unfreezing backbone for full fine-tuning <<<")
            unfreeze_backbone(model)
            # Re-initialize optimizer with all parameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE / 10,  # Lower LR for fine-tuning
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=config.LR_FACTOR,
                patience=config.LR_PATIENCE,
            )
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.log(f"  Trainable parameters: {trainable_params:,}")
            logger.log(f"  New learning rate: {config.LEARNING_RATE / 10:.2e}\n")

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log(f"\nEpoch {epoch}/{config.NUM_EPOCHS} (LR: {current_lr:.2e})")
        logger.log("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            epoch,
            logger,
            non_blocking=config.NON_BLOCKING,
        )

        # Evaluate
        test_loss, test_acc, test_auc = evaluate(
            model, device, test_loader, criterion, non_blocking=config.NON_BLOCKING
        )

        epoch_time = time() - epoch_start

        logger.log(f"\n  Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%")
        logger.log(f"  Test Loss:  {test_loss:.6f} | Test Acc:  {test_acc:.2f}% | AUC: {test_auc:.4f}")
        logger.log(f"  Epoch Time: {epoch_time:.1f}s")
        print(f"  Epoch {epoch}/{config.NUM_EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | AUC: {test_auc:.4f} | Time: {epoch_time:.1f}s")

        # Update learning rate scheduler
        scheduler.step(test_acc)

        # Check for improvement
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save best model
            logger.log("\n  *** New best accuracy! Saving best model ***")
            save_model(
                model,
                config.BEST_MODEL_PATH,
                config,
                epoch,
                test_acc,
                train_dataset.classes,
                logger,
            )
        else:
            epochs_without_improvement += 1
            logger.log(f"  No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping check
        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            logger.log(f"\n>>> Early stopping triggered after {epoch} epochs <<<")
            break

    total_time = time() - start_time

    # Save final model
    logger.log("\n")
    logger.log_separator()
    logger.log("SAVING FINAL MODEL")
    logger.log_separator()

    save_model(
        model,
        config.FINAL_MODEL_PATH,
        config,
        epoch,
        test_acc,
        train_dataset.classes,
        logger,
    )

    # Final results
    logger.log("\n")
    logger.log_separator()
    logger.log("TRAINING COMPLETE")
    logger.log_separator()

    logger.log("\nTiming:")
    logger.log(
        f"  Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    logger.log(f"  Average Time per Epoch: {total_time / epoch:.2f} seconds")

    logger.log("\nBest Results:")
    logger.log(f"  Best Test Accuracy: {best_accuracy:.2f}%")
    logger.log(f"  Best Epoch: {best_epoch}")

    # Final evaluation
    final_test_loss, final_test_acc, final_test_auc = evaluate(
        model, device, test_loader, criterion, non_blocking=config.NON_BLOCKING
    )
    logger.log("\nFinal Epoch Results:")
    logger.log(f"  Final Test Loss: {final_test_loss:.6f}")
    logger.log(f"  Final Test Accuracy: {final_test_acc:.2f}%")
    logger.log(f"  Final Test AUC (macro OvR): {final_test_auc:.4f}")

    # Save ROC curve plot
    plot_roc_curves(
        model, device, test_loader, train_dataset.classes,
        config.ROC_PLOT_PATH, non_blocking=config.NON_BLOCKING,
    )
    logger.log(f"\nROC curve saved to: {config.ROC_PLOT_PATH}")
    print(f"  ROC curve saved to: {config.ROC_PLOT_PATH}")

    logger.log("\n")
    logger.log_separator()
    logger.log("MODEL FILES")
    logger.log_separator()
    logger.log(f"\n>>> FOR PREDICTIONS, USE: {config.BEST_MODEL_PATH}")
    logger.log(
        f"    (Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch})"
    )
    logger.log(f"\n    Final epoch model: {config.FINAL_MODEL_PATH}")
    logger.log(f"    (Final accuracy: {final_test_acc:.2f}%, AUC: {final_test_auc:.4f} - for reference only)")
    logger.log_separator()

    logger.log(
        f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Print completion message to stdout
    print("\nTraining Complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best accuracy: {best_accuracy:.2f}%")
    print(f"  Final AUC (macro OvR): {final_test_auc:.4f}")
    print(f"  Log file: {config.LOG_FILE}")
    print(f"\n>>> FOR PREDICTIONS, USE: {config.BEST_MODEL_PATH}")

    return model


if __name__ == "__main__":
    model = main()
