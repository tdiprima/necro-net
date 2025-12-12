# training script
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger
from utils.data_loader import create_data_loaders_from_separate_datasets
from utils.dataset import ImgDataset
from utils.training import train_epoch, validate


def freeze_backbone(model):
    """Freeze all layers except the final fc layer."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def main():
    # Setup logging
    logger.add("training.log", rotation="10 MB", retention="7 days", level="INFO")
    logger.info("Starting ResNet50 training script")

    with open("config.json", "r") as f:
        config = json.load(f)

    TRAIN_ROOT = config["train_root"]
    TEST_ROOT = config["test_root"]
    CLASSES = sorted(
        [d.name for d in Path(TRAIN_ROOT).iterdir() if d.is_dir()]
    )  # auto-detect from train directory
    IMG_SIZE = tuple(config["img_size"])
    BATCH = config["batch_size"]
    EPOCHS = config["epochs"]
    LR = config["lr"]
    UNFREEZE_EPOCH = config.get("unfreeze_epoch", 6)
    EARLY_STOP_PATIENCE = config.get("early_stop_patience", 8)

    logger.info(
        f"Configuration loaded: {EPOCHS} epochs, batch size {BATCH}, learning rate {LR}"
    )
    logger.info(f"Classes detected: {CLASSES}")
    logger.info(f"Image size: {IMG_SIZE}")

    # build separate datasets
    train_dataset = ImgDataset(TRAIN_ROOT, CLASSES, img_size=IMG_SIZE, augment=True)
    val_dataset = ImgDataset(TEST_ROOT, CLASSES, img_size=IMG_SIZE, augment=False)
    train_loader, val_loader = create_data_loaders_from_separate_datasets(
        train_dataset, val_dataset, BATCH
    )

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # model = models.resnet50(weights=None)  # No pretrained weights
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)
    # Compute class weights for imbalanced data
    class_counts = [0] * len(CLASSES)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    class_weights = torch.tensor(
        [total / c for c in class_counts], dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum() * len(CLASSES)  # normalize
    logger.info(f"Class counts: {dict(zip(CLASSES, class_counts))}")
    logger.info(f"Class weights: {class_weights.tolist()}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.3, patience=2, min_lr=1e-6
    )

    logger.info("Model initialized: ResNet50 with ImageNet pretrained weights")

    # Freeze backbone for initial epochs
    freeze_backbone(model)
    logger.info(f"Backbone frozen. Will unfreeze at epoch {UNFREEZE_EPOCH}")

    best_acc = 0
    early_stop_counter = 0
    best_model_state = None
    logger.info(
        f"Starting training for {EPOCHS} epochs (early stop patience: {EARLY_STOP_PATIENCE})"
    )

    training_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze backbone after initial frozen epochs
        if epoch == UNFREEZE_EPOCH:
            unfreeze_backbone(model)
            # Lower LR when unfreezing for fine-tuning
            for param_group in opt.param_groups:
                param_group["lr"] = LR * 0.1
            logger.info(f"Backbone unfrozen at epoch {epoch}. LR reduced to {LR * 0.1}")

        logger.info(f"Starting epoch {epoch}/{EPOCHS}")
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, acc = validate(model, val_loader, loss_fn, device)

        scheduler.step(acc)  # Step on validation accuracy, not loss

        epoch_msg = f"Epoch {epoch}: train_loss {tr_loss:.4f} val_loss {val_loss:.4f} val_acc {acc:.4f}"
        print(epoch_msg)
        logger.info(epoch_msg)

        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join("models", "DecaResNet_v3.pth"))
            save_msg = f"New best accuracy: {acc:.4f}! Saved DecaResNet_v3.pth"
            print("Saved DecaResNet_v3.pth")
            logger.info(save_msg)
        else:
            early_stop_counter += 1
            logger.info(
                f"Current accuracy {acc:.4f} < best accuracy {best_acc:.4f} (early stop counter: {early_stop_counter}/{EARLY_STOP_PATIENCE})"
            )

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            print(f"Early stopping triggered after {epoch} epochs")
            break

        logger.info(f"Completed epoch {epoch}/{EPOCHS}")

    total_training_time = time.time() - training_start_time
    minutes, seconds = divmod(total_training_time, 60)
    hours, minutes = divmod(minutes, 60)
    time_msg = (
        f"Training completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    )
    print(time_msg)
    logger.info(time_msg)

    # export to ONNX
    logger.info("Starting ONNX export")
    dummy = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    torch.onnx.export(
        model, dummy, os.path.join("models", "DecaResNet_v3.onnx"), opset_version=11
    )
    print("Exported DecaResNet_v3.onnx")
    logger.info("ONNX export completed successfully")

    # export to TorchScript
    logger.info("Starting TorchScript export")
    traced_model = torch.jit.trace(model, dummy)
    traced_model.save(os.path.join("models", "DecaResNet_v3.pt"))
    print("Exported DecaResNet_v3.pt")
    logger.info("TorchScript export completed successfully")
    logger.info(f"Training completed! Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
