# training script
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger

from utils.data_loader import create_data_loaders_from_separate_datasets
from utils.dataset import ImgDataset
from utils.training import train_epoch, validate


def main():
    # Setup logging
    logger.add("training.log", rotation="10 MB", retention="7 days", level="INFO")
    logger.info("Starting training script")

    with open("../config.json", "r") as f:
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

    # model = models.resnet50(pretrained=True)
    # model = models.resnet50(weights=None)  # No pretrained weights
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    logger.info("Model initialized: ResNet50 with ImageNet pretrained weights")

    best_acc = 0
    early_stop_patience = 5
    early_stop_counter = 0
    best_model_state = None
    logger.info(f"Starting training for {EPOCHS} epochs")

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Starting epoch {epoch}/{EPOCHS}")
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, acc = validate(model, val_loader, loss_fn, device)

        scheduler.step(val_loss)

        epoch_msg = f"Epoch {epoch}: train_loss {tr_loss:.4f} val_loss {val_loss:.4f} val_acc {acc:.4f}"
        print(epoch_msg)
        logger.info(epoch_msg)

        if acc > best_acc:
            best_acc = acc
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(
                best_model_state, os.path.join("models", "DecaResNet_v2.pth")
            )
            save_msg = f"New best accuracy: {acc:.4f}! Saved DecaResNet_v2.pth"
            print("Saved DecaResNet_v2.pth")
            logger.info(save_msg)
        else:
            early_stop_counter += 1
            logger.info(f"Current accuracy {acc:.4f} < best accuracy {best_acc:.4f} (early stop counter: {early_stop_counter}/{early_stop_patience})")

        if early_stop_counter >= early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            print(f"Early stopping triggered after {epoch} epochs")
            break

        logger.info(f"Completed epoch {epoch}/{EPOCHS}")

    # export to ONNX
    logger.info("Starting ONNX export")
    dummy = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    torch.onnx.export(
        model, dummy, os.path.join("models", "DecaResNet_v2.onnx"), opset_version=11
    )
    print("Exported DecaResNet_v2.onnx")
    logger.info("ONNX export completed successfully")

    # export to TorchScript
    logger.info("Starting TorchScript export")
    traced_model = torch.jit.trace(model, dummy)
    traced_model.save(os.path.join("models", "DecaResNet_v2.pt"))
    print("Exported DecaResNet_v2.pt")
    logger.info("TorchScript export completed successfully")
    logger.info(f"Training completed! Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
