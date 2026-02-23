# PAAD ResNet50 Model Parameters

Configuration parameters defined in the `Config` class of `paad_resnet50_trainer.py`.

---

## Training Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `BATCH_SIZE` | `32` | Batch size for DataLoader |
| `LEARNING_RATE` | `1e-4` | Initial LR for fine-tuning |
| `NUM_EPOCHS` | `65` | Maximum training epochs |
| `WEIGHT_DECAY` | `1e-4` | L2 regularization coefficient |

## Learning Rate Scheduler

| Parameter | Value | Description |
|---|---|---|
| `LR_PATIENCE` | `3` | Epochs before reducing LR |
| `LR_FACTOR` | `0.5` | Multiplier when LR is reduced |

## Early Stopping

| Parameter | Value | Description |
|---|---|---|
| `EARLY_STOPPING_PATIENCE` | `7` | Epochs without improvement before stopping |

## DataLoader Settings

| Parameter | Value | Description |
|---|---|---|
| `NUM_WORKERS` | `4` | Parallel data loading workers |
| `PIN_MEMORY` | `True` | Faster CPU→GPU transfers |
| `NON_BLOCKING` | `True` | Async data transfers |

## Model Settings

| Parameter | Value | Description |
|---|---|---|
| `NUM_CLASSES` | `10` | Output classes |
| `PRETRAINED` | `True` | Use ImageNet pretrained weights |
| `FREEZE_BACKBONE_EPOCHS` | `3` | Epochs to freeze backbone before full fine-tuning |

## Image / Normalization Settings

| Parameter | Value | Description |
|---|---|---|
| `IMAGE_SIZE` | `224` | Input image dimensions (224×224) |
| `IMAGENET_MEAN` | `[0.485, 0.456, 0.406]` | Per-channel normalization mean |
| `IMAGENET_STD` | `[0.229, 0.224, 0.225]` | Per-channel normalization std |

---

## Notes

- When the backbone is unfrozen at epoch `FREEZE_BACKBONE_EPOCHS + 1`, the learning rate is automatically reduced to `LEARNING_RATE / 10` = `1e-5` for full fine-tuning (see `paad_resnet50_trainer.py`, line 599).
- On CPU-only systems, `NUM_WORKERS`, `PIN_MEMORY`, and `NON_BLOCKING` are all overridden to `0` / `False` at runtime.
