# Necro Net

I built this because I had to train models on real pathology images.  Testing ResNet50 versus ViT.

---

This repo trains image classifiers on pancreatic adenocarcinoma (PAAD) histology slides using two different architectures: a fine-tuned ResNet50 and a Vision Transformer (ViT). Both start from ImageNet pretrained weights and adapt to the target classes.

## ResNet50

The ResNet50 trainer lives in `ResNet50/raj_data_paad/`. It uses class-balanced loss, early stopping, and learning rate scheduling. The best model is exported to ONNX and TorchScript.

Configure paths and hyperparameters in `config.json`, then run:

```bash
cd ResNet50/raj_data_paad
uv run train_classify.py
```

## Vision Transformer (ViT)

The ViT trainer is in `ViT/`. It defaults to CPU-optimized settings (70 threads) but works on GPU too.

```bash
cd ViT
uv run train_raj_vit.py --epochs 8 --batch_size 16 --output_dir models
```

Override defaults as needed:

```bash
uv run train_raj_vit.py --epochs 20 --patience 5 --cpu_threads 72 --num_workers 20
```

## Setup

CPU:

```bash
uv sync --project pyproject_cpu.toml
```

GPU (CUDA):

```bash
uv sync --project pyproject_gpu.toml
```

Requires Python 3.10+.

<br>
