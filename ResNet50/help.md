## How to realistically get back to ~95.4% (do this in order)

### **Step 1: Add a learning-rate scheduler (this is non-negotiable)**

This alone can be worth **+3–4%**.

Best low-effort options:

**Option A: ReduceLROnPlateau (safest)**

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)
```

Trigger it on **validation accuracy**, not loss.

**Option B: Cosine annealing (if you feel spicy)**

```python
CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
```

If you do *nothing else*, do this.

---

### **Step 2: Train longer (30–40 epochs)**

Your model was **still improving at epoch 15**.

What I'd do:

* Set max epochs = **40**
* Early stopping patience = **6–8**

Histopath models often jump late once LR drops.

---

### **Step 3: Unfreeze the backbone (if you didn't already)**

If you kept ResNet frozen the whole time, that's a ceiling.

Best practice:

1. Epochs 1–5: freeze backbone
2. Epochs 6+: unfreeze everything
3. Lower LR when you unfreeze:

```python
lr = 1e-4
```

This is another **+2–3%** lever.

---

### **Step 4: Slightly better augmentation (don't go crazy)**

Histology hates aggressive augmentation, but mild stuff helps:

Recommended:

* RandomHorizontalFlip
* RandomVerticalFlip
* ColorJitter (very light)
* RandomResizedCrop (scale 0.9–1.0)

Avoid:

* Heavy rotations
* CutMix/MixUp (usually hurts morphology)

This is often worth **+1–2%**.

---

### **Step 5: Check class imbalance (sneaky accuracy killer)**

If one class is underrepresented:

* Use **class-weighted CrossEntropyLoss**
* Or **WeightedRandomSampler**

This can quietly add **+1–3%**, especially if "Tumor" or "Necrosis" is rare.
