import torch
from tqdm import tqdm


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for imgs, lbls in tqdm(loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, lbls)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()
    total = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
            total += loss.item()
            correct += (preds.argmax(1) == lbls).sum().item()
            total_samples += imgs.size(0)
    return total / len(loader), correct / total_samples
