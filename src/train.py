"""
Train the 4-class recycling classifier (plastic, paper, glass, metal)
using transfer learning on MobileNetV2.

Usage:
    python src/train.py --data data/ --epochs 15 --out models/classifier.pt

Expects data/train, data/val, data/test, each with 4 class subfolders
(output of prepare_data.py).
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASS_NAMES = ["glass", "metal", "paper", "plastic"]  # alphabetical, matches ImageFolder


def build_dataloaders(data_dir: Path, batch_size: int, img_size: int = 224):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tfms)

    assert train_ds.classes == sorted(train_ds.classes)
    print(f"Classes (index order): {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes: int, device: torch.device):
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        pretrained = True
    except Exception as e:
        # Some sandboxed/offline environments block download.pytorch.org.
        # Fall back to training from scratch rather than failing outright
        # -- on an unrestricted machine the pretrained path above will work.
        print(f"Could not download pretrained weights ({e}); training from scratch instead.")
        model = models.mobilenet_v2(weights=None)
        pretrained = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    if pretrained:
        # Freeze the pretrained feature extractor, only train the new head.
        # This is the right call for a small dataset (~400-600 images/class)
        # and a 2-week timeline: far less overfitting risk than fine-tuning
        # the whole network, and trains in minutes on CPU.
        for param in model.features.parameters():
            param.requires_grad = False
    # else: no pretrained features worth freezing, train the whole network.

    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--epochs", default=15, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--out", default="models/classifier.pt", type=Path)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        args.data, args.batch_size
    )
    model = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss()
    # Only optimize params that require grad: just the head when pretrained
    # features are frozen, or the whole network when training from scratch.
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_acc = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(val_loss)
        dt = time.time() - t0
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"val_loss {val_loss:.3f} acc {val_acc:.3f} | {dt:.1f}s"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc}
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "classes": classes},
                args.out,
            )
            print(f"  -> new best (val_acc={val_acc:.3f}), saved to {args.out}")

    # Final held-out test evaluation using the best checkpoint
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
    print(f"\nFinal test accuracy (best checkpoint): {test_acc:.3%}")
    print("Charter target: >= 85%. " + ("MET" if test_acc >= 0.85 else "NOT YET MET"))

    with open(args.out.with_suffix(".history.json"), "w") as f:
     json.dump({"history": history, "test_acc": test_acc, "classes": classes}, f, indent=2)


if __name__ == "__main__":
    main()
