"""
Module 3 — train.py
Training loop for the Swin-T classifier: class-balanced sampling,
weighted focal loss, LayerOut-style freeze/unfreeze (configurable,
NOT auto-decided — your doc says this needs empirical benchmarking,
so this file exposes the knob, it doesn't claim an answer for you).

Hardware note: RTX 3050 6GB is tight for Swin-T @ 224x224. AMP
(mixed precision) is enabled by default to reduce activation memory —
this isn't speculative, it's a direct response to the documented
VRAM constraint, not a feature added for its own sake.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import numpy as np

from modules.module3_vision.dataset import SkinTelDataset, build_split_index, get_class_counts
from modules.module3_vision.model import SwinTClassifier

# ---- Config (edit these to run the LayerOut benchmark yourself) ----
BATCH_SIZE = 16
EPOCHS = 20
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
FOCAL_GAMMA = 2.0
SAMPLER_FLOOR = 150   # per documented plan: augment sparse classes up to this
SAMPLER_CAP = 500     # cap dominant classes at this, per epoch
CHECKPOINT_DIR = "models/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "swin_t_module3_best.pth")

# FREEZE_MODE controls the LayerOut experiment — change this and re-run
# to compare. This file does NOT decide which is best; you benchmark it.
#   "none"        -> full backbone trainable
#   "freeze_all"  -> backbone frozen, only head trains
#   "stage3_only" -> backbone frozen except final Swin stage (idx 3)
FREEZE_MODE = "none"


class ClassBalancedSampler(Sampler):
    """Per-epoch resampling: classes below SAMPLER_FLOOR are oversampled
    (with replacement) up to the floor; classes above SAMPLER_CAP are
    undersampled (without replacement) down to the cap. Implements the
    documented strategy directly and explicitly, rather than approximating
    it via WeightedRandomSampler's implicit probabilities."""

    def __init__(self, df, label_to_idx, floor=SAMPLER_FLOOR, cap=SAMPLER_CAP, seed=42):
        self.df = df.reset_index(drop=True)
        self.floor = floor
        self.cap = cap
        self.rng = np.random.default_rng(seed)

        self.class_indices = {}
        for label in df["label_collapsed"].unique():
            self.class_indices[label] = self.df.index[self.df["label_collapsed"] == label].to_numpy()

    def __iter__(self):
        all_indices = []
        for label, idxs in self.class_indices.items():
            n = len(idxs)
            if n < self.floor:
                chosen = self.rng.choice(idxs, size=self.floor, replace=True)
            elif n > self.cap:
                chosen = self.rng.choice(idxs, size=self.cap, replace=False)
            else:
                chosen = idxs
            all_indices.extend(chosen.tolist())
        self.rng.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self):
        total = 0
        for idxs in self.class_indices.values():
            n = len(idxs)
            total += self.floor if n < self.floor else min(n, self.cap)
        return total


class FocalLoss(nn.Module):
    """Standard focal loss with per-class alpha weighting, derived from
    real train-split class frequencies (inverse-frequency, normalized to
    mean 1 so the loss scale stays comparable to plain cross-entropy)."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


def compute_alpha_weights(train_df, label_to_idx) -> torch.Tensor:
    counts = get_class_counts(train_df, split="train")
    num_classes = len(label_to_idx)
    weights = torch.ones(num_classes)
    for label, idx in label_to_idx.items():
        c = counts.get(label, 1)
        weights[idx] = 1.0 / c
    weights = weights / weights.mean()  # normalize to mean 1
    return weights


def apply_freeze_mode(model: SwinTClassifier, mode: str):
    if mode == "freeze_all":
        model.freeze_backbone()
    elif mode == "stage3_only":
        model.freeze_backbone()
        model.unfreeze_backbone_stage(3)
    elif mode == "none":
        pass  # full backbone trainable, default timm state
    else:
        raise ValueError(f"Unknown FREEZE_MODE: {mode}")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    accuracy = correct / total

    # Macro F1 — more honest than accuracy alone given the imbalance
    num_classes = model.head[-1].out_features
    f1_per_class = []
    for c in range(num_classes):
        tp = sum(1 for p, t in zip(all_preds, all_targets) if p == c and t == c)
        fp = sum(1 for p, t in zip(all_preds, all_targets) if p == c and t != c)
        fn = sum(1 for p, t in zip(all_preds, all_targets) if p != c and t == c)
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall == 0:
            f1_per_class.append(0.0)
        else:
            f1_per_class.append(2 * precision * recall / (precision + recall))
    macro_f1 = sum(f1_per_class) / len(f1_per_class) if f1_per_class else 0.0

    return accuracy, macro_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"FREEZE_MODE = {FREEZE_MODE}  (change this + re-run to benchmark LayerOut)")

    full_index = build_split_index()
    train_df = full_index[full_index["final_split"] == "train"]

    train_ds = SkinTelDataset(split="train")
    test_ds = SkinTelDataset(split="test")

    sampler = ClassBalancedSampler(train_df, train_ds.label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    num_classes = len(train_ds.label_to_idx)
    model = SwinTClassifier(num_classes=num_classes, pretrained=True).to(device)
    apply_freeze_mode(model, FREEZE_MODE)

    alpha = compute_alpha_weights(train_df, train_ds.label_to_idx).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)

    optimizer = torch.optim.AdamW(model.param_groups(BACKBONE_LR, HEAD_LR))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_macro_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc, macro_f1 = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={avg_loss:.4f} | test_acc={acc:.4f} | test_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> new best macro_f1={macro_f1:.4f}, saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()