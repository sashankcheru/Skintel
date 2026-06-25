"""
Module 3 — model.py
Swin-Tiny backbone (ImageNet-22k pretrained) + custom classification head.

Verified live (June 2026):
- timm tag: swin_tiny_patch4_window7_224.ms_in22k
- backbone.num_features == 768 (confirmed via forward pass, not assumed)
- Replaces the abandoned Swin_MC_best_model.pth checkpoint (architecture
  mismatch: that file is Swin-Base @ window12/384res/8-class head — see
  inspection log, not compatible with this pipeline's 224x224 input).
"""

import torch
import torch.nn as nn
import timm


class SwinTClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224.ms_in22k",
            pretrained=pretrained,
            num_classes=0,  # strip the 21,841-class IN22k head; we attach our own
        )

        feature_dim = self.backbone.num_features  # 768 for Swin-Tiny, read dynamically

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone(x)       # [B, 768] pooled feature vector
        logits = self.head(features)      # [B, num_classes]
        if return_features:
            return logits, features
        return logits

    def freeze_backbone(self):
        """Freeze all backbone params. Used as the baseline before
        LayerOut stochastic block unfreezing is benchmarked in the
        training loop (Module 3, step 3)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone_stage(self, stage_idx: int):
        """Unfreeze one Swin stage by index (0-3) for LayerOut-style
        partial fine-tuning. Stage 3 (final) is typically unfrozen first."""
        for name, p in self.backbone.named_parameters():
            if f"layers.{stage_idx}." in name:
                p.requires_grad = True

    def param_groups(self, backbone_lr: float, head_lr: float):
        """Separate LR groups — standard practice when fine-tuning a
        pretrained backbone alongside a freshly initialized head."""
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]


if __name__ == "__main__":
    model = SwinTClassifier(num_classes=128, pretrained=True)
    dummy = torch.randn(4, 3, 224, 224)
    logits = model(dummy)
    print("logits shape:", logits.shape)  # expect [4, 128]

    logits, feats = model(dummy, return_features=True)
    print("features shape:", feats.shape)  # expect [4, 768]