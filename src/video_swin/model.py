"""
Swin Transformer model for Video Emotion Recognition.

Architecture:
  - Backbone: timm's Swin Transformer (swin_base_patch4_window7_224 pretrained on ImageNet-22k)
  - Temporal aggregation: mean-pool over extracted frame features
  - Head: LayerNorm → Dropout → FC
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import NUM_CLASSES


class SwinVideoModel(nn.Module):
    """
    Frame-level Swin Transformer with temporal mean-pooling.

    Input shape:  (B, 3, T, H, W)
    Output shape: (B, NUM_CLASSES) logits
    """

    def __init__(
        self,
        backbone: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        freeze_stages: int = 2,   # 0 = full finetune, N = freeze first N stages
    ):
        super().__init__()
        self.num_frames = None  # set dynamically

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # remove classifier head → returns features
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        # ── Selective freezing ────────────────────────────────────────────────
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

        # ── Temporal aggregation head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def _freeze_stages(self, num_stages: int):
        """Freeze patch embedding and first `num_stages` Swin stages."""
        # Always freeze patch embed
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False

        # Freeze absolute position embedding if present
        if hasattr(self.backbone, "absolute_pos_embed"):
            self.backbone.absolute_pos_embed.requires_grad = False

        # Freeze specified Swin stages (layers)
        if hasattr(self.backbone, "layers"):
            for i, layer in enumerate(self.backbone.layers):
                if i < num_stages:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, T, H, W)
        returns: (B, num_classes) logits
        """
        B, C, T, H, W = x.shape

        # Reshape to process each frame independently
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)

        # Extract frame features
        feat = self.backbone(x)        # (B*T, D)
        feat = feat.reshape(B, T, -1)  # (B, T, D)

        # Temporal mean pooling
        feat = feat.mean(dim=1)        # (B, D)

        # Classification head
        logits = self.head(feat)       # (B, num_classes)
        return logits


def build_model(
    backbone: str = "swin_base_patch4_window7_224",
    pretrained: bool = True,
    dropout: float = 0.4,
    freeze_stages: int = 2,
) -> SwinVideoModel:
    """Factory function to instantiate the model."""
    model = SwinVideoModel(
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        freeze_stages=freeze_stages,
    )
    return model
