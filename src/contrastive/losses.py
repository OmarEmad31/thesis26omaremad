"""
losses.py — Contrastive Loss Functions
=======================================
InfoNCE  : Self-supervised contrastive loss (used in Phase 1 SSL)
SupCon   : Supervised contrastive loss (used in Phase 2 fine-tuning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Self-supervised InfoNCE (NT-Xent) loss.
    Used in Phase 1 — no labels required.
    Positive pair: two augmented views of the same sample.
    Negatives: all other samples in the batch (+ memory bank if used).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [B, D] normalized embeddings (two views of same batch)
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        B = z1.size(0)

        # [2B, D]
        z = torch.cat([z1, z2], dim=0)
        # [2B, 2B] similarity matrix
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))

        # Positive indices: z1[i] <-> z2[i]
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0, B,   device=z.device)
        ])
        return F.cross_entropy(sim, labels)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Used in Phase 2 — uses emotion labels.
    Pulls same-emotion embeddings together, pushes different ones apart.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features : [B, D] normalized projection embeddings
        labels   : [B]    emotion class indices
        """
        features = F.normalize(features, dim=1)
        B = features.size(0)

        sim = torch.mm(features, features.T) / self.temperature
        # Mask diagonal (self)
        mask_self = torch.eye(B, device=features.device).bool()
        sim.masked_fill_(mask_self, float('-inf'))

        # Positive mask: same label, different sample
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        pos_mask.fill_diagonal_(0)

        # Log-softmax over all non-self pairs
        log_prob = F.log_softmax(sim, dim=1)

        # Mean over positive pairs
        n_pos = pos_mask.sum(1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(1) / n_pos
        return loss.mean()
