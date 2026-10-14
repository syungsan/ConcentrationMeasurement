"""Optional training controls shared by concentration and situation models."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_optimizer(model, cfg):
    backbone = []
    other = []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("img_enc.backbone.") else other).append(parameter)
    groups = [{"params": other, "lr": cfg.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr * cfg.backbone_lr_scale})
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def freeze_batchnorm_stats(model, enabled):
    # Keep affine parameters trainable, but retain pretrained running statistics.
    if enabled:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def balanced_class_weights(labels):
    """Bounded inverse square-root frequency weights fitted on training labels only."""
    indices = torch.as_tensor(labels).round().long().clamp(1, 7) - 1
    counts = torch.bincount(indices, minlength=7).float()
    weights = counts.clamp_min(1).rsqrt()
    weights[counts == 0] = 0
    weights /= (weights[indices].mean().clamp_min(1e-8))
    return weights.clamp(max=3.0), counts


def concentration_loss(logits, score, labels, cfg, class_weights=None):
    if cfg.objective == "ordinal":
        thresholds = torch.arange(1, 7, device=labels.device).unsqueeze(0)
        targets = (labels.unsqueeze(1) - thresholds).clamp(0, 1)
        per_sample = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(-1)
        if cfg.score_loss_weight:
            per_sample = per_sample + cfg.score_loss_weight * F.smooth_l1_loss(score, labels, reduction="none")
    else:
        per_sample = (score - labels).square()
    if class_weights is not None:
        weights = class_weights[labels.round().long().clamp(1, 7) - 1]
        return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    return per_sample.mean()


def selection_value(metrics, metric):
    # All selection criteria are minimized; accuracy/kappa signs are reversed.
    return -metrics[metric] if metric in {"exact_accuracy", "qwk"} else metrics[metric]


def build_scheduler(optimizer, cfg):
    if cfg.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)
    return None
