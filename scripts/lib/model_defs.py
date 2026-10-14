# scripts/model_defs.py
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Literal, List

from .devices import default_device
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

Mode = Literal["image", "skeleton", "fusion"]
Agg = Literal["mean", "median"]
Objective = Literal["regression", "ordinal"]


@dataclass(frozen=True)
class Cfg:
    # dataset roots (複数授業)
    data_roots: List[Path]
    db_paths: List[Path]

    # raters aggregation
    raters: List[str]
    agg: Agg = "mean"
    label_source: Literal["individual", "consensus"] = "individual"
    objective: Objective = "ordinal"
    split_unit: Literal["window", "dataset", "session", "school"] = "school"
    holdout_groups: List[str] | None = None
    include_pilot_post: bool = False
    min_raters: int = 2
    max_label_std: float | None = None
    situation_teacher_forcing: float = 0.5
    use_situation_feature: bool = True

    # training
    mode: Mode = "fusion"
    epochs: int = 20
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = default_device()
    num_workers: int = 4

    # sequence
    T: int = 10
    sample_strategy: Literal["uniform", "random"] = "uniform"
    augmentation: Literal["standard", "mild", "none"] = "standard"
    grayscale: bool = False
    pose_features: Literal["raw", "motion"] = "raw"
    temporal_pool: Literal["last", "mean"] = "last"
    ordered_ordinal: bool = False
    backbone_lr_scale: float = 1.0
    freeze_batchnorm: bool = False
    lr_schedule: Literal["none", "plateau"] = "none"
    early_stopping_patience: int = 0
    selection_metric: Literal["mse", "mae", "exact_accuracy", "qwk"] = "mse"
    class_balance: bool = False
    score_loss_weight: float = 0.0
    situation_config: dict | None = None

    # image
    img_size: int = 224
    img_feat: int = 256
    imagenet_pretrain: bool = True
    image_roi: Literal["upper_body", "full_body"] = "upper_body"
    fusion_image_scale: float = 0.25

    # pose
    K: int = 17
    pose_dim: int = 3
    pose_feat: int = 128

    # lesson situation (one-hot: 聞く / 書く / 話し合う)
    situation_dim: int = 3

    # temporal
    temporal: Literal["gru", "transformer"] = "gru"
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.1

    # split
    val_ratio: float = 0.2
    no_validation: bool = False


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int, pretrained: bool):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> [B,512,1,1]
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.proj(h)


def motion_pose_features(pose: torch.Tensor) -> torch.Tensor:
    """Relative COCO geometry and inter-frame displacement; invalid joints are masked.

    Displacements are per sampled frame, not physical velocity or true eye gaze.
    Shoulder alignment reduces changing visible-joint normalization artifacts.
    """
    if pose.shape[-2:] != (17, 3):
        raise ValueError("motion features require COCO-17 [x,y,confidence]")
    xy = pose[..., :2].nan_to_num()
    conf = pose[..., 2].nan_to_num().clamp(0, 1)
    visible = conf >= 0.2
    shoulders = visible[..., 5] & visible[..., 6]
    center = (xy[..., 5, :] + xy[..., 6, :]) / 2
    width = (xy[..., 5, :] - xy[..., 6, :]).norm(dim=-1)
    valid = visible & shoulders.unsqueeze(-1) & (width > 1e-4).unsqueeze(-1)
    relative = ((xy - center.unsqueeze(-2)) / width.clamp_min(1e-4)[..., None, None]).clamp(-5, 5)
    relative = relative * valid.unsqueeze(-1)
    delta = torch.zeros_like(relative)
    delta[:, 1:] = (relative[:, 1:] - relative[:, :-1]) * (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1)
    # Bone directions describe torso, head relative to shoulders, and arms.
    edges = [(5, 6), (5, 11), (6, 12), (5, 7), (7, 9), (6, 8), (8, 10), (5, 0), (6, 0)]
    directions = []
    for a, b in edges:
        bone = xy[..., b, :] - xy[..., a, :]
        unit = bone / bone.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        directions.append(unit * (visible[..., a] & visible[..., b]).unsqueeze(-1))
    return torch.cat([relative.flatten(-2), delta.flatten(-2), conf, *directions], dim=-1)


class PoseEncoder(nn.Module):
    def __init__(self, K: int, D: int, out_dim: int, features: str = "raw"):
        super().__init__()
        self.features = features
        if features == "motion" and (K, D) != (17, 3):
            raise ValueError("motion features require COCO-17")
        inp = 103 if features == "motion" else K * D
        self.mlp = nn.Sequential(
            nn.Linear(inp, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:  # [B,T,K,D]
        B, T, K, D = pose.shape
        x = motion_pose_features(pose).reshape(B * T, -1) if self.features == "motion" else pose.reshape(B * T, K * D)
        return self.mlp(x)  # [B*T,out_dim]


class TemporalGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float, pool: str = "last"):
        super().__init__()
        self.pool = pool
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out.mean(dim=1) if self.pool == "mean" else out[:, -1, :]


class TemporalTransformer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float, nhead: int = 4, pool: str = "last"):
        super().__init__()
        self.pool = pool
        self.in_proj = nn.Linear(in_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pos = nn.Parameter(torch.randn(1, 512, hidden) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _B, T, _ = x.shape
        h = self.in_proj(x) + self.pos[:, :T, :]
        z = self.enc(h)
        return z.mean(dim=1) if self.pool == "mean" else z[:, -1, :]


class Regressor(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.mode = cfg.mode
        self.use_situation_feature = bool(cfg.use_situation_feature)

        feat_dim = 0
        self.img_enc: Optional[nn.Module] = None
        self.pose_enc: Optional[nn.Module] = None

        if cfg.mode in ("image", "fusion"):
            self.img_enc = ImageEncoder(cfg.img_feat, cfg.imagenet_pretrain)
            feat_dim += cfg.img_feat
        self.image_scale = float(cfg.fusion_image_scale if cfg.mode == "fusion" else 1.0)
        if cfg.mode in ("skeleton", "fusion"):
            self.pose_enc = PoseEncoder(cfg.K, cfg.pose_dim, cfg.pose_feat, cfg.pose_features)
            feat_dim += cfg.pose_feat

        feat_dim += cfg.situation_dim

        if cfg.temporal == "gru":
            self.temporal = TemporalGRU(feat_dim, cfg.hidden, cfg.layers, cfg.dropout, pool=cfg.temporal_pool)
            head_in = cfg.hidden
        else:
            self.temporal = TemporalTransformer(feat_dim, cfg.hidden, cfg.layers, cfg.dropout, pool=cfg.temporal_pool)
            head_in = cfg.hidden

        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(head_in // 2, 6 if cfg.objective == "ordinal" else 1),
        )
        self.objective = cfg.objective
        self.ordered_ordinal = cfg.ordered_ordinal
        if cfg.ordered_ordinal and cfg.objective == "ordinal":
            # Start thresholds symmetrically around score 4 rather than biasing
            # initial predictions downward through five positive increments.
            with torch.no_grad():
                self.head[-1].bias.zero_()
                self.head[-1].bias[0] = 2.5 * 0.6931471805599453

    def forward(
            self,
            frames: Optional[torch.Tensor],
            poses: Optional[torch.Tensor],
            situations: torch.Tensor,
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []

        if self.mode in ("image", "fusion"):
            assert frames is not None
            B, T, C, H, W = frames.shape
            x = frames.reshape(B * T, C, H, W)
            f = self.img_enc(x).reshape(B, T, -1)  # type: ignore[union-attr]
            feats.append(f * self.image_scale)

        if self.mode in ("skeleton", "fusion"):
            assert poses is not None
            B, T, K, D = poses.shape
            p = self.pose_enc(poses).reshape(B, T, -1)  # type: ignore[union-attr]
            feats.append(p)

        if not self.use_situation_feature:
            situations = torch.zeros_like(situations)
        if situations.ndim == 2:
            situations = situations.unsqueeze(1).expand(-1, T, -1)
        feats.append(situations)

        x = torch.cat(feats, dim=-1)
        h = self.temporal(x)
        raw = self.head(h)
        if self.objective == "ordinal":
            # P(y > k), k=1..6. The expected score is bounded to [1, 7].
            return 1.0 + torch.sigmoid(self._ordinal_logits(raw)).sum(dim=-1)
        return raw.squeeze(-1)

    def forward_ordinal_logits(
            self,
            frames: Optional[torch.Tensor],
            poses: Optional[torch.Tensor],
            situations: torch.Tensor,
    ) -> torch.Tensor:
        if self.objective != "ordinal":
            raise RuntimeError("ordinal logits requested from a regression model")
        feats: List[torch.Tensor] = []
        if self.mode in ("image", "fusion"):
            assert frames is not None
            B, T, C, H, W = frames.shape
            f = self.img_enc(frames.reshape(B * T, C, H, W)).reshape(B, T, -1)  # type: ignore[union-attr]
            feats.append(f * self.image_scale)
        if self.mode in ("skeleton", "fusion"):
            assert poses is not None
            B, T, _K, _D = poses.shape
            feats.append(self.pose_enc(poses).reshape(B, T, -1))  # type: ignore[union-attr]
        if not self.use_situation_feature:
            situations = torch.zeros_like(situations)
        if situations.ndim == 2:
            situations = situations.unsqueeze(1).expand(-1, T, -1)
        feats.append(situations)
        return self._ordinal_logits(self.head(self.temporal(torch.cat(feats, dim=-1))))

    def _ordinal_logits(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.ordered_ordinal:
            return raw
        # Enforce P(y>1) >= ... >= P(y>6). No additional checkpoint parameters.
        return torch.cat([raw[:, :1], raw[:, :1] - torch.nn.functional.softplus(raw[:, 1:]).cumsum(dim=-1)], dim=-1)


def situation_cfg(cfg: Cfg) -> Cfg:
    """Restore independent classifier settings; absent metadata keeps old behavior."""
    return replace(cfg, **cfg.situation_config, situation_config=None) if cfg.situation_config else cfg


class SituationClassifier(nn.Module):
    """Predict the classroom situation from a student's visual time series."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        cfg = situation_cfg(cfg)
        self.mode = cfg.mode
        feat_dim = 0
        self.img_enc: Optional[nn.Module] = None
        self.pose_enc: Optional[nn.Module] = None
        if cfg.mode in ("image", "fusion"):
            self.img_enc = ImageEncoder(cfg.img_feat, cfg.imagenet_pretrain)
            feat_dim += cfg.img_feat
        self.image_scale = float(cfg.fusion_image_scale if cfg.mode == "fusion" else 1.0)
        if cfg.mode in ("skeleton", "fusion"):
            self.pose_enc = PoseEncoder(cfg.K, cfg.pose_dim, cfg.pose_feat, cfg.pose_features)
            feat_dim += cfg.pose_feat
        if cfg.temporal == "gru":
            self.temporal = TemporalGRU(feat_dim, cfg.hidden, cfg.layers, cfg.dropout, pool=cfg.temporal_pool)
        else:
            self.temporal = TemporalTransformer(
                feat_dim, cfg.hidden, cfg.layers, cfg.dropout, pool=cfg.temporal_pool
            )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, cfg.situation_dim),
        )

    def forward(
            self,
            frames: Optional[torch.Tensor],
            poses: Optional[torch.Tensor],
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        if self.mode in ("image", "fusion"):
            assert frames is not None
            B, T, C, H, W = frames.shape
            f = self.img_enc(frames.reshape(B * T, C, H, W)).reshape(B, T, -1)  # type: ignore[union-attr]
            feats.append(f * self.image_scale)
        if self.mode in ("skeleton", "fusion"):
            assert poses is not None
            B, T, _K, _D = poses.shape
            feats.append(self.pose_enc(poses).reshape(B, T, -1))  # type: ignore[union-attr]
        return self.head(self.temporal(torch.cat(feats, dim=-1)))
