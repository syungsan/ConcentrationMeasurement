# scripts/model_defs.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List

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

    # training
    mode: Mode = "fusion"
    epochs: int = 20
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # sequence
    T: int = 10
    sample_strategy: Literal["uniform", "random"] = "uniform"

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


class PoseEncoder(nn.Module):
    def __init__(self, K: int, D: int, out_dim: int):
        super().__init__()
        inp = K * D
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
        x = pose.reshape(B * T, K * D)
        return self.mlp(x)  # [B*T,out_dim]


class TemporalGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out[:, -1, :]


class TemporalTransformer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float, nhead: int = 4):
        super().__init__()
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
        return z[:, -1, :]


class Regressor(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.mode = cfg.mode

        feat_dim = 0
        self.img_enc: Optional[nn.Module] = None
        self.pose_enc: Optional[nn.Module] = None

        if cfg.mode in ("image", "fusion"):
            self.img_enc = ImageEncoder(cfg.img_feat, cfg.imagenet_pretrain)
            feat_dim += cfg.img_feat
        self.image_scale = float(cfg.fusion_image_scale if cfg.mode == "fusion" else 1.0)
        if cfg.mode in ("skeleton", "fusion"):
            self.pose_enc = PoseEncoder(cfg.K, cfg.pose_dim, cfg.pose_feat)
            feat_dim += cfg.pose_feat

        feat_dim += cfg.situation_dim

        if cfg.temporal == "gru":
            self.temporal = TemporalGRU(feat_dim, cfg.hidden, cfg.layers, cfg.dropout)
            head_in = cfg.hidden
        else:
            self.temporal = TemporalTransformer(feat_dim, cfg.hidden, cfg.layers, cfg.dropout)
            head_in = cfg.hidden

        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(head_in // 2, 6 if cfg.objective == "ordinal" else 1),
        )
        self.objective = cfg.objective

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

        if situations.ndim == 2:
            situations = situations.unsqueeze(1).expand(-1, T, -1)
        feats.append(situations)

        x = torch.cat(feats, dim=-1)
        h = self.temporal(x)
        raw = self.head(h)
        if self.objective == "ordinal":
            # P(y > k), k=1..6. The expected score is bounded to [1, 7].
            return 1.0 + torch.sigmoid(raw).sum(dim=-1)
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
        if situations.ndim == 2:
            situations = situations.unsqueeze(1).expand(-1, T, -1)
        feats.append(situations)
        return self.head(self.temporal(torch.cat(feats, dim=-1)))


class SituationClassifier(nn.Module):
    """Predict the classroom situation from a student's visual time series."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.mode = cfg.mode
        feat_dim = 0
        self.img_enc: Optional[nn.Module] = None
        self.pose_enc: Optional[nn.Module] = None
        if cfg.mode in ("image", "fusion"):
            self.img_enc = ImageEncoder(cfg.img_feat, cfg.imagenet_pretrain)
            feat_dim += cfg.img_feat
        self.image_scale = float(cfg.fusion_image_scale if cfg.mode == "fusion" else 1.0)
        if cfg.mode in ("skeleton", "fusion"):
            self.pose_enc = PoseEncoder(cfg.K, cfg.pose_dim, cfg.pose_feat)
            feat_dim += cfg.pose_feat
        if cfg.temporal == "gru":
            self.temporal = TemporalGRU(feat_dim, cfg.hidden, cfg.layers, cfg.dropout)
        else:
            self.temporal = TemporalTransformer(
                feat_dim, cfg.hidden, cfg.layers, cfg.dropout
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
