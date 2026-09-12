# lib/infer.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, Sequence

import torch
from torchvision import transforms

from lib.model_defs import Cfg, Regressor, SituationClassifier, Mode
from lib.situation import SITUATIONS, situation_one_hot


def build_image_tf(img_size: int):
    # 07_train.py の val/推論時と同じ
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _to_path_list(xs):
    out = []
    for x in xs:
        if isinstance(x, Path):
            out.append(x)
        else:
            out.append(Path(str(x)))
    return out


def _build_cfg_from_ckpt_dict(d: Dict[str, Any]) -> Cfg:
    """
    ckpt["cfg"] が Path を含む/含まない両方に対応して Cfg を復元する。
    """
    dd = dict(d)
    if "data_roots" in dd:
        dd["data_roots"] = _to_path_list(dd["data_roots"])
    if "db_paths" in dd:
        dd["db_paths"] = _to_path_list(dd["db_paths"])
    return Cfg(**dd)


def load_ckpt(ckpt_path: Path, mode: Mode, device: str):
    """
    PyTorch 2.6 の weights_only 変更に対応。
    さらに ckpt が
      - {"cfg":..., "state_dict":..., "mode":...} 形式
      - {"cfg":..., "state_dicts": {"image":..., "skeleton":..., "fusion":...}} 形式
    どちらでもロードできる。
    """
    # まずは weights_only=True (安全) を試して、ダメなら weights_only=False にフォールバック
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if ckpt.get("feature_schema") not in {
        "image_pose_situation_v1", "image_pose_situation_v2",
        "image_pose_situation_v3", "image_pose_situation_v4",
    }:
        raise RuntimeError(
            "このチェックポイントは状況特徴量に対応していません。"
            "更新後の 07_train.py で再学習してください。"
        )

    base_cfg_dict = dict(ckpt["cfg"])
    if ckpt.get("feature_schema") == "image_pose_situation_v1":
        base_cfg_dict.setdefault("objective", "regression")
    cfg_base = _build_cfg_from_ckpt_dict(base_cfg_dict)
    cfg = replace(cfg_base, mode=mode)

    model = Regressor(cfg)

    # 形式吸収
    if "state_dicts" in ckpt:
        sd = ckpt["state_dicts"][mode]
    elif "states" in ckpt:  # 念のため
        sd = ckpt["states"][mode]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        raise KeyError("checkpoint has no state_dict / state_dicts")

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return ckpt, cfg, model


def load_situation_model(
        ckpt: Dict[str, Any], cfg: Cfg, mode: Mode, device: str,
) -> SituationClassifier:
    states = ckpt.get("situation_state_dicts")
    if not isinstance(states, dict) or mode not in states:
        raise RuntimeError(
            "checkpoint has no automatic situation model for "
            f"mode={mode}; retrain with 07_train.py and --situation_epochs > 0"
        )
    labels = tuple(ckpt.get("situation_labels", ()))
    if labels and labels != tuple(SITUATIONS):
        raise RuntimeError(f"situation label order mismatch: {labels}")
    model = SituationClassifier(cfg)
    model.load_state_dict(states[mode], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_situation_probs(
        model: SituationClassifier,
        frames: Optional[torch.Tensor],
        poses: Optional[torch.Tensor],
        device: str,
) -> torch.Tensor:
    if frames is not None:
        frames = frames.to(device)
    if poses is not None:
        poses = poses.to(device)
    return torch.softmax(model(frames, poses), dim=-1).mean(dim=0).detach().cpu()


class SituationProbabilitySmoother:
    """EMA aggregation of situation evidence from multiple tracked students."""

    def __init__(self, alpha: float = 0.15):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.value: Optional[torch.Tensor] = None

    def update(self, probabilities: torch.Tensor) -> torch.Tensor:
        probabilities = probabilities.detach().float().cpu()
        probabilities = probabilities / probabilities.sum().clamp_min(1e-8)
        if self.value is None:
            self.value = probabilities
        else:
            self.value = self.value * (1.0 - self.alpha) + probabilities * self.alpha
            self.value = self.value / self.value.sum().clamp_min(1e-8)
        return self.value.clone()

    def current(self) -> Optional[torch.Tensor]:
        return None if self.value is None else self.value.clone()


@torch.no_grad()
def predict_score(
        model,
        frames: Optional[torch.Tensor],  # [1,T,3,H,W]
        poses: Optional[torch.Tensor],   # [1,T,K,3]
        device: str,
        situation: str | None = None,
        situation_probs: Optional[Sequence[float] | torch.Tensor] = None,
) -> float:
    if frames is not None:
        frames = frames.to(device)
    if poses is not None:
        poses = poses.to(device)
    if situation_probs is not None:
        situation_tensor = torch.as_tensor(
            situation_probs, dtype=torch.float32, device=device
        ).reshape(1, -1)
        if situation_tensor.shape[1] != len(SITUATIONS):
            raise ValueError(f"expected {len(SITUATIONS)} situation probabilities")
        situation_tensor = situation_tensor / situation_tensor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    elif situation is not None:
        situation_tensor = situation_one_hot(situation).unsqueeze(0).to(device)
    else:
        raise ValueError("situation or situation_probs is required")
    yhat = model(frames, poses, situation_tensor).detach().cpu().item()
    return float(yhat)


@torch.no_grad()
def predict_score_with_uncertainty(
        model,
        frames: Optional[torch.Tensor],
        poses: Optional[torch.Tensor],
        device: str,
        situation: str,
) -> tuple[float, float]:
    """Return expected 1..7 score and normalized ordinal entropy (0..1)."""
    if frames is not None:
        frames = frames.to(device)
    if poses is not None:
        poses = poses.to(device)
    situation_tensor = situation_one_hot(situation).unsqueeze(0).to(device)
    score = float(model(frames, poses, situation_tensor).detach().cpu().item())
    if getattr(model, "objective", "regression") != "ordinal":
        return score, float("nan")
    logits = model.forward_ordinal_logits(frames, poses, situation_tensor)
    probabilities = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    entropy = -(
        probabilities * probabilities.log()
        + (1 - probabilities) * (1 - probabilities).log()
    ).mean() / 0.6931471805599453
    return score, float(entropy.detach().cpu().item())


def clamp_1to7(v: float) -> int:
    vv = int(round(v))
    return max(1, min(7, vv))
