# lib/infer.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import torch
from torchvision import transforms

from lib.model_defs import Cfg, Regressor, Mode
from lib.situation import situation_one_hot


def build_image_tf(img_size: int):
    # 06_train.py の val/推論時と同じ
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

    if ckpt.get("feature_schema") != "image_pose_situation_v1":
        raise RuntimeError(
            "このチェックポイントは状況特徴量に対応していません。"
            "更新後の 06_train.py で再学習してください。"
        )

    base_cfg_dict = ckpt["cfg"]
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


@torch.no_grad()
def predict_score(
        model,
        frames: Optional[torch.Tensor],  # [1,T,3,H,W]
        poses: Optional[torch.Tensor],   # [1,T,K,3]
        device: str,
        situation: str,
) -> float:
    if frames is not None:
        frames = frames.to(device)
    if poses is not None:
        poses = poses.to(device)
    situation_tensor = situation_one_hot(situation).unsqueeze(0).to(device)
    yhat = model(frames, poses, situation_tensor).detach().cpu().item()
    return float(yhat)


def clamp_1to7(v: float) -> int:
    vv = int(round(v))
    return max(1, min(7, vv))
