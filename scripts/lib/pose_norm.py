# lib/pose_norm.py
from __future__ import annotations
import numpy as np


def normalize_pose_kpts(
        kpts: np.ndarray,
        *,
        conf_thr: float = 0.2,
        eps: float = 1e-6,
) -> np.ndarray:
    """
    kpts: [K,3] (x,y,conf) in "crop" coordinate
    return: [K,3] normalized (x,y) and keep conf
      - center: mean of visible xy
      - scale: RMS radius of visible points
    """
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        raise ValueError("kpts must be [K,3] or [K,>=2]")

    out = kpts.astype(np.float32).copy()
    if out.shape[1] == 2:
        out = np.concatenate([out, np.ones((out.shape[0], 1), np.float32)], axis=1)

    xy = out[:, :2]
    conf = out[:, 2]

    vis = conf >= conf_thr
    if not np.any(vis):
        out[:, :2] = 0.0
        return out

    mu = xy[vis].mean(axis=0, keepdims=True)
    xy0 = xy - mu

    r2 = (xy0[vis] ** 2).sum(axis=1).mean()
    s = float(np.sqrt(max(r2, eps)))

    out[:, :2] = xy0 / s
    return out
