from __future__ import annotations

import numpy as np


def upper_body_crop(person_crop: np.ndarray, top_ratio: float = 0.68) -> np.ndarray:
    """Return a head/upper-body crop from a full-person BGR/RGB crop.

    The detector crop remains the coordinate space for pose estimation. This
    helper is only for the RGB image branch so clothing/background shortcuts are
    reduced while head, shoulder, arm, and desk-facing cues remain available.
    """
    if person_crop.size == 0:
        return person_crop
    h, _w = person_crop.shape[:2]
    y2 = max(1, min(h, int(round(h * float(top_ratio)))))
    return person_crop[:y2, :].copy()


def image_model_crop(person_crop: np.ndarray, roi: str = "upper_body") -> np.ndarray:
    roi = str(roi or "upper_body").strip().lower().replace("-", "_")
    if roi in {"full", "full_body", "person"}:
        return person_crop
    if roi in {"upper", "upper_body", "head_upper"}:
        return upper_body_crop(person_crop)
    raise ValueError(f"unknown image ROI: {roi}")
