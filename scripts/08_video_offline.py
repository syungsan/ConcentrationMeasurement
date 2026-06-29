# scripts/08_video_offline.py
from __future__ import annotations

import time
SCRIPT_STARTED_AT = time.perf_counter()

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Deque, Tuple, List, Any

import cv2
import numpy as np
import torch
from PIL import Image
from collections import deque

from ultralytics import YOLO

from lib.runroot import get_data_root, rpath
from lib.infer import load_ckpt, build_image_tf, predict_score, clamp_1to7
from lib.situation import SITUATIONS
from lib.pose_norm import normalize_pose_kpts


# -------------------------
# Track smoothing / hold
# -------------------------
@dataclass
class TrackState:
    bbox: np.ndarray         # [x1,y1,x2,y2] float32 (frame coords)
    score: float             # raw score (float)
    last_seen: int           # frame index
    det_conf: float = 1.0
    kpts_frame: Optional[np.ndarray] = None  # [K,3] x,y,conf in FRAME coordinates


def ema(old: float, new: float, alpha: float) -> float:
    return old * (1.0 - alpha) + new * alpha


def ema_bbox(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return old * (1.0 - alpha) + new * alpha


def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W - 1, int(x2)))
    y2 = max(0, min(H - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# -------------------------
# Skeleton (COCO-17) bones
# -------------------------
# COCO17 indices (Ultralytics YOLOv8-pose):
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
COCO17_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms + shoulders
    (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


# -------------------------
# Drawing helpers
# -------------------------
def draw_label(img, x, y, text, *, font_scale=1.2, thickness=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(x); y = int(y)
    cv2.rectangle(img, (x, y - th - baseline - 8), (x + tw + 10, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_skeleton(
        img: np.ndarray,
        kpts_xyc: np.ndarray,
        *,
        conf_thr: float = 0.2,
        radius: int = 3,
        line_thick: int = 2,
):
    """
    kpts_xyc: [K,3] x,y,conf in FRAME coordinates
    """
    if kpts_xyc is None or kpts_xyc.ndim != 2 or kpts_xyc.shape[1] < 2:
        return
    K = kpts_xyc.shape[0]

    # lines
    if K >= 17:
        for a, b in COCO17_EDGES:
            if a >= K or b >= K:
                continue
            xa, ya = float(kpts_xyc[a, 0]), float(kpts_xyc[a, 1])
            xb, yb = float(kpts_xyc[b, 0]), float(kpts_xyc[b, 1])
            ca = float(kpts_xyc[a, 2]) if kpts_xyc.shape[1] >= 3 else 1.0
            cb = float(kpts_xyc[b, 2]) if kpts_xyc.shape[1] >= 3 else 1.0
            if ca < conf_thr or cb < conf_thr:
                continue
            cv2.line(
                img,
                (int(round(xa)), int(round(ya))),
                (int(round(xb)), int(round(yb))),
                (255, 255, 0),
                int(line_thick),
                cv2.LINE_AA,
            )

    # points
    for i in range(K):
        x = float(kpts_xyc[i, 0])
        y = float(kpts_xyc[i, 1])
        c = float(kpts_xyc[i, 2]) if kpts_xyc.shape[1] >= 3 else 1.0
        if c < conf_thr:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), int(radius), (255, 255, 0), -1, cv2.LINE_AA)


# -------------------------
# Eye mosaic helpers
# -------------------------
def _valid_kpt(kpts: np.ndarray, idx: int, conf_thr: float) -> Optional[Tuple[float, float]]:
    if kpts is None or kpts.ndim != 2 or kpts.shape[1] < 2:
        return None
    if idx < 0 or idx >= kpts.shape[0]:
        return None
    x = float(kpts[idx, 0])
    y = float(kpts[idx, 1])
    c = float(kpts[idx, 2]) if kpts.shape[1] >= 3 else 1.0
    if c < conf_thr:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def apply_pixelate(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, *, mosaic_scale: float):
    """
    mosaic_scale: 0.03〜0.15くらい推奨（小さいほど強い）
    """
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 2 or rw < 2:
        return

    s = float(mosaic_scale)
    dw = max(1, int(round(rw * s)))
    dh = max(1, int(round(rh * s)))
    small = cv2.resize(roi, (dw, dh), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pix


def mosaic_eyes_from_kpts(
        frame: np.ndarray,
        kpts_frame: np.ndarray,
        *,
        conf_thr: float = 0.25,
        pad: float = 1.8,
        mosaic_scale: float = 0.06,
):
    """
    COCO17:
      left_eye=1, right_eye=2, nose=0, left_ear=3, right_ear=4
    """
    if kpts_frame is None or kpts_frame.ndim != 2 or kpts_frame.shape[0] < 5:
        return

    le = _valid_kpt(kpts_frame, 1, conf_thr)
    re = _valid_kpt(kpts_frame, 2, conf_thr)
    nose = _valid_kpt(kpts_frame, 0, conf_thr)

    if le is None and re is None:
        return

    xs = []
    ys = []
    if le is not None:
        xs.append(le[0]); ys.append(le[1])
    if re is not None:
        xs.append(re[0]); ys.append(re[1])

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    if le is not None and re is not None:
        eye_dist = float(math.hypot(le[0] - re[0], le[1] - re[1]))
    else:
        if nose is not None:
            eye_dist = float(math.hypot(cx - nose[0], cy - nose[1])) * 1.2
        else:
            eye_dist = 25.0

    w = max(10.0, eye_dist * pad * 1.2)
    h = max(10.0, eye_dist * pad * 0.9)

    if nose is not None:
        cy = 0.7 * cy + 0.3 * float(nose[1])

    x1 = int(round(cx - w * 0.5))
    y1 = int(round(cy - h * 0.5))
    x2 = int(round(cx + w * 0.5))
    y2 = int(round(cy + h * 0.5))

    apply_pixelate(frame, x1, y1, x2, y2, mosaic_scale=mosaic_scale)


# -------------------------
# Feature helpers
# -------------------------
def make_pose_empty(K: int = 17, D: int = 3) -> np.ndarray:
    return np.zeros((K, D), dtype=np.float32)


def crop_resize_for_pose(crop_bgr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    s = float(long_side) / float(max(h, w))
    if s >= 1.0:
        return crop_bgr
    nh, nw = int(round(h * s)), int(round(w * s))
    return cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def pose_to_frame_coords(
        kpts_xyc_in_poseimg: np.ndarray,  # [K,3] on crop_pose_in coords
        poseimg_wh: Tuple[int, int],      # (w,h) of crop_pose_in
        crop_wh: Tuple[int, int],         # (w,h) of original crop
        crop_xy1: Tuple[int, int],        # (x1,y1) on frame
) -> np.ndarray:
    pw, ph = poseimg_wh
    cw, ch = crop_wh
    x1, y1 = crop_xy1

    if pw <= 0 or ph <= 0 or cw <= 0 or ch <= 0:
        return kpts_xyc_in_poseimg

    sx = float(cw) / float(pw)
    sy = float(ch) / float(ph)

    out = kpts_xyc_in_poseimg.copy()
    out[:, 0] = out[:, 0] * sx + float(x1)
    out[:, 1] = out[:, 1] * sy + float(y1)
    return out


def pad_seq(seq: List[Any], T_: int) -> List[Any]:
    if len(seq) >= T_:
        return seq[-T_:]
    if len(seq) == 0:
        return []
    last = seq[-1]
    return seq + [last] * (T_ - len(seq))


# -------------------------
# Stable ID remap (reduce ID switches)
# -------------------------
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(1e-6, area_a + area_b - inter)
    return float(inter / union)


def _center(b):
    x1, y1, x2, y2 = [float(v) for v in b]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def _kpt_feat(kpts_frame: Optional[np.ndarray], conf_thr: float = 0.2) -> np.ndarray:
    """
    kpts_frame: [K,3] in frame coords
    -> translation/scale normalized feature (flattened)
    """
    if kpts_frame is None or kpts_frame.ndim != 2 or kpts_frame.shape[1] < 2:
        return np.zeros((0,), dtype=np.float32)

    xy = kpts_frame[:, :2].astype(np.float32)

    if kpts_frame.shape[1] >= 3:
        c = kpts_frame[:, 2].astype(np.float32)
        m = c >= conf_thr
        if int(m.sum()) >= 4:
            xy = xy[m]

    if xy.shape[0] < 4:
        return np.zeros((0,), dtype=np.float32)

    mu = xy.mean(axis=0, keepdims=True)
    z = xy - mu
    s = float(np.sqrt((z ** 2).sum(axis=1).mean()) + 1e-6)
    z = z / s
    return z.reshape(-1)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 1.0
    na = float(np.linalg.norm(a) + 1e-6)
    nb = float(np.linalg.norm(b) + 1e-6)
    return float(1.0 - (a @ b) / (na * nb))


def _hsv_hist(frame_bgr: np.ndarray, bbox, bins=16) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((bins * 3,), dtype=np.float32)

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((bins * 3,), dtype=np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hs = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256]).reshape(-1).astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        hs.append(hist)
    return np.concatenate(hs, axis=0)


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 1.0
    return float(np.abs(a - b).mean())


@dataclass
class StableTrack:
    stable_id: int
    bbox: np.ndarray
    last_frame: int
    kfeat: np.ndarray
    hist: np.ndarray


class StableIDMapper:
    """
    Map detections to stable IDs with greedy assignment (fast).
    Cost = w_iou*(1-iou) + w_center*center_dist_norm + w_kpt*cos_dist + w_hist*l1
    """
    def __init__(
            self,
            *,
            max_age: int = 40,
            iou_min: float = 0.05,
            w_iou: float = 2.5,
            w_center: float = 1.0,
            w_kpt: float = 1.6,
            w_hist: float = 0.7,
            kpt_conf: float = 0.2,
            center_norm: float = 500.0,
            hist_bins: int = 16,
    ):
        self.max_age = int(max_age)
        self.iou_min = float(iou_min)
        self.w_iou = float(w_iou)
        self.w_center = float(w_center)
        self.w_kpt = float(w_kpt)
        self.w_hist = float(w_hist)
        self.kpt_conf = float(kpt_conf)
        self.center_norm = float(center_norm)
        self.hist_bins = int(hist_bins)

        self._next_id = 1
        self.tracks: Dict[int, StableTrack] = {}

    def gc(self, frame_idx: int):
        dead = [sid for sid, tr in self.tracks.items() if frame_idx - tr.last_frame > self.max_age]
        for sid in dead:
            del self.tracks[sid]

    def assign(
            self,
            frame_bgr: np.ndarray,
            frame_idx: int,
            dets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        dets item needs:
          bbox(np.ndarray float32 [x1,y1,x2,y2]),
          kpts_frame(optional np.ndarray [K,3])
        returns dets with 'stable_id'
        """
        self.gc(frame_idx)

        # precompute det features
        for d in dets:
            bbox = d["bbox"]
            d["center"] = _center(bbox)
            d["kfeat"] = _kpt_feat(d.get("kpts_frame", None), conf_thr=self.kpt_conf)
            d["hist"] = _hsv_hist(frame_bgr, bbox, bins=self.hist_bins)

        # no existing track -> assign new
        sids = list(self.tracks.keys())
        if not sids:
            for d in dets:
                sid = self._next_id; self._next_id += 1
                self.tracks[sid] = StableTrack(
                    stable_id=sid,
                    bbox=d["bbox"].copy(),
                    last_frame=frame_idx,
                    kfeat=d["kfeat"],
                    hist=d["hist"],
                )
                d["stable_id"] = sid
            return dets

        # candidate pairs: (cost, det_index, stable_id)
        pairs = []
        for j, d in enumerate(dets):
            best_for_track = []
            for sid in sids:
                tr = self.tracks[sid]
                iou = _iou(tr.bbox, d["bbox"])
                if iou < self.iou_min:
                    continue
                cdist = float(np.linalg.norm(_center(tr.bbox) - d["center"])) / self.center_norm
                kdist = _cos_dist(tr.kfeat, d["kfeat"])
                hdist = _l1(tr.hist, d["hist"])
                cost = self.w_iou * (1.0 - iou) + self.w_center * cdist + self.w_kpt * kdist + self.w_hist * hdist
                best_for_track.append((cost, sid))
            if best_for_track:
                cost, sid = min(best_for_track, key=lambda x: x[0])
                pairs.append((cost, j, sid))

        pairs.sort(key=lambda x: x[0])

        used_sid = set()
        used_det = set()

        for cost, j, sid in pairs:
            if j in used_det or sid in used_sid:
                continue
            used_det.add(j); used_sid.add(sid)
            dets[j]["stable_id"] = sid

            tr = self.tracks[sid]
            tr.bbox = dets[j]["bbox"].copy()
            tr.last_frame = frame_idx
            tr.kfeat = dets[j]["kfeat"]
            tr.hist = dets[j]["hist"]

        # remaining dets -> new ids
        for j, d in enumerate(dets):
            if "stable_id" in d:
                continue
            sid = self._next_id; self._next_id += 1
            self.tracks[sid] = StableTrack(
                stable_id=sid,
                bbox=d["bbox"].copy(),
                last_frame=frame_idx,
                kfeat=d["kfeat"],
                hist=d["hist"],
            )
            d["stable_id"] = sid

        return dets


# -------------------------
# Logging
# -------------------------
LOG_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_path TEXT NOT NULL,
  frame_idx INTEGER NOT NULL,
  t REAL NOT NULL,

  track_id INTEGER NOT NULL,
  stable_id INTEGER,

  score REAL NOT NULL,
  score_clamped INTEGER NOT NULL,
  situation TEXT NOT NULL CHECK(situation IN ('聞く', '書く', '話し合う')),
  det_conf REAL,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL
);

CREATE INDEX IF NOT EXISTS idx_pred_video_t ON predictions(video_path, t);
CREATE INDEX IF NOT EXISTS idx_pred_video_track ON predictions(video_path, track_id);
CREATE INDEX IF NOT EXISTS idx_pred_video_stable ON predictions(video_path, stable_id);
"""


def open_log_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(LOG_SCHEMA)
    conn.commit()
    return conn


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True, help="学習済みckpt (.pt)")
    ap.add_argument("--mode", type=str, required=True, choices=["image", "skeleton", "fusion"])
    ap.add_argument("--situation", type=str, required=True, choices=SITUATIONS)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--video", type=str, default=None)

    ap.add_argument("--det_model", type=str, default=None)
    ap.add_argument("--pose_model", type=str, default=None)

    ap.add_argument("--tracker", type=str, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)

    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--save_assets", action="store_true")
    ap.add_argument("--assets_dir", type=str, default="outputs/assets")

    ap.add_argument("--annotate_out", type=str, default=None)

    ap.add_argument("--log_db", type=str, default="outputs/pred_log.sqlite")
    ap.add_argument("--log_every", type=int, default=30)

    # flicker control
    ap.add_argument("--max_age", type=int, default=12)
    ap.add_argument("--bbox_alpha", type=float, default=0.6)
    ap.add_argument("--score_alpha", type=float, default=0.3)
    ap.add_argument("--class_alpha", type=float, default=0.2)

    # draw sizes
    ap.add_argument("--box_thick", type=int, default=3)
    ap.add_argument("--label_scale", type=float, default=1.2)
    ap.add_argument("--label_thick", type=int, default=3)
    ap.add_argument("--class_scale", type=float, default=1.6)
    ap.add_argument("--class_thick", type=int, default=4)

    # skeleton draw
    ap.add_argument("--draw_skeleton", action="store_true", help="骨格を表示（skeleton/fusionで有効）")
    ap.add_argument("--skel_radius", type=int, default=3)
    ap.add_argument("--skel_line", type=int, default=2)
    ap.add_argument("--skel_conf", type=float, default=0.2)

    # eye mosaic
    ap.add_argument("--mosaic_eyes", action="store_true", help="顔keypointから目周辺をモザイク（draw_skeletonと併用可）")
    ap.add_argument("--mosaic_conf", type=float, default=0.25, help="顔kpt conf閾値")
    ap.add_argument("--mosaic_pad", type=float, default=1.8, help="目領域の拡張倍率")
    ap.add_argument("--mosaic_scale", type=float, default=0.06, help="モザイク強さ(小さいほど強い) 例:0.04〜0.10")

    # stable id remap
    ap.add_argument("--stable_id", action="store_true", help="ID入れ替わり抑制：stable_idを再割当して表示/ログする")
    ap.add_argument("--stable_max_age", type=int, default=40)
    ap.add_argument("--stable_iou_min", type=float, default=0.05)
    ap.add_argument("--stable_w_iou", type=float, default=2.5)
    ap.add_argument("--stable_w_center", type=float, default=1.0)
    ap.add_argument("--stable_w_kpt", type=float, default=1.6)
    ap.add_argument("--stable_w_hist", type=float, default=0.7)
    ap.add_argument("--stable_kpt_conf", type=float, default=0.2)
    ap.add_argument("--stable_center_norm", type=float, default=500.0)
    ap.add_argument("--stable_hist_bins", type=int, default=16)

    return ap.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    cfg_yaml = repo_root / "config.yaml"
    if not cfg_yaml.exists():
        raise FileNotFoundError(f"config.yaml not found: {cfg_yaml}")

    import yaml
    ycfg = yaml.safe_load(cfg_yaml.read_text(encoding="utf-8"))

    data_root = get_data_root(repo_root) if args.data_root is None else Path(args.data_root).resolve()

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = (data_root / video_path).resolve()
    else:
        video_path = rpath(data_root, ycfg["paths"]["raw_video"])
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    _ckpt, train_cfg, reg = load_ckpt(Path(args.ckpt).resolve(), mode=args.mode, device=args.device)

    T = args.T if args.T > 0 else int(train_cfg.T)
    img_tf = build_image_tf(int(train_cfg.img_size))

    det_model_path = args.det_model or ycfg["detection_tracking"]["model"]
    pose_model_path = args.pose_model or ycfg["pose"]["model"]
    det_model_path = str((repo_root / det_model_path).resolve()) if not Path(det_model_path).is_absolute() else det_model_path
    pose_model_path = str((repo_root / pose_model_path).resolve()) if not Path(pose_model_path).is_absolute() else pose_model_path

    det = YOLO(det_model_path)
    pose = YOLO(pose_model_path)

    trk_cfg = ycfg["detection_tracking"]
    tracker = args.tracker or trk_cfg.get("tracker", "botsort.yaml")
    tracker_path = (repo_root / tracker).resolve()
    tracker_str = str(tracker_path) if tracker_path.exists() else str(tracker)

    imgsz = int(args.imgsz) if args.imgsz else int(trk_cfg.get("imgsz", 1280))
    conf = float(args.conf) if args.conf is not None else float(trk_cfg.get("conf", 0.35))
    iou = float(args.iou) if args.iou is not None else float(trk_cfg.get("iou", 0.5))
    classes = trk_cfg.get("classes", [0])

    pose_cfg = ycfg["pose"]
    pose_conf = float(pose_cfg.get("conf", 0.25))
    pose_imgsz = int(pose_cfg.get("imgsz", 384))
    crop_long_side = int(ycfg["sampling"].get("crop_long_side", 384))

    assets_dir = Path(args.assets_dir).resolve()
    if args.save_assets:
        (assets_dir / "crops").mkdir(parents=True, exist_ok=True)
        (assets_dir / "poses").mkdir(parents=True, exist_ok=True)

    log_db_path = Path(args.log_db).resolve()
    conn = open_log_db(log_db_path)
    cur = conn.cursor()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if args.annotate_out:
        out_path = Path(args.annotate_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))

    img_buf: Dict[int, Deque[torch.Tensor]] = {}
    pose_buf: Dict[int, Deque[np.ndarray]] = {}
    tracks: Dict[int, TrackState] = {}

    class_avg_ema: Optional[float] = None

    mapper = None
    if args.stable_id:
        mapper = StableIDMapper(
            max_age=int(args.stable_max_age),
            iou_min=float(args.stable_iou_min),
            w_iou=float(args.stable_w_iou),
            w_center=float(args.stable_w_center),
            w_kpt=float(args.stable_w_kpt),
            w_hist=float(args.stable_w_hist),
            kpt_conf=float(args.stable_kpt_conf),
            center_norm=float(args.stable_center_norm),
            hist_bins=int(args.stable_hist_bins),
        )

    results = det.track(
        source=str(video_path),
        stream=True,
        persist=True,
        verbose=False,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        tracker=tracker_str,
        classes=classes,
    )

    frame_idx = -1
    committed = 0

    for r in results:
        frame_idx += 1
        t = float(frame_idx) / float(fps)

        frame = r.orig_img
        if frame is None:
            continue
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        frame = frame.copy()

        if r.boxes is None or r.boxes.id is None:
            tids = np.array([], dtype=np.int32)
            xyxy = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
        else:
            tids = r.boxes.id.cpu().numpy().astype(np.int32)
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = r.boxes.conf.cpu().numpy().astype(np.float32)

        # --- update buffers/states using tracker tid ---
        for i, tid in enumerate(tids.tolist()):
            x1, y1, x2, y2 = xyxy[i].tolist()
            det_conf = float(confs[i])

            bb = clamp_bbox(x1, y1, x2, y2, W, H)
            if bb is None:
                continue
            x1i, y1i, x2i, y2i = bb

            crop = frame[y1i:y2i, x1i:x2i].copy()
            if crop.size == 0:
                continue

            if args.mode in ("image", "fusion"):
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                ft = img_tf(pil)
                img_buf.setdefault(tid, deque(maxlen=T)).append(ft)

            kpts_for_draw = None

            if (args.mode in ("skeleton", "fusion")) or (args.draw_skeleton and args.mode in ("skeleton", "fusion")) or args.mosaic_eyes or (args.stable_id and args.mode in ("skeleton", "fusion")):
                crop_pose_in = crop_resize_for_pose(crop, crop_long_side)
                ph, pw = crop_pose_in.shape[:2]
                ch, cw = crop.shape[:2]

                pr = pose.predict(crop_pose_in, conf=pose_conf, imgsz=pose_imgsz, verbose=False)[0]

                arr = make_pose_empty(K=int(train_cfg.K), D=int(train_cfg.pose_dim))
                if pr.keypoints is not None and len(pr.keypoints.data) > 0:
                    arr = pr.keypoints.data[0].cpu().numpy().astype(np.float32)

                # model input normalize
                try:
                    arr_norm = normalize_pose_kpts(arr, conf_thr=float(args.skel_conf))
                except Exception:
                    arr_norm = arr.copy()
                    arr_norm[:, :2] = 0.0

                if args.mode in ("skeleton", "fusion"):
                    pose_buf.setdefault(tid, deque(maxlen=T)).append(arr_norm)

                # draw coords: raw arr -> frame coords
                kpts_for_draw = pose_to_frame_coords(
                    kpts_xyc_in_poseimg=arr,
                    poseimg_wh=(pw, ph),
                    crop_wh=(cw, ch),
                    crop_xy1=(x1i, y1i),
                )

            frames_in: Optional[torch.Tensor] = None
            poses_in: Optional[torch.Tensor] = None

            if args.mode in ("image", "fusion"):
                seq = pad_seq(list(img_buf.get(tid, [])), T)
                if len(seq) > 0:
                    frames_in = torch.stack(seq, dim=0).unsqueeze(0)

            if args.mode in ("skeleton", "fusion"):
                seqp = pad_seq(list(pose_buf.get(tid, [])), T)
                if len(seqp) > 0:
                    poses_in = torch.from_numpy(np.stack(seqp, axis=0)).unsqueeze(0)

            if (args.mode == "image" and frames_in is None) or (args.mode == "skeleton" and poses_in is None) or (args.mode == "fusion" and (frames_in is None or poses_in is None)):
                score_raw = float("nan")
            else:
                score_raw = predict_score(reg, frames_in, poses_in, device=args.device, situation=args.situation)

            bbox_new = np.array([x1, y1, x2, y2], dtype=np.float32)
            if tid in tracks:
                st = tracks[tid]
                st.bbox = ema_bbox(st.bbox, bbox_new, float(args.bbox_alpha))
                if not math.isnan(score_raw):
                    st.score = ema(st.score, float(score_raw), float(args.score_alpha))
                st.last_seen = frame_idx
                st.det_conf = det_conf
                if kpts_for_draw is not None:
                    st.kpts_frame = kpts_for_draw
            else:
                init_score = float(score_raw) if not math.isnan(score_raw) else 5.0
                tracks[tid] = TrackState(
                    bbox=bbox_new,
                    score=init_score,
                    last_seen=frame_idx,
                    det_conf=det_conf,
                    kpts_frame=kpts_for_draw,
                )

            if args.save_assets:
                key = f"f{frame_idx:06d}_tid{tid}"
                crop_path = assets_dir / "crops" / f"{key}.jpg"
                pose_path = assets_dir / "poses" / f"{key}.json"
                cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                if args.mode in ("skeleton", "fusion"):
                    last_pose = pose_buf.get(tid, deque())[-1] if pose_buf.get(tid, None) else make_pose_empty()
                    pose_path.write_text(
                        json.dumps({"keypoints": last_pose.tolist(), "format": "x,y,conf", "space": "crop"}, ensure_ascii=False),
                        encoding="utf-8"
                    )

        # alive tracks
        alive: List[Tuple[int, TrackState]] = []
        for tid, st in list(tracks.items()):
            if frame_idx - st.last_seen <= int(args.max_age):
                alive.append((tid, st))
            else:
                del tracks[tid]
                img_buf.pop(tid, None)
                pose_buf.pop(tid, None)

        # stable id mapping (optional)
        tid_to_sid: Dict[int, int] = {}
        if mapper is not None:
            dets = [{"tid": tid, "bbox": st.bbox.astype(np.float32), "kpts_frame": st.kpts_frame} for tid, st in alive]
            dets = mapper.assign(frame, frame_idx, dets)
            tid_to_sid = {int(d["tid"]): int(d["stable_id"]) for d in dets}

        # class avg
        scores_now = [float(st.score) for _tid, st in alive if not math.isnan(float(st.score))]
        if scores_now:
            avg_now = float(np.mean(scores_now))
            class_avg_ema = avg_now if class_avg_ema is None else ema(class_avg_ema, avg_now, float(args.class_alpha))
            avg_i = clamp_1to7(class_avg_ema)
            draw_label(
                frame, 20, 55,
                f"Class Avg: {avg_i}   (n={len(scores_now)})",
                font_scale=float(args.class_scale),
                thickness=int(args.class_thick),
            )
        else:
            draw_label(
                frame, 20, 55,
                "Class Avg: -   (n=0)",
                font_scale=float(args.class_scale),
                thickness=int(args.class_thick),
            )

        # draw + mosaic
        for tid, st in alive:
            x1, y1, x2, y2 = st.bbox.astype(int)
            bb = clamp_bbox(x1, y1, x2, y2, W, H)
            if bb is None:
                continue
            x1i, y1i, x2i, y2i = bb

            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), int(args.box_thick))

            score_i = clamp_1to7(float(st.score))
            show_id = tid_to_sid.get(tid, tid) if (args.stable_id and mapper is not None) else tid
            draw_label(
                frame, x1i, y1i,
                f"ID:{int(show_id)}  score:{score_i}",
                font_scale=float(args.label_scale),
                thickness=int(args.label_thick),
            )

            if args.mosaic_eyes and st.kpts_frame is not None:
                mosaic_eyes_from_kpts(
                    frame,
                    st.kpts_frame,
                    conf_thr=float(args.mosaic_conf),
                    pad=float(args.mosaic_pad),
                    mosaic_scale=float(args.mosaic_scale),
                )

            if args.draw_skeleton and st.kpts_frame is not None and args.mode in ("skeleton", "fusion"):
                draw_skeleton(
                    frame,
                    st.kpts_frame,
                    conf_thr=float(args.skel_conf),
                    radius=int(args.skel_radius),
                    line_thick=int(args.skel_line),
                )

        # log DB
        for tid, st in alive:
            x1, y1, x2, y2 = st.bbox.tolist()
            score = float(st.score)
            sid = tid_to_sid.get(tid, None) if (args.stable_id and mapper is not None) else None
            cur.execute(
                """
                INSERT INTO predictions(video_path, frame_idx, t, track_id, stable_id, score, score_clamped, situation, det_conf, x1,y1,x2,y2)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(video_path),
                    int(frame_idx),
                    float(t),
                    int(tid),
                    (int(sid) if sid is not None else None),
                    float(score),
                    int(clamp_1to7(score)),
                    args.situation,
                    float(st.det_conf),
                    float(x1), float(y1), float(x2), float(y2),
                )
            )

        if frame_idx % int(args.log_every) == 0:
            conn.commit()
            committed += 1

        if writer is not None:
            writer.write(frame)

        if frame_idx % 120 == 0:
            print(f"frame={frame_idx} t={t:.2f}s alive={len(alive)} commits={committed} stable_id={bool(args.stable_id)}")

    conn.commit()
    conn.close()
    if writer is not None:
        writer.release()

    print("DONE.")
    if args.annotate_out:
        print("annotated video:", str(Path(args.annotate_out).resolve()))
    print("log db:", str(Path(args.log_db).resolve()))


if __name__ == "__main__":
    try:
        main()
        import winsound
        try:
            winsound.PlaySound("mei_kara_mei_switch1.wav", winsound.SND_FILENAME)
        except Exception as e:
            print(f"[WARN] 音声を再生できませんでした: {e}")
    finally:
        elapsed = time.perf_counter() - SCRIPT_STARTED_AT
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"所要時間: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} ({elapsed:.2f}秒)")

# command

# 目モザイク + 骨格線 + stable_id（おすすめ）
# python scripts/08_video_offline.py --ckpt models/skeleton_modes.pt --mode skeleton --data_root datasets/hara --annotate_out outputs/annot.mp4 --draw_skeleton --mosaic_eyes --stable_id

# stable_id を強めたいとき（交差が多い教室向け）
# python scripts/08_video_offline.py --ckpt models/skeleton_modes.pt --mode skeleton --situation 聞く --data_root datasets/hara --annotate_out outputs/annot.mp4 --draw_skeleton --stable_id --stable_iou_min 0.10 --stable_w_iou 3.0 --stable_w_center 1.2
