# scripts/08_realtime.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal, Tuple, List, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from lib.db import connect, init_db, upsert_video, ensure_track
from lib.image_roi import image_model_crop
from lib.face_exclusion import FaceDbMatcher, OnlineFaceExclusionTracker
from lib.runroot import get_data_root, rpath
from lib.pose_norm import normalize_pose_kpts
from lib.seq_buffer import MultiTrackBuffer
from lib.infer import (
    load_ckpt, load_situation_model, build_image_tf, predict_score,
    predict_situation_probs, SituationProbabilitySmoother, clamp_1to7,
)
from lib.situation import SITUATIONS

Mode = Literal["image", "skeleton", "fusion"]
SITUATION_DISPLAY = ("listen", "write", "discuss")


# -------------------------
# Skeleton (COCO-17) bones
# -------------------------
COCO17_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# -------------------------
# Utils
# -------------------------
def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= 0:
        return img
    s = float(long_side) / float(max(h, w))
    if abs(s - 1.0) < 1e-6:
        return img
    nh, nw = int(round(h * s)), int(round(w * s))
    if nh <= 1 or nw <= 1:
        return img
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W - 1, int(x2)))
    y2 = max(0, min(H - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def posix_rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def ema(old: float, new: float, alpha: float) -> float:
    return old * (1.0 - alpha) + new * alpha


def ema_bbox(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return old * (1.0 - alpha) + new * alpha


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
    # COCO17: left_eye=1, right_eye=2, nose=0
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
# Window helpers
# -------------------------
def setup_window(win_name: str, win_w: int, win_h: int, fullscreen: bool):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if win_w > 0 and win_h > 0:
        try:
            cv2.resizeWindow(win_name, int(win_w), int(win_h))
        except Exception:
            pass

    if fullscreen:
        try:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass


def set_fullscreen(win_name: str, enable: bool):
    try:
        cv2.setWindowProperty(
            win_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if enable else cv2.WINDOW_NORMAL,
        )
    except Exception:
        pass


# -------------------------
# Stable ID remap (reduce ID switches)  [NEW]
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
    last_t: float
    kfeat: np.ndarray
    hist: np.ndarray


class StableIDMapper:
    """
    Greedy assignment for speed.
    Cost = w_iou*(1-iou) + w_center*center_dist_norm + w_kpt*cos_dist + w_hist*l1
    """
    def __init__(
            self,
            *,
            max_age_sec: float = 3.5,
            iou_min: float = 0.05,
            w_iou: float = 2.5,
            w_center: float = 1.0,
            w_kpt: float = 1.6,
            w_hist: float = 0.7,
            kpt_conf: float = 0.2,
            center_norm: float = 500.0,
            hist_bins: int = 16,
    ):
        self.max_age_sec = float(max_age_sec)
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

    def gc(self, t: float):
        dead = [sid for sid, tr in self.tracks.items() if (t - tr.last_t) > self.max_age_sec]
        for sid in dead:
            del self.tracks[sid]

    def assign(self, frame_bgr: np.ndarray, t: float, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.gc(t)

        # features
        for d in dets:
            bbox = d["bbox"]
            d["center"] = _center(bbox)
            d["kfeat"] = _kpt_feat(d.get("kpts_frame", None), conf_thr=self.kpt_conf)
            d["hist"] = _hsv_hist(frame_bgr, bbox, bins=self.hist_bins)

        sids = list(self.tracks.keys())
        if not sids:
            for d in dets:
                sid = self._next_id; self._next_id += 1
                self.tracks[sid] = StableTrack(
                    stable_id=sid,
                    bbox=d["bbox"].copy(),
                    last_t=float(t),
                    kfeat=d["kfeat"],
                    hist=d["hist"],
                )
                d["stable_id"] = sid
            return dets

        pairs = []
        for j, d in enumerate(dets):
            best = []
            for sid in sids:
                tr = self.tracks[sid]
                iou = _iou(tr.bbox, d["bbox"])
                if iou < self.iou_min:
                    continue
                cdist = float(np.linalg.norm(_center(tr.bbox) - d["center"])) / self.center_norm
                kdist = _cos_dist(tr.kfeat, d["kfeat"])
                hdist = _l1(tr.hist, d["hist"])
                cost = self.w_iou * (1.0 - iou) + self.w_center * cdist + self.w_kpt * kdist + self.w_hist * hdist
                best.append((cost, sid))
            if best:
                cost, sid = min(best, key=lambda x: x[0])
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
            tr.last_t = float(t)
            tr.kfeat = dets[j]["kfeat"]
            tr.hist = dets[j]["hist"]

        for j, d in enumerate(dets):
            if "stable_id" in d:
                continue
            sid = self._next_id; self._next_id += 1
            self.tracks[sid] = StableTrack(
                stable_id=sid,
                bbox=d["bbox"].copy(),
                last_t=float(t),
                kfeat=d["kfeat"],
                hist=d["hist"],
            )
            d["stable_id"] = sid

        return dets


# -------------------------
# States (for stable drawing)
# -------------------------
@dataclass
class TrackDrawState:
    bbox: np.ndarray              # float32 [x1,y1,x2,y2] in frame coords
    score: float                  # raw score (ema)
    last_seen_t: float            # seconds
    det_conf: float = 1.0
    kpts_frame: Optional[np.ndarray] = None  # [K,3] in frame coords
    excluded: bool = False
    excluded_name: Optional[str] = None


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="07_train.py で保存した .pt")
    ap.add_argument("--mode", type=str, default="fusion", choices=["image", "skeleton", "fusion"])
    ap.add_argument(
        "--situation", type=str, choices=SITUATIONS, default=None,
        help="Manual override. Omit to estimate the situation automatically.",
    )
    ap.add_argument("--situation_alpha", type=float, default=0.15)
    situation_feature_group = ap.add_mutually_exclusive_group()
    situation_feature_group.add_argument(
        "--use-situation-feature", "--use_situation_feature",
        dest="use_situation_feature", action="store_true",
        help="Feed situation probabilities into concentration prediction.",
    )
    situation_feature_group.add_argument(
        "--no-situation-feature", "--no_situation_feature",
        dest="use_situation_feature", action="store_false",
        help="Keep situation estimation/logging/display, but exclude it from concentration prediction.",
    )
    ap.set_defaults(use_situation_feature=None)
    ap.add_argument("--source", type=str, default="0", help="0(webcam) or video path")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--show", action="store_true", help="ウィンドウ表示")
    ap.add_argument("--save_assets", action="store_true", help="crop/pose を保存してDBに記録")

    # window controls
    ap.add_argument("--win_name", type=str, default="focus realtime", help="表示ウィンドウ名")
    ap.add_argument("--win_w", type=int, default=0, help="ウィンドウ幅（0なら未指定）")
    ap.add_argument("--win_h", type=int, default=0, help="ウィンドウ高さ（0なら未指定）")
    ap.add_argument("--fullscreen", action="store_true", help="起動時に全画面（fキーでトグル可）")

    # lightweight controls
    ap.add_argument("--ttl", type=float, default=3.0, help="表示/状態保持TTL秒（点滅抑制）")
    ap.add_argument("--sample_fps", type=float, default=0.0, help="0ならconfig。軽量化なら 1～2 推奨")
    ap.add_argument("--det_imgsz", type=int, default=0, help="0ならconfig。軽量化なら 640/960 推奨")
    ap.add_argument("--pose_imgsz", type=int, default=0, help="0ならconfig。軽量化なら 256～320 推奨")
    ap.add_argument("--crop_long", type=int, default=0, help="0ならconfig。軽量化なら 256～320 推奨")
    ap.add_argument("--max_tracks_draw", type=int, default=30, help="描画上限（多人数の時に軽量化）")

    # smoothing
    ap.add_argument("--bbox_alpha", type=float, default=0.55)
    ap.add_argument("--score_alpha", type=float, default=0.30)
    ap.add_argument("--class_alpha", type=float, default=0.20)

    # draw sizes
    ap.add_argument("--box_thick", type=int, default=3)
    ap.add_argument("--label_scale", type=float, default=1.0)
    ap.add_argument("--label_thick", type=int, default=3)
    ap.add_argument("--class_scale", type=float, default=1.6)
    ap.add_argument("--class_thick", type=int, default=4)

    # skeleton
    ap.add_argument("--draw_skeleton", action="store_true", help="骨格を表示（重いのでデフォルトOFF）")
    ap.add_argument("--skel_radius", type=int, default=3)
    ap.add_argument("--skel_line", type=int, default=2)
    ap.add_argument("--skel_conf", type=float, default=0.2)

    # eye mosaic
    ap.add_argument("--mosaic_eyes", action="store_true", help="顔keypointから目周辺をモザイク（skeleton/fusionで有効）")
    ap.add_argument("--mosaic_conf", type=float, default=0.25, help="顔kpt conf閾値")
    ap.add_argument("--mosaic_pad", type=float, default=1.8, help="目領域の拡張倍率")
    ap.add_argument("--mosaic_scale", type=float, default=0.06, help="モザイク強さ(小さいほど強い) 例:0.04〜0.10")

    # stable id
    ap.add_argument("--stable_id", action="store_true", help="ID入れ替わり抑制：stable_idで表示する（tidはDBに保存）")
    ap.add_argument("--stable_age", type=float, default=3.5, help="stable_idの保持秒（長いほど粘る）")
    ap.add_argument("--stable_iou_min", type=float, default=0.05)
    ap.add_argument("--stable_w_iou", type=float, default=2.5)
    ap.add_argument("--stable_w_center", type=float, default=1.0)
    ap.add_argument("--stable_w_kpt", type=float, default=1.6)
    ap.add_argument("--stable_w_hist", type=float, default=0.7)
    ap.add_argument("--stable_kpt_conf", type=float, default=0.2)
    ap.add_argument("--stable_center_norm", type=float, default=500.0)
    ap.add_argument("--stable_hist_bins", type=int, default=16)

    ap.add_argument("--commit_every", type=int, default=15)
    return ap.parse_args()


def main():
    args = parse_args()

    here = Path(__file__).resolve()
    repo_root = here.parent.parent

    import yaml
    cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))

    data_root = Path(args.data_root).resolve() if args.data_root else get_data_root(repo_root)
    db_path = rpath(data_root, cfg["paths"]["db_path"])
    crop_dir = rpath(data_root, cfg["paths"]["crop_dir"])
    pose_dir = rpath(data_root, cfg["paths"]["pose_dir"])
    if args.save_assets:
        crop_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

    detcfg = cfg["detection_tracking"]
    posecfg = cfg["pose"]
    sampcfg = cfg["sampling"]
    segcfg = cfg["segments"]

    sample_fps = float(args.sample_fps) if args.sample_fps > 0 else float(sampcfg["sample_fps"])
    window_sec = float(segcfg["window_sec"])
    T = max(1, int(round(sample_fps * window_sec)))

    crop_long = int(args.crop_long) if args.crop_long > 0 else int(sampcfg["crop_long_side"])
    det_imgsz = int(args.det_imgsz) if args.det_imgsz > 0 else int(detcfg["imgsz"])
    pose_imgsz = int(args.pose_imgsz) if args.pose_imgsz > 0 else int(posecfg["imgsz"])

    det_model = YOLO(str((repo_root / detcfg["model"]).resolve()))

    face_tracker = None
    face_cfg = cfg.get("face_exclusion", {})
    if bool(face_cfg.get("enabled", False)):
        face_db_path = (repo_root / str(face_cfg.get("db_path", "models/face_db.npz"))).resolve()
        if not face_db_path.exists():
            raise FileNotFoundError(f"face DB not found: {face_db_path}")
        face_tracker = OnlineFaceExclusionTracker(
            FaceDbMatcher(
                face_db_path,
                threshold=float(face_cfg.get("similarity_threshold", 0.50)),
                device=str(face_cfg.get("device", args.device)),
            ),
            min_matches=int(face_cfg.get("min_matches", 2)),
            min_match_ratio=float(face_cfg.get("min_match_ratio", 0.50)),
            max_frames=int(face_cfg.get("max_frames", 8)),
        )
        print(f"face exclusion: enabled, DB={face_db_path}")

    need_pose = (args.mode in ("skeleton", "fusion"))
    draw_pose = bool(args.draw_skeleton) and (args.mode in ("skeleton", "fusion"))
    need_mosaic = bool(args.mosaic_eyes) and (args.mode in ("skeleton", "fusion"))
    need_stable_kpt = bool(args.stable_id) and (args.mode in ("skeleton", "fusion"))
    pose_model = YOLO(str((repo_root / posecfg["model"]).resolve())) if (need_pose or draw_pose or need_mosaic or need_stable_kpt) else None

    ckpt, train_cfg, reg = load_ckpt(Path(args.ckpt).resolve(), mode=args.mode, device=args.device)
    use_situation_feature = (
        bool(train_cfg.use_situation_feature)
        if args.use_situation_feature is None else bool(args.use_situation_feature)
    )
    reg.use_situation_feature = use_situation_feature
    print(f"use_situation_feature={use_situation_feature}")
    image_roi = str(getattr(train_cfg, "image_roi", sampcfg.get("image_roi", "upper_body")))
    situation_model = None
    situation_smoother = SituationProbabilitySmoother(args.situation_alpha)
    if args.situation is None:
        situation_model = load_situation_model(
            ckpt, train_cfg, mode=args.mode, device=args.device
        )
    current_situation = args.situation or SITUATIONS[0]
    current_situation_probs = torch.zeros(len(SITUATIONS), dtype=torch.float32)
    current_situation_probs[SITUATIONS.index(current_situation)] = 1.0
    img_tf = build_image_tf(int(train_cfg.img_size))

    conn = connect(db_path)
    init_db(conn)
    prediction_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(predictions)")
    }
    if "situation_confidence" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN situation_confidence REAL")
    if "situation_probs" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN situation_probs TEXT")
    conn.commit()

    src = args.source
    cap = cv2.VideoCapture(0 if src.strip() == "0" else src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_id = upsert_video(conn, f"realtime:{src}", float(fps), int(W), int(H), int(0))

    buffers = MultiTrackBuffer(mode=args.mode, T=T, ttl_sec=float(args.ttl))
    latest: Dict[int, TrackDrawState] = {}

    mapper = None
    if args.stable_id:
        mapper = StableIDMapper(
            max_age_sec=float(args.stable_age),
            iou_min=float(args.stable_iou_min),
            w_iou=float(args.stable_w_iou),
            w_center=float(args.stable_w_center),
            w_kpt=float(args.stable_w_kpt),
            w_hist=float(args.stable_w_hist),
            kpt_conf=float(args.stable_kpt_conf),
            center_norm=float(args.stable_center_norm),
            hist_bins=int(args.stable_hist_bins),
        )

    last_sample_t = -1e9
    frame_idx = 0
    sample_count = 0
    class_avg_ema: Optional[float] = None

    tracker_cfg = detcfg["tracker"]
    tracker_path = (repo_root / tracker_cfg).resolve()
    tracker_str = str(tracker_path) if tracker_path.exists() else str(tracker_cfg)

    # window
    fullscreen_state = bool(args.fullscreen)
    if args.show:
        setup_window(args.win_name, int(args.win_w), int(args.win_h), fullscreen_state)

    print(f"[realtime] mode={args.mode} T={T} sample_fps={sample_fps} det_imgsz={det_imgsz} pose_imgsz={pose_imgsz} crop_long={crop_long} image_roi={image_roi}")
    print(f"[realtime] draw_skeleton={bool(args.draw_skeleton)} mosaic_eyes={bool(args.mosaic_eyes)} stable_id={bool(args.stable_id)} save_assets={bool(args.save_assets)} ttl={args.ttl}s")
    if args.show:
        print(f"[realtime] window='{args.win_name}' size=({args.win_w},{args.win_h}) fullscreen={fullscreen_state} (toggle: f)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = frame_idx / float(fps)

            # --- sampling ---
            do_sample = (t - last_sample_t) >= (1.0 / sample_fps - 1e-6)
            if do_sample:
                last_sample_t = t
                sample_count += 1

                rs = det_model.track(
                    source=frame,
                    conf=float(detcfg["conf"]),
                    iou=float(detcfg["iou"]),
                    imgsz=int(det_imgsz),
                    tracker=tracker_str,
                    persist=True,
                    classes=detcfg.get("classes", None),
                    verbose=False,
                )
                r = rs[0]

                if r.boxes is not None and r.boxes.id is not None:
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                    confs = r.boxes.conf.cpu().numpy().astype(np.float32)
                    tids = r.boxes.id.cpu().numpy().astype(int)

                    for i, tid in enumerate(tids.tolist()):
                        ensure_track(conn, video_id, int(tid))

                        x1, y1, x2, y2 = xyxy[i].tolist()
                        det_conf = float(confs[i])

                        bb = clamp_bbox(x1, y1, x2, y2, W, H)
                        if bb is None:
                            continue
                        x1i, y1i, x2i, y2i = bb

                        crop0 = frame[y1i:y2i, x1i:x2i].copy()
                        if crop0.size == 0:
                            continue

                        excluded = False
                        excluded_name = None
                        if face_tracker is not None:
                            excluded, excluded_name = face_tracker.update(tid, crop0)
                        bbox_new = np.array([x1, y1, x2, y2], dtype=np.float32)
                        if excluded:
                            buffers.buf.pop(int(tid), None)
                            if tid in latest:
                                st = latest[tid]
                                st.bbox = ema_bbox(st.bbox, bbox_new, float(args.bbox_alpha))
                                st.score = float("nan")
                                st.last_seen_t = float(t)
                                st.det_conf = det_conf
                                st.excluded = True
                                st.excluded_name = excluded_name
                            else:
                                latest[tid] = TrackDrawState(
                                    bbox=bbox_new,
                                    score=float("nan"),
                                    last_seen_t=float(t),
                                    det_conf=det_conf,
                                    excluded=True,
                                    excluded_name=excluded_name,
                                )
                            continue
                        if tid in latest:
                            latest[tid].excluded = False
                            latest[tid].excluded_name = None

                        crop_h0, crop_w0 = crop0.shape[:2]
                        pose_crop = resize_long_side(crop0, crop_long)
                        crop = resize_long_side(image_model_crop(crop0, image_roi), crop_long)

                        frame_tensor = None
                        if args.mode in ("image", "fusion"):
                            from PIL import Image
                            im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            frame_tensor = img_tf(im)

                        pose_tensor = None
                        pose_json = {}
                        kpts_frame_for_draw = None

                        if (args.mode in ("skeleton", "fusion")) and (pose_model is not None):
                            pr = pose_model.predict(
                                pose_crop,
                                conf=float(posecfg["conf"]),
                                imgsz=int(pose_imgsz),
                                verbose=False
                            )[0]

                            K = int(train_cfg.K)
                            if pr.keypoints is not None and len(pr.keypoints.data) > 0:
                                kpts = pr.keypoints.data[0].cpu().numpy().astype(np.float32)
                            else:
                                kpts = np.zeros((K, 3), dtype=np.float32)

                            kpts_norm = normalize_pose_kpts(kpts, conf_thr=float(args.skel_conf))
                            pose_tensor = torch.from_numpy(kpts_norm)

                            if args.save_assets:
                                pose_json = {"keypoints": kpts_norm.tolist(), "format": "x,y,conf", "space": "person_crop", "normalized": "mean+RMS"}

                            # map resized-crop coords -> crop0 coords -> frame coords
                            if draw_pose or need_mosaic or need_stable_kpt:
                                ch, cw = pose_crop.shape[:2]
                                kpts_on_crop0 = kpts.copy()
                                if cw > 0 and ch > 0:
                                    sx0 = float(crop_w0) / float(cw)
                                    sy0 = float(crop_h0) / float(ch)
                                    kpts_on_crop0[:, 0] *= sx0
                                    kpts_on_crop0[:, 1] *= sy0
                                kpts_frame_for_draw = kpts_on_crop0.copy()
                                kpts_frame_for_draw[:, 0] += float(x1i)
                                kpts_frame_for_draw[:, 1] += float(y1i)

                        # push buffer
                        buf = buffers.get(int(tid))
                        buf.push(frame_tensor=frame_tensor, pose_tensor=pose_tensor, t=t)

                        if buf.ready():
                            frames_b, poses_b = buf.get_batch()
                            if situation_model is not None:
                                local_probs = predict_situation_probs(
                                    situation_model, frames_b, poses_b, args.device
                                )
                                situation_probs = situation_smoother.update(local_probs)
                                current_situation_probs = situation_probs
                                current_situation = SITUATIONS[int(situation_probs.argmax().item())]
                                yhat = predict_score(
                                    reg, frames_b, poses_b, args.device,
                                    situation_probs=situation_probs,
                                )
                            else:
                                current_situation = str(args.situation)
                                yhat = predict_score(
                                    reg, frames_b, poses_b, args.device,
                                    situation=current_situation,
                                )

                            bbox_new = np.array([x1, y1, x2, y2], dtype=np.float32)
                            if tid in latest:
                                st = latest[tid]
                                st.excluded = False
                                st.excluded_name = None
                                st.bbox = ema_bbox(st.bbox, bbox_new, float(args.bbox_alpha))
                                st.score = ema(st.score, float(yhat), float(args.score_alpha))
                                st.last_seen_t = float(t)
                                st.det_conf = det_conf
                                if kpts_frame_for_draw is not None:
                                    st.kpts_frame = kpts_frame_for_draw
                            else:
                                latest[tid] = TrackDrawState(
                                    bbox=bbox_new,
                                    score=float(yhat),
                                    last_seen_t=float(t),
                                    det_conf=det_conf,
                                    kpts_frame=kpts_frame_for_draw,
                                    excluded=False,
                                )

                            crop_rel = None
                            pose_rel = None
                            if args.save_assets:
                                t_key = int(round(t * 1000))
                                crop_path = crop_dir / f"rt_f{frame_idx}_t{t_key}ms_tid{tid}.jpg"
                                pose_path = pose_dir / f"rt_f{frame_idx}_t{t_key}ms_tid{tid}.json"
                                cv2.imwrite(str(crop_path), pose_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                pose_path.write_text(json.dumps(pose_json, ensure_ascii=False), encoding="utf-8")
                                crop_rel = posix_rel(crop_path, data_root)
                                pose_rel = posix_rel(pose_path, data_root)

                            conn.execute(
                                """
                                INSERT INTO predictions(
                                    video_id, frame_idx, t, track_id, mode, yhat, score_int, situation,
                                    situation_confidence, situation_probs,
                                    x1,y1,x2,y2, det_conf, crop_path, pose_path
                                )
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                """,
                                (
                                    video_id, int(frame_idx), float(t), int(tid), str(args.mode),
                                    float(yhat), int(clamp_1to7(yhat)), current_situation,
                                    float(current_situation_probs.max().item()),
                                    json.dumps(current_situation_probs.tolist()),
                                    float(x1), float(y1), float(x2), float(y2),
                                    float(det_conf),
                                    crop_rel, pose_rel
                                )
                            )

                if sample_count % int(max(1, args.commit_every)) == 0:
                    conn.commit()

                buffers.gc(t)

                # TTL GC (tid-states)
                for tid in list(latest.keys()):
                    if (t - latest[tid].last_seen_t) > float(args.ttl):
                        del latest[tid]
                        if face_tracker is not None:
                            face_tracker.forget(tid)

            # --- stable_id mapping for display (every frame) ---
            tid_to_sid: Dict[int, int] = {}
            if mapper is not None:
                dets = [{"tid": int(tid), "bbox": st.bbox.astype(np.float32), "kpts_frame": st.kpts_frame} for tid, st in latest.items()]
                dets = mapper.assign(frame, t, dets)
                tid_to_sid = {int(d["tid"]): int(d["stable_id"]) for d in dets}

            # --- draw ---
            scores = [
                st.score for st in latest.values()
                if not st.excluded and not math.isnan(float(st.score))
            ]
            if scores:
                avg_now = float(np.mean(scores))
                class_avg_ema = avg_now if class_avg_ema is None else ema(class_avg_ema, avg_now, float(args.class_alpha))
                avg_display = min(7.0, max(1.0, float(class_avg_ema)))
                draw_label(frame, 20, 55, f"Class Avg: {avg_display:.2f}   (n={len(scores)})",
                           font_scale=float(args.class_scale), thickness=int(args.class_thick))
            else:
                draw_label(frame, 20, 55, "Class Avg: -   (n=0)",
                           font_scale=float(args.class_scale), thickness=int(args.class_thick))

            situation_index = SITUATIONS.index(current_situation)
            draw_label(
                frame, 20, 105,
                f"Situation: {SITUATION_DISPLAY[situation_index]} "
                f"({float(current_situation_probs.max().item()):.2f})",
                font_scale=1.0, thickness=2,
            )

            items = list(latest.items())
            items.sort(key=lambda kv: kv[1].det_conf, reverse=True)
            items = items[: int(args.max_tracks_draw)]

            for tid, st in items:
                x1, y1, x2, y2 = st.bbox.astype(int)
                bb = clamp_bbox(x1, y1, x2, y2, W, H)
                if bb is None:
                    continue
                x1i, y1i, x2i, y2i = bb

                show_id = int(tid_to_sid.get(int(tid), int(tid))) if mapper is not None else int(tid)
                if st.excluded:
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 128, 255), int(args.box_thick))
                    suffix = f" ({st.excluded_name})" if st.excluded_name else ""
                    draw_label(
                        frame, x1i, y1i, f"ID:{show_id}  EXCLUDED{suffix}",
                        font_scale=float(args.label_scale),
                        thickness=int(args.label_thick),
                    )
                elif not math.isnan(float(st.score)):
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), int(args.box_thick))
                    score_i = clamp_1to7(float(st.score))
                    draw_label(frame, x1i, y1i, f"ID:{show_id}  score:{score_i}",
                               font_scale=float(args.label_scale), thickness=int(args.label_thick))
                else:
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), int(args.box_thick))
                    draw_label(
                        frame, x1i, y1i, f"ID:{show_id}  warm-up",
                        font_scale=max(0.50, float(args.label_scale) * 0.66),
                        thickness=1,
                    )

                if args.mosaic_eyes and (args.mode in ("skeleton", "fusion")) and st.kpts_frame is not None:
                    mosaic_eyes_from_kpts(
                        frame, st.kpts_frame,
                        conf_thr=float(args.mosaic_conf),
                        pad=float(args.mosaic_pad),
                        mosaic_scale=float(args.mosaic_scale),
                    )

                if args.draw_skeleton and st.kpts_frame is not None:
                    draw_skeleton(
                        frame, st.kpts_frame,
                        conf_thr=float(args.skel_conf),
                        radius=int(args.skel_radius),
                        line_thick=int(args.skel_line),
                    )

            if args.show:
                cv2.imshow(args.win_name, frame)
                k = cv2.waitKey(1) & 0xFF

                if k == ord("f"):
                    fullscreen_state = not fullscreen_state
                    set_fullscreen(args.win_name, fullscreen_state)

                if k == 27 or k == ord("q"):
                    break

            frame_idx += 1

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        conn.commit()
        conn.close()
        print("DONE realtime. DB:", db_path)


if __name__ == "__main__":
    main()

# examples:
# stable_id + fullscreen toggle
# python scripts/08_realtime.py --ckpt models/gru_fusion_modes.pt --mode fusion --source 0 --show --fullscreen --stable_id --data_root realtime
#
# stable_id + mosaic (skeleton/fusion)
# python scripts/08_realtime.py --ckpt models/skeleton_modes.pt --mode skeleton --source 0 --show --stable_id --mosaic_eyes --data_root realtime
#
# tune stable matcher (more strict):
# python scripts/08_realtime.py --ckpt models/skeleton_modes.pt --mode skeleton --situation 聞く --source 0 --show --stable_id --stable_iou_min 0.10 --stable_w_iou 3.0 --stable_w_center 1.2 --data_root realtime
