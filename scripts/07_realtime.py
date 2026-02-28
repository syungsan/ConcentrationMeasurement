# scripts/07_realtime.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal, Tuple, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from lib.db import connect, init_db, upsert_video, ensure_track
from lib.runroot import get_data_root, rpath
from lib.pose_norm import normalize_pose_kpts
from lib.seq_buffer import MultiTrackBuffer
from lib.infer import load_ckpt, build_image_tf, predict_score, clamp_1to10

Mode = Literal["image", "skeleton", "fusion"]


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
    """
    COCO17: left_eye=1, right_eye=2, nose=0
    目周辺を推定してモザイクする（face bbox不要）
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
# States (for stable drawing)
# -------------------------
@dataclass
class TrackDrawState:
    bbox: np.ndarray              # float32 [x1,y1,x2,y2] in frame coords
    score: float                  # raw score (ema)
    last_seen_t: float            # seconds
    det_conf: float = 1.0
    kpts_frame: Optional[np.ndarray] = None  # [K,3] in frame coords


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="06_train.py で保存した .pt")
    ap.add_argument("--mode", type=str, default="fusion", choices=["image", "skeleton", "fusion"])
    ap.add_argument("--source", type=str, default="0", help="0(webcam) or video path")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--show", action="store_true", help="ウィンドウ表示")
    ap.add_argument("--save_assets", action="store_true", help="crop/pose を保存してDBに記録")

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

    # eye mosaic (only when pose is available: mode skeleton/fusion)
    ap.add_argument("--mosaic_eyes", action="store_true", help="顔keypointから目周辺をモザイク（skeleton/fusionで有効）")
    ap.add_argument("--mosaic_conf", type=float, default=0.25, help="顔kpt conf閾値")
    ap.add_argument("--mosaic_pad", type=float, default=1.8, help="目領域の拡張倍率")
    ap.add_argument("--mosaic_scale", type=float, default=0.06, help="モザイク強さ(小さいほど強い) 例:0.04〜0.10")

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

    # models
    det_model = YOLO(str((repo_root / detcfg["model"]).resolve()))

    need_pose = (args.mode in ("skeleton", "fusion"))  # for model input
    draw_pose = bool(args.draw_skeleton) and (args.mode in ("skeleton", "fusion"))
    need_mosaic = bool(args.mosaic_eyes) and (args.mode in ("skeleton", "fusion"))
    pose_model = YOLO(str((repo_root / posecfg["model"]).resolve())) if (need_pose or draw_pose or need_mosaic) else None

    # regressor
    _ckpt, train_cfg, reg = load_ckpt(Path(args.ckpt).resolve(), mode=args.mode, device=args.device)
    img_tf = build_image_tf(int(train_cfg.img_size))

    # db
    conn = connect(db_path)
    init_db(conn)

    # video source
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

    last_sample_t = -1e9
    frame_idx = 0
    sample_count = 0
    class_avg_ema: Optional[float] = None

    tracker_cfg = detcfg["tracker"]
    tracker_path = (repo_root / tracker_cfg).resolve()
    tracker_str = str(tracker_path) if tracker_path.exists() else str(tracker_cfg)

    print(f"[realtime] mode={args.mode} T={T} sample_fps={sample_fps} det_imgsz={det_imgsz} pose_imgsz={pose_imgsz} crop_long={crop_long}")
    print(f"[realtime] draw_skeleton={bool(args.draw_skeleton)} mosaic_eyes={bool(args.mosaic_eyes)} save_assets={bool(args.save_assets)} ttl={args.ttl}s")

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

                        # keep original crop0 size for mapping
                        crop_h0, crop_w0 = crop0.shape[:2]

                        # downscale crop for speed
                        crop = resize_long_side(crop0, crop_long)

                        # image tensor
                        frame_tensor = None
                        if args.mode in ("image", "fusion"):
                            from PIL import Image
                            im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            frame_tensor = img_tf(im)

                        # pose tensor + kpts for draw/mosaic
                        pose_tensor = None
                        pose_json = {}
                        kpts_frame_for_draw = None

                        if (args.mode in ("skeleton", "fusion")) and (pose_model is not None):
                            pr = pose_model.predict(
                                crop,
                                conf=float(posecfg["conf"]),
                                imgsz=int(pose_imgsz),
                                verbose=False
                            )[0]

                            K = int(train_cfg.K)
                            if pr.keypoints is not None and len(pr.keypoints.data) > 0:
                                kpts = pr.keypoints.data[0].cpu().numpy().astype(np.float32)  # on resized-crop coords
                            else:
                                kpts = np.zeros((K, 3), dtype=np.float32)

                            # normalize for model input
                            kpts_norm = normalize_pose_kpts(kpts, conf_thr=float(args.skel_conf))
                            pose_tensor = torch.from_numpy(kpts_norm)

                            if args.save_assets:
                                pose_json = {
                                    "keypoints": kpts_norm.tolist(),
                                    "format": "x,y,conf",
                                    "space": "crop",
                                    "normalized": "mean+RMS",
                                }

                            # map resized-crop coords -> crop0 coords -> frame coords
                            if draw_pose or need_mosaic:
                                ch, cw = crop.shape[:2]
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
                            yhat = predict_score(reg, frames_b, poses_b, args.device)
                            _score_int = clamp_1to10(yhat)

                            bbox_new = np.array([x1, y1, x2, y2], dtype=np.float32)
                            if tid in latest:
                                st = latest[tid]
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
                                )

                            # save assets + DB insert (optional)
                            crop_rel = None
                            pose_rel = None
                            if args.save_assets:
                                t_key = int(round(t * 1000))
                                crop_path = crop_dir / f"rt_f{frame_idx}_t{t_key}ms_tid{tid}.jpg"
                                pose_path = pose_dir / f"rt_f{frame_idx}_t{t_key}ms_tid{tid}.json"
                                cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                pose_path.write_text(json.dumps(pose_json, ensure_ascii=False), encoding="utf-8")
                                crop_rel = posix_rel(crop_path, data_root)
                                pose_rel = posix_rel(pose_path, data_root)

                            conn.execute(
                                """
                                INSERT INTO predictions(
                                    video_id, frame_idx, t, track_id, mode, yhat, score_int,
                                    x1,y1,x2,y2, det_conf, crop_path, pose_path
                                )
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                """,
                                (
                                    video_id, int(frame_idx), float(t), int(tid), str(args.mode),
                                    float(yhat), int(clamp_1to10(yhat)),
                                    float(x1), float(y1), float(x2), float(y2),
                                    float(det_conf),
                                    crop_rel, pose_rel
                                )
                            )

                if sample_count % int(max(1, args.commit_every)) == 0:
                    conn.commit()

                buffers.gc(t)

                # TTL GC
                for tid in list(latest.keys()):
                    if (t - latest[tid].last_seen_t) > float(args.ttl):
                        del latest[tid]

            # --- draw (every frame) ---
            scores = [st.score for st in latest.values() if not math.isnan(float(st.score))]
            if scores:
                avg_now = float(np.mean(scores))
                class_avg_ema = avg_now if class_avg_ema is None else ema(class_avg_ema, avg_now, float(args.class_alpha))
                avg_i = clamp_1to10(class_avg_ema)
                draw_label(
                    frame, 20, 55,
                    f"Class Avg: {avg_i}   (n={len(scores)})",
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

            # draw top N
            items = list(latest.items())
            items.sort(key=lambda kv: kv[1].det_conf, reverse=True)
            items = items[: int(args.max_tracks_draw)]

            for tid, st in items:
                x1, y1, x2, y2 = st.bbox.astype(int)
                bb = clamp_bbox(x1, y1, x2, y2, W, H)
                if bb is None:
                    continue
                x1i, y1i, x2i, y2i = bb

                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), int(args.box_thick))
                score_i = clamp_1to10(float(st.score))
                draw_label(
                    frame, x1i, y1i,
                    f"ID:{int(tid)}  score:{score_i}",
                    font_scale=float(args.label_scale),
                    thickness=int(args.label_thick),
                )

                # mosaic first (so skeleton stays visible if enabled)
                if args.mosaic_eyes and (args.mode in ("skeleton", "fusion")) and st.kpts_frame is not None:
                    mosaic_eyes_from_kpts(
                        frame,
                        st.kpts_frame,
                        conf_thr=float(args.mosaic_conf),
                        pad=float(args.mosaic_pad),
                        mosaic_scale=float(args.mosaic_scale),
                    )

                if args.draw_skeleton and st.kpts_frame is not None:
                    draw_skeleton(
                        frame,
                        st.kpts_frame,
                        conf_thr=float(args.skel_conf),
                        radius=int(args.skel_radius),
                        line_thick=int(args.skel_line),
                    )

            if args.show:
                cv2.imshow("focus realtime", frame)
                k = cv2.waitKey(1) & 0xFF
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
# skeleton + mosaic
# python scripts/07_realtime.py --ckpt models/skeleton_modes.pt --mode skeleton --source 0 --show \
#   --sample_fps 2 --det_imgsz 960 --pose_imgsz 320 --crop_long 320 --mosaic_eyes --mosaic_scale 0.06
#
# fusion + skeleton + mosaic (heavier)
# python scripts/07_realtime.py --ckpt models/skeleton_modes.pt --mode fusion --source 0 --show \
#   --sample_fps 2 --det_imgsz 960 --pose_imgsz 320 --crop_long 320 --draw_skeleton --mosaic_eyes
