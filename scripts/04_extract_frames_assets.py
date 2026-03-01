from pathlib import Path
import json
import math
import yaml

import cv2
import numpy as np
from ultralytics import YOLO

from lib.db import connect, init_db
from lib.video import read_frame_at
from lib.runroot import get_data_root, rpath  # 既に追加済みの想定

here = Path(__file__).resolve()
repo_root = here.parent.parent  # config.yaml がある場所（固定）


def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return img
    nh, nw = int(round(h * s)), int(round(w * s))
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


def main():
    cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))

    # データ実体 root（未指定なら従来互換で repo_root）
    data_root = get_data_root(repo_root)

    raw_video = rpath(data_root, cfg["paths"]["raw_video"])
    db_path = rpath(data_root, cfg["paths"]["db_path"])

    crop_dir = rpath(data_root, cfg["paths"]["crop_dir"])
    pose_dir = rpath(data_root, cfg["paths"]["pose_dir"])
    crop_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    sample_fps = float(cfg["sampling"]["sample_fps"])
    crop_long = int(cfg["sampling"]["crop_long_side"])

    posecfg = cfg["pose"]
    pose_model = YOLO(str(rpath(repo_root, posecfg["model"])))  # model は repo_root 側に置く運用が多いので repo_root

    conn = connect(db_path)
    init_db(conn)
    cur = conn.cursor()

    v = cur.execute("SELECT id, fps, width, height, path FROM videos ORDER BY id DESC LIMIT 1").fetchone()
    if v is None:
        raise RuntimeError("No video in DB. Run 02_detect_track.py first.")
    video_id, fps_db, W, H, video_path = int(v[0]), float(v[1]), int(v[2]), int(v[3]), str(v[4])

    if not raw_video.exists():
        # DBに保存された path が違うケースもあるので、最後の保険
        vp = Path(video_path)
        if vp.exists():
            raw_video = vp
        else:
            raise FileNotFoundError(f"raw_video not found: {raw_video} (and DB path not found: {video_path})")

    cap = cv2.VideoCapture(str(raw_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {raw_video}")

    segs = cur.execute("""
                       SELECT s.id, s.track_id, s.t_start, s.t_end
                       FROM segments s
                       WHERE s.video_id=?
                       ORDER BY s.id
                       """, (video_id,)).fetchall()

    print("segments:", len(segs), "db:", db_path)

    dt = 1.0 / sample_fps

    for idx, (seg_id, tid, t_start, t_end) in enumerate(segs):
        seg_id = int(seg_id); tid = int(tid)
        t_start = float(t_start); t_end = float(t_end)

        # 既に生成済みならスキップ
        exists = cur.execute("SELECT 1 FROM segment_frames WHERE segment_id=? LIMIT 1", (seg_id,)).fetchone()
        if exists:
            continue

        n_steps = max(1, int(math.floor((t_end - t_start) * sample_fps)))
        ts = [t_start + (k + 0.5) * dt for k in range(n_steps)]
        ts = [min(max(t, t_start), t_end) for t in ts]

        for t in ts:
            det = cur.execute("""
                              SELECT frame_idx, x1,y1,x2,y2, conf, t
                              FROM detections
                              WHERE video_id=? AND track_id=?
                              ORDER BY ABS(t-?)
                                  LIMIT 1
                              """, (video_id, tid, float(t))).fetchone()
            if det is None:
                continue

            frame_idx, x1, y1, x2, y2, conf, det_t = det
            frame_idx = int(frame_idx)

            ok, frame = read_frame_at(cap, frame_idx)
            if not ok:
                continue

            bb = clamp_bbox(x1, y1, x2, y2, W, H)
            if bb is None:
                continue
            x1i, y1i, x2i, y2i = bb

            crop = frame[y1i:y2i, x1i:x2i].copy()
            crop = resize_long_side(crop, crop_long)

            # 保存ファイル名（小数の揺れ対策でミリ秒相当に丸め）
            t_key = int(round(t * 1000))
            crop_path = crop_dir / f"seg{seg_id}_t{t_key}ms_tid{tid}.jpg"
            pose_path = pose_dir / f"seg{seg_id}_t{t_key}ms_tid{tid}.json"

            cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            pose_json = {}
            pr = pose_model.predict(
                crop,
                conf=float(posecfg["conf"]),
                imgsz=int(posecfg["imgsz"]),
                verbose=False
            )[0]
            if pr.keypoints is not None and len(pr.keypoints.data) > 0:
                kpts = pr.keypoints.data[0].cpu().numpy().tolist()
                pose_json = {"keypoints": kpts, "format": "x,y,conf", "space": "crop"}

            pose_path.write_text(json.dumps(pose_json, ensure_ascii=False))

            # DBには data_root からの相対パスで保存
            crop_rel = posix_rel(crop_path, data_root)
            pose_rel = posix_rel(pose_path, data_root)

            cur.execute("""
                        INSERT OR IGNORE INTO segment_frames(
                    segment_id, t, frame_idx, x1,y1,x2,y2, crop_path, pose_path
                ) VALUES(?,?,?,?,?,?,?,?,?)
                        """, (
                            seg_id, float(t), frame_idx,
                            float(x1), float(y1), float(x2), float(y2),
                            crop_rel, pose_rel
                        ))

        conn.commit()
        if idx % 50 == 0:
            print(f"processed segments: {idx}/{len(segs)}")

    cap.release()
    conn.commit()
    print("DONE extract segment_frames. DB:", db_path)


if __name__ == "__main__":
    main()

    import winsound
    try:
        winsound.PlaySound("mei_kara_mei_switch1.wav", winsound.SND_FILENAME)
    except Exception as e:
        print(f"[WARN] 音声を再生できませんでした: {e}")

# command
# python scripts/04_extract_frames_assets.py --data_root datasets/lesson_001