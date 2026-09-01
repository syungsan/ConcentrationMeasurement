import time
SCRIPT_STARTED_AT = time.perf_counter()

from pathlib import Path
import json
import math
import yaml

import cv2
import numpy as np
from ultralytics import YOLO

from lib.db import connect, init_db
from lib.face_exclusion import (
    FaceDbMatcher,
    decide_face_exclusion,
    update_auto_exclusion,
)
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


def print_progress(
    completed: int,
    total: int,
    started_at: float,
    *,
    auto_excluded: int,
    force: bool = False,
) -> None:
    """セグメント処理の進捗、速度、推定残り時間を表示する。"""
    if total <= 0:
        return
    interval = max(1, total // 100)
    if not force and completed % interval != 0:
        return
    elapsed = max(0.0, time.perf_counter() - started_at)
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    percent = completed * 100.0 / total
    print(
        f"[PROGRESS] {completed}/{total} ({percent:5.1f}%) "
        f"elapsed={elapsed:,.1f}s eta={remaining:,.1f}s "
        f"rate={rate:.2f} segments/s excluded={auto_excluded}",
        flush=True,
    )


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

    facecfg = cfg.get("face_exclusion", {})
    face_matcher = None
    face_max_frames = int(facecfg.get("max_frames", 8))
    if bool(facecfg.get("enabled", False)):
        face_db_path = rpath(repo_root, str(facecfg.get("db_path", "models/face_db.npz")))
        if not face_db_path.exists():
            raise FileNotFoundError(f"face DB not found: {face_db_path}")
        face_matcher = FaceDbMatcher(
            face_db_path,
            threshold=float(facecfg.get("similarity_threshold", 0.50)),
            device=str(facecfg.get("device", "cuda")),
        )
        print(f"face exclusion: enabled, DB={face_db_path}")

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
                       SELECT s.id, s.track_id, w.t_start, w.t_end
                       FROM segments s
                       JOIN windows w ON w.id=s.window_id
                       WHERE w.video_id=?
                       ORDER BY s.id
                       """, (video_id,)).fetchall()

    print("segments:", len(segs), "db:", db_path)

    dt = 1.0 / sample_fps
    auto_excluded = 0
    progress_started_at = time.perf_counter()

    for idx, (seg_id, tid, t_start, t_end) in enumerate(segs):
        seg_id = int(seg_id); tid = int(tid)
        t_start = float(t_start); t_end = float(t_end)

        existing_rows = cur.execute("""
            SELECT crop_path FROM segment_frames
            WHERE segment_id=? AND crop_path IS NOT NULL
            ORDER BY t LIMIT ?
        """, (seg_id, face_max_frames)).fetchall()
        if existing_rows:
            if face_matcher is not None:
                images = []
                for (stored_path,) in existing_rows:
                    path = rpath(data_root, str(stored_path))
                    image = cv2.imread(str(path))
                    if image is not None:
                        images.append(image)
                matches = [face_matcher.match(image)[0] for image in images]
                excluded, name, count, ratio = decide_face_exclusion(
                    matches,
                    int(facecfg.get("min_matches", 2)),
                    float(facecfg.get("min_match_ratio", 0.50)),
                )
                update_auto_exclusion(
                    conn, seg_id, excluded=excluded, matched_name=name,
                )
                if excluded:
                    auto_excluded += 1
                    print(
                        f"[FACE EXCLUDE] segment={seg_id} track={tid} "
                        f"name={name} matches={count}/{len(matches)} ratio={ratio:.2f}"
                    )
                conn.commit()
            print_progress(
                idx + 1,
                len(segs),
                progress_started_at,
                auto_excluded=auto_excluded,
                force=(idx + 1 == len(segs)),
            )
            continue

        n_steps = max(1, int(math.floor((t_end - t_start) * sample_fps)))
        ts = [t_start + (k + 0.5) * dt for k in range(n_steps)]
        ts = [min(max(t, t_start), t_end) for t in ts]

        face_images: list[np.ndarray] = []
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
            if len(face_images) < face_max_frames:
                face_images.append(crop)

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

        if face_matcher is not None:
            matches = [face_matcher.match(image)[0] for image in face_images]
            excluded, name, count, ratio = decide_face_exclusion(
                matches,
                int(facecfg.get("min_matches", 2)),
                float(facecfg.get("min_match_ratio", 0.50)),
            )
            update_auto_exclusion(
                conn, seg_id, excluded=excluded, matched_name=name,
            )
            if excluded:
                auto_excluded += 1
                print(
                    f"[FACE EXCLUDE] segment={seg_id} track={tid} "
                    f"name={name} matches={count}/{len(matches)} ratio={ratio:.2f}"
                )

        conn.commit()
        print_progress(
            idx + 1,
            len(segs),
            progress_started_at,
            auto_excluded=auto_excluded,
            force=(idx + 1 == len(segs)),
        )

    cap.release()
    conn.commit()
    print(
        "DONE extract segment_frames. DB:", db_path,
        "auto-excluded segments:", auto_excluded,
    )


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
# python scripts/04_extract_frames_assets.py --data_root datasets/lesson_001
