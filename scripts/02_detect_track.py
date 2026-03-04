from pathlib import Path
import yaml

from ultralytics import YOLO

from lib.db import connect, init_db, upsert_video, ensure_track
from lib.video import get_video_meta
from lib.runroot import get_data_root, rpath

here = Path(__file__).resolve()
repo_root = here.parent.parent  # config.yaml がある場所（固定）


def resolve_repo_path(repo_root: Path, p: str) -> Path:
    """models/ など repo に置く前提のパス解決（config.yamlは相対パス前提）"""
    return (repo_root / p).resolve()


def main():
    cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))

    # データ実体 root（未指定なら従来互換で repo_root）
    data_root = get_data_root(repo_root)

    raw_video = rpath(data_root, cfg["paths"]["raw_video"])
    db_path = rpath(data_root, cfg["paths"]["db_path"])
    detcfg = cfg["detection_tracking"]

    if not raw_video.exists():
        raise FileNotFoundError(f"raw video not found: {raw_video}")

    # DB open/init
    conn = connect(db_path)
    init_db(conn)

    # meta
    meta = get_video_meta(raw_video)

    # 重要: video path を data_root 側の実パスで保存（後で追える）
    video_id = upsert_video(
        conn,
        str(raw_video),
        float(meta.fps),
        int(meta.width),
        int(meta.height),
        int(meta.frame_count),
    )

    # YOLO model は repo_root 側にある想定（models/）
    model_path = resolve_repo_path(repo_root, detcfg["model"])
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    model = YOLO(str(model_path))

    # tracker も repo_root 側の想定（botsort.yaml 等）
    tracker_path = resolve_repo_path(repo_root, detcfg["tracker"])
    if not tracker_path.exists():
        # Ultralytics同梱を使う場合は相対でも動くことがあるので、警告してそのまま渡す
        tracker_path = Path(detcfg["tracker"])

    # Ultralytics track: generator でフレーム順に result が来る
    results = model.track(
        source=str(raw_video),
        conf=float(detcfg["conf"]),
        iou=float(detcfg["iou"]),
        imgsz=int(detcfg["imgsz"]),
        tracker=str(tracker_path),
        persist=True,
        classes=detcfg.get("classes", None),
        verbose=False,
        stream=True,
    )

    cur = conn.cursor()
    fps = float(meta.fps)
    frame_idx = 0
    inserted = 0

    for r in results:
        t = frame_idx / fps

        if r.boxes is None or r.boxes.id is None:
            frame_idx += 1
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        tids = r.boxes.id.cpu().numpy().astype(int)

        for i, tid in enumerate(tids):
            ensure_track(conn, video_id, int(tid))

            x1, y1, x2, y2 = xyxy[i].tolist()
            conf = float(confs[i])

            cur.execute(
                """
                INSERT INTO detections(video_id, frame_idx, t, track_id, x1,y1,x2,y2, conf)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (video_id, frame_idx, float(t), int(tid), float(x1), float(y1), float(x2), float(y2), conf),
            )
            inserted += 1

        if frame_idx % 50 == 0:
            conn.commit()
            print(f"frame={frame_idx} t={t:.2f}s inserted={inserted}")

        frame_idx += 1

    conn.commit()
    print("DONE detections:", inserted, "rows. DB:", db_path)


if __name__ == "__main__":
    main()

    import winsound
    try:
        winsound.PlaySound("mei_kara_mei_switch1.wav", winsound.SND_FILENAME)
    except Exception as e:
        print(f"[WARN] 音声を再生できませんでした: {e}")

# command
# python scripts/02_detect_track.py --data_root datasets/lesson_001
