from pathlib import Path
import yaml

from lib.db import connect, init_db
from lib.segment import build_segments
from lib.runroot import get_data_root, rpath

here = Path(__file__).resolve()
repo_root = here.parent.parent  # config.yaml がある場所（固定）


def main():
    cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))

    # データ実体 root（未指定なら従来互換で repo_root）
    data_root = get_data_root(repo_root)

    db_path = rpath(data_root, cfg["paths"]["db_path"])
    scfg = cfg["segments"]

    conn = connect(db_path)
    init_db(conn)
    cur = conn.cursor()

    # videos は「最後に追加された1本」を対象（授業フォルダごとにDBが分かれている想定）
    video_row = cur.execute("SELECT id, fps FROM videos ORDER BY id DESC LIMIT 1").fetchone()
    if video_row is None:
        raise RuntimeError("No video in DB. Run 02_detect_track.py first.")

    video_id, fps = int(video_row[0]), float(video_row[1])

    # track一覧
    tracks = cur.execute("SELECT track_id FROM tracks WHERE video_id=?", (video_id,)).fetchall()
    tracks = [int(r[0]) for r in tracks]

    total = 0
    for tid in tracks:
        times = cur.execute(
            "SELECT t FROM detections WHERE video_id=? AND track_id=? ORDER BY t",
            (video_id, tid)
        ).fetchall()
        times = [float(r[0]) for r in times]

        segs = build_segments(
            times,
            window=float(scfg["window_sec"]),
            step=float(scfg["step_sec"]),
            min_presence_ratio=float(scfg["min_presence_ratio"]),
        )

        for (t_start, t_end) in segs:
            cur.execute(
                "INSERT OR IGNORE INTO segments(video_id, track_id, t_start, t_end) VALUES(?,?,?,?)",
                (video_id, tid, float(t_start), float(t_end)),
            )
            total += 1

    conn.commit()
    print("DONE segments:", total, "DB:", db_path)


if __name__ == "__main__":
    main()

    import winsound
    try:
        winsound.PlaySound("mei_kara_mei_switch1.wav", winsound.SND_FILENAME)
    except Exception as e:
        print(f"[WARN] 音声を再生できませんでした: {e}")

# command
# python scripts/03_make_segments.py --data_root datasets/lesson_001
