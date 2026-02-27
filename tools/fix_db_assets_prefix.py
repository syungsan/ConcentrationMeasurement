# tools/fix_db_assets_prefix.py
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path

def norm(s: str) -> str:
    return str(s).replace("\\", "/").strip()

def rewrite_one(s: str) -> str:
    s = norm(s)
    # 代表的なズレを全部吸収
    s = s.replace("data/assets/", "assets/")
    if s.startswith("data/"):
        s = s.replace("data/", "", 1)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="dataset.sqlite path")
    ap.add_argument("--dry", action="store_true", help="dry run")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    changed = 0

    # segment_frames の crop_path / pose_path を直す
    rows = cur.execute("SELECT id, crop_path, pose_path FROM segment_frames").fetchall()
    for _id, crop, pose in rows:
        crop2 = rewrite_one(crop) if crop else crop
        pose2 = rewrite_one(pose) if pose else pose
        if crop2 != crop or pose2 != pose:
            changed += 1
            if not args.dry:
                cur.execute(
                    "UPDATE segment_frames SET crop_path=?, pose_path=? WHERE id=?",
                    (crop2, pose2, int(_id))
                )

    if not args.dry:
        conn.commit()
    conn.close()

    print(f"DONE changed rows={changed} db={db_path}")

if __name__ == "__main__":
    main()
