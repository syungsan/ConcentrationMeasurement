from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def parse_time(value: str) -> float:
    value = str(value).strip()
    if not value:
        raise ValueError("blank time")
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60.0 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    raise ValueError(f"invalid time: {value}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(predictions)")
    }
    if "situation_override" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN situation_override TEXT")
    if "situation_override_note" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN situation_override_note TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS situation_overrides (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          video_path TEXT,
          start_sec REAL NOT NULL,
          end_sec REAL NOT NULL,
          situation TEXT NOT NULL,
          note TEXT,
          applied_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def apply_ab(
        db_path: Path,
        start_sec: float,
        end_sec: float,
        *,
        video_filter: str | None,
        clear_existing: bool,
) -> dict[str, int | float | str | None]:
    if end_sec <= start_sec:
        raise ValueError("end must be greater than start")
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        where = ["t>=?", "t<?"]
        params: list[object] = [float(start_sec), float(end_sec)]
        if video_filter:
            where.append("video_path LIKE ?")
            params.append(f"%{video_filter}%")
        where_sql = " AND ".join(where)
        with conn:
            if clear_existing:
                conn.execute(
                    f"UPDATE predictions SET situation_override=NULL, "
                    f"situation_override_note=NULL WHERE {where_sql}",
                    params,
                )
            cur = conn.execute(
                f"UPDATE predictions SET situation_override='AB', "
                f"situation_override_note='Acti-Break' WHERE {where_sql}",
                params,
            )
            updated = int(cur.rowcount if cur.rowcount is not None else 0)
            conn.execute("""
                INSERT INTO situation_overrides(
                  video_path, start_sec, end_sec, situation, note
                ) VALUES(?,?,?,?,?)
            """, (
                video_filter, float(start_sec), float(end_sec),
                "AB", "Acti-Break",
            ))
        total_ab = int(conn.execute("""
            SELECT COUNT(*) FROM predictions WHERE situation_override='AB'
        """).fetchone()[0])
        return {
            "db": str(db_path),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "video_filter": video_filter,
            "updated_rows": updated,
            "total_ab_rows": total_ab,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="09_video_offline.py の pred_log.sqlite にAB区間を後付けします。"
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--start", required=True, help="秒、mm:ss、hh:mm:ss")
    parser.add_argument("--end", required=True, help="秒、mm:ss、hh:mm:ss")
    parser.add_argument(
        "--video-filter",
        default=None,
        help="複数動画が同じDBにある場合の video_path 部分一致フィルタ",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="指定区間内の既存overrideを消してからABを適用",
    )
    args = parser.parse_args()
    result = apply_ab(
        args.db.resolve(),
        parse_time(args.start),
        parse_time(args.end),
        video_filter=args.video_filter,
        clear_existing=args.clear_existing,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
