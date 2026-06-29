from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def migrate(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=OFF")

    if table_exists(conn, "windows") and "window_id" in columns(conn, "segments"):
        window_columns = columns(conn, "windows")
        additions = {
            "situation_set_by": "TEXT",
            "situation_updated_at": "TEXT",
            "situation_locked": "INTEGER NOT NULL DEFAULT 0 CHECK(situation_locked IN (0, 1))",
        }
        for name, definition in additions.items():
            if name not in window_columns:
                conn.execute(f"ALTER TABLE windows ADD COLUMN {name} {definition}")
        conn.commit()
        print(f"already migrated: {db_path}")
        conn.close()
        return

    required = {"videos", "segments", "segment_frames", "labels"}
    missing = sorted(name for name in required if not table_exists(conn, name))
    if missing:
        conn.close()
        raise RuntimeError(f"migration source tables are missing: {missing}")

    try:
        conn.execute("BEGIN IMMEDIATE")

        legacy_tables = ["segments", "segment_frames", "labels"]
        for optional in ("window_skips", "label_events", "label_window_meta"):
            if table_exists(conn, optional):
                legacy_tables.append(optional)
        for table in legacy_tables:
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")

        for index in ("idx_seg_video_track", "idx_segframes_segment"):
            conn.execute(f"DROP INDEX IF EXISTS {index}")

        conn.execute("""
            CREATE TABLE windows (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              video_id INTEGER NOT NULL,
              t_start REAL NOT NULL,
              t_end REAL NOT NULL,
              situation TEXT CHECK(situation IN ('聞く', '書く', '話し合う')),
              situation_set_by TEXT,
              situation_updated_at TEXT,
              situation_locked INTEGER NOT NULL DEFAULT 0 CHECK(situation_locked IN (0, 1)),
              UNIQUE(video_id, t_start, t_end),
              FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE segments (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              window_id INTEGER NOT NULL,
              track_id INTEGER NOT NULL,
              UNIQUE(window_id, track_id),
              FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE segment_frames (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              segment_id INTEGER NOT NULL,
              t REAL NOT NULL,
              frame_idx INTEGER NOT NULL,
              x1 REAL, y1 REAL, x2 REAL, y2 REAL,
              crop_path TEXT,
              pose_path TEXT,
              UNIQUE(segment_id, t),
              FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE labels (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              segment_id INTEGER NOT NULL,
              rater TEXT NOT NULL,
              score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 7),
              note TEXT,
              created_at TEXT DEFAULT (datetime('now')),
              updated_at TEXT DEFAULT (datetime('now')),
              UNIQUE(segment_id, rater),
              FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE window_skips (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              rater TEXT NOT NULL,
              window_id INTEGER NOT NULL,
              reason TEXT,
              created_at TEXT DEFAULT (datetime('now')),
              UNIQUE(rater, window_id),
              FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE label_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT DEFAULT (datetime('now')),
              rater TEXT NOT NULL,
              action TEXT NOT NULL,
              segment_id INTEGER,
              old_score INTEGER,
              new_score INTEGER,
              note TEXT,
              FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE SET NULL
            )
        """)

        conn.execute("""
            INSERT INTO windows(video_id, t_start, t_end, situation)
            SELECT video_id, t_start, t_end, MAX(situation)
            FROM segments_legacy
            GROUP BY video_id, t_start, t_end
        """)
        conn.execute("""
            INSERT INTO segments(id, window_id, track_id)
            SELECT s.id, w.id, s.track_id
            FROM segments_legacy s
            JOIN windows w ON w.video_id=s.video_id
                          AND w.t_start=s.t_start AND w.t_end=s.t_end
        """)
        conn.execute("""
            INSERT INTO segment_frames(id, segment_id, t, frame_idx, x1, y1, x2, y2, crop_path, pose_path)
            SELECT id, segment_id, t, frame_idx, x1, y1, x2, y2, crop_path, pose_path
            FROM segment_frames_legacy
        """)

        if table_exists(conn, "label_window_meta_legacy"):
            conn.execute("""
                INSERT INTO labels(id, segment_id, rater, score, note, created_at, updated_at)
                SELECT l.id, l.segment_id, l.rater, l.score, m.note,
                       l.created_at, l.created_at
                FROM labels_legacy l
                LEFT JOIN label_window_meta_legacy m
                       ON m.segment_id=l.segment_id AND m.rater=l.rater
            """)
        else:
            conn.execute("""
                INSERT INTO labels(id, segment_id, rater, score, created_at, updated_at)
                SELECT id, segment_id, rater, score, created_at, created_at
                FROM labels_legacy
            """)

        if table_exists(conn, "window_skips_legacy"):
            conn.execute("""
                INSERT INTO window_skips(rater, window_id, reason, created_at)
                SELECT ws.rater, w.id, ws.reason, ws.created_at
                FROM window_skips_legacy ws
                JOIN windows w ON w.t_start=ws.t_start AND w.t_end=ws.t_end
            """)
        if table_exists(conn, "label_events_legacy"):
            conn.execute("""
                INSERT INTO label_events(id, created_at, rater, action, segment_id, old_score, new_score, note)
                SELECT id, created_at, rater, action, segment_id, old_score, new_score, note
                FROM label_events_legacy
            """)

        for table in reversed(legacy_tables):
            conn.execute(f"DROP TABLE {table}_legacy")

        conn.execute("CREATE INDEX idx_windows_video_time ON windows(video_id, t_start, t_end)")
        conn.execute("CREATE INDEX idx_seg_window_track ON segments(window_id, track_id)")
        conn.execute("CREATE INDEX idx_segframes_segment ON segment_frames(segment_id)")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.close()

    print(f"migrated: {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    migrate(args.db.resolve())


if __name__ == "__main__":
    main()
