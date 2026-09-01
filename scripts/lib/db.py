# lib/db.py
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS videos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT UNIQUE,
  fps REAL,
  width INTEGER,
  height INTEGER,
  frame_count INTEGER
);

CREATE TABLE IF NOT EXISTS tracks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id INTEGER NOT NULL,
  track_id INTEGER NOT NULL,
  UNIQUE(video_id, track_id)
);

CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id INTEGER NOT NULL,
  frame_idx INTEGER NOT NULL,
  t REAL NOT NULL,
  track_id INTEGER NOT NULL,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  conf REAL
);

-- segment: track_id ごとの時間区間（ラベル単位）
CREATE TABLE IF NOT EXISTS windows (
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
);

-- person segment belonging to one shared lesson window
CREATE TABLE IF NOT EXISTS segments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  window_id INTEGER NOT NULL,
  track_id INTEGER NOT NULL,
  UNIQUE(window_id, track_id),
  FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
);

-- 時系列サンプル（tごとのクロップ・姿勢）
CREATE TABLE IF NOT EXISTS segment_frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  segment_id INTEGER NOT NULL,
  t REAL NOT NULL,
  frame_idx INTEGER NOT NULL,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  crop_path TEXT,
  pose_path TEXT,
  UNIQUE(segment_id, t),
  FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  segment_id INTEGER NOT NULL,
  rater TEXT NOT NULL,
  score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 7),
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now')),
  UNIQUE(segment_id, rater),
  FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS window_skips (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  rater TEXT NOT NULL,
  window_id INTEGER NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  UNIQUE(rater, window_id),
  FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
);

-- 区間内で学習・評価対象にしない人物（先生、保護者など）
CREATE TABLE IF NOT EXISTS excluded_segments (
  segment_id INTEGER PRIMARY KEY,
  reason TEXT NOT NULL DEFAULT 'adult',
  excluded_by TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS label_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT DEFAULT (datetime('now')),
  rater TEXT NOT NULL,
  action TEXT NOT NULL,
  segment_id INTEGER,
  old_score INTEGER,
  new_score INTEGER,
  FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE SET NULL
);

-- ★推論ログ（集中度予測）
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT DEFAULT (datetime('now')),
  video_id INTEGER NOT NULL,
  frame_idx INTEGER NOT NULL,
  t REAL NOT NULL,
  track_id INTEGER NOT NULL,
  mode TEXT NOT NULL,              -- image/skeleton/fusion
  yhat REAL NOT NULL,              -- 回帰出力
  score_int INTEGER NOT NULL,       -- 1..7 に丸めた値
  situation TEXT NOT NULL CHECK(situation IN ('聞く', '書く', '話し合う')),
  situation_confidence REAL,
  situation_probs TEXT,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  det_conf REAL,
  crop_path TEXT,                  -- 保存した場合のみ
  pose_path TEXT                   -- 保存した場合のみ
);

CREATE INDEX IF NOT EXISTS idx_det_video_track_t ON detections(video_id, track_id, t);
CREATE INDEX IF NOT EXISTS idx_windows_video_time ON windows(video_id, t_start, t_end);
CREATE INDEX IF NOT EXISTS idx_seg_window_track ON segments(window_id, track_id);
CREATE INDEX IF NOT EXISTS idx_segframes_segment ON segment_frames(segment_id);
CREATE INDEX IF NOT EXISTS idx_excluded_segments_segment ON excluded_segments(segment_id);

-- ★推論検索用
CREATE INDEX IF NOT EXISTS idx_pred_video_track_t ON predictions(video_id, track_id, t);
CREATE INDEX IF NOT EXISTS idx_pred_video_t ON predictions(video_id, t);
"""

def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()

def upsert_video(conn: sqlite3.Connection, path: str, fps: float, w: int, h: int, nframes: int) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO videos(path, fps, width, height, frame_count) VALUES(?,?,?,?,?)",
        (path, fps, w, h, nframes)
    )
    conn.commit()
    cur.execute("SELECT id FROM videos WHERE path=?", (path,))
    return int(cur.fetchone()[0])

def ensure_track(conn: sqlite3.Connection, video_id: int, track_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO tracks(video_id, track_id) VALUES(?,?)",
        (video_id, track_id)
    )
    conn.commit()
