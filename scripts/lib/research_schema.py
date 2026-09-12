from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Iterable


SCHEMA_VERSION = 1
VALID_PHASES = ("normal", "pre", "post", "pilot_post")
VALID_ROLES = ("development", "external_validation", "trial")


RESEARCH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_metadata (
  id INTEGER PRIMARY KEY CHECK(id = 1),
  schema_version INTEGER NOT NULL,
  school_id TEXT NOT NULL,
  grade TEXT NOT NULL,
  class_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  intervention_phase TEXT NOT NULL
    CHECK(intervention_phase IN ('normal','pre','post','pilot_post')),
  study_role TEXT NOT NULL
    CHECK(study_role IN ('development','external_validation','trial')),
  minutes_since_intervention REAL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS blind_assignments (
  rater TEXT NOT NULL,
  segment_id INTEGER NOT NULL,
  blind_code TEXT NOT NULL,
  display_order INTEGER NOT NULL,
  assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(rater, segment_id),
  UNIQUE(rater, blind_code),
  UNIQUE(rater, display_order),
  FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS label_observations (
  segment_id INTEGER NOT NULL,
  rater TEXT NOT NULL,
  source_clip_start REAL,
  source_clip_end REAL,
  source_clip_sec REAL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(segment_id, rater),
  FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_blind_rater_order
  ON blind_assignments(rater, display_order);
CREATE INDEX IF NOT EXISTS idx_observation_rater
  ON label_observations(rater);
"""


@dataclass(frozen=True)
class ResearchMetadata:
    school_id: str
    grade: str
    class_id: str
    session_id: str
    intervention_phase: str = "normal"
    study_role: str = "development"
    minutes_since_intervention: float | None = None

    def validate(self) -> None:
        required = {
            "school_id": self.school_id,
            "grade": self.grade,
            "class_id": self.class_id,
            "session_id": self.session_id,
        }
        blank = [name for name, value in required.items() if not str(value).strip()]
        if blank:
            raise ValueError(f"blank research metadata fields: {', '.join(blank)}")
        if self.intervention_phase not in VALID_PHASES:
            raise ValueError(f"invalid intervention_phase: {self.intervention_phase}")
        if self.study_role not in VALID_ROLES:
            raise ValueError(f"invalid study_role: {self.study_role}")
        if self.intervention_phase in {"post", "pilot_post"}:
            if self.minutes_since_intervention is None:
                raise ValueError("post-intervention data requires minutes_since_intervention")
            if self.minutes_since_intervention < 0:
                raise ValueError("minutes_since_intervention must be >= 0")


def ensure_research_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(RESEARCH_SCHEMA_SQL)
    conn.execute("DROP INDEX IF EXISTS idx_observation_rater")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_observation_rater ON label_observations(rater)"
    )
    legacy_columns = {
        "research_metadata": (
            "teacher_id", "subject", "intervention_id", "recorded_at", "notes",
        ),
        "label_observations": (
            "observable", "confidence", "protocol_version", "on_task",
            "attention_direction", "task_appropriate_action", "off_task",
        ),
        "labels": ("note",),
        "window_skips": ("reason",),
        "label_events": ("note",),
    }
    tables = {
        str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    for table, columns in legacy_columns.items():
        if table not in tables:
            continue
        existing = {
            str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
        }
        for column in columns:
            if column in existing:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    for table in (
        "weather", "weather_observations", "temperature_observations",
        "room_temperature", "environment_observations",
    ):
        if table in tables:
            conn.execute(f"DROP TABLE {table}")
    observation_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(label_observations)")
    }
    for name in ("source_clip_start", "source_clip_end", "source_clip_sec"):
        if name not in observation_columns:
            conn.execute(f"ALTER TABLE label_observations ADD COLUMN {name} REAL")
    conn.commit()


def set_metadata(conn: sqlite3.Connection, value: ResearchMetadata) -> None:
    value.validate()
    ensure_research_schema(conn)
    conn.execute(
        """
        INSERT INTO research_metadata(
          id, schema_version, school_id, grade, class_id, session_id,
          intervention_phase, study_role, minutes_since_intervention, updated_at
        ) VALUES(1,?,?,?,?,?,?,?,?,datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          schema_version=excluded.schema_version,
          school_id=excluded.school_id, grade=excluded.grade,
          class_id=excluded.class_id, session_id=excluded.session_id,
          intervention_phase=excluded.intervention_phase,
          study_role=excluded.study_role,
          minutes_since_intervention=excluded.minutes_since_intervention,
          updated_at=datetime('now')
        """,
        (
            SCHEMA_VERSION, value.school_id.strip(), value.grade.strip(),
            value.class_id.strip(), value.session_id.strip(),
            value.intervention_phase, value.study_role,
            value.minutes_since_intervention,
        ),
    )
    conn.commit()


def get_metadata(conn: sqlite3.Connection) -> dict[str, object] | None:
    ensure_research_schema(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM research_metadata WHERE id=1").fetchone()
    return dict(row) if row else None


def metadata_json(conn: sqlite3.Connection) -> str:
    return json.dumps(get_metadata(conn), ensure_ascii=False, indent=2)


def eligible_for_training(metadata: dict[str, object], include_pilot_post: bool) -> bool:
    role = str(metadata.get("study_role", ""))
    phase = str(metadata.get("intervention_phase", ""))
    if role != "development":
        return False
    return phase == "normal" or (include_pilot_post and phase == "pilot_post")


def ensure_blind_assignments(
        conn: sqlite3.Connection, rater: str,
) -> int:
    """Create a stable chronological evaluation order on first use."""
    rater = rater.strip()
    if not rater:
        raise ValueError("rater must not be blank")
    ensure_research_schema(conn)
    existing = int(conn.execute(
        "SELECT COUNT(*) FROM blind_assignments WHERE rater=?", (rater,)
    ).fetchone()[0])
    if existing:
        return existing
    segment_ids: list[int] = []
    for (segment_id,) in conn.execute("""
        SELECT s.id FROM windows w
        JOIN segments s ON s.window_id=w.id
        WHERE w.situation IS NOT NULL AND w.situation_locked=1
        ORDER BY w.video_id, w.t_start, w.t_end, s.track_id, s.id
    """):
        segment_ids.append(int(segment_id))
    if not segment_ids:
        raise RuntimeError("no locked situation windows are available for rating")
    order = 0
    with conn:
        for segment_id in segment_ids:
            order += 1
            blind_code = hashlib.sha256(
                f"{rater}:{segment_id}:classroom_engagement_v2".encode("utf-8")
            ).hexdigest()[:12].upper()
            conn.execute("""
                INSERT INTO blind_assignments(
                  rater,segment_id,blind_code,display_order
                ) VALUES(?,?,?,?)
            """, (rater, segment_id, blind_code, order))
    return order


def require_tables(conn: sqlite3.Connection, tables: Iterable[str]) -> None:
    actual = {
        str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = sorted(set(tables) - actual)
    if missing:
        raise RuntimeError(f"missing tables: {missing}")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None
