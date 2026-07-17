from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lib.research_schema import (  # noqa: E402
    ResearchMetadata,
    eligible_for_training,
    ensure_blind_assignments,
    ensure_research_schema,
    get_metadata,
    set_metadata,
)


class ResearchSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript("""
            CREATE TABLE videos(
              id INTEGER PRIMARY KEY, fps REAL, width INTEGER,
              height INTEGER, frame_count INTEGER
            );
            CREATE TABLE windows(
              id INTEGER PRIMARY KEY, t_start REAL, t_end REAL,
              situation TEXT, situation_locked INTEGER
            );
            CREATE TABLE segments(
              id INTEGER PRIMARY KEY, window_id INTEGER, track_id INTEGER
            );
            INSERT INTO videos VALUES(1,24,1920,1080,1000);
            INSERT INTO windows VALUES(1,0,5,'listen',1),(2,5,10,'write',1);
            INSERT INTO segments VALUES(1,1,10),(2,1,11),(3,2,10);
        """)

    def tearDown(self) -> None:
        self.conn.close()

    def test_normal_development_is_trainable(self) -> None:
        set_metadata(self.conn, ResearchMetadata(
            school_id="school-a", grade="5", class_id="5A", session_id="s1"
        ))
        metadata = get_metadata(self.conn)
        assert metadata is not None
        self.assertTrue(eligible_for_training(metadata, include_pilot_post=False))

    def test_trial_and_main_post_are_never_trainable(self) -> None:
        for role, phase, minutes in [
            ("trial", "normal", None),
            ("development", "post", 3.0),
        ]:
            set_metadata(self.conn, ResearchMetadata(
                school_id="school-a", grade="5", class_id="5A", session_id="s1",
                study_role=role, intervention_phase=phase,
                minutes_since_intervention=minutes,
            ))
            metadata = get_metadata(self.conn)
            assert metadata is not None
            self.assertFalse(eligible_for_training(metadata, include_pilot_post=True))

    def test_pilot_post_requires_time_and_explicit_opt_in(self) -> None:
        with self.assertRaises(ValueError):
            ResearchMetadata(
                school_id="s", grade="5", class_id="a", session_id="x",
                intervention_phase="pilot_post",
            ).validate()
        value = ResearchMetadata(
            school_id="s", grade="5", class_id="a", session_id="x",
            intervention_phase="pilot_post", minutes_since_intervention=5,
        )
        set_metadata(self.conn, value)
        metadata = get_metadata(self.conn)
        assert metadata is not None
        self.assertFalse(eligible_for_training(metadata, include_pilot_post=False))
        self.assertTrue(eligible_for_training(metadata, include_pilot_post=True))

    def test_schema_is_idempotent(self) -> None:
        ensure_research_schema(self.conn)
        ensure_research_schema(self.conn)
        tables = {
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        self.assertIn("blind_assignments", tables)
        self.assertIn("label_observations", tables)
        observation_columns = {
            row[1] for row in self.conn.execute(
                "PRAGMA table_info(label_observations)"
            )
        }
        self.assertIn("source_clip_start", observation_columns)
        self.assertIn("source_clip_end", observation_columns)
        self.assertIn("source_clip_sec", observation_columns)
        self.assertNotIn("observable", observation_columns)
        self.assertNotIn("confidence", observation_columns)
        self.assertNotIn("protocol_version", observation_columns)
        self.assertNotIn("on_task", observation_columns)
        metadata_columns = {
            row[1] for row in self.conn.execute(
                "PRAGMA table_info(research_metadata)"
            )
        }
        self.assertNotIn("teacher_id", metadata_columns)
        self.assertNotIn("subject", metadata_columns)
        self.assertNotIn("notes", metadata_columns)

    def test_blind_order_is_created_on_first_rater_use(self) -> None:
        first = ensure_blind_assignments(self.conn, "rater-a")
        second = ensure_blind_assignments(self.conn, "rater-a")
        self.assertEqual(first, 3)
        self.assertEqual(second, 3)
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM blind_assignments WHERE rater='rater-a'"
            ).fetchone()[0],
            3,
        )


if __name__ == "__main__":
    unittest.main()
