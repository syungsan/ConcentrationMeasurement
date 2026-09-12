from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lib.face_exclusion import (  # noqa: E402
    OnlineFaceExclusionTracker,
    decide_face_exclusion,
    update_auto_exclusion,
)


class FaceExclusionTests(unittest.TestCase):
    def test_online_tracker_recovers_after_track_id_switch(self) -> None:
        class FakeMatcher:
            def __init__(self):
                self.results = iter(["teacher", "teacher", None, None, None])

            def match(self, _image):
                return next(self.results), 1.0

        tracker = OnlineFaceExclusionTracker(
            FakeMatcher(), min_matches=2, min_match_ratio=0.5, max_frames=4
        )
        image = object()
        self.assertEqual(tracker.update(1, image), (False, None))
        self.assertEqual(tracker.update(1, image), (True, "teacher"))
        self.assertEqual(tracker.update(1, image), (True, "teacher"))
        self.assertEqual(tracker.update(1, image), (True, "teacher"))
        self.assertEqual(tracker.update(1, image), (False, None))

    def test_requires_count_and_ratio(self) -> None:
        self.assertEqual(
            decide_face_exclusion(["teacher", "teacher", None, None], 2, 0.5),
            (True, "teacher", 2, 0.5),
        )
        self.assertFalse(
            decide_face_exclusion(["teacher", None, None, None], 2, 0.5)[0]
        )

    def test_manual_exclusion_is_not_overwritten_or_deleted(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE excluded_segments(
              segment_id INTEGER PRIMARY KEY,
              reason TEXT NOT NULL,
              excluded_by TEXT
            )
        """)
        conn.execute(
            "INSERT INTO excluded_segments VALUES(1, 'adult', 'rater-a')"
        )
        update_auto_exclusion(
            conn, 1, excluded=True, matched_name="registered-adult"
        )
        update_auto_exclusion(conn, 1, excluded=False, matched_name=None)
        self.assertEqual(
            conn.execute(
                "SELECT reason, excluded_by FROM excluded_segments WHERE segment_id=1"
            ).fetchone(),
            ("adult", "rater-a"),
        )
        conn.close()

    def test_automatic_exclusion_can_be_refreshed(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE excluded_segments(
              segment_id INTEGER PRIMARY KEY,
              reason TEXT NOT NULL,
              excluded_by TEXT
            )
        """)
        update_auto_exclusion(conn, 2, excluded=True, matched_name="teacher")
        self.assertEqual(
            conn.execute(
                "SELECT reason FROM excluded_segments WHERE segment_id=2"
            ).fetchone()[0],
            "face_db:teacher",
        )
        update_auto_exclusion(conn, 2, excluded=False, matched_name=None)
        self.assertIsNone(
            conn.execute(
                "SELECT 1 FROM excluded_segments WHERE segment_id=2"
            ).fetchone()
        )
        conn.close()


if __name__ == "__main__":
    unittest.main()
