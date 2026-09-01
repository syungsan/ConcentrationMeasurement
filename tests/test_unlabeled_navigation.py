from __future__ import annotations

import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GUI_PATH = ROOT / "scripts" / "06_label_gui.py"
sys.path.insert(0, str(ROOT / "scripts"))


def load_gui_module():
    spec = importlib.util.spec_from_file_location("label_gui", GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UnlabeledNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gui = load_gui_module()

    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript("""
            CREATE TABLE windows (
                id INTEGER PRIMARY KEY,
                video_id INTEGER NOT NULL,
                t_start REAL NOT NULL,
                t_end REAL NOT NULL,
                situation TEXT
            );
            CREATE TABLE segments (
                id INTEGER PRIMARY KEY,
                window_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL
            );
            CREATE TABLE labels (
                id INTEGER PRIMARY KEY,
                segment_id INTEGER NOT NULL,
                rater TEXT NOT NULL
            );
            CREATE TABLE excluded_segments (
                segment_id INTEGER PRIMARY KEY
            );
            INSERT INTO windows VALUES
                (1, 1, 0.0, 5.0, 'listen'),
                (2, 1, 5.0, 10.0, 'listen'),
                (3, 1, 10.0, 15.0, 'listen');
            INSERT INTO segments VALUES (1, 1, 10), (2, 2, 20), (3, 3, 30);
            INSERT INTO labels VALUES (1, 1, 'rater-a');
        """)
        self.app = SimpleNamespace(cur=self.conn.cursor())

    def tearDown(self) -> None:
        self.conn.close()

    def test_combined_rating_clip_finds_unlabeled_base_windows(self) -> None:
        has_unlabeled = self.gui.LabelFastApp._window_has_unlabeled
        self.assertTrue(has_unlabeled(self.app, "rater-a", 0.0, 15.0))

    def test_excluded_segments_are_not_treated_as_unlabeled(self) -> None:
        self.conn.execute("INSERT INTO excluded_segments VALUES (2)")
        self.conn.execute("INSERT INTO excluded_segments VALUES (3)")
        has_unlabeled = self.gui.LabelFastApp._window_has_unlabeled
        self.assertFalse(has_unlabeled(self.app, "rater-a", 0.0, 15.0))

    def test_combined_rating_clip_matches_its_situation(self) -> None:
        matches = self.gui.LabelFastApp._window_matches_situation
        self.assertTrue(matches(self.app, "rater-a", 0.0, 15.0, "listen"))
        self.assertFalse(matches(self.app, "rater-a", 0.0, 15.0, "write"))

    def test_seek_time_loads_matching_clip_and_preserves_position(self) -> None:
        loaded = []
        positions = []
        app = SimpleNamespace(
            ensure_ready=lambda: True,
            rater=lambda: "rater-a",
            task="rating",
            windows=[(0.0, 15.0), (15.0, 30.0)],
            window_idx=0,
            load_window=lambda t0, t1, rater, keep_play_state: loaded.append(
                (t0, t1, rater, keep_play_state)
            ),
            player=SimpleNamespace(setPosition=positions.append),
        )

        self.gui.LabelFastApp.load_window_at_time(app, 22.5)

        self.assertEqual(app.window_idx, 1)
        self.assertEqual(loaded, [(15.0, 30.0, "rater-a", True)])
        self.assertEqual(positions, [22500])


if __name__ == "__main__":
    unittest.main()
