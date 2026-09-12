from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lib.rating_clips import build_rating_clips, sample_animation_paths  # noqa: E402


class RatingClipTests(unittest.TestCase):
    def test_combines_six_five_second_windows(self) -> None:
        windows = [(i * 5, (i + 1) * 5, "listen", 1) for i in range(8)]
        self.assertEqual(build_rating_clips(windows, 30), [(0.0, 30.0), (30.0, 40.0)])

    def test_does_not_cross_situation_boundary(self) -> None:
        windows = [
            (0, 5, "listen", 1), (5, 10, "listen", 1),
            (10, 15, "write", 1), (15, 20, "write", 1),
        ]
        self.assertEqual(build_rating_clips(windows, 30), [(0.0, 10.0), (10.0, 20.0)])

    def test_does_not_cross_gap_or_lock_boundary(self) -> None:
        windows = [
            (0, 5, "listen", 1), (6, 11, "listen", 1),
            (11, 16, "listen", 0),
        ]
        self.assertEqual(
            build_rating_clips(windows, 30),
            [(0.0, 5.0), (6.0, 11.0), (11.0, 16.0)],
        )

    def test_animation_samples_two_frames_per_second_up_to_sixty(self) -> None:
        rows = [(index * 0.25, f"frame-{index}") for index in range(120)]
        paths = sample_animation_paths(rows, 0.0, 30.0, fps=2.0, max_frames=60)
        self.assertEqual(len(paths), 60)
        self.assertEqual(len(set(paths)), 60)


if __name__ == "__main__":
    unittest.main()
