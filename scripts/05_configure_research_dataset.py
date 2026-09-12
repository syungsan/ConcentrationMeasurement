from __future__ import annotations

import time
SCRIPT_STARTED_AT = time.perf_counter()

import argparse
import sqlite3
from pathlib import Path

from lib.research_schema import ResearchMetadata, metadata_json, set_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = ("normal", "pilot_post", "pre", "post")


def resolve_dataset(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="05: Register the minimum research metadata for one dataset."
    )
    parser.add_argument("dataset", help="dataset folder, e.g. datasets/lesson_001")
    parser.add_argument("school_id", help="anonymous school ID, e.g. school_a")
    parser.add_argument("grade", help="grade, e.g. 5")
    parser.add_argument("class_id", help="anonymous class ID, e.g. 5A")
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument(
        "--minutes-after", type=float,
        help="Minutes after intervention; required only for pilot_post/post.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset(args.dataset)
    db = dataset / "db" / "dataset.sqlite"
    if not db.exists():
        raise FileNotFoundError(db)
    minutes_after = args.minutes_after
    if args.phase in {"pilot_post", "post"} and minutes_after is None:
        raise ValueError(
            f"phase={args.phase} requires --minutes-after (use 0 for immediately after)"
        )
    role = "trial" if args.phase in {"pre", "post"} else "development"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        set_metadata(conn, ResearchMetadata(
            school_id=args.school_id,
            grade=args.grade,
            class_id=args.class_id,
            session_id=dataset.name,
            intervention_phase=args.phase,
            study_role=role,
            minutes_since_intervention=minutes_after,
        ))
        print(metadata_json(conn))
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
        from lib.completion_sound import play_completion_sound
        play_completion_sound()
    finally:
        elapsed = time.perf_counter() - SCRIPT_STARTED_AT
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"所要時間: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} ({elapsed:.2f}秒)")

# command
# python scripts\05_configure_research_dataset.py datasets\lesson_001 school_a 5 5A normal
