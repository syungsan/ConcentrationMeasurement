"""Read-only audit of actual stored time series; no model or GPU required."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from statistics import median


def audit(db: Path) -> dict:
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        rows = conn.execute("""
            SELECT s.id, w.t_end-w.t_start, COUNT(sf.t),
                   MIN(sf.t), MAX(sf.t), COUNT(DISTINCT sf.t),
                   SUM(CASE WHEN sf.t < w.t_start OR sf.t >= w.t_end THEN 1 ELSE 0 END),
                   EXISTS(SELECT 1 FROM labels l WHERE l.segment_id=s.id)
            FROM segments s JOIN windows w ON w.id=s.window_id
            LEFT JOIN segment_frames sf ON sf.segment_id=s.id
            GROUP BY s.id
        """).fetchall()
        label_counts = conn.execute(
            "SELECT rater, COUNT(*), MIN(score), MAX(score) FROM labels GROUP BY rater"
        ).fetchall()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        excluded = conn.execute("SELECT COUNT(*) FROM excluded_segments").fetchone()[0] if "excluded_segments" in tables else 0
    conn.close()

    def summarize(selected: list) -> dict:
        spans = [r[4]-r[3] for r in selected if r[2]]
        return {
            "segments": len(selected),
            "non_5s_windows": sum(abs(r[1]-5.0) > 1e-5 for r in selected),
            "no_frames": sum(r[2] == 0 for r in selected),
            "fewer_than_10_frames": sum(r[2] < 10 for r in selected),
            "exactly_10_frames": sum(r[2] == 10 for r in selected),
            "more_than_10_frames": sum(r[2] > 10 for r in selected),
            "duplicate_timestamps": sum(r[2] != r[5] for r in selected),
            "frames_outside_window": sum(r[6] or 0 for r in selected),
            "span_min_sec": min(spans) if spans else None,
            "span_median_sec": median(spans) if spans else None,
            "span_max_sec": max(spans) if spans else None,
        }
    return {"db": str(db.resolve()), "all": summarize(rows),
            "labeled": summarize([r for r in rows if r[7]]),
            "excluded_segments": excluded,
            "raters": [{"name": r[0], "n": r[1], "min": r[2], "max": r[3]} for r in label_counts]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [audit(db) for db in sorted(args.datasets.glob("*/db/dataset.sqlite"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print(Path(result["db"]).parents[1].name, json.dumps(result["labeled"]))


if __name__ == "__main__":
    main()
