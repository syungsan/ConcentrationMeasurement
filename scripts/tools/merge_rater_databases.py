from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sqlite3
from pathlib import Path


def open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def require_schema(conn: sqlite3.Connection, path: Path) -> None:
    required = {"videos", "windows", "segments", "labels"}
    tables = {
        str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = sorted(required - tables)
    if missing:
        raise RuntimeError(f"{path}: required tables are missing: {missing}")
    segment_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(segments)")}
    if "window_id" not in segment_columns:
        raise RuntimeError(f"{path}: old schema; run migrate_windows_schema.py first")


def segment_map(conn: sqlite3.Connection) -> dict[tuple, int]:
    rows = conn.execute("""
        SELECT v.fps, v.width, v.height, v.frame_count,
               w.t_start, w.t_end, s.track_id, s.id
        FROM segments s
        JOIN windows w ON w.id=s.window_id
        JOIN videos v ON v.id=w.video_id
    """).fetchall()
    return {
        (
            round(float(fps or 0.0), 6), int(width or 0), int(height or 0), int(frame_count or 0),
            round(float(t0), 6), round(float(t1), 6), int(track_id),
        ): int(segment_id)
        for fps, width, height, frame_count, t0, t1, track_id, segment_id in rows
    }


def situation_map(conn: sqlite3.Connection) -> dict[tuple, tuple[str | None, int]]:
    rows = conn.execute("""
        SELECT v.fps, v.width, v.height, v.frame_count,
               w.t_start, w.t_end, w.situation, w.situation_locked
        FROM windows w JOIN videos v ON v.id=w.video_id
    """).fetchall()
    return {
        (
            round(float(fps or 0.0), 6), int(width or 0), int(height or 0), int(frame_count or 0),
            round(float(t0), 6), round(float(t1), 6),
        ): (str(situation) if situation is not None else None, int(locked))
        for fps, width, height, frame_count, t0, t1, situation, locked in rows
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_database(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    src = open_readonly(source)
    dst = sqlite3.connect(str(output))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()


def rebuild_consensus(conn: sqlite3.Connection) -> int:
    conn.execute("DROP TABLE IF EXISTS label_consensus")
    conn.execute("""
        CREATE TABLE label_consensus (
          segment_id INTEGER PRIMARY KEY,
          mean_score REAL NOT NULL,
          rater_count INTEGER NOT NULL,
          score_std REAL NOT NULL,
          score_min INTEGER NOT NULL,
          score_max INTEGER NOT NULL,
          updated_at TEXT DEFAULT (datetime('now')),
          FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
        )
    """)
    grouped: dict[int, list[float]] = {}
    for segment_id, score in conn.execute("SELECT segment_id, score FROM labels ORDER BY segment_id"):
        grouped.setdefault(int(segment_id), []).append(float(score))
    for segment_id, scores in grouped.items():
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        conn.execute("""
            INSERT INTO label_consensus(
                segment_id, mean_score, rater_count, score_std, score_min, score_max
            ) VALUES(?,?,?,?,?,?)
        """, (
            segment_id, mean, len(scores), math.sqrt(variance),
            int(min(scores)), int(max(scores)),
        ))
    return len(grouped)


def write_consensus_csv(conn: sqlite3.Connection, output_db: Path) -> Path:
    csv_path = output_db.with_name(f"{output_db.stem}_consensus.csv")
    rows = conn.execute("""
        SELECT w.t_start, w.t_end, s.track_id, w.situation,
               c.mean_score, c.rater_count, c.score_std, c.score_min, c.score_max
        FROM label_consensus c
        JOIN segments s ON s.id=c.segment_id
        JOIN windows w ON w.id=s.window_id
        ORDER BY w.t_start, s.track_id
    """).fetchall()
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "t_start", "t_end", "track_id", "situation", "mean_score",
            "rater_count", "score_std", "score_min", "score_max",
        ])
        writer.writerows(rows)
    return csv_path


def merge(base_db: Path, inputs: list[Path], output: Path, overwrite: bool) -> dict[str, object]:
    base_db = base_db.resolve()
    inputs = [path.resolve() for path in inputs]
    output = output.resolve()
    if output == base_db or output in inputs:
        raise RuntimeError("output must be different from base/input databases")

    base_conn = open_readonly(base_db)
    try:
        require_schema(base_conn, base_db)
        base_segments = segment_map(base_conn)
        base_situations = situation_map(base_conn)
    finally:
        base_conn.close()

    if not base_segments:
        raise RuntimeError("base database has no segments")
    if any(situation is None or not locked for situation, locked in base_situations.values()):
        raise RuntimeError("base database has unset or unlocked shared situations")

    copy_database(base_db, output, overwrite)
    out = sqlite3.connect(str(output))
    out.execute("PRAGMA foreign_keys=ON")
    imported = 0
    duplicates = 0
    raters: set[str] = set()

    try:
        out.execute("""
            CREATE TABLE IF NOT EXISTS merge_sources (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source_name TEXT NOT NULL,
              source_sha256 TEXT NOT NULL UNIQUE,
              imported_at TEXT DEFAULT (datetime('now')),
              label_count INTEGER NOT NULL,
              raters TEXT NOT NULL
            )
        """)
        with out:
            for input_path in inputs:
                source_hash = file_sha256(input_path)
                if out.execute(
                    "SELECT 1 FROM merge_sources WHERE source_sha256=?", (source_hash,)
                ).fetchone():
                    duplicates += 1
                    continue

                src = open_readonly(input_path)
                try:
                    require_schema(src, input_path)
                    src_segments = segment_map(src)
                    if set(src_segments) != set(base_segments):
                        raise RuntimeError(f"different dataset topology: {input_path}")
                    if situation_map(src) != base_situations:
                        raise RuntimeError(f"shared situations differ from base: {input_path}")

                    reverse_src = {segment_id: key for key, segment_id in src_segments.items()}
                    source_raters: set[str] = set()
                    source_count = 0
                    for segment_id, rater, score, note, created_at, updated_at in src.execute("""
                        SELECT segment_id, rater, score, note, created_at, updated_at
                        FROM labels ORDER BY id
                    """):
                        key = reverse_src.get(int(segment_id))
                        if key is None:
                            raise RuntimeError(f"unknown segment {segment_id}: {input_path}")
                        destination_segment = base_segments[key]
                        rater = str(rater).strip()
                        if not rater:
                            raise RuntimeError(f"blank rater name: {input_path}")
                        existing = out.execute(
                            "SELECT score, COALESCE(note,'') FROM labels WHERE segment_id=? AND rater=?",
                            (destination_segment, rater),
                        ).fetchone()
                        normalized_note = str(note or "")
                        if existing:
                            if int(existing[0]) != int(score) or str(existing[1]) != normalized_note:
                                raise RuntimeError(
                                    f"conflicting label: rater={rater}, segment={destination_segment}, source={input_path}"
                                )
                            duplicates += 1
                            continue
                        out.execute("""
                            INSERT INTO labels(
                                segment_id, rater, score, note, created_at, updated_at
                            ) VALUES(?,?,?,?,?,?)
                        """, (
                            destination_segment, rater, int(score), note,
                            created_at, updated_at or created_at,
                        ))
                        imported += 1
                        source_count += 1
                        source_raters.add(rater)
                        raters.add(rater)
                    out.execute("""
                        INSERT INTO merge_sources(source_name, source_sha256, label_count, raters)
                        VALUES(?,?,?,?)
                    """, (input_path.name, source_hash, source_count, ",".join(sorted(source_raters))))
                finally:
                    src.close()

            consensus_count = rebuild_consensus(out)
            raters = {
                str(row[0]) for row in out.execute(
                    "SELECT DISTINCT rater FROM labels ORDER BY rater"
                )
            }
        if out.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign key check failed after merge")
        csv_path = write_consensus_csv(out, output)
    except Exception:
        out.close()
        if output.exists():
            output.unlink()
        raise
    out.close()
    return {
        "output": output,
        "consensus_csv": csv_path,
        "imported_labels": imported,
        "duplicates": duplicates,
        "raters": sorted(raters),
        "consensus_segments": consensus_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-db", required=True, type=Path)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = merge(args.base_db, args.inputs, args.output, args.overwrite)
    print(f"output: {result['output']}")
    print(f"consensus CSV: {result['consensus_csv']}")
    print(f"imported labels: {result['imported_labels']}")
    print(f"duplicate labels/sources: {result['duplicates']}")
    print(f"raters: {', '.join(result['raters']) or '(none)'}")
    print(f"consensus segments: {result['consensus_segments']}")


if __name__ == "__main__":
    main()
