from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "WPy64-312101"


def resolve_dataset(value: Path) -> Path:
    path = value if value.is_absolute() else REPO_ROOT / value
    return path.resolve()


def resolve_crop(source_root: Path, db_path: Path, value: str) -> Path | None:
    if not value:
        return None
    normalized = str(value).replace("\\", "/").strip()
    raw = Path(normalized)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([
        source_root / raw,
        db_path.parent / raw,
        db_path.parent.parent / raw,
    ])
    if normalized.startswith("data/"):
        candidates.append(source_root / normalized.removeprefix("data/"))
    if "data/assets/" in normalized:
        candidates.append(source_root / normalized.replace("data/assets/", "assets/"))
    candidates.append(source_root / "assets" / "crops" / raw.name)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
        except OSError:
            continue
    return None


def validate_source(conn: sqlite3.Connection, db_path: Path) -> None:
    tables = {
        str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    required = {"videos", "windows", "segments", "segment_frames"}
    missing = sorted(required - tables)
    if missing:
        raise RuntimeError(f"{db_path}: missing tables: {missing}")
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(windows)")}
    if not {"situation_set_by", "situation_locked"} <= columns:
        raise RuntimeError(f"{db_path}: run migrate_windows_schema.py first")
    pending = int(conn.execute("""
        SELECT COUNT(*) FROM windows
        WHERE situation IS NULL OR situation_set_by IS NULL OR situation_locked <> 1
    """).fetchone()[0])
    if pending:
        raise RuntimeError(
            f"{db_path}: {pending} windows are unset, unconfirmed, or unlocked"
        )


def create_slim_database(
        source_root: Path, destination_root: Path,
) -> dict[str, int]:
    source_db = source_root / "db" / "dataset.sqlite"
    if not source_db.exists():
        raise FileNotFoundError(source_db)
    destination_db = destination_root / "db" / "dataset.sqlite"
    destination_db.parent.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
    validate_source(src, source_db)
    dst = sqlite3.connect(str(destination_db))
    dst.execute("PRAGMA foreign_keys=ON")
    dst.executescript("""
        CREATE TABLE videos (
          id INTEGER PRIMARY KEY,
          path TEXT UNIQUE,
          fps REAL, width INTEGER, height INTEGER, frame_count INTEGER
        );
        CREATE TABLE windows (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL,
          t_start REAL NOT NULL,
          t_end REAL NOT NULL,
          situation TEXT NOT NULL CHECK(situation IN ('聞く', '書く', '話し合う')),
          situation_set_by TEXT NOT NULL,
          situation_updated_at TEXT,
          situation_locked INTEGER NOT NULL DEFAULT 1 CHECK(situation_locked IN (0,1)),
          UNIQUE(video_id, t_start, t_end),
          FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
        );
        CREATE TABLE segments (
          id INTEGER PRIMARY KEY,
          window_id INTEGER NOT NULL,
          track_id INTEGER NOT NULL,
          UNIQUE(window_id, track_id),
          FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
        );
        CREATE TABLE segment_frames (
          id INTEGER PRIMARY KEY,
          segment_id INTEGER NOT NULL,
          t REAL NOT NULL,
          frame_idx INTEGER NOT NULL,
          x1 REAL, y1 REAL, x2 REAL, y2 REAL,
          crop_path TEXT,
          pose_path TEXT,
          UNIQUE(segment_id, t),
          FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
        );
        CREATE TABLE labels (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          segment_id INTEGER NOT NULL,
          rater TEXT NOT NULL,
          score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 7),
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT DEFAULT (datetime('now')),
          UNIQUE(segment_id, rater),
          FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
        );
        CREATE TABLE window_skips (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          rater TEXT NOT NULL,
          window_id INTEGER NOT NULL,
          created_at TEXT DEFAULT (datetime('now')),
          UNIQUE(rater, window_id),
          FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
        );
        CREATE TABLE label_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT DEFAULT (datetime('now')),
          rater TEXT NOT NULL,
          action TEXT NOT NULL,
          segment_id INTEGER,
          old_score INTEGER,
          new_score INTEGER,
          FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE SET NULL
        );
        CREATE TABLE blind_assignments (
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
        CREATE TABLE label_observations (
          segment_id INTEGER NOT NULL,
          rater TEXT NOT NULL,
          source_clip_start REAL,
          source_clip_end REAL,
          source_clip_sec REAL,
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY(segment_id, rater),
          FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_windows_video_time ON windows(video_id, t_start, t_end);
        CREATE INDEX idx_seg_window_track ON segments(window_id, track_id);
        CREATE INDEX idx_segframes_segment ON segment_frames(segment_id);
    """)

    for row in src.execute("SELECT id, path, fps, width, height, frame_count FROM videos"):
        video_id, _path, fps, width, height, frame_count = row
        dst.execute(
            "INSERT INTO videos(id,path,fps,width,height,frame_count) VALUES(?,?,?,?,?,?)",
            (video_id, "videos/proxy.mp4", fps, width, height, frame_count),
        )
    dst.executemany("""
        INSERT INTO windows(
            id, video_id, t_start, t_end, situation, situation_set_by,
            situation_updated_at, situation_locked
        ) VALUES(?,?,?,?,?,?,?,?)
    """, src.execute("""
        SELECT id, video_id, t_start, t_end, situation, situation_set_by,
               situation_updated_at, situation_locked
        FROM windows ORDER BY id
    """).fetchall())
    dst.executemany(
        "INSERT INTO segments(id,window_id,track_id) VALUES(?,?,?)",
        src.execute("SELECT id,window_id,track_id FROM segments ORDER BY id").fetchall(),
    )

    crop_dir = destination_root / "assets" / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    copied_by_source: dict[Path, str] = {}
    used_names: dict[str, Path] = {}
    frame_count = 0
    crop_count = 0
    missing: list[str] = []
    rows = src.execute("""
        SELECT id, segment_id, t, frame_idx, x1, y1, x2, y2, crop_path
        FROM segment_frames ORDER BY id
    """)
    for frame_id, segment_id, t, source_frame_idx, x1, y1, x2, y2, crop_path in rows:
        resolved = resolve_crop(source_root, source_db, str(crop_path or ""))
        packaged_path = None
        if resolved is None:
            missing.append(str(crop_path or "(empty)"))
        else:
            packaged_name = copied_by_source.get(resolved)
            if packaged_name is None:
                packaged_name = resolved.name
                owner = used_names.get(packaged_name.lower())
                if owner is not None and owner != resolved:
                    packaged_name = f"sf{int(frame_id)}_{resolved.name}"
                shutil.copy2(resolved, crop_dir / packaged_name)
                copied_by_source[resolved] = packaged_name
                used_names[packaged_name.lower()] = resolved
                crop_count += 1
            packaged_path = f"assets/crops/{packaged_name}"
        dst.execute("""
            INSERT INTO segment_frames(
                id, segment_id, t, frame_idx, x1, y1, x2, y2, crop_path, pose_path
            ) VALUES(?,?,?,?,?,?,?,?,?,NULL)
        """, (
            frame_id, segment_id, t, source_frame_idx,
            x1, y1, x2, y2, packaged_path,
        ))
        frame_count += 1

    if missing:
        src.close()
        dst.close()
        destination_db.unlink(missing_ok=True)
        sample = "\n".join(missing[:10])
        raise RuntimeError(f"{len(missing)} crop files are missing. Examples:\n{sample}")

    dst.commit()
    violations = dst.execute("PRAGMA foreign_key_check").fetchall()
    src.close()
    dst.close()
    if violations:
        raise RuntimeError(f"foreign key violations in packaged DB: {violations[:10]}")
    return {"segment_frames": frame_count, "crops": crop_count}


def copy_runtime(destination: Path) -> None:
    if not RUNTIME_DIR.exists():
        raise FileNotFoundError(RUNTIME_DIR)
    if os.name == "nt":
        result = subprocess.run([
            "robocopy", str(RUNTIME_DIR), str(destination), "/E",
            "/R:2", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
        ])
        if result.returncode >= 8:
            raise RuntimeError(f"robocopy failed with exit code {result.returncode}")
    else:
        shutil.copytree(RUNTIME_DIR, destination)


def dataset_destination(source_root: Path, package_root: Path) -> Path:
    try:
        relative = source_root.relative_to(REPO_ROOT / "datasets")
    except ValueError:
        relative = Path(source_root.name)
    return package_root / "datasets" / relative


def build_package(
        datasets: list[Path], output: Path,
        overwrite: bool, skip_runtime: bool, make_zip: bool,
) -> Path:
    skip_runtime = skip_runtime or os.name != "nt"
    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    try:
        shutil.copy2(REPO_ROOT / "launcher.py", output / "launcher.py")
        shutil.copy2(REPO_ROOT / "launch_labeler.bat", output / "launch_labeler.bat")
        shutil.copy2(REPO_ROOT / "config.yaml", output / "config.yaml")
        (output / "requirements_labeler.txt").write_text("PySide6\nnumpy\nPyYAML\n", encoding="utf-8")
        (output / "START_HERE.txt").write_text(
            "Windows（同梱Pythonあり）: launch_labeler.bat\n"
            "Mac / Pythonなしの配布: Tkinter対応Python 3.12の仮想環境で\n"
            "python -m pip install -r requirements_labeler.txt\n"
            "python launcher.py\n", encoding="utf-8",
        )
        (output / "scripts" / "lib").mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "scripts" / "06_label_gui.py", output / "scripts" / "06_label_gui.py")
        shutil.copy2(REPO_ROOT / "scripts" / "lib" / "situation.py", output / "scripts" / "lib" / "situation.py")
        shutil.copy2(REPO_ROOT / "scripts" / "lib" / "image_roi.py", output / "scripts" / "lib" / "image_roi.py")
        shutil.copy2(
            REPO_ROOT / "scripts" / "lib" / "research_schema.py",
            output / "scripts" / "lib" / "research_schema.py",
        )
        shutil.copy2(
            REPO_ROOT / "scripts" / "lib" / "rating_clips.py",
            output / "scripts" / "lib" / "rating_clips.py",
        )

        dataset_results = []
        seen_destinations: set[Path] = set()
        for source in datasets:
            source_root = resolve_dataset(source)
            destination_root = dataset_destination(source_root, output)
            if destination_root in seen_destinations:
                raise RuntimeError(f"duplicate packaged dataset path: {destination_root}")
            seen_destinations.add(destination_root)
            destination_root.mkdir(parents=True)
            video = source_root / "videos" / "proxy.mp4"
            if not video.exists():
                raise FileNotFoundError(video)
            (destination_root / "videos").mkdir(parents=True)
            shutil.copy2(video, destination_root / "videos" / "proxy.mp4")
            proxy_ids = source_root / "videos" / "proxy_ids.mp4"
            if proxy_ids.exists():
                shutil.copy2(proxy_ids, destination_root / "videos" / "proxy_ids.mp4")
            counts = create_slim_database(source_root, destination_root)
            dataset_results.append({
                "dataset": destination_root.relative_to(output / "datasets").as_posix(),
                **counts,
            })

        if not skip_runtime:
            copy_runtime(output / "WPy64-312101")

        package_config = {
            "distribution_mode": "evaluator",
            "evaluator_name": "",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "datasets": dataset_results,
            "runtime_included": not skip_runtime,
        }
        (output / "package_config.json").write_text(
            json.dumps(package_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        '''
        (output / "RETURN_INSTRUCTIONS.txt").write_text(
            "採点後はすべての画面を終了し、datasets内の db\\dataset.sqlite を"
            "管理者へ渡してください。\n",
            encoding="utf-8-sig",
        )
        '''
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    if make_zip:
        shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true", help="テスト用: WinPythonをコピーしない")
    parser.add_argument("--zip", action="store_true", help="生成後にzipも作成")
    args = parser.parse_args()
    output = build_package(
        args.datasets, args.output,
        args.overwrite, args.skip_runtime, args.zip,
    )
    print(f"package: {output}")
    if args.zip:
        print(f"zip: {output}.zip")


if __name__ == "__main__":
    main()
