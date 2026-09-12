from __future__ import annotations

from .platform_tools import find_video_tool, find_subtitle_ffmpeg

import json
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ffmpeg_bin(name: str) -> Path:
    return find_video_tool(_repo_root(), name)


def _probe_video(path: Path) -> tuple[int, int, float]:
    ffprobe = _ffmpeg_bin("ffprobe")
    result = subprocess.run(
        [
            str(ffprobe), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    stream = json.loads(result.stdout)["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = str(stream.get("r_frame_rate", "24/1")).split("/")
    fps = float(num) / max(1.0, float(den))
    return width, height, fps


def _ass_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    hours = int(sec // 3600)
    sec -= hours * 3600
    minutes = int(sec // 60)
    sec -= minutes * 60
    seconds = int(sec)
    centis = int(round((sec - seconds) * 100))
    if centis >= 100:
        seconds += 1
        centis -= 100
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centis:02d}"


def _escape_ass_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _write_ass(
        *,
        ass_path: Path,
        width: int,
        height: int,
        label_fps: float,
        detections_by_frame: dict[int, dict[int, tuple[int, int, int, int]]],
) -> int:
    frame_sec = 1.0 / max(1.0, label_fps)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: ID,Arial,18,&H77FFFFFF,&H77FFFFFF,&H55201811,&HAA201811,"
        "1,0,0,0,100,100,0,0,3,1,0,7,0,0,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    count = 0
    for frame_idx, labels in sorted(detections_by_frame.items()):
        start = frame_idx / max(1.0, label_fps)
        end = start + frame_sec
        for track_id, (x1, y1, x2, y2) in sorted(labels.items()):
            box_w = max(1, int(x2) - int(x1))
            box_h = max(1, int(y2) - int(y1))
            x = int(x1) + max(6, int(round(box_w * 0.06)))
            y = int(y1) - max(4, int(round(box_h * 0.04)))
            x = max(4, min(int(x), width - 70))
            y = max(22, min(int(y), height - 24))
            text = _escape_ass_text(f"ID {track_id}")
            lines.append(
                "Dialogue: 0,"
                f"{_ass_time(start)},{_ass_time(end)},ID,,0,0,0,,"
                f"{{\\pos({x},{y})}}{text}"
            )
            count += 1
    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def make_proxy_with_track_ids(
        *,
        db_path: Path,
        proxy_video: Path,
        output_video: Path,
        label_fps: float = 4.0,
) -> int:
    """Bake subtle track IDs into a copy of the proxy video using ffmpeg ASS subtitles."""
    if not proxy_video.exists():
        raise FileNotFoundError(proxy_video)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    width, height, _proxy_fps = _probe_video(proxy_video)
    label_fps = max(1.0, float(label_fps))
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            str(row[0]) for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "detections" not in tables:
            return 0
        row = conn.execute("""
            SELECT id, COALESCE(width,0), COALESCE(height,0)
            FROM videos ORDER BY id DESC LIMIT 1
        """).fetchone()
        if not row:
            return 0
        video_id = int(row[0])
        src_w = int(row[1] or 0)
        src_h = int(row[2] or 0)
        sx = width / src_w if src_w > 0 else 1.0
        sy = height / src_h if src_h > 0 else 1.0

        detections_by_frame: dict[int, dict[int, tuple[int, int, int, int]]] = defaultdict(dict)
        for track_id, t, x1, y1, x2, y2 in conn.execute("""
            SELECT track_id, t, x1, y1, x2, y2
            FROM detections
            WHERE video_id=?
            ORDER BY t, track_id
        """, (video_id,)):
            frame_idx = max(0, int(round(float(t) * label_fps)))
            detections_by_frame[frame_idx][int(track_id)] = (
                int(round(float(x1) * sx)),
                int(round(float(y1) * sy)),
                int(round(float(x2) * sx)),
                int(round(float(y2) * sy)),
            )
    finally:
        conn.close()

    output_video.parent.mkdir(parents=True, exist_ok=True)
    ass_path = output_video.with_name(output_video.stem + ".ass")
    count = _write_ass(
        ass_path=ass_path,
        width=width,
        height=height,
        label_fps=label_fps,
        detections_by_frame=detections_by_frame,
    )
    if count == 0:
        return 0

    tmp_video = output_video.with_name(output_video.stem + ".tmp.mp4")
    ffmpeg = find_subtitle_ffmpeg(_repo_root())
    subprocess.run(
        [
            str(ffmpeg), "-y",
            "-i", str(proxy_video),
            "-vf", f"subtitles={ass_path.name}",
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "copy",
            str(tmp_video),
        ],
        cwd=str(output_video.parent),
        check=True,
    )
    if output_video.exists():
        output_video.unlink()
    tmp_video.replace(output_video)
    try:
        ass_path.unlink()
    except OSError:
        pass
    return count
