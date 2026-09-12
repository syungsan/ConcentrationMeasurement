from __future__ import annotations

from typing import Iterable


def build_rating_clips(
    windows: Iterable[tuple[float, float, object, int]], max_seconds: float
) -> list[tuple[float, float]]:
    """Combine adjacent base windows without crossing situation/lock boundaries."""
    records = list(windows)
    if max_seconds <= 0:
        return [(float(t0), float(t1)) for t0, t1, _s, _locked in records]
    runs: list[tuple[float, float, object, int]] = []
    previous_center: float | None = None
    for raw_t0, raw_t1, situation, raw_locked in records:
        t0, t1, locked = float(raw_t0), float(raw_t1), int(raw_locked)
        center = (t0 + t1) / 2.0
        if t1 <= t0:
            raise ValueError(f"invalid window: {t0}-{t1}")
        if not runs:
            runs.append((t0, t1, situation, locked))
        else:
            run_start, run_end, run_situation, run_locked = runs[-1]
            overlaps_or_touches = t0 <= run_end + 1e-4
            same_context = situation == run_situation and locked == run_locked
            if overlaps_or_touches and same_context:
                runs[-1] = (run_start, max(run_end, t1), run_situation, run_locked)
            else:
                new_start = t0
                if overlaps_or_touches and previous_center is not None:
                    boundary = (previous_center + center) / 2.0
                    boundary = max(run_start, min(boundary, max(run_end, t1)))
                    runs[-1] = (run_start, boundary, run_situation, run_locked)
                    new_start = boundary
                runs.append((new_start, t1, situation, locked))
        previous_center = center

    clips: list[tuple[float, float]] = []
    for run_start, run_end, _situation, _locked in runs:
        clip_start = run_start
        while clip_start < run_end - 1e-6:
            clip_end = min(clip_start + max_seconds, run_end)
            clips.append((clip_start, clip_end))
            clip_start = clip_end
    centers = [(float(t0) + float(t1)) / 2.0 for t0, t1, _s, _locked in records]
    nonempty: list[tuple[float, float]] = []
    for clip_start, clip_end in clips:
        if any(clip_start <= center < clip_end for center in centers):
            nonempty.append((clip_start, clip_end))
        elif nonempty and abs(nonempty[-1][1] - clip_start) < 1e-4:
            nonempty[-1] = (nonempty[-1][0], clip_end)
    return nonempty


def sample_animation_paths(
    rows: Iterable[tuple[float, str]],
    start: float,
    end: float,
    fps: float = 2.0,
    max_frames: int = 60,
) -> list[str]:
    """Select the nearest available crop for uniform animation timestamps."""
    candidates = [
        (float(t), str(path)) for t, path in rows
        if path and start <= float(t) < end
    ]
    if not candidates or end <= start or fps <= 0 or max_frames <= 0:
        return []
    unique: list[tuple[float, str]] = []
    used: set[str] = set()
    for item in sorted(candidates):
        if item[1] not in used:
            unique.append(item)
            used.add(item[1])
    count = min(
        max_frames,
        max(1, int(round((end - start) * fps))),
        len(unique),
    )
    if count == 1:
        return [unique[len(unique) // 2][1]]
    indices = [
        round(index * (len(unique) - 1) / (count - 1))
        for index in range(count)
    ]
    return [unique[index][1] for index in indices]
