"""Locate native video tools on Windows, macOS and Linux."""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_video_tool(repo_root: Path, name: str = "ffmpeg") -> Path:
    filename = name + (".exe" if os.name == "nt" else "")
    bundled = repo_root / "ffmpeg" / "bin" / filename
    if bundled.is_file():
        return bundled.resolve()
    installed = shutil.which(name)
    if installed:
        return Path(installed)
    raise FileNotFoundError(
        f"{name} was not found. Install ffmpeg and add it to PATH "
        "(macOS: brew install ffmpeg)."
    )


def find_subtitle_ffmpeg(repo_root: Path) -> Path:
    candidates = []
    try:
        candidates.append(find_video_tool(repo_root))
    except FileNotFoundError:
        pass
    if sys.platform == "darwin":
        candidates.extend([
            Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"),
            Path("/usr/local/opt/ffmpeg-full/bin/ffmpeg"),
        ])
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        result = subprocess.run(
            [str(candidate), "-hide_banner", "-filters"],
            capture_output=True, text=True, check=True,
        )
        if any(len(parts := line.split()) > 1 and parts[1] == "subtitles"
               for line in result.stdout.splitlines()):
            return candidate
    raise RuntimeError(
        "ID動画にはsubtitlesフィルター（libass）対応のffmpegが必要です。"
        "Macでは brew install ffmpeg-full を実行してください。"
        "検出結果の再作成は不要です。build_proxy_id_video.pyだけを再実行してください。"
    )
