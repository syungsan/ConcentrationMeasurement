import subprocess
from pathlib import Path
import yaml

from lib.runroot import get_data_root, rpath

here = Path(__file__).resolve()
repo_root = here.parent.parent  # config.yaml が置いてある場所（従来どおり）

def make_proxy(ffmpeg_bin: Path, src: Path, dst: Path, width: int, fps: int, crf: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_bin), "-y",
        "-i", str(src),
        "-vf", f"scale={width}:-2",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
        str(dst),
    ]
    subprocess.run(cmd, check=True)

def main():
    # config.yaml は repo_root に1つだけ
    cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))

    # データ実体のrootは実行時に切り替える（未指定なら従来互換で repo_root）
    data_root = get_data_root(repo_root)

    raw = rpath(data_root, cfg["paths"]["raw_video"])
    proxy = rpath(data_root, cfg["paths"]["proxy_video"])
    p = cfg["proxy"]

    ffmpeg_bin = (repo_root / "ffmpeg/bin/ffmpeg.exe").resolve()
    if not ffmpeg_bin.exists():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg_bin}")

    if not raw.exists():
        raise FileNotFoundError(f"raw video not found: {raw}")

    make_proxy(ffmpeg_bin, raw, proxy, int(p["width"]), int(p["fps"]), int(p["crf"]))
    print("OK proxy:", proxy)

if __name__ == "__main__":
    main()

# command
# python scripts/01_make_proxy.py --data_root datasets/lesson_001