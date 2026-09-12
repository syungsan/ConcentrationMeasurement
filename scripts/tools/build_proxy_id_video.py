from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.proxy_id_video import make_proxy_with_track_ids  # noqa: E402
from lib.runroot import rpath  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label-fps", type=float, default=4.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = args.dataset.resolve()
    cfg_path = args.config if args.config.is_absolute() else repo_root / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    db_path = rpath(dataset_root, cfg["paths"]["db_path"])
    proxy_video = rpath(dataset_root, cfg["paths"]["proxy_video"])
    output = args.output
    if output is None:
        output = proxy_video.with_name("proxy_ids.mp4")
    elif not output.is_absolute():
        output = dataset_root / output

    drawn = make_proxy_with_track_ids(
        db_path=db_path,
        proxy_video=proxy_video,
        output_video=output.resolve(),
        label_fps=args.label_fps,
    )
    print(f"output: {output.resolve()}")
    print(f"drawn labels: {drawn}")
    if drawn == 0:
        print("No IDs were drawn. Run 02_detect_track.py first and check detections.")


if __name__ == "__main__":
    main()
