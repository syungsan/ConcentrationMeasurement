# lib/runroot.py
from __future__ import annotations

import argparse
from pathlib import Path


def get_repo_root(file_path: str | Path) -> Path:
    """
    scripts/xx.py から呼ぶ想定:
      here = Path(__file__).resolve()
      repo_root = here.parent.parent
    を毎回書くのが面倒なら使う補助（任意）。
    """
    p = Path(file_path).resolve()
    return p.parent.parent


def get_data_root(repo_root: Path) -> Path:
    """
    実行時に --data_root が指定されたらそれを「この実行のデータroot」にする。
    指定がなければ repo_root をデータrootとして扱う（従来互換）。
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data_root", type=str, default=None)
    args, _ = ap.parse_known_args()

    if args.data_root:
        return Path(args.data_root).resolve()
    return repo_root.resolve()


def rpath(data_root: Path, rel: str) -> Path:
    """
    data_root を基準に config.yaml の paths 相対パスを解決する。
    """
    return (data_root / rel).resolve()
