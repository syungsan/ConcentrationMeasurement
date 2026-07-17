from __future__ import annotations

import time
SCRIPT_STARTED_AT = time.perf_counter()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "school_id", "class_id", "session_id", "phase", "time_bin",
    "classroom_score",
}


def validate(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise RuntimeError(f"missing columns: {missing}")
    phases = set(df["phase"].dropna().astype(str))
    if not {"pre", "post"} <= phases:
        raise RuntimeError("both pre and post rows are required")


def paired_class_effects(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["school_id", "class_id"]
    means = df.groupby(keys + ["phase"], as_index=False).agg(
        score=("classroom_score", "mean"),
        n_bins=("classroom_score", "size"),
    )
    wide = means.pivot(index=keys, columns="phase", values="score").reset_index()
    counts = means.pivot(index=keys, columns="phase", values="n_bins").reset_index()
    wide = wide.dropna(subset=["pre", "post"]).copy()
    wide["effect_post_minus_pre"] = wide["post"] - wide["pre"]
    wide = wide.merge(
        counts.rename(columns={"pre": "n_pre", "post": "n_post"}), on=keys
    )
    return wide


def cluster_bootstrap(
    effects: pd.DataFrame, iterations: int, seed: int
) -> dict[str, float | int]:
    values = effects["effect_post_minus_pre"].to_numpy(dtype=float)
    if len(values) < 2:
        raise RuntimeError("at least two classes with pre/post observations are required")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "n_classes": int(len(values)),
        "mean_effect": float(values.mean()),
        "median_class_effect": float(np.median(values)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_iterations": int(iterations),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class-clustered pre/post analysis for a frozen measurement model."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    validate(df)
    df["classroom_score"] = pd.to_numeric(df["classroom_score"], errors="raise")
    effects = paired_class_effects(df)
    summary = cluster_bootstrap(effects, args.bootstrap, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.out_dir / "class_effects.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / "effect_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("NOTE: This paired cluster analysis does not adjust for calendar time or controls.")
    print("Use the exported class-level data in a prespecified mixed-effects/stepped-wedge model for confirmatory inference.")


if __name__ == "__main__":
    try:
        main()
        import winsound
        try:
            winsound.PlaySound("mei_kara_mei_switch1.wav", winsound.SND_FILENAME)
        except Exception as e:
            print(f"[WARN] 音声を再生できませんでした: {e}")
    finally:
        elapsed = time.perf_counter() - SCRIPT_STARTED_AT
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"所要時間: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} ({elapsed:.2f}秒)")

# command
# python scripts\11_intervention_analysis.py --input outputs\trial_classroom_timeseries.csv --out-dir outputs\trial_analysis
