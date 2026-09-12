from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"school_id", "phase", "human_score", "model_score"}


def phase_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    human = frame["human_score"].to_numpy(dtype=float)
    model = frame["model_score"].to_numpy(dtype=float)
    error = model - human
    correlation = float(np.corrcoef(human, model)[0, 1]) if len(frame) > 1 else float("nan")
    return {
        "n": int(len(frame)),
        "mae": float(np.mean(np.abs(error))),
        "bias_model_minus_human": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson": correlation,
    }


def bootstrap_bias_difference(
    frame: pd.DataFrame, iterations: int, seed: int
) -> dict[str, float]:
    # Resample schools, retaining within-school dependence.
    schools = frame["school_id"].drop_duplicates().tolist()
    if len(schools) < 2:
        raise RuntimeError("audit requires at least two schools")
    grouped = {school: frame[frame["school_id"] == school] for school in schools}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        selected = rng.choice(schools, size=len(schools), replace=True)
        sample = pd.concat([grouped[school] for school in selected], ignore_index=True)
        bias = sample.assign(error=sample["model_score"] - sample["human_score"]).groupby(
            "phase"
        )["error"].mean()
        if "pre" in bias and "post" in bias:
            values.append(float(bias["post"] - bias["pre"]))
    if not values:
        raise RuntimeError("both pre and post observations are required in bootstrap samples")
    arr = np.asarray(values)
    return {
        "post_minus_pre_bias": float(arr.mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether model-vs-human error changes after intervention."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing columns: {missing}")
    frame = frame[frame["phase"].isin(["pre", "post"])].copy()
    result = {
        "pre": phase_metrics(frame[frame["phase"] == "pre"]),
        "post": phase_metrics(frame[frame["phase"] == "post"]),
        "bias_shift": bootstrap_bias_difference(frame, args.bootstrap, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
