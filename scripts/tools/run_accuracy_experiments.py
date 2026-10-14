"""Run controlled accuracy ablations with the same dataset holdout and seeds."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import os
import queue
import threading
import time
import statistics
import math
from datetime import datetime
from uuid import uuid4
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_with_progress(command, log, *, title):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    messages = queue.Queue()

    def read_output():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        while True:
            try:
                line = messages.get(timeout=15)
            except queue.Empty:
                print(f"[{title}] running, elapsed={time.monotonic()-started:.0f}s (waiting for training output)", flush=True)
                continue
            if line is None:
                break
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        reader.join(timeout=2)
        process.stdout.close()


def experiments():
    # Each adjacent ablation changes one conceptual factor.
    base = ["--augmentation", "none", "--situation_teacher_forcing", "0"]
    tuned = base + ["--temporal_pool", "mean", "--backbone_lr_scale", "0.1",
                    "--freeze_batchnorm", "--lr_schedule", "plateau",
                    "--early_stopping_patience", "10"]
    motion = tuned + ["--pose_features", "motion"]
    ordered = motion + ["--ordered_ordinal", "--score_loss_weight", "0.2"]
    return {
        "baseline": base,
        "optimization": tuned,
        "motion": motion,
        "ordered": ordered,
        "balanced": ordered + ["--class_balance"],
        "mild": ordered + ["--augmentation", "mild"],
        "standard": ordered + ["--augmentation", "standard"],
        "no_situation": ordered + ["--no-situation-feature"],
        "grayscale": ordered + ["--grayscale"],
    }



def summarize_cv(rows):
    """Rank by mean lesson-level exact accuracy; MAE breaks ties.

    Epoch medians are rounded up and are only a practical refit heuristic.
    """
    candidates = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        accuracy = [float(row["val_exact_accuracy"]) for row in group]
        candidate = {
            "variant": variant, "runs": len(group),
            "mean_exact_accuracy": statistics.mean(accuracy),
            "std_exact_accuracy": statistics.stdev(accuracy) if len(accuracy) > 1 else 0.0,
            "mean_mae": statistics.mean(float(row["val_mae"]) for row in group),
            "mean_qwk": statistics.mean(float(row["val_qwk"]) for row in group),
            "recommended_epochs": max(1, math.ceil(statistics.median(int(row["epoch"]) for row in group))),
            "recommended_situation_epochs": math.ceil(statistics.median(int(row["situation_epoch"]) for row in group)),
            "fold_best_epochs": [int(row["epoch"]) for row in group],
        }
        candidates.append(candidate)
    candidates.sort(key=lambda row: (-row["mean_exact_accuracy"], row["mean_mae"]))
    return {"selection_rule": "mean lesson exact accuracy, then mean MAE", "selected": candidates[0], "candidates": candidates}


def read_prior_results(directory):
    rows = []
    paths = [directory / "summary.csv", *sorted(directory.glob("additions/*/summary.csv"))]
    for path in paths:
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows.extend(csv.DictReader(stream))
    return rows


def append_plan(directory, chosen, output):
    """Use saved baseline commands to retain the original splits and training settings."""
    source = json.loads((directory / "experiment_plan.json").read_text(encoding="utf-8"))
    references = [row for row in source if row["variant"] == "baseline"]
    if not references:
        raise ValueError("Append requires baseline entries in the original experiment_plan.json")
    prior = read_prior_results(directory)
    keys = {(row["variant"], str(row["seed"]), str(row.get("fold") or "")) for row in prior}
    plan = []
    for reference in references:
        for variant in chosen:
            fold = reference.get("fold")
            key = (variant, str(reference["seed"]), str(fold) if fold is not None else "")
            if key in keys:
                raise FileExistsError(f"Result already exists: variant={variant}, seed={reference['seed']}, fold={fold}")
            suffix = f"_fold{fold}" if fold is not None else ""
            checkpoint = output / f"{variant}_seed{reference['seed']}{suffix}.pt"
            command = reference["command"].copy()
            command[0] = sys.executable
            command[1] = str(ROOT / "scripts/07_train.py")
            command[command.index("--save_name") + 1] = str(checkpoint)
            command.extend(experiments()[variant])
            plan.append({"variant": variant, "seed": reference["seed"], "fold": fold,
                         "checkpoint": str(checkpoint), "command": command,
                         "reference_checkpoint": reference["checkpoint"]})
    return plan, prior


def write_summary(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_folds(dataset_count, fold_count=0):
    count = fold_count or dataset_count
    if not 2 <= count <= dataset_count:
        raise ValueError("cv_folds must be between 2 and the number of datasets (0 means leave-one-dataset-out)")
    return [[index for index in range(dataset_count) if index % count == fold] for fold in range(count)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_roots", default="")
    ap.add_argument("--append", action="store_true",
                    help="Add only chosen variants in a new subdirectory, inheriting original experiment settings.")
    ap.add_argument("--raters", default="tadano")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--variants", default=",".join(experiments()))
    ap.add_argument("--cross_validation", choices=["none", "dataset"], default="none",
                    help="dataset: rotate each of N dataset roots as the validation set.")
    ap.add_argument("--cv_folds", type=int, default=0,
                    help="Group whole datasets into this many folds; 0 validates each dataset individually.")
    ap.add_argument("--situation_patience", type=int, default=None,
                    help="Stop situation training after this many epochs without validation-loss improvement.")
    ap.add_argument("--concentration_patience", type=int, default=None,
                    help="Override concentration early-stopping patience for all chosen variants.")
    ap.add_argument("--selection_metric", choices=["mse", "mae", "exact_accuracy", "qwk"], default="mse")
    ap.add_argument("--final_train", action="store_true", help="After CV, train selected variant on all data at median best epochs.")
    ap.add_argument("--situation_epochs", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--situation_profile", choices=["legacy", "shared"], default="legacy")
    ap.add_argument("--output_dir", default="outputs/accuracy_experiments")
    ap.add_argument("--split_unit", choices=["dataset", "session", "window"], default="dataset")
    ap.add_argument("--holdout_groups", default=None, help="Default dataset:0 for dataset split; otherwise random seeded holdout.")
    ap.add_argument("--execute", action="store_true", help="Without this flag, only save/print the plan.")
    args = ap.parse_args()
    if not args.append and not args.data_roots:
        ap.error("--data_roots is required unless --append is used")
    if args.append and args.final_train:
        ap.error("--append cannot automatically refit a final model; compare combined results first")
    if args.final_train and (not args.execute or args.cross_validation != "dataset"):
        ap.error("--final_train requires --execute and --cross_validation dataset")
    if args.cross_validation == "dataset":
        roots = [value for value in args.data_roots.split(",") if value.strip()]
        if len(roots) < 2:
            ap.error("dataset cross-validation requires at least two datasets")
        if args.holdout_groups:
            ap.error("cross-validation chooses holdouts automatically; omit --holdout_groups")
        args.split_unit = "dataset"
        try:
            folds = dataset_folds(len(roots), args.cv_folds)
        except ValueError as error:
            ap.error(str(error))
    elif args.cv_folds:
        ap.error("--cv_folds requires --cross_validation dataset")
    if args.situation_patience is not None and args.situation_patience < 0:
        ap.error("--situation_patience must be nonnegative")
    if args.concentration_patience is not None and args.concentration_patience < 0:
        ap.error("--concentration_patience must be nonnegative")
    if args.holdout_groups is None:
        args.holdout_groups = "dataset:0" if args.split_unit == "dataset" else ""
    variants = experiments()
    chosen = args.variants.split(",")
    unknown = set(chosen) - variants.keys()
    if unknown:
        ap.error(f"unknown variants: {sorted(unknown)}")
    seeds = [int(s) for s in args.seeds.split(",")]
    original_output = Path(args.output_dir).resolve()
    output = (original_output / "additions" / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
              if args.append else original_output)
    output.mkdir(parents=True, exist_ok=True)
    plan = []
    for seed in ([] if args.append else seeds):
        for variant in chosen:
            for fold in (range(len(folds)) if args.cross_validation == "dataset" else [None]):
                suffix = f"_fold{fold}" if fold is not None else ""
                checkpoint = output / f"{variant}_seed{seed}{suffix}.pt"
                command = [sys.executable, str(ROOT / "scripts/07_train.py"),
                           "--data_roots", args.data_roots, "--raters", args.raters,
                           "--label_source", "individual", "--include_pilot_post",
                           "--mode", "fusion", "--temporal", "transformer",
                           "--T", "10", "--sample_strategy", "uniform",
                           "--image_roi", "upper_body", "--fusion_image_scale", "0.5",
                           "--lr", "1e-4", "--selection_metric", args.selection_metric,
                           "--situation_epochs", str(args.situation_epochs),
                           "--epochs", str(args.epochs), "--num_workers", str(args.num_workers),
                           "--seed", str(seed), "--split_unit", args.split_unit,
                           "--situation_profile", args.situation_profile,
                           "--holdout_groups", ",".join(f"dataset:{index}" for index in folds[fold]) if fold is not None else args.holdout_groups,
                           "--save_name", str(checkpoint), *variants[variant]]
                if args.situation_patience is not None:
                    command.extend(["--situation_patience", str(args.situation_patience)])
                if args.concentration_patience is not None:
                    command.extend(["--early_stopping_patience", str(args.concentration_patience)])
                plan.append({"variant": variant, "seed": seed, "fold": fold, "checkpoint": str(checkpoint), "command": command})
    prior = []
    if args.append:
        plan, prior = append_plan(original_output, chosen, output)
        command = plan[0]["command"]
        args.selection_metric = command[command.index("--selection_metric") + 1]
        args.situation_profile = command[command.index("--situation_profile") + 1]
        args.cross_validation = "dataset" if any(run["fold"] is not None for run in plan) else "none"
        print(f"Append: original files preserved at {original_output}; new results at {output}", flush=True)
    retained_summary = []
    if not args.append and args.execute:
        selected_keys = {(run["variant"], str(run["seed"]), str(run["fold"]) if run["fold"] is not None else "") for run in plan}
        summary_path = output / "summary.csv"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8-sig", newline="") as stream:
                retained_summary = [row for row in csv.DictReader(stream)
                                    if (row["variant"], str(row["seed"]), str(row.get("fold") or "")) not in selected_keys]
        print(f"Overwrite mode: rerunning {len(plan)} selected experiments in {output}", flush=True)
        # Remove stale selected rows before starting, so interrupted runs cannot
        # present their previous metrics as results of this rerun.
        if summary_path.exists():
            write_summary(summary_path, retained_summary)
    (output / "experiment_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = []
    overall_started = time.monotonic()
    for index, run in enumerate(plan, 1):
        title = f"{index}/{len(plan)} {run['variant']} seed={run['seed']} fold={run['fold']}"
        print(f"\n[{title}] {'START' if args.execute else 'PLAN'}; completed={index-1}/{len(plan)}; total elapsed={time.monotonic()-overall_started:.0f}s", flush=True)
        print(subprocess.list2cmdline(run["command"]), flush=True)
        if not args.execute:
            continue
        with (output / f"{Path(run['checkpoint']).stem}.log").open("w", encoding="utf-8") as log:
            run_with_progress(run["command"], log, title=title)
        report = Path(run["checkpoint"]).with_suffix("")
        report = report.parent / (report.name + "_training_report")
        if args.append:
            reference = Path(run["reference_checkpoint"])
            old_manifest = reference.with_name(reference.stem + "_training_report") / "fusion_split_manifest.json"
            new_manifest = report / "fusion_split_manifest.json"
            if old_manifest.exists():
                old_split = json.loads(old_manifest.read_text(encoding="utf-8"))
                new_split = json.loads(new_manifest.read_text(encoding="utf-8"))
                if any(sorted(old_split[key]) != sorted(new_split[key]) for key in ("train_items", "val_items")):
                    raise RuntimeError("Data split changed from the original experiment; new run was saved but not merged into comparison summary.")
        histories = json.loads((report / "history.json").read_text(encoding="utf-8"))
        history = histories["concentration"]["fusion"]
        situation_history = histories.get("situation", {}).get("fusion", [])
        best_situation = min(situation_history, key=lambda row: row["val_loss"]) if situation_history else {}
        best = min(history, key=lambda row: -row["val_" + args.selection_metric] if args.selection_metric in {"exact_accuracy", "qwk"} else row["val_" + args.selection_metric])
        summary.append({"variant": run["variant"], "seed": run["seed"], "fold": run["fold"],
                        "situation_epoch": best_situation.get("epoch", 0),
                        "situation_val_accuracy": best_situation.get("val_accuracy", ""),
                        "situation_profile": args.situation_profile,
                        **{key: best[key] for key in ["epoch", "val_mse", "val_mae", "val_qwk", "val_exact_accuracy", "val_constant_baseline"]}})
        write_summary(output / "summary.csv", retained_summary + summary)
        if args.append:
            write_summary(output / "combined_summary.csv", prior + summary)
        print(f"[{title}] COMPLETE; exact={best['val_exact_accuracy']:.4f}; situation_acc={best_situation.get('val_accuracy', 'disabled')}; summary={output / 'summary.csv'}", flush=True)
    if args.execute and args.cross_validation == "dataset":
        recommendation = summarize_cv(prior + summary if args.append else summary)
        (output / "cv_recommendation.json").write_text(json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8")
        winner = recommendation["selected"]
        # An older condition may win after appending; preserve its original command.
        original_plan = (json.loads((original_output / "experiment_plan.json").read_text(encoding="utf-8"))
                         if args.append else [])
        previous_plans = []
        if args.append:
            for path in original_output.glob("additions/*/experiment_plan.json"):
                previous_plans.extend(json.loads(path.read_text(encoding="utf-8")))
        command = next(run["command"].copy() for run in plan + original_plan + previous_plans if run["variant"] == winner["variant"])
        def set_argument(flag, value):
            if flag in command:
                # Some ablations deliberately override an earlier argument.
                indices = [i for i, arg in enumerate(command) if arg == flag]
                for i in indices:
                    command[i+1] = str(value)
            else:
                command.extend([flag, str(value)])
        set_argument("--epochs", winner["recommended_epochs"])
        set_argument("--situation_epochs", winner["recommended_situation_epochs"])
        set_argument("--save_name", output / "final_all_data.pt")
        command.append("--no_validation")
        (output / "final_training_command.json").write_text(json.dumps(command, indent=2, ensure_ascii=False), encoding="utf-8")
        print("CV recommendation:", json.dumps(winner, ensure_ascii=False), flush=True)
        print("Final all-data command:", subprocess.list2cmdline(command), flush=True)
        if args.final_train:
            with (output / "final_all_data.log").open("w", encoding="utf-8") as log:
                run_with_progress(command, log, title="FINAL all data")
    if not args.execute:
        print(f"Plan saved: {output / 'experiment_plan.json'}; add --execute to train.")


if __name__ == "__main__":
    main()
