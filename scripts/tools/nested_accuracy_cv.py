"""Nested lesson CV orchestration. No training/evaluation dependencies at import time."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def split_values(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def build_nested_plan(args):
    from run_accuracy_experiments import ROOT, experiments, grouped_dataset_folds

    if args.append or args.holdout_groups:
        raise ValueError("nested CV chooses its own splits; --append/--holdout_groups are unsupported")
    roots = split_values(args.data_roots)
    dbs = split_values(args.db_paths)
    groups = split_values(args.dataset_groups) if args.dataset_groups else [f"dataset:{i}" for i in range(len(roots))]
    if len(groups) != len(roots) or (dbs and len(dbs) != len(roots)):
        raise ValueError("roots, explicit database paths and lesson groups must align")
    if len(set(groups)) < 3:
        raise ValueError("nested CV requires at least three independent lesson groups")
    if len({str(Path(root).resolve()).casefold() for root in roots}) != len(roots):
        raise ValueError("duplicate data_roots would leak data across folds")
    if dbs and len({str(Path(db).resolve()).casefold() for db in dbs}) != len(dbs):
        raise ValueError("duplicate db_paths would leak data across folds")
    variants = split_values(args.variants)
    if not variants or set(variants) - experiments().keys() or len(set(variants)) != len(variants):
        raise ValueError("specify distinct, supported --variants")
    seeds = [int(seed) for seed in split_values(args.seeds)]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("specify distinct integer --seeds")
    if args.epochs < 1 or args.situation_epochs < 0:
        raise ValueError("epochs must be positive and situation_epochs nonnegative")
    if any(value is not None and value < 0 for value in (args.situation_patience, args.concentration_patience)):
        raise ValueError("patience must be nonnegative")
    outer = grouped_dataset_folds(groups, args.cv_folds)
    output = Path(args.output_dir).resolve()

    def selection_command(indices, directory, seed):
        selected_groups = [groups[i] for i in indices]
        # Validate before launching any expensive training.
        grouped_dataset_folds(selected_groups, args.inner_folds)
        command = [sys.executable, str(ROOT / "scripts/tools/run_accuracy_experiments.py"),
                   "--cross_validation", "dataset", "--cv_folds", str(args.inner_folds),
                   "--data_roots", ",".join(roots[i] for i in indices),
                   "--dataset_groups", ",".join(selected_groups),
                   "--label_source", args.label_source, "--raters", args.raters,
                   "--variants", ",".join(variants), "--seeds", str(seed),
                   "--selection_metric", args.selection_metric,
                   "--epochs", str(args.epochs), "--situation_epochs", str(args.situation_epochs),
                   "--num_workers", str(args.num_workers), "--situation_profile", args.situation_profile,
                   "--output_dir", str(directory), "--execute", "--final_train"]
        if dbs:
            command += ["--db_paths", ",".join(dbs[i] for i in indices)]
        for flag, value in (("--batch_size", args.batch_size),
                            ("--situation_patience", args.situation_patience),
                            ("--concentration_patience", args.concentration_patience)):
            if value is not None:
                command += [flag, str(value)]
        return command

    runs = []
    for seed in seeds:
        for fold, test_indices in enumerate(outer):
            train_indices = [i for i in range(len(roots)) if i not in test_indices]
            directory = output / f"seed{seed}" / f"outer{fold}"
            inner = directory / "inner"
            evaluations = []
            for group in dict.fromkeys(groups[i] for i in test_indices):
                indices = [i for i in test_indices if groups[i] == group]
                evaluation_dir = directory / f"test_group{len(evaluations)}"
                command = [sys.executable, str(ROOT / "scripts/tools/evaluate_labeled_dataset.py"),
                           "--model", str(inner / "final_all_data.pt"),
                           "--data_roots", ",".join(roots[i] for i in indices),
                           "--label_source", args.label_source, "--raters", args.raters,
                           "--situation-source", "predicted", "--num_workers", str(args.num_workers),
                           "--batch_size", str(args.batch_size or 16), "--output-dir", str(evaluation_dir)]
                if dbs:
                    command += ["--db_paths", ",".join(dbs[i] for i in indices)]
                evaluations.append({"group": group, "indices": indices,
                                    "output_dir": str(evaluation_dir), "command": command})
            runs.append({"seed": seed, "outer_fold": fold, "train_indices": train_indices,
                         "test_indices": test_indices, "inner_command": selection_command(train_indices, inner, seed),
                         "inner_dir": str(inner), "evaluations": evaluations})
    # Deployment selection uses a fresh full-data development CV, never outer scores.
    deployment = []
    if args.final_train:
        for seed in seeds:
            directory = output / "deployment" / f"seed{seed}"
            deployment.append({"seed": seed, "command": selection_command(list(range(len(roots))), directory, seed)})
    return {"protocol": "nested_lesson_cv_v1", "data_roots": roots, "db_paths": dbs,
            "dataset_groups": groups, "variants": variants, "seeds": seeds,
            "epoch_selection": args.selection_metric,
            "variant_selection": "unweighted mean across inner folds: exact accuracy, then MAE",
            "refit_epochs": "ceiling of median inner best epochs; no validation, fixed learning rate",
            "runs": runs, "deployment": deployment}


def summarize_tests(rows):
    """Equal lesson weighting within each seed; do not treat repeated seeds as new lessons."""
    metrics = ("exact_accuracy", "within_1_accuracy", "within_2_accuracy", "mae", "rmse",
               "quadratic_weighted_kappa", "pearson", "spearman", "bias_pred_minus_true")
    by_seed = []
    for seed in sorted({row["seed"] for row in rows}):
        selected = [row for row in rows if row["seed"] == seed]
        stats = {}
        for key in metrics:
            values = [float(row[key]) for row in selected if key in row and math.isfinite(float(row[key]))]
            stats[key] = {"mean": statistics.mean(values) if values else None,
                          "std_across_lessons": statistics.stdev(values) if len(values) > 1 else None,
                          "valid_lessons": len(values)}
        by_seed.append({"seed": seed, "lessons": len(selected), "metrics": stats})
    return {"aggregation": "equal weight per held-out lesson; seeds reported separately",
            "by_seed": by_seed, "test_runs": len(rows)}


def selection_finished(command):
    """Recognize completed refits, including runs made before resume was added."""
    def arg(cmd, flag, default=None):
        return cmd[cmd.index(flag) + 1] if flag in cmd else default
    directory = Path(arg(command, "--output_dir"))
    try:
        saved = json.loads((directory / "final_training_command.json").read_text(encoding="utf-8"))
        winner = json.loads((directory / "cv_recommendation.json").read_text(encoding="utf-8"))["selected"]
        history = json.loads((directory / "final_all_data_training_report/history.json").read_text(encoding="utf-8"))
        checkpoint = directory / "final_all_data.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or "--no_validation" not in saved:
            return False
        for flag in ("--data_roots", "--db_paths", "--label_source", "--raters", "--seed"):
            expected = arg(command, "--seeds") if flag == "--seed" else arg(command, flag)
            if arg(saved, flag) != expected:
                return False
        if winner["variant"] not in split_values(arg(command, "--variants")):
            return False
        for stage, flag, key in (("concentration", "--epochs", "recommended_epochs"),
                                 ("situation", "--situation_epochs", "recommended_situation_epochs")):
            epochs = int(arg(saved, flag))
            records = history.get(stage, {}).get("fusion", [])
            if epochs != int(winner[key]) or len(records) != epochs:
                return False
            if records and (records[-1]["epoch"] != epochs or not records[-1].get("no_validation")):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_nested(args):
    from run_accuracy_experiments import run_with_progress, write_summary

    plan = build_nested_plan(args)
    output = Path(args.output_dir).resolve()
    resume = getattr(args, "resume", False)
    # A nested run is an auditable experiment; avoid mixing stale test results.
    if output.exists() and any(output.iterdir()):
        existing = list(output.iterdir())
        saved = output / "nested_plan.json"
        if (not saved.is_file() or json.loads(saved.read_text(encoding="utf-8")) != plan
                or (existing != [saved] and not resume)):
            raise ValueError("nested output_dir contains results or a different plan; use a new directory")
    elif resume:
        raise ValueError("--resume requires an existing matching nested_plan.json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "nested_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    inner_trainings = sum(len(set(plan["dataset_groups"][i] for i in run["train_indices"]))
                          if not args.inner_folds else args.inner_folds for run in plan["runs"]) * len(plan["variants"])
    print(f"Nested CV: {len(plan['runs'])} outer refits, {inner_trainings} inner training runs; "
          f"{len(plan['deployment'])} optional deployment CV/refits. Plan: {output / 'nested_plan.json'}", flush=True)
    if not args.execute:
        print("Plan only; no training or test evaluation started. Add --execute to run this unchanged plan.", flush=True)
        return
    if (output / "completed.json").exists():
        (output / "completed.json").unlink()
    rows = []
    for run in plan["runs"]:
        title = f"seed={run['seed']} outer={run['outer_fold']}"
        print(f"[{title}] inner selection and outer-training refit", flush=True)
        reused_refit = resume and selection_finished(run["inner_command"])
        if reused_refit:
            print(f"[{title}] REUSE completed inner CV and refit", flush=True)
        else:
            with (output / f"seed{run['seed']}_outer{run['outer_fold']}_inner.log").open("a" if resume else "w", encoding="utf-8") as log:
                run_with_progress(run["inner_command"], log, title=title)
        recommendation = json.loads((Path(run["inner_dir"]) / "cv_recommendation.json").read_text(encoding="utf-8"))["selected"]
        for index, evaluation in enumerate(run["evaluations"]):
            marker = Path(evaluation["output_dir"]) / "nested_evaluation_complete.json"
            reuse = (reused_refit and marker.is_file()
                     and json.loads(marker.read_text(encoding="utf-8")) == evaluation["command"]
                     and (marker.parent / "metrics.json").is_file())
            if not reuse:
                if marker.exists():
                    marker.unlink()
                with (output / f"seed{run['seed']}_outer{run['outer_fold']}_test{index}.log").open("a" if resume else "w", encoding="utf-8") as log:
                    run_with_progress(evaluation["command"], log, title=f"{title} TEST {evaluation['group']}")
                marker.write_text(json.dumps(evaluation["command"]), encoding="utf-8")
            metrics = json.loads((Path(evaluation["output_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            rows.append({"seed": run["seed"], "outer_fold": run["outer_fold"], "lesson_group": evaluation["group"],
                         "selected_variant": recommendation["variant"],
                         "epochs": recommendation["recommended_epochs"],
                         "situation_epochs": recommendation["recommended_situation_epochs"], **metrics})
            write_summary(output / "outer_test_summary.csv", rows)
            (output / "outer_test_aggregate.json").write_text(json.dumps(summarize_tests(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    for run in plan["deployment"]:
        if resume and selection_finished(run["command"]):
            print(f"[DEPLOYMENT seed={run['seed']}] REUSE completed CV/refit", flush=True)
            continue
        with (output / f"deployment_seed{run['seed']}.log").open("w", encoding="utf-8") as log:
            run_with_progress(run["command"], log, title=f"DEPLOYMENT seed={run['seed']}")
    (output / "completed.json").write_text(json.dumps({"outer_test_runs": len(rows), "deployment_runs": len(plan["deployment"])}), encoding="utf-8")
    print(f"Nested CV complete: {output / 'outer_test_summary.csv'}", flush=True)
