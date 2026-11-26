"""Protocol tests run without importing torch or launching GPU training."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tools"))
import run_accuracy_experiments as runner
import nested_accuracy_cv as nested


def options(directory, **overrides):
    args = dict(append=False, holdout_groups=None, data_roots="a1,a2,b,c,d",
                db_paths="a1.sqlite,a2.sqlite,b.sqlite,c.sqlite,d.sqlite",
                dataset_groups="A,A,B,C,D", variants="standard,mild", seeds="42",
                epochs=50, situation_epochs=20, situation_patience=3,
                concentration_patience=5, cv_folds=0, inner_folds=0,
                output_dir=str(directory), label_source="consensus", raters="tadano",
                selection_metric="mse", num_workers=0, situation_profile="legacy",
                batch_size=16, final_train=False, execute=False)
    args.update(overrides)
    return argparse.Namespace(**args)


def argument(command, name):
    return command[command.index(name) + 1]


class NestedCVTests(unittest.TestCase):
    def test_outer_test_is_absent_from_inner_and_refit_inputs(self):
        plan = nested.build_nested_plan(options("unused", final_train=True))
        self.assertEqual(len(plan["runs"]), 4)
        for run in plan["runs"]:
            train, test = set(run["train_indices"]), set(run["test_indices"])
            self.assertFalse(train & test)
            self.assertEqual(train | test, set(range(5)))
            command = run["inner_command"]
            self.assertEqual(argument(command, "--data_roots").split(","),
                             [plan["data_roots"][i] for i in sorted(train)])
            self.assertEqual(argument(command, "--db_paths").split(","),
                             [plan["db_paths"][i] for i in sorted(train)])
            self.assertIn("--final_train", command)
            if 0 in test:
                self.assertIn(1, test)
            for evaluation in run["evaluations"]:
                self.assertTrue(set(evaluation["indices"]) <= test)
                self.assertEqual(argument(evaluation["command"], "--situation-source"), "predicted")
        self.assertEqual(len(plan["deployment"]), 1)
        self.assertEqual(argument(plan["deployment"][0]["command"], "--data_roots"), "a1,a2,b,c,d")

    def test_grouped_inner_plan_holds_all_parts_together(self):
        self.assertEqual(runner.grouped_dataset_folds(["A", "A", "B", "C"]), [[0, 1], [2], [3]])
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["runner", "--data_roots", "a1,a2,b,c", "--dataset_groups", "A,A,B,C",
                    "--cross_validation", "dataset", "--variants", "standard", "--output_dir", tmp]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            plan = json.loads((Path(tmp) / "experiment_plan.json").read_text())
            self.assertEqual([argument(run["command"], "--holdout_groups") for run in plan],
                             ["dataset:0,dataset:1", "dataset:2", "dataset:3"])

    def test_invalid_protocols_fail_before_training(self):
        for changes in ({"dataset_groups": "A,A,A,A,B"}, {"inner_folds": 4},
                        {"cv_folds": 2, "dataset_groups": "A,A,B,C,C"},
                        {"data_roots": "a1,a1,b,c,d"}, {"db_paths": "a,a,b,c,d"},
                        {"variants": "unknown"}, {"seeds": "42,42"}, {"append": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                nested.build_nested_plan(options("unused", **changes))

    def test_mock_execution_separates_test_results_and_never_selects_from_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            args = options(output, execute=True, final_train=True, cv_folds=2, inner_folds=2)
            calls = []

            def fake_run(command, log, *, title):
                calls.append(command)
                if "--model" not in command:
                    directory = Path(argument(command, "--output_dir"))
                    directory.mkdir(parents=True)
                    (directory / "cv_recommendation.json").write_text(json.dumps({"selected": {
                        "variant": "mild", "recommended_epochs": 4, "recommended_situation_epochs": 3}}))
                else:
                    directory = Path(argument(command, "--output-dir"))
                    directory.mkdir(parents=True)
                    (directory / "metrics.json").write_text(json.dumps({"n": 20, "mae": 1.2, "exact_accuracy": .3}))

            with patch.object(runner, "run_with_progress", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                nested.run_nested(args)
            self.assertEqual(len(calls), 7)  # 2 inner/refits, 4 test lessons, 1 deployment CV/refit
            self.assertTrue((output / "completed.json").exists())
            aggregate = json.loads((output / "outer_test_aggregate.json").read_text())
            self.assertEqual(aggregate["by_seed"][0]["lessons"], 4)
            self.assertAlmostEqual(aggregate["by_seed"][0]["metrics"]["mae"]["mean"], 1.2)
            self.assertNotIn("outer_test", " ".join(calls[-1]))
            with self.assertRaises(ValueError):
                nested.run_nested(args)

    def test_dry_plan_can_be_executed_without_changing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = options(Path(tmp) / "plan")
            with patch.object(runner, "run_with_progress", side_effect=AssertionError("training started")), contextlib.redirect_stdout(io.StringIO()):
                nested.run_nested(args)
                nested.run_nested(args)
            self.assertFalse((Path(args.output_dir) / "outer_test_summary.csv").exists())

    def test_nonfinite_metrics_are_not_averaged_and_seeds_stay_separate(self):
        result = nested.summarize_tests([{"seed": 42, "mae": 1, "pearson": float("nan")},
                                         {"seed": 42, "mae": 3, "pearson": .5},
                                         {"seed": 43, "mae": 5}])
        self.assertEqual(len(result["by_seed"]), 2)
        self.assertEqual(result["by_seed"][0]["metrics"]["mae"]["mean"], 2)
        self.assertEqual(result["by_seed"][0]["metrics"]["pearson"]["valid_lessons"], 1)

    def test_resume_reuses_completed_inner_stage_and_retries_failed_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = options(Path(tmp) / "resume", execute=True)
            with contextlib.redirect_stdout(io.StringIO()):
                args.execute = False
                nested.run_nested(args)
            args.execute = True
            args.resume = True
            calls = []

            def evaluate_only(command, log, *, title):
                self.assertIn("--model", command)
                calls.append(command)
                directory = Path(argument(command, "--output-dir"))
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "metrics.json").write_text(json.dumps({"n": 1, "mae": 1}))

            for run in nested.build_nested_plan(args)["runs"]:
                directory = Path(run["inner_dir"])
                directory.mkdir(parents=True)
                (directory / "cv_recommendation.json").write_text(json.dumps({"selected": {
                    "variant": "mild", "recommended_epochs": 4, "recommended_situation_epochs": 3}}))
            with patch.object(nested, "selection_finished", return_value=True), patch.object(runner, "run_with_progress", side_effect=evaluate_only), contextlib.redirect_stdout(io.StringIO()):
                nested.run_nested(args)
                self.assertEqual(len(calls), 4)
                nested.run_nested(args)
                self.assertEqual(len(calls), 4)  # completed evaluations are not repeated
            args.epochs = 51
            with self.assertRaises(ValueError):
                nested.run_nested(args)


if __name__ == "__main__":
    unittest.main()
