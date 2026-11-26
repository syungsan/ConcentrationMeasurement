from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lib.model_defs import Cfg, Regressor, SituationClassifier, motion_pose_features
from lib.training_options import balanced_class_weights, concentration_loss, build_optimizer, freeze_batchnorm_stats
from lib.infer import load_ckpt, load_situation_model


class AccuracyImprovementTests(unittest.TestCase):
    def test_reduced_dataset_cv_keeps_whole_datasets(self):
        spec = importlib.util.spec_from_file_location("quick_cv_runner", ROOT / "scripts/tools/run_accuracy_experiments.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        self.assertEqual(runner.dataset_folds(5, 2), [[0, 2, 4], [1, 3]])
        self.assertEqual(runner.dataset_folds(3), [[0], [1], [2]])
        with self.assertRaises(ValueError):
            runner.dataset_folds(3, 4)
        import json
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            argv = ["runner", "--data_roots", "a,b,c,d", "--cross_validation", "dataset",
                    "--cv_folds", "2", "--variants", "no_situation,mild", "--seeds", "42",
                    "--situation_patience", "3", "--output_dir", tmp]
            with patch.object(sys, "argv", argv):
                runner.main()
            plan = json.loads((Path(tmp) / "experiment_plan.json").read_text())
            self.assertEqual(len(plan), 4)
            for variant in ("no_situation", "mild"):
                selected = [run for run in plan if run["variant"] == variant]
                holdouts = [run["command"][run["command"].index("--holdout_groups")+1] for run in selected]
                self.assertEqual(holdouts, ["dataset:0,dataset:2", "dataset:1,dataset:3"])
                self.assertIn("--situation_patience", selected[0]["command"])

    def test_no_situation_classifier_does_not_change_training(self):
        spec = importlib.util.spec_from_file_location("independent_fixture", ROOT / "tests/test_training_audit.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        train = fixture.train
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            ds = fixture.dataset(Path(tmp), augment="none")
            cfg = replace(ds.cfg, mode="skeleton", use_situation_feature=False, hidden=16,
                          layers=1, epochs=2, num_workers=0, device="cpu")
            situation_state = SituationClassifier(cfg).state_dict()
            with patch.object(train, "fetch_items_window_split", return_value=([(0, 1)], [(0, 1)])):
                _, without, _ = train.train_one(cfg)
                with patch.object(train, "SituationClassifier", side_effect=AssertionError("independent classifier must not enter concentration training")):
                    _, with_classifier, _ = train.train_one(cfg, situation_state)
            for key in without:
                torch.testing.assert_close(without[key], with_classifier[key], rtol=0, atol=0)
            model = Regressor(cfg).eval()
            model.load_state_dict(with_classifier)
            poses = torch.zeros(2, 10, 17, 3)
            torch.testing.assert_close(model(None, poses, torch.tensor([[1.,0,0],[0,1.,0]])),
                                       model(None, poses, torch.tensor([[0.,0,1.],[1.,0,0]])))

    def test_evaluation_records_independent_situation(self):
        spec = importlib.util.spec_from_file_location("independent_eval", ROOT / "scripts/tools/evaluate_labeled_dataset.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from types import SimpleNamespace
        class Score(torch.nn.Module):
            def forward(self, frames, poses, situations):
                return torch.full((situations.shape[0],), 4.)
        class Situation(torch.nn.Module):
            def forward(self, frames, poses):
                return torch.tensor([[0., 5., 0.]])
        loader = [([0], None, None, torch.tensor([[1.,0,0]]), torch.tensor([4.]))]
        cfg = SimpleNamespace(device="cpu", db_paths=[Path("test.sqlite")], data_roots=[Path("test")])
        with patch.object(module, "load_segment_metadata", return_value={"situation": "聞く"}):
            rows = module.predict_rows(SimpleNamespace(SITUATIONS=["聞く", "書く", "話し合う"]),
                                       Score(), loader, [(0, 1)], cfg, Situation())
        self.assertEqual(rows[0]["predicted_situation"], "書く")
        self.assertGreater(rows[0]["situation_confidence"], .98)
        self.assertEqual(rows[0]["pred_score"], 4.)

    def test_standard_append_preserves_original_results(self):
        spec = importlib.util.spec_from_file_location("append_runner", ROOT / "scripts/tools/run_accuracy_experiments.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        import json
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            root = Path(tmp)
            reference = {"variant": "baseline", "seed": 43, "fold": None,
                         "checkpoint": str(root / "baseline_seed43.pt"),
                         "command": [sys.executable, "train.py", "--data_roots", "a,b", "--seed", "43",
                                     "--split_unit", "window", "--epochs", "17", "--selection_metric", "exact_accuracy",
                                     "--situation_profile", "legacy", "--save_name", str(root / "baseline_seed43.pt")]}
            (root / "experiment_plan.json").write_text(json.dumps([reference]))
            prior = {"variant": "baseline", "seed": 43, "fold": "", "epoch": 2,
                     "val_mse": 1., "val_mae": .5, "val_qwk": .4, "val_exact_accuracy": .6, "val_constant_baseline": .3}
            runner.write_summary(root / "summary.csv", [prior])
            (root / "baseline_seed43.pt").write_bytes(b"old model")
            original = {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
            commands = []
            def fake_train(command, log, **kwargs):
                commands.append(command)
                checkpoint = Path(command[command.index("--save_name") + 1])
                checkpoint.touch()
                report = checkpoint.with_name(checkpoint.stem + "_training_report")
                report.mkdir()
                (report / "history.json").write_text(json.dumps({"concentration": {"fusion": [prior]}}))
            argv = ["runner", "--append", "--variants", "standard", "--output_dir", str(root), "--execute"]
            with patch.object(sys, "argv", argv), patch.object(runner, "run_with_progress", side_effect=fake_train):
                runner.main()
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][commands[0].index("--epochs") + 1], "17")
            self.assertEqual(commands[0][-2:], ["--augmentation", "standard"])
            for path, data in original.items():
                self.assertEqual(path.read_bytes(), data)
            combined = list(root.glob("additions/*/combined_summary.csv"))
            self.assertEqual(len(combined), 1)
            with combined[0].open(encoding="utf-8-sig", newline="") as stream:
                rows = list(runner.csv.DictReader(stream))
            self.assertEqual([row["variant"] for row in rows], ["baseline", "standard"])
            with patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    runner.main()

    def cfg(self, **kwargs):
        return Cfg(data_roots=[], db_paths=[], raters=[], mode="skeleton", hidden=16,
                   pose_feat=8, layers=1, dropout=0, imagenet_pretrain=False, **kwargs)

    def test_motion_geometry_invariance_and_visibility(self):
        torch.manual_seed(4)
        pose = torch.randn(2, 5, 17, 3)
        pose[..., 2] = 1
        features = motion_pose_features(pose)
        moved = pose.clone()
        # Per-frame translation/positive scale should not invent relative motion.
        moved[..., :2] = moved[..., :2] * torch.rand(2, 5, 1, 1).add(1) + torch.randn(2, 5, 1, 2)
        torch.testing.assert_close(features, motion_pose_features(moved), atol=2e-5, rtol=2e-5)
        self.assertEqual(features.shape, (2, 5, 103))
        self.assertTrue(torch.equal(features[:, 0, 34:68], torch.zeros(2, 34)))
        pose[..., 2] = 0
        self.assertTrue(torch.equal(motion_pose_features(pose), torch.zeros_like(features)))
        self.assertTrue(torch.isfinite(motion_pose_features(torch.full_like(pose, float("nan")))).all())

    def test_ordered_thresholds_forward_and_gradients(self):
        for temporal in ("gru", "transformer"):
            cfg = self.cfg(temporal=temporal, temporal_pool="mean", pose_features="motion", ordered_ordinal=True)
            model = Regressor(cfg).eval()
            pose = torch.randn(3, 5, 17, 3)
            pose[..., 2] = 1
            situations = torch.eye(3)
            logits = model.forward_ordinal_logits(None, pose, situations)
            self.assertTrue((logits[:, :-1] >= logits[:, 1:]).all())
            score = model(None, pose, situations)
            torch.testing.assert_close(score, 1 + logits.sigmoid().sum(-1))
            labels = torch.tensor([1., 4.5, 7.])
            cfg = replace(cfg, score_loss_weight=0.2)
            loss = concentration_loss(logits, score, labels, cfg)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertTrue(((score >= 1) & (score <= 7)).all())

    def test_training_only_weights(self):
        weights, counts = balanced_class_weights([1.] * 16 + [7.])
        self.assertGreater(weights[6], weights[0])
        self.assertEqual(counts.tolist(), [16, 0, 0, 0, 0, 0, 1])
        self.assertTrue(torch.isfinite(weights).all())
        self.assertLessEqual(weights.max(), 3)

    def test_independent_situation_settings_and_checkpoint(self):
        cfg = self.cfg(pose_features="motion", temporal_pool="mean",
                       situation_config={"pose_features": "raw", "temporal_pool": "last", "backbone_lr_scale": 1.0, "freeze_batchnorm": False})
        concentration = Regressor(cfg)
        situation = SituationClassifier(cfg)
        self.assertEqual(concentration.pose_enc.features, "motion")
        self.assertEqual(situation.pose_enc.features, "raw")
        self.assertEqual(situation.temporal.pool, "last")
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            path = Path(tmp) / "independent.pt"
            torch.save(dict(cfg=cfg.__dict__, mode="skeleton", state_dict=concentration.state_dict(),
                            situation_state_dicts={"skeleton": situation.state_dict()},
                            feature_schema="image_pose_situation_v5"), path)
            ckpt, restored, _ = load_ckpt(path, "skeleton", "cpu")
            loaded = load_situation_model(ckpt, restored, "skeleton", "cpu")
            self.assertEqual(loaded.pose_enc.features, "raw")

    def test_streaming_child_logs_and_failure(self):
        spec = importlib.util.spec_from_file_location("progress_runner", ROOT / "scripts/tools/run_accuracy_experiments.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        import io
        from contextlib import redirect_stdout
        log, terminal = io.StringIO(), io.StringIO()
        with redirect_stdout(terminal):
            runner.run_with_progress([sys.executable, "-u", "-c", "print('epoch 1/2'); print('epoch 2/2')"], log, title="test")
        self.assertIn("epoch 1/2", terminal.getvalue())
        self.assertEqual(terminal.getvalue(), log.getvalue())
        with self.assertRaises(runner.subprocess.CalledProcessError):
            runner.run_with_progress([sys.executable, "-c", "raise SystemExit(3)"], io.StringIO(), title="failure")

    def test_optimizer_backbone_and_bn(self):
        cfg = replace(self.cfg(), mode="image", backbone_lr_scale=0.1)
        model = Regressor(cfg).train()
        opt = build_optimizer(model, cfg)
        self.assertEqual([group["lr"] for group in opt.param_groups], [cfg.lr, cfg.lr * 0.1])
        freeze_batchnorm_stats(model, True)
        bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
        self.assertFalse(bn.training)
        self.assertTrue(bn.weight.requires_grad)

    def test_new_and_legacy_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            for improved in (False, True):
                cfg = self.cfg(pose_features="motion" if improved else "raw", temporal_pool="mean" if improved else "last", ordered_ordinal=improved)
                model = Regressor(cfg).eval()
                situation = SituationClassifier(cfg).eval()
                fields = dict(cfg.__dict__)
                if not improved:
                    for key in ("pose_features", "temporal_pool", "ordered_ordinal"):
                        fields.pop(key)
                path = Path(tmp) / "model.pt"
                torch.save(dict(cfg=fields, state_dict=model.state_dict(),
                                situation_state_dicts={"skeleton": situation.state_dict()},
                                feature_schema="image_pose_situation_v5" if improved else "image_pose_situation_v4"), path)
                ckpt, loaded_cfg, loaded = load_ckpt(path, "skeleton", "cpu")
                loaded_situation = load_situation_model(ckpt, loaded_cfg, "skeleton", "cpu")
                pose = torch.randn(2, 4, 17, 3)
                pose[..., 2] = 1
                probs = torch.ones(2, 3) / 3
                torch.testing.assert_close(model(None, pose, probs), loaded(None, pose, probs))
                torch.testing.assert_close(situation(None, pose), loaded_situation(None, pose))

    def test_training_and_validation_integration(self):
        spec = importlib.util.spec_from_file_location("audit_fixture", ROOT / "tests/test_training_audit.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        train = fixture.train
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            ds = fixture.dataset(Path(tmp), augment="none")
            cfg = replace(ds.cfg, mode="skeleton", pose_features="motion", temporal_pool="mean",
                          ordered_ordinal=True, class_balance=True, score_loss_weight=0.1,
                          epochs=2, num_workers=0, batch_size=1, hidden=16, layers=1,
                          device="cpu", lr_schedule="plateau", early_stopping_patience=1,
                          selection_metric="exact_accuracy")
            with patch.object(train, "fetch_items_window_split", return_value=([(0, 1)], [(0, 1)])):
                result, state, history = train.train_one(cfg)
                _, situation_state, _ = train.train_situation_one(cfg, 1)
            self.assertTrue(torch.isfinite(torch.tensor(result)))
            self.assertTrue(state and situation_state)
            self.assertIn("val_exact_accuracy", history[0])
            self.assertEqual(history[0]["val_constant_baseline"], 1.0)

    def test_all_data_training_without_validation(self):
        spec = importlib.util.spec_from_file_location("full_data_fixture", ROOT / "tests/test_training_audit.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        train = fixture.train
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            ds = fixture.dataset(Path(tmp), augment="none")
            import sqlite3
            with sqlite3.connect(ds.cfg.db_paths[0]) as connection:
                connection.executescript("""
                    ALTER TABLE windows ADD COLUMN video_id INTEGER DEFAULT 1;
                    ALTER TABLE windows ADD COLUMN t_start REAL DEFAULT 0;
                    ALTER TABLE windows ADD COLUMN t_end REAL DEFAULT 5;
                    ALTER TABLE segments ADD COLUMN track_id INTEGER DEFAULT 1;
                """)
            connection.close()
            cfg = replace(ds.cfg, no_validation=True, mode="skeleton", device="cpu",
                          split_unit="school", holdout_groups=["school:any"],
                          num_workers=0, epochs=2, hidden=16, layers=1,
                          lr_schedule="plateau", early_stopping_patience=1)
            tr, va = train.fetch_items_window_split(cfg)
            self.assertEqual(tr, [(0, 1)])
            self.assertEqual(va, [])
            with patch.object(train, "evaluate", side_effect=AssertionError("validation must not run")), \
                 patch.object(train, "evaluate_situation", side_effect=AssertionError("validation must not run")):
                score, state, history = train.train_one(cfg)
                metrics, situation_state, situation_history = train.train_situation_one(cfg, 2)
            self.assertIsNone(score)
            self.assertEqual(len(history), 2)
            self.assertEqual(len(situation_history), 2)
            self.assertFalse(metrics["validation_enabled"])
            self.assertNotIn("val_mse", history[-1])
            self.assertTrue(state and situation_state)

    def test_variable_dataset_cv_and_epoch_recommendation(self):
        spec = importlib.util.spec_from_file_location("cv_runner", ROOT / "scripts/tools/run_accuracy_experiments.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        import json
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            argv = ["runner", "--data_roots", "a,b,c", "--variants", "baseline", "--cross_validation", "dataset", "--output_dir", tmp]
            with patch.object(sys, "argv", argv):
                runner.main()
            plan = json.loads((Path(tmp) / "experiment_plan.json").read_text())
            self.assertEqual(len(plan), 3)
            self.assertEqual([row["fold"] for row in plan], [0, 1, 2])
            for i, row in enumerate(plan):
                command = row["command"]
                self.assertEqual(command[command.index("--holdout_groups")+1], f"dataset:{i}")
            rows = [{"variant": "baseline", "epoch": ep, "situation_epoch": sit,
                     "val_exact_accuracy": .5, "val_mae": .8, "val_qwk": .4}
                    for ep, sit in [(3, 2), (5, 3), (20, 4)]]
            recommendation = runner.summarize_cv(rows)["selected"]
            self.assertEqual(recommendation["recommended_epochs"], 5)
            self.assertEqual(recommendation["recommended_situation_epochs"], 3)
            executed = []
            def fake_cv_train(command, log, **kwargs):
                executed.append(command)
                checkpoint = Path(command[command.index("--save_name") + 1])
                checkpoint.touch()
                if "--no_validation" in command:
                    return
                fold = int(command[command.index("--holdout_groups") + 1].split(":")[1])
                report = checkpoint.with_name(checkpoint.stem + "_training_report")
                report.mkdir()
                epoch = [3, 5, 20][fold]
                row = {"epoch": epoch, "val_mse": 1., "val_mae": .8, "val_qwk": .4,
                       "val_exact_accuracy": .5, "val_constant_baseline": .3}
                situation = {"epoch": [2, 3, 4][fold], "val_loss": 1., "val_accuracy": .7}
                (report / "history.json").write_text(json.dumps({"concentration": {"fusion": [row]}, "situation": {"fusion": [situation]}}))
            merged_options = ["--db_paths", "a.sqlite,b.sqlite,c.sqlite", "--label_source", "consensus"]
            with patch.object(sys, "argv", argv + merged_options + ["--execute", "--final_train"]), \
                 patch.object(runner, "run_with_progress", side_effect=fake_cv_train):
                runner.main()
            self.assertEqual(len(executed), 4)
            for command in executed:
                self.assertEqual(command[command.index("--db_paths") + 1], "a.sqlite,b.sqlite,c.sqlite")
                self.assertEqual(command[command.index("--label_source") + 1], "consensus")
            final = executed[-1]
            self.assertIn("--no_validation", final)
            self.assertEqual(final[final.index("--epochs")+1], "5")
            self.assertEqual(final[final.index("--situation_epochs")+1], "3")

    def test_mild_augmentation_preserves_geometry(self):
        spec = importlib.util.spec_from_file_location("mild_fixture", ROOT / "tests/test_training_audit.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            ds = fixture.dataset(Path(tmp), augment="mild", grayscale=True)
            try:
                frames, poses, _, _ = ds[0]
                self.assertEqual(tuple(frames.shape), (10, 3, 32, 32))
                torch.testing.assert_close(frames, frames[:1].expand_as(frames))
                self.assertFalse(ds.color_jitter.saturation)
                self.assertTrue(torch.equal(poses, torch.zeros_like(poses)))
            finally:
                ds.close()

    def test_experiment_runner_summarizes_saved_history(self):
        spec = importlib.util.spec_from_file_location("experiment_runner", ROOT / "scripts/tools/run_accuracy_experiments.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        import json
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            def fake_train(command, log, **kwargs):
                checkpoint = Path(command[command.index("--save_name") + 1])
                checkpoint.touch()
                report = checkpoint.with_name(checkpoint.stem + "_training_report")
                report.mkdir(exist_ok=True)
                row = {"epoch": 3, "val_mse": 1.0, "val_mae": .5, "val_qwk": .3,
                       "val_exact_accuracy": .4, "val_constant_baseline": .39}
                (report / "history.json").write_text(json.dumps({"concentration": {"fusion": [row]}}))
            argv = ["runner", "--data_roots", "a,b", "--variants", "baseline,motion",
                    "--output_dir", tmp, "--execute"]
            with patch.object(sys, "argv", argv), patch.object(runner, "run_with_progress", side_effect=fake_train):
                runner.main()
                summary = (Path(tmp) / "summary.csv").read_text(encoding="utf-8-sig")
                self.assertIn("baseline,42", summary)
                self.assertIn("motion,42", summary)
                # Repeating a normal run overwrites selected results without
                # duplicating their summary rows or deleting other variants.
                extra = {"variant": "untouched", "seed": 42, "fold": "", "epoch": 1,
                         "val_mse": 2., "val_mae": 1., "val_qwk": .1,
                         "val_exact_accuracy": .2, "val_constant_baseline": .39}
                existing = runner.read_prior_results(Path(tmp))
                runner.write_summary(Path(tmp) / "summary.csv", existing + [extra])
                other = Path(tmp) / "untouched.pt"
                other.write_bytes(b"preserve")
                runner.main()
                rows = runner.read_prior_results(Path(tmp))
                self.assertEqual(len(rows), 3)
                self.assertEqual(other.read_bytes(), b"preserve")
                self.assertEqual(sum(row["variant"] == "baseline" for row in rows), 1)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
