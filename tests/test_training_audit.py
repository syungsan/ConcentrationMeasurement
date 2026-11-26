from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("train_audit", ROOT / "scripts/07_train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
from lib.infer import _build_cfg_from_ckpt_dict, build_image_tf
from lib.image_roi import image_model_crop
from lib.model_defs import Cfg, TemporalTransformer


def dataset(tmp_path, *, augment="standard", is_train=True, grayscale=False):
    db = tmp_path / "dataset.sqlite"
    pixels = np.arange(96 * 80 * 3, dtype=np.uint8).reshape(96, 80, 3)
    Image.fromarray(pixels).save(tmp_path / "frame.png")
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE labels(segment_id INTEGER, rater TEXT, score REAL);
            CREATE TABLE segments(id INTEGER, window_id INTEGER);
            CREATE TABLE windows(id INTEGER, situation TEXT);
            CREATE TABLE segment_frames(segment_id INTEGER, t REAL, crop_path TEXT, pose_path TEXT);
            INSERT INTO labels VALUES(1, 'rater', 4);
            INSERT INTO segments VALUES(1, 1);
            INSERT INTO windows VALUES(1, '聞く');
        """)
        conn.executemany("INSERT INTO segment_frames VALUES(1, ?, 'frame.png', '')",
                         [(0.25 + 0.5 * i,) for i in range(10)])
    conn.close()
    cfg = Cfg(data_roots=[tmp_path], db_paths=[db], raters=["rater"],
              mode="fusion", T=10, img_size=32, augmentation=augment, grayscale=grayscale)
    return train.MultiSQLiteSegmentDataset(cfg, [(0, 1)], is_train)


def test_augmentation_does_not_invent_motion(tmp_path):
    ds = dataset(tmp_path)
    try:
        for seed in range(4):
            train.set_seed(seed)
            frames, poses, _, _ = ds[0]
            assert frames.shape == (10, 3, 32, 32)
            assert poses.shape == (10, 17, 3)
            for frame in frames[1:]:
                torch.testing.assert_close(frame, frames[0])
    finally:
        for conn in ds._conns.values():
            conn.close()


def test_no_augmentation_matches_evaluation(tmp_path):
    ds = dataset(tmp_path, augment="none")
    evaluation = train.MultiSQLiteSegmentDataset(ds.cfg, ds.items, False)
    try:
        torch.testing.assert_close(ds[0][0], evaluation[0][0])
        torch.testing.assert_close(ds[0][1], evaluation[0][1])
    finally:
        for instance in (ds, evaluation):
            for conn in instance._conns.values():
                conn.close()


def test_missing_evaluation_asset_cannot_change_identity(tmp_path):
    ds = dataset(tmp_path, is_train=False)
    (tmp_path / "frame.png").unlink()
    try:
        with unittest.TestCase().assertRaises(FileNotFoundError):
            ds[0]
    finally:
        for conn in ds._conns.values():
            conn.close()


def test_best_cpu_state_survives_further_training():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    state = train.snapshot_state(model)
    expected = {key: value.clone() for key, value in state.items()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(7)
        model(torch.randn(4, 3))
    for key, value in state.items():
        torch.testing.assert_close(value, expected[key])


def test_legacy_fusion_scale_is_unity():
    old_cfg = dict(data_roots=[], db_paths=[], raters=[])
    assert _build_cfg_from_ckpt_dict(old_cfg).fusion_image_scale == 1.0
    assert _build_cfg_from_ckpt_dict(old_cfg).image_roi == "full_body"
    assert _build_cfg_from_ckpt_dict({**old_cfg, "fusion_image_scale": 0.5}).fusion_image_scale == 0.5


def test_duplicate_database_is_rejected_before_reading(tmp_path):
    path = tmp_path / "db/dataset.sqlite"
    alias = tmp_path / "db/../db/dataset.sqlite"
    cfg = Cfg(data_roots=[tmp_path, tmp_path], db_paths=[path, alias], raters=["sample"])
    with unittest.TestCase().assertRaisesRegex(ValueError, "Duplicate training database"):
        train.fetch_items_window_split(cfg)
    train.validate_unique_databases([path, tmp_path / "another.sqlite"])


def test_split_manifest_records_actual_dataset_membership(tmp_path):
    cfg = Cfg(data_roots=[tmp_path, tmp_path / "second"],
              db_paths=[tmp_path / "first.sqlite", tmp_path / "second.sqlite"], raters=["sample"])
    manifest = train.build_split_manifest(cfg, [(0, 1), (0, 2)], [(0, 3)])
    assert manifest["dataset_counts"][0]["train"] == 2
    assert manifest["dataset_counts"][0]["validation"] == 1
    assert manifest["dataset_counts"][1]["train"] == 0
    assert manifest["dataset_counts"][1]["validation"] == 0
    assert manifest["train_items"] == [[0, 1], [0, 2]]


def test_uniform_sequence_spans_window_and_transformer_uses_order():
    assert train.choose_T_indices(10, 10, "uniform", True) == list(range(10))
    assert train.choose_T_indices(20, 10, "uniform", True)[-1] == 19
    assert train.choose_T_indices(9, 10, "uniform", True) == list(range(9)) + [8]
    torch.manual_seed(42)
    temporal = TemporalTransformer(3, 16, 1, 0).eval()
    sequence = torch.randn(1, 10, 3)
    permuted = sequence.clone()
    permuted[:, :9] = sequence[:, :9].flip(1)  # final frame unchanged
    with torch.no_grad():
        assert not torch.allclose(temporal(sequence), temporal(permuted), atol=1e-7, rtol=0)


def test_grayscale_is_preserved_with_and_without_augmentation(tmp_path):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    previous = None
    for augment, is_train in [("standard", True), ("none", True), ("standard", False)]:
        ds = dataset(tmp_path, augment=augment, is_train=is_train, grayscale=True)
        try:
            for seed in (1, 2):
                train.set_seed(seed)
                frames = ds[0][0]
                pixels = frames * std + mean
                assert frames.shape == (10, 3, 32, 32)
                torch.testing.assert_close(pixels[:, 0], pixels[:, 1], atol=1e-6, rtol=0)
                torch.testing.assert_close(pixels[:, 0], pixels[:, 2], atol=1e-6, rtol=0)
                for frame in frames[1:]:
                    torch.testing.assert_close(frame, frames[0])
                if augment == "standard" and is_train:
                    if previous is not None:
                        assert not torch.equal(previous, frames)
                    previous = frames
        finally:
            for conn in ds._conns.values():
                conn.close()
        # Reuse the fixture directory for the next condition.
        (tmp_path / "dataset.sqlite").unlink()


def test_grayscale_validation_and_live_inference_match(tmp_path):
    ds = dataset(tmp_path, is_train=False, grayscale=True)
    try:
        with Image.open(tmp_path / "frame.png") as image:
            cropped = Image.fromarray(image_model_crop(np.asarray(image.convert("RGB")), ds.cfg.image_roi))
        expected = build_image_tf(ds.cfg.img_size, grayscale=True)(cropped)
        torch.testing.assert_close(ds[0][0][0], expected)
        # RGB remains the backward-compatible default and retains color.
        assert not torch.equal(build_image_tf(ds.cfg.img_size)(cropped), expected)
    finally:
        for conn in ds._conns.values():
            conn.close()


def test_grayscale_checkpoint_setting_and_cli_option():
    from unittest.mock import patch
    cfg = {"data_roots": [], "db_paths": [], "raters": []}
    assert _build_cfg_from_ckpt_dict(cfg).grayscale is False
    assert _build_cfg_from_ckpt_dict({**cfg, "grayscale": True}).grayscale is True
    with patch.object(sys, "argv", ["07_train.py", "--data_roots", "lesson", "--grayscale", "--augmentation", "standard"]):
        args = train.parse_args()
        assert args.grayscale is True and args.augmentation == "standard"


if __name__ == "__main__":
    scratch = ROOT / "outputs" / "system_audit_20261008"
    scratch.mkdir(parents=True, exist_ok=True)
    suite = unittest.TestSuite()
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            def run(test=test):
                if test.__code__.co_argcount:
                    with TemporaryDirectory(dir=scratch) as directory:
                        test(Path(directory))
                else:
                    test()
            suite.addTest(unittest.FunctionTestCase(run, description=name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
