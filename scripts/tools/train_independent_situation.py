"""Add a separately trained situation classifier without retraining concentration."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lib.devices import resolve_device
from lib.infer import load_ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.resolve() == args.model.resolve() or args.output.exists():
        raise FileExistsError("Specify a new output checkpoint; existing files are preserved")
    report = args.output.with_name(args.output.stem + "_situation_report")
    if report.exists():
        raise FileExistsError(report)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    raw = torch.load(args.model, map_location="cpu", weights_only=False)
    mode = raw.get("mode", "fusion")
    ckpt, cfg, _ = load_ckpt(args.model, mode, "cpu")
    if cfg.use_situation_feature:
        raise ValueError("This tool is for no_situation models (use_situation_feature=False)")
    overrides = {"pose_features": "raw", "temporal_pool": "last", "backbone_lr_scale": 1.0,
                 "freeze_batchnorm": False, "lr_schedule": "none", "early_stopping_patience": 0,
                 "augmentation": "none"}
    training_cfg = replace(cfg, situation_config=overrides, device=resolve_device(args.device),
                           num_workers=args.num_workers)
    spec = importlib.util.spec_from_file_location("independent_training", ROOT / "scripts/07_train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    metrics, state, history = train.train_situation_one(training_cfg, args.epochs)
    # Concentration weights, configuration and original metrics remain intact.
    ckpt["cfg"] = {**ckpt["cfg"], "situation_config": overrides}
    ckpt["situation_state_dicts"] = {**ckpt.get("situation_state_dicts", {}), mode: state}
    ckpt["situation_history"] = {**ckpt.get("situation_history", {}), mode: history}
    ckpt["situation_results"] = {**ckpt.get("situation_results", {}), mode: metrics}
    ckpt["situation_labels"] = list(train.SITUATIONS)
    ckpt["independent_situation_source"] = str(args.model.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train.write_training_artifacts(report, {}, {mode: history})
    torch.save(ckpt, args.output)
    print(f"Saved independent classifier: {args.output.resolve()}; concentration weights preserved", flush=True)


if __name__ == "__main__":
    main()
