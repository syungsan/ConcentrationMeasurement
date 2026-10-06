from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.devices import default_device, resolve_device
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from torch.utils.data import DataLoader, Dataset


CLASSES = list(range(1, 8))
JAPANESE_FONT_CANDIDATES = (
    "Noto Sans JP",
    "Noto Sans CJK JP",
    "Yu Gothic",
    "YuGothic",
    "Meiryo",
    "MS Gothic",
    "IPAexGothic",
    "IPAGothic",
)


def configure_plot_theme(*, style: str) -> None:
    """Configure seaborn with a font that can render Japanese situation names."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (name for name in JAPANESE_FONT_CANDIDATES if name in available_fonts),
        "sans-serif",
    )
    sns.set_theme(style=style, rc={"font.family": font_family})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained concentration model on labeled dataset.sqlite files. "
            "This creates predictions, regression metrics, 1-7 class metrics, and plots."
        )
    )
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint saved by scripts/07_train.py.")
    parser.add_argument("--data_roots", required=True, help="Comma-separated dataset roots for the test set.")
    parser.add_argument(
        "--db_paths",
        default="",
        help="Optional comma-separated DB paths. Default: each data_root + config.yaml paths.db_path.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["image", "skeleton", "fusion"], default="")
    parser.add_argument("--label_source", choices=["individual", "consensus"], default="")
    parser.add_argument("--raters", default="", help="Required when label_source=individual unless checkpoint cfg has raters.")
    parser.add_argument("--agg", choices=["mean", "median"], default="")
    parser.add_argument("--min_raters", type=int)
    parser.add_argument("--max_label_std", type=float)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--situation-source",
        choices=["true", "predicted"],
        default="predicted",
        help="true uses DB situation labels. predicted uses the situation classifier stored in the checkpoint.",
    )
    situation_feature_group = parser.add_mutually_exclusive_group()
    situation_feature_group.add_argument(
        "--use-situation-feature", "--use_situation_feature",
        dest="use_situation_feature", action="store_true",
        help="Feed situation labels/probabilities into concentration prediction.",
    )
    situation_feature_group.add_argument(
        "--no-situation-feature", "--no_situation_feature",
        dest="use_situation_feature", action="store_false",
        help="Still classify/record situation, but exclude it from concentration prediction.",
    )
    parser.set_defaults(use_situation_feature=None)
    return parser.parse_args()


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "config.yaml").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("could not find repository root")


def load_train_module(root: Path):
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("train07_for_eval", scripts_dir / "07_train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/07_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_infer_module(root: Path):
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from lib.infer import load_ckpt, load_situation_model

    return load_ckpt, load_situation_model


def load_yaml_cfg(root: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def split_paths(value: str) -> list[Path]:
    return [Path(s.strip()).resolve() for s in value.split(",") if s.strip()]


def score_to_class(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 1, 7).astype(int)


def collect_labeled_items(train_mod: Any, cfg: Any) -> list[tuple[int, int]]:
    items: list[tuple[int, int]] = []
    for db_i, db_path in enumerate(cfg.db_paths):
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        if cfg.label_source == "consensus":
            filters = ["w.situation IN (?, ?, ?)", "c.rater_count >= ?"]
            params: list[Any] = list(train_mod.SITUATIONS) + [cfg.min_raters]
            if cfg.max_label_std is not None:
                filters.append("c.score_std <= ?")
                params.append(cfg.max_label_std)
            rows = cur.execute(
                f"""
                SELECT s.id
                FROM segments s
                JOIN windows w ON w.id = s.window_id
                JOIN segment_frames sf ON sf.segment_id = s.id
                JOIN label_consensus c ON c.segment_id = s.id
                WHERE {' AND '.join(filters)}
                GROUP BY s.id
                ORDER BY w.video_id, w.t_start, s.track_id
                """,
                params,
            ).fetchall()
        else:
            if not cfg.raters:
                raise RuntimeError("raters are required for label_source=individual")
            qmarks = ",".join(["?"] * len(cfg.raters))
            rows = cur.execute(
                f"""
                SELECT s.id
                FROM segments s
                JOIN windows w ON w.id = s.window_id
                JOIN segment_frames sf ON sf.segment_id = s.id
                JOIN labels l ON l.segment_id = s.id AND l.rater IN ({qmarks})
                WHERE w.situation IN (?, ?, ?)
                GROUP BY s.id
                ORDER BY w.video_id, w.t_start, s.track_id
                """,
                cfg.raters + list(train_mod.SITUATIONS),
            ).fetchall()
        conn.close()
        items.extend((db_i, int(row[0])) for row in rows)
    if not items:
        raise RuntimeError("no labeled test items found")
    return items


def load_segment_metadata(db_path: Path, segment_id: int) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        """
        SELECT s.id, s.track_id, w.video_id, w.t_start, w.t_end, w.situation
        FROM segments s
        JOIN windows w ON w.id = s.window_id
        WHERE s.id = ?
        """,
        (int(segment_id),),
    ).fetchone()
    conn.close()
    if row is None:
        return {"segment_id": segment_id}
    return {
        "segment_id": int(row[0]),
        "track_id": int(row[1]),
        "video_id": int(row[2]),
        "t_start": float(row[3]),
        "t_end": float(row[4]),
        "situation": str(row[5]),
    }


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset[index]


def make_indexed_collate(train_mod: Any):
    def collate(batch):
        indices = [int(item[0]) for item in batch]
        samples = [item[1] for item in batch]
        frames, poses, situations, y = train_mod.collate_fn(samples)
        return indices, frames, poses, situations, y

    return collate


@torch.no_grad()
def predict_rows(
    train_mod: Any,
    model: torch.nn.Module,
    loader: DataLoader,
    items: list[tuple[int, int]],
    cfg: Any,
    situation_model: torch.nn.Module | None,
) -> list[dict[str, Any]]:
    model.eval()
    if situation_model is not None:
        situation_model.eval()
    rows: list[dict[str, Any]] = []
    for indices, frames, poses, situations, y in loader:
        if frames is not None:
            frames = frames.to(cfg.device)
        if poses is not None:
            poses = poses.to(cfg.device)
        situations = situations.to(cfg.device)
        y = y.to(cfg.device)
        if situation_model is not None:
            logits = situation_model(frames, poses)
            situations = torch.softmax(logits, dim=-1).nan_to_num(
                nan=1.0 / max(1, len(train_mod.SITUATIONS)),
                posinf=1.0 / max(1, len(train_mod.SITUATIONS)),
                neginf=0.0,
            )
        pred = model(frames, poses, situations).detach().cpu().numpy()
        truth = y.detach().cpu().numpy()
        for local_i, true_score, pred_score in zip(indices, truth, pred):
            db_i, segment_id = items[int(local_i)]
            meta = load_segment_metadata(cfg.db_paths[db_i], segment_id)
            rows.append(
                {
                    "db_index": db_i,
                    "db_path": str(cfg.db_paths[db_i]),
                    "data_root": str(cfg.data_roots[db_i]),
                    **meta,
                    "true_score": float(true_score),
                    "pred_score": float(pred_score),
                    "score_error": float(pred_score - true_score),
                }
            )
    return rows


def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["true_score"].to_numpy(dtype=float)
    y_pred = frame["pred_score"].to_numpy(dtype=float)
    true_class = frame["true_class"].to_numpy(dtype=int)
    pred_class = frame["pred_class"].to_numpy(dtype=int)
    precision, recall, f1, _support = precision_recall_fscore_support(
        true_class,
        pred_class,
        labels=CLASSES,
        average="weighted",
        zero_division=0,
    )
    return {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "bias_pred_minus_true": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)) if len(frame) >= 2 else float("nan"),
        "pearson": float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")),
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
        "exact_accuracy": float(accuracy_score(true_class, pred_class)),
        "within_1_accuracy": float(np.mean(np.abs(pred_class - true_class) <= 1)),
        "within_2_accuracy": float(np.mean(np.abs(pred_class - true_class) <= 2)),
        "mean_absolute_class_error": float(np.mean(np.abs(pred_class - true_class))),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "quadratic_weighted_kappa": float(cohen_kappa_score(true_class, pred_class, weights="quadratic")),
    }


def save_confusion_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    cm = confusion_matrix(frame["true_class"], frame["pred_class"], labels=CLASSES)
    cm_frame = pd.DataFrame(cm, index=[f"true_{i}" for i in CLASSES], columns=[f"pred_{i}" for i in CLASSES])
    cm_frame.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    row_norm = cm_frame.div(cm_frame.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    row_norm.to_csv(output_dir / "confusion_matrix_row_normalized.csv", encoding="utf-8-sig")

    configure_plot_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_frame, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    save_figure(fig, output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(row_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=True, ax=ax)
    ax.set_title("Confusion Matrix Row Normalized")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    save_figure(fig, output_dir / "confusion_matrix_row_normalized.png", dpi=180)
    plt.close(fig)


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".svg"))


def save_regression_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    configure_plot_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(data=frame, x="true_score", y="pred_score", hue="situation", ax=ax)
    ax.plot([1, 7], [1, 7], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0.75, 7.25)
    ax.set_ylim(0.75, 7.25)
    ax.set_title("True vs Predicted Score")
    fig.tight_layout()
    save_figure(fig, output_dir / "true_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(frame["score_error"], kde=True, bins=20, ax=ax)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("pred_score - true_score")
    fig.tight_layout()
    save_figure(fig, output_dir / "error_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    root = repo_root()
    train_mod = load_train_module(root)
    load_ckpt, load_situation_model = load_infer_module(root)
    ycfg = load_yaml_cfg(root)

    mode = args.mode
    if not mode:
        raw_ckpt = torch.load(str(args.model), map_location="cpu", weights_only=False)
        mode = str(raw_ckpt.get("mode", "fusion"))

    ckpt, model_cfg, model = load_ckpt(args.model, mode, args.device)
    model.to(args.device)
    use_situation_feature = (
        bool(model_cfg.use_situation_feature)
        if args.use_situation_feature is None else bool(args.use_situation_feature)
    )
    model.use_situation_feature = use_situation_feature

    data_roots = split_paths(args.data_roots)
    if args.db_paths.strip():
        db_paths = split_paths(args.db_paths)
    else:
        db_paths = [(root_path / Path(ycfg["paths"]["db_path"])).resolve() for root_path in data_roots]
    if len(data_roots) != len(db_paths):
        raise RuntimeError("data_roots and db_paths must have the same number of entries")
    for db_path in db_paths:
        if not db_path.exists():
            raise FileNotFoundError(db_path)

    label_source = args.label_source or model_cfg.label_source
    raters = [s.strip() for s in args.raters.split(",") if s.strip()] or list(model_cfg.raters)
    agg = args.agg or model_cfg.agg
    min_raters = args.min_raters if args.min_raters is not None else model_cfg.min_raters
    max_label_std = args.max_label_std if args.max_label_std is not None else model_cfg.max_label_std

    eval_cfg = replace(
        model_cfg,
        data_roots=data_roots,
        db_paths=db_paths,
        label_source=label_source,
        raters=raters,
        agg=agg,
        mode=mode,
        min_raters=min_raters,
        max_label_std=max_label_std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        sample_strategy="uniform",
    )

    situation_model = None
    if args.situation_source == "predicted":
        situation_model = load_situation_model(ckpt, eval_cfg, mode, args.device)

    items = collect_labeled_items(train_mod, eval_cfg)
    dataset = train_mod.MultiSQLiteSegmentDataset(eval_cfg, items, is_train=False)
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_indexed_collate(train_mod),
    )
    rows = predict_rows(train_mod, model, loader, items, eval_cfg, situation_model)
    frame = pd.DataFrame(rows)
    frame["true_class"] = score_to_class(frame["true_score"].to_numpy(dtype=float))
    frame["pred_class"] = score_to_class(frame["pred_score"].to_numpy(dtype=float))
    frame["class_error"] = frame["pred_class"] - frame["true_class"]
    frame["abs_class_error"] = frame["class_error"].abs()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    metrics = compute_metrics(frame)
    metrics_json_path = args.output_dir / "metrics.json"
    metrics_csv_path = args.output_dir / "metrics.csv"
    metrics_json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    report = classification_report(
        frame["true_class"],
        frame["pred_class"],
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(args.output_dir / "classification_report.csv", encoding="utf-8-sig")

    save_confusion_outputs(frame, args.output_dir)
    save_regression_plots(frame, args.output_dir)

    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": str(args.model.resolve()),
                "mode": mode,
                "data_roots": [str(p) for p in data_roots],
                "db_paths": [str(p) for p in db_paths],
                "label_source": label_source,
                "raters": raters,
                "situation_source": args.situation_source,
                "use_situation_feature": use_situation_feature,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {metrics_json_path.resolve()}")
    print(f"Saved metrics table to: {metrics_csv_path.resolve()}")
    print(f"Saved evaluation outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
