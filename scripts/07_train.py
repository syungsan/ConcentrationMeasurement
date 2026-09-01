# scripts/07_train.py
from __future__ import annotations

import time
SCRIPT_STARTED_AT = time.perf_counter()

import argparse
import csv
import json
import math
import random
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

from lib.image_roi import image_model_crop
from lib.pose_norm import normalize_pose_kpts
from lib.model_defs import Cfg, Regressor, SituationClassifier, Agg  # ★分離import
from lib.situation import SITUATIONS, situation_one_hot
from lib.research_schema import eligible_for_training, get_metadata


# -------------------------
# Repo root helper
# -------------------------
def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(6):
        if (p / "config.yaml").exists():
            return p
        p = p.parent
    return start.resolve()


def load_yaml_cfg(repo_root: Path) -> dict:
    import yaml
    return yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))


def rpath(base: Path, p: str) -> Path:
    return (base / Path(p)).resolve()


def write_csv_dicts(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".svg"))


def write_line_plot(
        path: Path,
        rows: List[Dict[str, object]],
        series: List[Tuple[str, str]],
        *,
        title: str,
        y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for key, label in series:
        xs: List[float] = []
        ys: List[float] = []
        for row in rows:
            try:
                x = float(row["epoch"])
                y = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=label)
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(y_label)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, path, dpi=160)
    plt.close(fig)


def write_training_artifacts(
        report_dir: Path,
        concentration_history: Dict[str, List[Dict[str, object]]],
        situation_history: Dict[str, List[Dict[str, object]]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        report_dir / "history.json",
        {
            "concentration": concentration_history,
            "situation": situation_history,
        },
    )
    for mode, rows in concentration_history.items():
        write_csv_dicts(report_dir / f"concentration_{mode}_history.csv", rows)
        write_line_plot(
            report_dir / f"concentration_{mode}_loss.png",
            rows,
            [("train_loss", "train_loss"), ("val_mse", "val_mse")],
            title=f"Concentration Loss ({mode})",
            y_label="loss",
        )
        write_line_plot(
            report_dir / f"concentration_{mode}_error.png",
            rows,
            [("val_rmse", "val_rmse"), ("val_mae", "val_mae")],
            title=f"Validation Error ({mode})",
            y_label="score error",
        )
        write_line_plot(
            report_dir / f"concentration_{mode}_ordered_metrics.png",
            rows,
            [("val_qwk", "val_qwk"), ("val_spearman", "val_spearman"), ("val_r2", "val_r2")],
            title=f"Validation Ordered Metrics ({mode})",
            y_label="metric",
        )
    for mode, rows in situation_history.items():
        write_csv_dicts(report_dir / f"situation_{mode}_history.csv", rows)
        write_line_plot(
            report_dir / f"situation_{mode}_loss.png",
            rows,
            [("val_loss", "val_loss")],
            title=f"Situation Loss ({mode})",
            y_label="loss",
        )
        write_line_plot(
            report_dir / f"situation_{mode}_accuracy.png",
            rows,
            [("val_accuracy", "val_accuracy")],
            title=f"Situation Accuracy ({mode})",
            y_label="accuracy",
        )


# -------------------------
# Helpers
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_T_indices(n: int, T: int, strategy: str, is_train: bool) -> List[int]:
    if n <= 0:
        raise ValueError("n must be > 0")
    if n >= T:
        if strategy == "random" and is_train:
            start = random.randint(0, n - T)
            return list(range(start, start + T))
        idxs = np.linspace(0, n - 1, T).round().astype(int).tolist()
        return idxs
    return list(range(n)) + [n - 1] * (T - n)


def agg_scores(scores: List[float], how: Agg) -> float:
    arr = np.array(scores, dtype=np.float32)
    if how == "median":
        return float(np.median(arr))
    return float(np.mean(arr))


def _norm_path_str(p: str) -> str:
    return str(p).replace("\\", "/").strip()


def resolve_asset_path(p: str, *, dataset_root: Path, db_path: Path) -> Optional[Path]:
    if not p:
        return None

    s = _norm_path_str(p)
    pp = Path(s)

    candidates: List[Path] = []

    if pp.is_absolute():
        candidates.append(pp)

    candidates.append((dataset_root / pp).resolve())

    candidates.append((db_path.parent / pp).resolve())
    candidates.append((db_path.parent.parent / pp).resolve())

    if "data/assets/" in s:
        s2 = s.replace("data/assets/", "assets/")
        candidates.append((dataset_root / s2).resolve())

    if s.startswith("data/"):
        s2 = s.replace("data/", "", 1)
        candidates.append((dataset_root / s2).resolve())

    name = pp.name
    if name:
        candidates.append((dataset_root / "assets" / "crops" / name).resolve())
        candidates.append((dataset_root / "assets" / "poses" / name).resolve())

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            pass
    return None


# -------------------------
# SQLite multi-db Dataset
# -------------------------
class MultiSQLiteSegmentDataset(Dataset):
    """
    1 sample = (db_index, segment_id)

    - crop_path / pose_path の解決を強くする
    - 欠損アセットが混ざっても DataLoader を落とさずにスキップ（リサンプル）する
    """
    def __init__(self, cfg: Cfg, items: List[Tuple[int, int]], is_train: bool):
        self.cfg = cfg
        self.items = items
        self.is_train = is_train

        aug = []
        if is_train:
            aug += [
                transforms.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.35, hue=0.03),
                transforms.RandomGrayscale(p=0.25),
            ]
        else:
            aug += [
                transforms.Resize(cfg.img_size + 32),
                transforms.CenterCrop(cfg.img_size),
            ]
        self.img_tf = transforms.Compose(
            aug + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )

        self._conns: Dict[int, sqlite3.Connection] = {}

    def _get_conn(self, db_i: int) -> sqlite3.Connection:
        if db_i not in self._conns:
            c = sqlite3.connect(str(self.cfg.db_paths[db_i]))
            c.execute("PRAGMA foreign_keys = ON;")
            self._conns[db_i] = c
        return self._conns[db_i]

    def __len__(self):
        return len(self.items)

    def _resolve_path(self, db_i: int, p: str) -> Path:
        dataset_root = self.cfg.data_roots[db_i]
        db_path = self.cfg.db_paths[db_i]
        rp = resolve_asset_path(p, dataset_root=dataset_root, db_path=db_path)
        if rp is None:
            raise FileNotFoundError(f"Missing asset: {p}")
        return rp

    def _load_pose(self, db_i: int, pose_path: str) -> np.ndarray:
        K = self.cfg.K
        D = self.cfg.pose_dim
        out = np.zeros((K, D), dtype=np.float32)

        if not pose_path:
            return out

        try:
            p = self._resolve_path(db_i, pose_path)
        except FileNotFoundError:
            return out

        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            kpts = j.get("keypoints", None)
            if not kpts:
                return out
            arr = np.array(kpts, dtype=np.float32)
            kk = min(K, arr.shape[0])
            dd = min(D, arr.shape[1]) if arr.ndim == 2 else 0
            if arr.ndim == 2 and dd > 0:
                out[:kk, :dd] = arr[:kk, :dd]
        except Exception:
            return out

        # ★ normalize (train/infer一致)
        try:
            out = normalize_pose_kpts(out, conf_thr=0.2)
        except Exception:
            out[:, :2] = 0.0
        return out

    def _load_label(self, cur: sqlite3.Cursor, seg_id: int) -> Optional[float]:
        if self.cfg.label_source == "consensus":
            row = cur.execute(
                "SELECT mean_score FROM label_consensus WHERE segment_id=?",
                (int(seg_id),),
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else None
        qmarks = ",".join(["?"] * len(self.cfg.raters))
        rows = cur.execute(
            f"SELECT score FROM labels WHERE segment_id=? AND rater IN ({qmarks})",
            [int(seg_id)] + self.cfg.raters
        ).fetchall()
        if not rows:
            return None
        scores = [float(r[0]) for r in rows if r and r[0] is not None]
        if not scores:
            return None
        return agg_scores(scores, self.cfg.agg)

    def _load_frames_and_poses(self, db_i: int, seg_id: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, float]:
        conn = self._get_conn(db_i)
        cur = conn.cursor()

        y = self._load_label(cur, seg_id)
        if y is None:
            raise RuntimeError(f"segment {seg_id} has no label for raters={self.cfg.raters}")

        situation_row = cur.execute(
            "SELECT w.situation FROM segments s JOIN windows w ON w.id=s.window_id WHERE s.id=?",
            (int(seg_id),)
        ).fetchone()
        if not situation_row or not situation_row[0]:
            raise RuntimeError(f"segment {seg_id} has no situation")
        situation = situation_one_hot(str(situation_row[0]))

        rows = cur.execute(
            "SELECT t, crop_path, pose_path FROM segment_frames WHERE segment_id=? ORDER BY t",
            (int(seg_id),)
        ).fetchall()
        if not rows:
            raise RuntimeError(f"segment {seg_id} has no segment_frames")

        idxs = choose_T_indices(len(rows), self.cfg.T, self.cfg.sample_strategy, self.is_train)

        frames = None
        poses = None

        if self.cfg.mode in ("image", "fusion"):
            imgs: List[torch.Tensor] = []
            for i in idxs:
                _t, crop_path, _pose_path = rows[i]
                p = self._resolve_path(db_i, crop_path)
                im = Image.open(p).convert("RGB")
                im = Image.fromarray(image_model_crop(np.asarray(im), self.cfg.image_roi))
                imgs.append(self.img_tf(im))
            frames = torch.stack(imgs, dim=0)  # [T,3,H,W]

        if self.cfg.mode in ("skeleton", "fusion"):
            ps: List[np.ndarray] = []
            for i in idxs:
                _t, _crop_path, pose_path = rows[i]
                ps.append(self._load_pose(db_i, pose_path))
            poses = torch.from_numpy(np.stack(ps, axis=0))  # [T,K,3]

        return frames, poses, situation, float(y)

    def __getitem__(self, idx: int):
        for _ in range(10):
            db_i, seg_id = self.items[idx]
            try:
                frames, poses, situation, y = self._load_frames_and_poses(int(db_i), int(seg_id))
                return frames, poses, situation, torch.tensor(y, dtype=torch.float32)
            except FileNotFoundError:
                idx = random.randint(0, len(self.items) - 1)
                continue

        raise FileNotFoundError("Too many missing assets. Please fix DB paths or assets layout.")


def collate_fn(batch):
    frames_list, poses_list, situation_list, y_list = zip(*batch)

    frames = None
    if frames_list[0] is not None:
        frames = torch.stack(frames_list, dim=0)  # [B,T,3,H,W]

    poses = None
    if poses_list[0] is not None:
        poses = torch.stack(poses_list, dim=0)  # [B,T,K,3]

    situations = torch.stack(situation_list, dim=0)  # [B,3]
    y = torch.stack(y_list, dim=0)  # [B]
    return frames, poses, situations, y


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(
        model: nn.Module, loader: DataLoader, device: str,
        situation_model: Optional[SituationClassifier] = None,
) -> Dict[str, float]:
    model.eval()
    preds, ys = [], []
    for frames, poses, situations, y in loader:
        if frames is not None:
            frames = frames.to(device)
        if poses is not None:
            poses = poses.to(device)
        situations = situations.to(device)
        if situation_model is not None:
            situation_logits = situation_model(frames, poses)
            situations = torch.softmax(situation_logits, dim=-1).nan_to_num(
                nan=1.0 / max(1, len(SITUATIONS)),
                posinf=1.0 / max(1, len(SITUATIONS)),
                neginf=0.0,
            )
        y = y.to(device)
        yhat = model(frames, poses, situations)
        preds.append(yhat.detach().cpu())
        ys.append(y.detach().cpu())
    pred = torch.cat(preds)
    y = torch.cat(ys)
    finite_mask = torch.isfinite(pred) & torch.isfinite(y)
    nonfinite_pred = int((~torch.isfinite(pred)).sum().item())
    nonfinite_target = int((~torch.isfinite(y)).sum().item())
    if not bool(finite_mask.all()):
        pred = pred[finite_mask]
        y = y[finite_mask]
    if pred.numel() == 0:
        return {
            "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
            "r2": float("nan"), "spearman": float("nan"), "qwk": float("nan"),
            "nonfinite_pred": nonfinite_pred, "nonfinite_target": nonfinite_target,
        }
    mse = F.mse_loss(pred, y).item()
    mae = F.l1_loss(pred, y).item()
    rmse = math.sqrt(mse)

    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum().clamp_min(1e-8)
    ss_res = ((y - pred) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item()
    pred_np = pred.numpy()
    y_np = y.numpy()
    spearman = _spearman(y_np, pred_np)
    qwk = _quadratic_weighted_kappa(
        np.clip(np.rint(y_np), 1, 7).astype(int),
        np.clip(np.rint(pred_np), 1, 7).astype(int),
    )
    return {
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
        "spearman": spearman, "qwk": qwk,
        "nonfinite_pred": nonfinite_pred, "nonfinite_target": nonfinite_target,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ar, br = _rankdata(a), _rankdata(b)
    if np.std(ar) == 0 or np.std(br) == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def _quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n_class = 7
    observed = np.zeros((n_class, n_class), dtype=np.float64)
    for x, yv in zip(a, b):
        if not np.isfinite(x) or not np.isfinite(yv):
            continue
        xi = int(x)
        yi = int(yv)
        if not (1 <= xi <= n_class and 1 <= yi <= n_class):
            continue
        observed[xi - 1, yi - 1] += 1
    if observed.sum() == 0:
        return float("nan")
    hist_a = observed.sum(axis=1)
    hist_b = observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b) / observed.sum()
    indices = np.arange(n_class)
    weights = ((indices[:, None] - indices[None, :]) / (n_class - 1)) ** 2
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else 0.0


def fetch_items_window_split(cfg: Cfg) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    set_seed(cfg.seed)

    grouped_items: Dict[str, List[Tuple[int, int]]] = {}

    for db_i, db_path in enumerate(cfg.db_paths):
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        has_excluded_segments = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='excluded_segments'"
        ).fetchone() is not None
        exclusion_filter = """
            AND NOT EXISTS (
                SELECT 1 FROM excluded_segments es
                WHERE es.segment_id=s.id
            )
        """ if has_excluded_segments else ""

        metadata = get_metadata(conn)
        if cfg.split_unit in {"school", "session"} and metadata is None:
            raise RuntimeError(
                f"{db_path}: research metadata is required for split_unit={cfg.split_unit}; "
                "run 05_configure_research_dataset.py"
            )
        if metadata is not None and not eligible_for_training(metadata, cfg.include_pilot_post):
            print(
                f"[exclude] {db_path}: role={metadata['study_role']} "
                f"phase={metadata['intervention_phase']}"
            )
            conn.close()
            continue

        if cfg.label_source == "consensus":
            filters = ["w.situation IN (?, ?, ?)", "c.rater_count >= ?"]
            params: list[object] = list(SITUATIONS) + [cfg.min_raters]
            if cfg.max_label_std is not None:
                filters.append("c.score_std <= ?")
                params.append(cfg.max_label_std)
            rows = cur.execute(f"""
                SELECT s.id, w.video_id, w.t_start, w.t_end
                FROM segments s
                JOIN windows w ON w.id = s.window_id
                JOIN segment_frames sf ON sf.segment_id = s.id
                JOIN label_consensus c ON c.segment_id = s.id
                WHERE {' AND '.join(filters)} {exclusion_filter}
                GROUP BY s.id
                ORDER BY w.video_id, w.t_start, s.track_id
            """, params).fetchall()
        else:
            qmarks = ",".join(["?"] * len(cfg.raters))
            rows = cur.execute(f"""
                SELECT s.id, w.video_id, w.t_start, w.t_end
                FROM segments s
                JOIN windows w ON w.id = s.window_id
                JOIN segment_frames sf ON sf.segment_id = s.id
                JOIN labels l ON l.segment_id = s.id AND l.rater IN ({qmarks})
                WHERE w.situation IN (?, ?, ?) {exclusion_filter}
                GROUP BY s.id
                ORDER BY w.video_id, w.t_start, s.track_id
            """, cfg.raters + list(SITUATIONS)).fetchall()

        conn.close()

        for seg_id, video_id, t0, t1 in rows:
            if cfg.split_unit == "window":
                group = f"window:{db_i}:{int(video_id)}:{float(t0):.6f}:{float(t1):.6f}"
            elif cfg.split_unit == "dataset":
                group = f"dataset:{db_i}"
            elif cfg.split_unit == "session":
                group = f"session:{metadata['school_id']}:{metadata['session_id']}"  # type: ignore[index]
            else:
                group = f"school:{metadata['school_id']}"  # type: ignore[index]
            grouped_items.setdefault(group, []).append((db_i, int(seg_id)))

    groups = sorted(grouped_items)
    if len(groups) < 2:
        raise RuntimeError(
            f"split_unit={cfg.split_unit} produced {len(groups)} group(s); at least 2 are required"
        )
    holdout = set(cfg.holdout_groups or [])
    if holdout:
        unknown = holdout - set(groups)
        if unknown:
            raise RuntimeError(f"unknown holdout groups: {sorted(unknown)}; available={groups}")
        val_groups = holdout
    else:
        random.shuffle(groups)
        n_val = max(1, int(round(len(groups) * cfg.val_ratio)))
        n_val = min(n_val, len(groups) - 1)
        val_groups = set(groups[:n_val])

    tr_items = [item for group, items in grouped_items.items() if group not in val_groups for item in items]
    va_items = [item for group, items in grouped_items.items() if group in val_groups for item in items]
    print(f"split_unit={cfg.split_unit} train_groups={sorted(set(groups)-val_groups)}")
    print(f"split_unit={cfg.split_unit} val_groups={sorted(val_groups)}")

    return tr_items, va_items


def train_one(
        cfg: Cfg,
        situation_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[float, Dict[str, torch.Tensor], List[Dict[str, object]]]:
    set_seed(cfg.seed)

    tr_items, va_items = fetch_items_window_split(cfg)
    if len(tr_items) == 0 or len(va_items) == 0:
        raise RuntimeError(f"Not enough data. train={len(tr_items)} val={len(va_items)}")

    ds_tr = MultiSQLiteSegmentDataset(cfg, tr_items, is_train=True)
    ds_va = MultiSQLiteSegmentDataset(cfg, va_items, is_train=False)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model = Regressor(cfg).to(cfg.device)
    situation_model: Optional[SituationClassifier] = None
    if situation_state is not None:
        situation_model = SituationClassifier(cfg).to(cfg.device)
        situation_model.load_state_dict(situation_state, strict=True)
        situation_model.eval()
        for parameter in situation_model.parameters():
            parameter.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_mse = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, object]] = []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        skipped_train_batches = 0
        train_nonfinite_pred = 0
        train_nonfinite_loss = 0
        train_nonfinite_grad = 0

        for frames, poses, situations, y in dl_tr:
            if frames is not None:
                frames = frames.to(cfg.device, non_blocking=True)
            if poses is not None:
                poses = poses.to(cfg.device, non_blocking=True)
            situations = situations.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            if situation_model is not None:
                with torch.no_grad():
                    predicted_situations = torch.softmax(
                        situation_model(frames, poses), dim=-1
                    ).nan_to_num(
                        nan=1.0 / max(1, len(SITUATIONS)),
                        posinf=1.0 / max(1, len(SITUATIONS)),
                        neginf=0.0,
                    )
                keep_truth = (
                    torch.rand(len(y), 1, device=cfg.device)
                    < cfg.situation_teacher_forcing
                )
                situations = torch.where(
                    keep_truth, situations, predicted_situations
                )

            opt.zero_grad(set_to_none=True)
            if cfg.objective == "ordinal":
                logits = model.forward_ordinal_logits(frames, poses, situations)
                yhat = 1.0 + torch.sigmoid(logits).sum(dim=-1)
                if not torch.isfinite(logits).all() or not torch.isfinite(yhat).all():
                    skipped_train_batches += 1
                    train_nonfinite_pred += int((~torch.isfinite(yhat)).sum().item())
                    continue
                thresholds = torch.arange(1, 7, device=y.device).unsqueeze(0)
                # Consensus labels may be fractional. A score of 4.5 therefore
                # yields full targets below 4 and a 0.5 target at threshold 4.
                ordinal_target = (y.unsqueeze(1) - thresholds).clamp(0.0, 1.0)
                loss = F.binary_cross_entropy_with_logits(logits, ordinal_target)
            else:
                yhat = model(frames, poses, situations)
                if not torch.isfinite(yhat).all():
                    skipped_train_batches += 1
                    train_nonfinite_pred += int((~torch.isfinite(yhat)).sum().item())
                    continue
                loss = loss_fn(yhat, y)
            if not torch.isfinite(loss):
                skipped_train_batches += 1
                train_nonfinite_loss += int(y.size(0))
                continue
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                skipped_train_batches += 1
                train_nonfinite_grad += int(y.size(0))
                continue
            opt.step()

            total += loss.item() * y.size(0)
            n += y.size(0)

        tr_mse = total / n if n > 0 else float("nan")
        # Validation reflects autonomous operation: only predicted situation
        # probabilities are supplied when a classifier is available.
        va = evaluate(model, dl_va, cfg.device, situation_model)
        history.append({
            "epoch": ep,
            "mode": cfg.mode,
            "objective": cfg.objective,
            "train_loss": tr_mse,
            "train_samples": n,
            "train_skipped_batches": skipped_train_batches,
            "train_nonfinite_pred": train_nonfinite_pred,
            "train_nonfinite_loss": train_nonfinite_loss,
            "train_nonfinite_grad": train_nonfinite_grad,
            "val_mse": va["mse"],
            "val_rmse": va["rmse"],
            "val_mae": va["mae"],
            "val_r2": va["r2"],
            "val_spearman": va["spearman"],
            "val_qwk": va["qwk"],
            "val_nonfinite_pred": va.get("nonfinite_pred", 0),
            "val_nonfinite_target": va.get("nonfinite_target", 0),
        })

        print(
            f"[{cfg.mode}] ep{ep:03d} train_mse={tr_mse:.4f} "
            f"val_rmse={va['rmse']:.4f} val_mae={va['mae']:.4f} "
            f"val_r2={va['r2']:.4f} val_rho={va['spearman']:.4f} val_qwk={va['qwk']:.4f} "
            f"val_nonfinite_pred={va.get('nonfinite_pred', 0)} "
            f"train_skipped={skipped_train_batches}"
        )

        if not math.isfinite(va["mse"]):
            if best_state is not None:
                model.load_state_dict(best_state, strict=True)
            print(
                f"[WARN] {cfg.mode} ep{ep:03d}: validation became non-finite; "
                "restored best weights and stopped this run."
            )
            break

        if va["mse"] < best_mse:
            best_mse = va["mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    return best_mse, best_state, history


@torch.no_grad()
def evaluate_situation(
        model: SituationClassifier, loader: DataLoader, device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    skipped = 0
    nonfinite_logits = 0
    for frames, poses, situations, _y in loader:
        if frames is not None:
            frames = frames.to(device)
        if poses is not None:
            poses = poses.to(device)
        target = situations.argmax(dim=1).to(device)
        logits = model(frames, poses)
        finite_rows = torch.isfinite(logits).all(dim=1)
        nonfinite_logits += int((~finite_rows).sum().item())
        if not bool(finite_rows.all()):
            logits = logits[finite_rows]
            target = target[finite_rows]
        if target.numel() == 0:
            skipped += 1
            continue
        loss = F.cross_entropy(logits, target, reduction="sum")
        if not torch.isfinite(loss):
            skipped += 1
            nonfinite_logits += int(target.numel())
            continue
        total_loss += loss.item()
        correct += int((logits.argmax(dim=1) == target).sum().item())
        total += int(target.numel())
    return {
        "loss": total_loss / total if total > 0 else float("nan"),
        "accuracy": correct / total if total > 0 else float("nan"),
        "nonfinite_logits": nonfinite_logits,
        "skipped_batches": skipped,
    }


def train_situation_one(
        cfg: Cfg, epochs: int,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], List[Dict[str, object]]]:
    set_seed(cfg.seed)
    tr_items, va_items = fetch_items_window_split(cfg)
    ds_tr = MultiSQLiteSegmentDataset(cfg, tr_items, is_train=True)
    ds_va = MultiSQLiteSegmentDataset(cfg, va_items, is_train=False)
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    dl_va = DataLoader(
        ds_va, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    model = SituationClassifier(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        skipped_train_batches = 0
        train_nonfinite_logits = 0
        for frames, poses, situations, _y in dl_tr:
            if frames is not None:
                frames = frames.to(cfg.device, non_blocking=True)
            if poses is not None:
                poses = poses.to(cfg.device, non_blocking=True)
            target = situations.argmax(dim=1).to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames, poses)
            if not torch.isfinite(logits).all():
                train_nonfinite_logits += int((~torch.isfinite(logits).all(dim=1)).sum().item())
                skipped_train_batches += 1
                continue
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                skipped_train_batches += 1
                train_nonfinite_logits += int(target.numel())
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate_situation(model, dl_va, cfg.device)
        history.append({
            "epoch": epoch,
            "mode": cfg.mode,
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_nonfinite_logits": metrics.get("nonfinite_logits", 0),
            "val_skipped_batches": metrics.get("skipped_batches", 0),
            "train_nonfinite_logits": train_nonfinite_logits,
            "train_skipped_batches": skipped_train_batches,
        })
        print(
            f"[situation:{cfg.mode}] ep{epoch:03d} "
            f"val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f} "
            f"val_nonfinite_logits={metrics.get('nonfinite_logits', 0)} "
            f"train_nonfinite_logits={train_nonfinite_logits} train_skipped={skipped_train_batches}"
        )
        if math.isfinite(metrics["loss"]) and metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(
            "Situation classifier did not produce a finite validation loss. "
            "Try lowering --lr, reducing transformer size/layers, or checking input features for NaN/inf."
        )
    return best_metrics, best_state, history


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_roots",
        type=str,
        required=True,
        help="授業データrootをカンマ区切りで複数指定 例: datasets/lesson_001,datasets/lesson_002,datasets/lesson_003"
    )
    ap.add_argument(
        "--db_paths", type=str, default="",
        help="data_rootsに対応するDBをカンマ区切りで直接指定（マージ後DB用）",
    )
    ap.add_argument(
        "--raters",
        type=str,
        default="",
        help="ラベラー名をカンマ区切りで複数指定 例: teacherA,teacherB"
    )
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "median"])
    ap.add_argument("--label_source", type=str, default="individual", choices=["individual", "consensus"],
                    help="individual=評価者別labels / consensus=マージ後の平均ラベル")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--temporal", type=str, default="gru", choices=["gru", "transformer"])
    ap.add_argument("--mode", type=str, default="fusion", choices=["image", "skeleton", "fusion"])
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--objective", choices=["ordinal", "regression"], default="ordinal")
    ap.add_argument(
        "--split_unit", choices=["school", "session", "dataset", "window"],
        default="school",
        help="Validation grouping. Use school for external-generalization development.",
    )
    ap.add_argument(
        "--holdout_groups", default="",
        help="Exact group IDs to reserve, e.g. school:SCHOOL_B,school:SCHOOL_C",
    )
    ap.add_argument(
        "--include_pilot_post", action="store_true",
        help="Allow development-role pilot_post datasets in training/validation.",
    )
    ap.add_argument("--min_raters", type=int, default=2)
    ap.add_argument("--max_label_std", type=float)
    ap.add_argument(
        "--situation_epochs", type=int, default=10,
        help="Epochs for the automatic situation classifier; 0 disables it.",
    )
    ap.add_argument(
        "--situation_teacher_forcing", type=float, default=0.5,
        help="Fraction of concentration batches using true situation labels.",
    )
    situation_feature_group = ap.add_mutually_exclusive_group()
    situation_feature_group.add_argument(
        "--use-situation-feature", "--use_situation_feature",
        dest="use_situation_feature", action="store_true",
        help="Feed situation probabilities into the concentration model (default).",
    )
    situation_feature_group.add_argument(
        "--no-situation-feature", "--no_situation_feature",
        dest="use_situation_feature", action="store_false",
        help="Keep situation classification artifacts, but exclude it from concentration prediction.",
    )
    ap.set_defaults(use_situation_feature=True)

    ap.add_argument("--sample_strategy", type=str, default="uniform", choices=["uniform", "random"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--image_roi", type=str, default="", choices=["", "upper_body", "full_body"])
    ap.add_argument(
        "--fusion_image_scale", type=float, default=0.25,
        help="Fusion時のRGB特徴の重み。skeleton中心なら0.1〜0.35推奨。",
    )
    ap.add_argument("--T", type=int, default=0, help="0なら configから自動推定（window_sec*sample_fps）")

    ap.add_argument("--save_name", type=str, default="best_allmodes.pt")
    ap.add_argument(
        "--report_dir",
        type=str,
        default="",
        help="Directory for training history CSV/JSON/PNG/SVG outputs. Default: <save_name stem>_training_report",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.situation_teacher_forcing <= 1.0:
        raise ValueError("situation_teacher_forcing must be between 0 and 1")

    file_here = Path(__file__).resolve()
    repo_root = find_repo_root(file_here.parent)
    ycfg = load_yaml_cfg(repo_root)

    data_roots = [Path(s).resolve() for s in args.data_roots.split(",") if s.strip()]
    if not data_roots:
        raise RuntimeError("no data_roots")

    if args.db_paths.strip():
        db_paths = [Path(s.strip()).resolve() for s in args.db_paths.split(",") if s.strip()]
        if len(db_paths) != len(data_roots):
            raise RuntimeError("db_paths must have the same number of entries as data_roots")
    else:
        db_paths = [rpath(dr, ycfg["paths"]["db_path"]) for dr in data_roots]
    for p in db_paths:
        if not p.exists():
            raise FileNotFoundError(f"DB not found: {p}")

    # T 自動推定: window_sec * sample_fps
    sample_fps = float(ycfg["sampling"]["sample_fps"])
    window_sec = float(ycfg["segments"]["window_sec"])
    T_default = max(1, int(round(sample_fps * window_sec)))
    T = args.T if args.T > 0 else T_default
    image_roi = args.image_roi or str(ycfg.get("sampling", {}).get("image_roi", "upper_body"))

    raters = [s.strip() for s in args.raters.split(",") if s.strip()]
    if args.label_source == "individual" and not raters:
        raise RuntimeError("no raters")

    base = Cfg(
        data_roots=data_roots,
        db_paths=db_paths,
        raters=raters,
        agg=args.agg,  # type: ignore[arg-type]
        label_source=args.label_source,  # type: ignore[arg-type]
        objective=args.objective,
        split_unit=args.split_unit,
        holdout_groups=[s.strip() for s in args.holdout_groups.split(",") if s.strip()],
        include_pilot_post=args.include_pilot_post,
        min_raters=args.min_raters,
        max_label_std=args.max_label_std,
        situation_teacher_forcing=args.situation_teacher_forcing,
        use_situation_feature=args.use_situation_feature,
        mode=args.mode,  # type: ignore[arg-type]
        temporal=args.temporal,  # type: ignore[arg-type]
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        sample_strategy=args.sample_strategy,  # type: ignore[arg-type]
        img_size=args.img_size,
        image_roi=image_roi,  # type: ignore[arg-type]
        fusion_image_scale=args.fusion_image_scale,
        T=T,
        val_ratio=args.val_ratio,
    )

    results: Dict[str, float] = {}
    states: Dict[str, Dict[str, torch.Tensor]] = {}
    situation_results: Dict[str, Dict[str, float]] = {}
    situation_states: Dict[str, Dict[str, torch.Tensor]] = {}
    concentration_history: Dict[str, List[Dict[str, object]]] = {}
    situation_history: Dict[str, List[Dict[str, object]]] = {}

    # --mode が指定されていればそれだけ、指定がなければ3モード全部
    modes_to_run = [args.mode] if args.mode else ["image", "skeleton", "fusion"]

    for m in modes_to_run:
        cfg = replace(base, mode=m)
        if args.situation_epochs > 0:
            sit_metrics, sit_state, sit_history = train_situation_one(cfg, args.situation_epochs)
            situation_results[m] = sit_metrics
            situation_states[m] = sit_state
            situation_history[m] = sit_history
        best, state, history = train_one(
            cfg, situation_state=situation_states.get(m)
        )
        results[m] = best
        states[m] = state
        concentration_history[m] = history
        print(f"BEST[{m}] mse={best:.6f}")

    out = repo_root / args.save_name
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir.strip()
        else out.with_name(f"{out.stem}_training_report")
    )
    write_training_artifacts(report_dir, concentration_history, situation_history)
    print("training report:", report_dir)

    # 保存する state_dict は「最後に回した mode」のもの
    # （単体実行ならその mode、全実行なら fusion を優先して保存、無ければ最後）
    if "fusion" in states:
        save_state = states["fusion"]
        save_mode = "fusion"
    else:
        save_mode = modes_to_run[-1]
        save_state = states[save_mode]

    # 07_train.py の保存直前
    cfg_dict = dict(base.__dict__)
    cfg_dict["data_roots"] = [str(p) for p in cfg_dict["data_roots"]]
    cfg_dict["db_paths"]   = [str(p) for p in cfg_dict["db_paths"]]

    torch.save(
        {
            "cfg": cfg_dict,
            "results": results,
            "mode": save_mode,
            "state_dict": save_state,
            "situation_results": situation_results,
            "situation_state_dicts": situation_states,
            "training_history": concentration_history,
            "situation_history": situation_history,
            "training_report_dir": str(report_dir),
            "situation_labels": list(SITUATIONS),
            "feature_schema": "image_pose_situation_v4",
        },
        out
    )
    print("saved:", out)
    print("DONE:", results)


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
# 初期小規模実験
# 複数評価者の合意DBを使う場合：
# python scripts\07_train.py --data_roots datasets\lesson_train --db_paths merged\lesson_train.sqlite --label_source consensus --split_unit window --mode skeleton --save_name models\pilot_class.pt
# 評価者が1名だけの場合：
# python scripts\07_train.py --data_roots datasets\lesson_train --label_source individual --raters evaluator01 --split_unit window --mode skeleton --save_name models\pilot_class.pt
# python.exe scripts\07_train.py --data_roots datasets\lesson_train --label_source individual --raters evaluator01 --split_unit window --mode fusion --fusion_image_scale 0.5 --save_name models\pilot_fusion_stable.pt

# --temporal transformer
# 複数評価者の複数DBを使う場合の前処理：
# python.exe scripts\tools\merge_rater_databases.py --base-db datasets\lesson_001\db\dataset.sqlite --inputs returned_dbs\lesson_001_tanaka.sqlite returned_dbs\lesson_001_suzuki.sqlite returned_dbs\lesson_001_sato.sqlite --output merged\lesson_001_merged.sqlite --overwrite
# マージ後
# python.exe scripts\07_train.py --data_roots datasets\lesson_001 --db_paths merged\lesson_001_merged.sqlite --label_source consensus --mode fusion

# python.exe scripts\07_train.py --data_roots datasets\20260227_unnan_nishi_5-1_1,datasets\20260227_unnan_nishi_5-1_1,datasets\20260227_unnan_nishi_5-1_2,datasets\20260311_unnan_nishi_5-1_1,datasets\20260313_unnan_nishi_5-1_1_a --label_source individual --raters sample --split_unit window --mode fusion --fusion_image_scale 0.5 --save_name models\pre_test_2_transformer_fusion.pt --temporal transformer --lr 1e-4 --include_pilot_post
