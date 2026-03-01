#0 scripts/06_train.py
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

from lib.pose_norm import normalize_pose_kpts
from lib.model_defs import Cfg, Regressor, Agg  # ★分離import


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

    def _load_frames_and_poses(self, db_i: int, seg_id: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
        conn = self._get_conn(db_i)
        cur = conn.cursor()

        y = self._load_label(cur, seg_id)
        if y is None:
            raise RuntimeError(f"segment {seg_id} has no label for raters={self.cfg.raters}")

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
                imgs.append(self.img_tf(im))
            frames = torch.stack(imgs, dim=0)  # [T,3,H,W]

        if self.cfg.mode in ("skeleton", "fusion"):
            ps: List[np.ndarray] = []
            for i in idxs:
                _t, _crop_path, pose_path = rows[i]
                ps.append(self._load_pose(db_i, pose_path))
            poses = torch.from_numpy(np.stack(ps, axis=0))  # [T,K,3]

        return frames, poses, float(y)

    def __getitem__(self, idx: int):
        for _ in range(10):
            db_i, seg_id = self.items[idx]
            try:
                frames, poses, y = self._load_frames_and_poses(int(db_i), int(seg_id))
                return frames, poses, torch.tensor(y, dtype=torch.float32)
            except FileNotFoundError:
                idx = random.randint(0, len(self.items) - 1)
                continue

        raise FileNotFoundError("Too many missing assets. Please fix DB paths or assets layout.")


def collate_fn(batch):
    frames_list, poses_list, y_list = zip(*batch)

    frames = None
    if frames_list[0] is not None:
        frames = torch.stack(frames_list, dim=0)  # [B,T,3,H,W]

    poses = None
    if poses_list[0] is not None:
        poses = torch.stack(poses_list, dim=0)  # [B,T,K,3]

    y = torch.stack(y_list, dim=0)  # [B]
    return frames, poses, y


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    preds, ys = [], []
    for frames, poses, y in loader:
        if frames is not None:
            frames = frames.to(device)
        if poses is not None:
            poses = poses.to(device)
        y = y.to(device)
        yhat = model(frames, poses)
        preds.append(yhat.detach().cpu())
        ys.append(y.detach().cpu())
    pred = torch.cat(preds)
    y = torch.cat(ys)
    mse = F.mse_loss(pred, y).item()
    mae = F.l1_loss(pred, y).item()
    rmse = math.sqrt(mse)

    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum().clamp_min(1e-8)
    ss_res = ((y - pred) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item()
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def fetch_items_window_split(cfg: Cfg) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    set_seed(cfg.seed)

    all_windows: Dict[Tuple[int, int, float, float], List[Tuple[int, int]]] = {}

    for db_i, db_path in enumerate(cfg.db_paths):
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        qmarks = ",".join(["?"] * len(cfg.raters))

        rows = cur.execute(f"""
            SELECT s.id, s.video_id, s.t_start, s.t_end
            FROM segments s
            JOIN segment_frames sf ON sf.segment_id = s.id
            JOIN labels l ON l.segment_id = s.id AND l.rater IN ({qmarks})
            GROUP BY s.id
            ORDER BY s.video_id, s.t_start, s.track_id
        """, cfg.raters).fetchall()

        conn.close()

        for seg_id, video_id, t0, t1 in rows:
            key = (db_i, int(video_id), float(t0), float(t1))
            all_windows.setdefault(key, []).append((db_i, int(seg_id)))

    win_keys = list(all_windows.keys())
    random.shuffle(win_keys)

    n_val = max(1, int(len(win_keys) * cfg.val_ratio))
    val_keys = set(win_keys[:n_val])

    tr_items: List[Tuple[int, int]] = []
    va_items: List[Tuple[int, int]] = []
    for k, items in all_windows.items():
        if k in val_keys:
            va_items.extend(items)
        else:
            tr_items.extend(items)

    return tr_items, va_items


def train_one(cfg: Cfg) -> Tuple[float, Dict[str, torch.Tensor]]:
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_mse = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for frames, poses, y in dl_tr:
            if frames is not None:
                frames = frames.to(cfg.device, non_blocking=True)
            if poses is not None:
                poses = poses.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            yhat = model(frames, poses)
            loss = loss_fn(yhat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item() * y.size(0)
            n += y.size(0)

        tr_mse = total / max(1, n)
        va = evaluate(model, dl_va, cfg.device)

        print(
            f"[{cfg.mode}] ep{ep:03d} train_mse={tr_mse:.4f} "
            f"val_rmse={va['rmse']:.4f} val_mae={va['mae']:.4f} val_r2={va['r2']:.4f}"
        )

        if va["mse"] < best_mse:
            best_mse = va["mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    return best_mse, best_state


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_roots",
        type=str,
        required=True,
        help="授業データrootをカンマ区切りで複数指定 例: datasets/hara,datasets/minamoto,datasets/miyazaki"
    )
    ap.add_argument(
        "--raters",
        type=str,
        required=True,
        help="ラベラー名をカンマ区切りで複数指定 例: teacherA,teacherB"
    )
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "median"])

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--temporal", type=str, default="gru", choices=["gru", "transformer"])
    ap.add_argument("--mode", type=str, default="fusion", choices=["image", "skeleton", "fusion"])
    ap.add_argument("--val_ratio", type=float, default=0.2)

    ap.add_argument("--sample_strategy", type=str, default="uniform", choices=["uniform", "random"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--T", type=int, default=0, help="0なら configから自動推定（window_sec*sample_fps）")

    ap.add_argument("--save_name", type=str, default="best_allmodes.pt")
    return ap.parse_args()


def main():
    args = parse_args()

    file_here = Path(__file__).resolve()
    repo_root = find_repo_root(file_here.parent)
    ycfg = load_yaml_cfg(repo_root)

    data_roots = [Path(s).resolve() for s in args.data_roots.split(",") if s.strip()]
    if not data_roots:
        raise RuntimeError("no data_roots")

    db_paths = [rpath(dr, ycfg["paths"]["db_path"]) for dr in data_roots]
    for p in db_paths:
        if not p.exists():
            raise FileNotFoundError(f"DB not found: {p}")

    # T 自動推定: window_sec * sample_fps
    sample_fps = float(ycfg["sampling"]["sample_fps"])
    window_sec = float(ycfg["segments"]["window_sec"])
    T_default = max(1, int(round(sample_fps * window_sec)))
    T = args.T if args.T > 0 else T_default

    raters = [s.strip() for s in args.raters.split(",") if s.strip()]
    if not raters:
        raise RuntimeError("no raters")

    base = Cfg(
        data_roots=data_roots,
        db_paths=db_paths,
        raters=raters,
        agg=args.agg,  # type: ignore[arg-type]
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
        T=T,
        val_ratio=args.val_ratio,
    )

    results: Dict[str, float] = {}
    states: Dict[str, Dict[str, torch.Tensor]] = {}

    # --mode が指定されていればそれだけ、指定がなければ3モード全部
    modes_to_run = [args.mode] if args.mode else ["image", "skeleton", "fusion"]

    for m in modes_to_run:
        cfg = replace(base, mode=m)
        best, state = train_one(cfg)
        results[m] = best
        states[m] = state
        print(f"BEST[{m}] mse={best:.6f}")

    out = repo_root / args.save_name

    # 保存する state_dict は「最後に回した mode」のもの
    # （単体実行ならその mode、全実行なら fusion を優先して保存、無ければ最後）
    if "fusion" in states:
        save_state = states["fusion"]
        save_mode = "fusion"
    else:
        save_mode = modes_to_run[-1]
        save_state = states[save_mode]

    # 06_train.py の保存直前
    cfg_dict = dict(base.__dict__)
    cfg_dict["data_roots"] = [str(p) for p in cfg_dict["data_roots"]]
    cfg_dict["db_paths"]   = [str(p) for p in cfg_dict["db_paths"]]

    torch.save(
        {"cfg": cfg_dict, "results": results, "mode": save_mode, "state_dict": save_state},
        out
    )
    print("saved:", out)
    print("DONE:", results)


if __name__ == "__main__":
    main()

    import winsound
    try:
        winsound.PlaySound("mei_kara_mei_switch1.wav", winsound.SND_FILENAME)
    except Exception as e:
        print(f"[WARN] 音声を再生できませんでした: {e}")

# command
# python scripts/06_train.py --data_roots datasets/hara,datasets/minamoto,datasets/miyazaki --raters teacherA,teacherB --agg mean --epochs 20 --batch_size 16 --temporal gru --mode skeleton --save_name models/skeleton_modes.pt
