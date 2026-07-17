# scripts/10_aggregate_predictions.py
from __future__ import annotations

import time
SCRIPT_STARTED_AT = time.perf_counter()

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, required=True, help="outputs/pred_log.sqlite")
    ap.add_argument("--out_dir", type=str, default="outputs", help="出力先")
    ap.add_argument("--video_path", type=str, default=None, help="特定動画だけ集計したい場合に指定（部分一致OK）")
    ap.add_argument("--bin_sec", type=float, default=5.0, help="時間ビン（秒）例:5.0")
    ap.add_argument("--min_samples_track", type=int, default=30, help="人物別集計の最低サンプル数")
    return ap.parse_args()


def load_df(db_path: Path, video_filter: str | None) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(predictions)")}
    confidence_sql = (
        "situation_confidence" if "situation_confidence" in columns
        else "NULL AS situation_confidence"
    )
    situation_sql = (
        "COALESCE(situation_override, situation) AS situation"
        if "situation_override" in columns
        else "situation"
    )
    q = f"""
        SELECT video_path, frame_idx, t, track_id, {situation_sql},
               {confidence_sql}, score, score_clamped, det_conf, x1, y1, x2, y2
        FROM predictions
        """
    df = pd.read_sql_query(q, conn)
    conn.close()

    if video_filter:
        # 部分一致
        df = df[df["video_path"].astype(str).str.contains(video_filter, na=False)]

    # 型
    df["t"] = df["t"].astype(float)
    df["score"] = df["score"].astype(float)
    df["score_clamped"] = df["score_clamped"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df


def overall_stats(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame([{
            "n_rows": 0,
            "n_tracks": 0,
            "t_min": np.nan,
            "t_max": np.nan,
            "score_mean": np.nan,
            "score_std": np.nan,
            "score_p10": np.nan,
            "score_p50": np.nan,
            "score_p90": np.nan,
            "clamped_mean": np.nan,
        }])

    s = df["score"].to_numpy()
    sc = df["score_clamped"].to_numpy()
    out = {
        "n_rows": int(len(df)),
        "n_tracks": int(df["track_id"].nunique()),
        "t_min": float(df["t"].min()),
        "t_max": float(df["t"].max()),
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
        "score_p10": float(np.percentile(s, 10)),
        "score_p50": float(np.percentile(s, 50)),
        "score_p90": float(np.percentile(s, 90)),
        "clamped_mean": float(np.mean(sc)),
    }
    return pd.DataFrame([out])


def by_track(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "situation", "track_id", "n", "t_min", "t_max",
            "score_mean", "score_std", "score_p50", "clamped_mean"
        ])

    g = df.groupby(["situation", "track_id"], as_index=False).agg(
        n=("score", "size"),
        t_min=("t", "min"),
        t_max=("t", "max"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        score_p50=("score", "median"),
        clamped_mean=("score_clamped", "mean"),
    )
    g["duration_sec"] = g["t_max"] - g["t_min"]
    g = g[g["n"] >= int(min_samples)].copy()
    g = g.sort_values(["score_mean", "n"], ascending=[False, False]).reset_index(drop=True)
    return g


def timeseries_per_sec(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["situation", "sec", "n_tracks", "score_mean", "score_p50", "clamped_mean"])

    df2 = df.copy()
    df2["sec"] = np.floor(df2["t"]).astype(int)

    # 同じ sec の同じ track が複数行あるので、まず track内で sec ごとに平均→その後クラス平均
    per_track_sec = df2.groupby(["situation", "sec", "track_id"], as_index=False).agg(
        score=("score", "mean"),
        clamped=("score_clamped", "mean"),
    )

    ts = per_track_sec.groupby(["situation", "sec"], as_index=False).agg(
        n_tracks=("track_id", "nunique"),
        score_mean=("score", "mean"),
        score_p50=("score", "median"),
        clamped_mean=("clamped", "mean"),
    )
    return ts.sort_values("sec").reset_index(drop=True)


def timeseries_binned(df: pd.DataFrame, bin_sec: float) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "situation", "bin_start", "bin_end", "n_tracks", "score_mean",
            "score_trimmed_mean", "score_p50", "score_std", "low_score_ratio",
            "clamped_mean",
        ])

    df2 = df.copy()
    b = float(bin_sec)
    df2["bin"] = np.floor(df2["t"] / b).astype(int)

    per_track_bin = df2.groupby(["situation", "bin", "track_id"], as_index=False).agg(
        score=("score", "mean"),
        clamped=("score_clamped", "mean"),
        situation_confidence=("situation_confidence", "mean"),
        t_min=("t", "min"),
        t_max=("t", "max"),
    )

    ts = per_track_bin.groupby(["situation", "bin"], as_index=False).agg(
        n_tracks=("track_id", "nunique"),
        score_mean=("score", "mean"),
        score_p50=("score", "median"),
        score_std=("score", "std"),
        situation_confidence=("situation_confidence", "mean"),
        clamped_mean=("clamped", "mean"),
    )
    robust = per_track_bin.groupby(["situation", "bin"])["score"].apply(
        lambda values: _trimmed_mean(values.to_numpy(dtype=float), 0.1)
    )
    low_ratio = per_track_bin.groupby(["situation", "bin"])["score"].apply(
        lambda values: float((values <= 3.0).mean())
    )
    ts = ts.merge(robust.rename("score_trimmed_mean"), on=["situation", "bin"])
    ts = ts.merge(low_ratio.rename("low_score_ratio"), on=["situation", "bin"])
    ts["bin_start"] = ts["bin"] * b
    ts["bin_end"] = (ts["bin"] + 1) * b
    ts = ts[[
        "situation", "bin_start", "bin_end", "n_tracks", "score_mean",
        "score_trimmed_mean", "score_p50", "score_std", "low_score_ratio",
        "situation_confidence", "clamped_mean",
    ]]
    return ts.sort_values(["situation", "bin_start"]).reset_index(drop=True)


def _trimmed_mean(values: np.ndarray, proportion: float) -> float:
    values = np.sort(values[np.isfinite(values)])
    if len(values) == 0:
        return float("nan")
    trim = int(np.floor(len(values) * proportion))
    if trim and 2 * trim < len(values):
        values = values[trim:-trim]
    return float(np.mean(values))


def main():
    args = parse_args()
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(db_path, args.video_path)

    overall = overall_stats(df)
    bytrk = by_track(df, args.min_samples_track)
    ts_sec = timeseries_per_sec(df)
    ts_bin = timeseries_binned(df, args.bin_sec)

    (out_dir / "agg_overall.csv").write_text(overall.to_csv(index=False), encoding="utf-8")
    (out_dir / "agg_by_track.csv").write_text(bytrk.to_csv(index=False), encoding="utf-8")
    (out_dir / "agg_timeseries_sec.csv").write_text(ts_sec.to_csv(index=False), encoding="utf-8")
    (out_dir / "agg_timeseries_bin.csv").write_text(ts_bin.to_csv(index=False), encoding="utf-8")
    xlsx_path = out_dir / "agg_timeseries_sec.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        ts_sec.to_excel(writer, sheet_name="agg_timeseries_sec", index=False)
        ts_bin.to_excel(writer, sheet_name="agg_timeseries_bin", index=False)
        overall.to_excel(writer, sheet_name="overall", index=False)
        bytrk.to_excel(writer, sheet_name="by_track", index=False)

    print("Saved:")
    print(" -", out_dir / "agg_overall.csv")
    print(" -", out_dir / "agg_by_track.csv")
    print(" -", out_dir / "agg_timeseries_sec.csv")
    print(" -", out_dir / "agg_timeseries_bin.csv")
    print(" -", xlsx_path)

    if len(overall):
        row = overall.iloc[0].to_dict()
        print("\nOverall:")
        print(row)

    if len(bytrk):
        print("\nTop tracks (by score_mean):")
        print(bytrk.head(10).to_string(index=False))


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
# python scripts/10_aggregate_predictions.py --db outputs/pred_log.sqlite --out_dir outputs --bin_sec 5
