"""Plot human concentration ratings from a 06_label_gui.py dataset database."""
from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from contextlib import closing
from pathlib import Path


def load_series(db_path: Path, raters: list[str] | None = None,
                track_id: int | None = None) -> tuple[list[dict], dict[int, str]]:
    """Read only; mean across labeled, non-excluded tracks within each window."""
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'videos', 'windows', 'segments', 'labels'} <= tables:
            raise ValueError('videos/windows/segments/labels がある dataset DB を指定してください。')
        if 'window_id' not in {r[1] for r in conn.execute('PRAGMA table_info(segments)')}:
            raise ValueError('旧形式の DB です。migrate_windows_schema.py で移行してください。')
        names = [r[0] for r in conn.execute('SELECT DISTINCT rater FROM labels ORDER BY rater')]
        selected = list(dict.fromkeys(raters)) if raters else names
        unknown = set(selected) - set(names)
        if unknown:
            raise ValueError('評価者が見つかりません: ' + ', '.join(sorted(unknown)))
        if not selected:
            raise ValueError('この DB には集中度の評価がありません。')
        videos = dict(conn.execute('SELECT id, path FROM videos'))
        windows = conn.execute('SELECT id, video_id, t_start, t_end FROM windows ORDER BY video_id,t_start,t_end,id').fetchall()
        predicates = ['l.score IS NOT NULL']
        params = []
        if 'window_skips' in tables:
            predicates.append('NOT EXISTS (SELECT 1 FROM window_skips ws WHERE ws.window_id=s.window_id AND ws.rater=l.rater)')
        if 'excluded_segments' in tables:
            predicates.append('NOT EXISTS (SELECT 1 FROM excluded_segments e WHERE e.segment_id=s.id)')
        if track_id is not None:
            predicates.append('s.track_id=?')
            params.append(track_id)
        rows = conn.execute('''SELECT s.window_id,l.rater,AVG(l.score),COUNT(*)
            FROM labels l JOIN segments s ON s.id=l.segment_id
            WHERE ''' + ' AND '.join(predicates) + ' GROUP BY s.window_id,l.rater', params).fetchall()
        values = {(wid, name): (float(score), int(n)) for wid, name, score, n in rows}
        result = []
        for wid, vid, start, end in windows:
            for name in selected:
                score, count = values.get((wid, name), (None, 0))
                result.append(dict(video_id=vid, window_id=wid, t_start=float(start),
                                   t_end=float(end), rater=name, score=score, n_labeled=count))
        if not any(row['score'] is not None for row in result):
            raise ValueError('指定条件で有効な評価がありません（未評価・スキップ・対象外を除く）。')
        return result, videos


def line_points(rows: list[dict]) -> tuple[list[float], list[float]]:
    """NaNs break the line at unrated windows and at physical time gaps."""
    xs, ys = [], []
    previous_end = None
    for row in rows:
        start, end = row['t_start'], row['t_end']
        if previous_end is not None and start > previous_end + 1e-6:
            xs.append((previous_end + start) / 120.0)
            ys.append(math.nan)
        xs.append((start + end) / 120.0)
        ys.append(row['score'] if row['score'] is not None else math.nan)
        previous_end = end
    return xs, ys


def smooth_values(values: list[float], window: int = 1) -> list[float]:
    """Centered moving mean within each finite run; never bridge missing data."""
    if window < 1 or window % 2 == 0:
        raise ValueError('smooth_window は1以上の奇数を指定してください。')
    result = list(values)
    radius = window // 2
    start = 0
    while start < len(values):
        if not math.isfinite(values[start]):
            start += 1
            continue
        end = start
        while end < len(values) and math.isfinite(values[end]):
            end += 1
        prefix = [0.0]
        for value in values[start:end]:
            prefix.append(prefix[-1] + value)
        for index in range(start, end):
            left = max(start, index - radius)
            right = min(end, index + radius + 1)
            result[index] = (prefix[right - start] - prefix[left - start]) / (right - left)
        start = end
    return result


def write_outputs(rows: list[dict], videos: dict[int, str], out_dir: Path,
                  title: str, track_id: int | None = None, show: bool = False,
                  smooth_window: int = 1) -> list[Path]:
    smooth_values([], smooth_window)  # validate before writing outputs
    import matplotlib
    matplotlib.use('QtAgg' if show else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Meiryo', 'Yu Gothic', 'Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans'):
        if name in available:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'path'
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    csv_path = out_dir / 'ratings_timeseries.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(csv_path)
    for vid in sorted({r['video_id'] for r in rows}):
        subset = [r for r in rows if r['video_id'] == vid]
        if not any(r['score'] is not None for r in subset):
            continue
        names = sorted({r['rater'] for r in subset})
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.get_cmap('tab20')
        styles = ('-', '--', '-.', ':')
        for index, name in enumerate(names):
            series = [r for r in subset if r['rater'] == name]
            xs, ys = line_points(series)
            ys = smooth_values(ys, smooth_window)
            ax.plot(xs, ys, label=name, marker='o', markersize=3,
                    linewidth=1.6, color=colors(index % 20), linestyle=styles[(index // 20) % 4])
        scope = '評価済み児童の平均' if track_id is None else f'児童 track_id={track_id}'
        video_name = Path(videos.get(vid) or str(vid)).name
        smoothing = f' / 移動平均 {smooth_window}区間' if smooth_window > 1 else ''
        ax.set_title(f'{title}\n{video_name} / {scope}{smoothing}')
        ax.set_xlabel('動画内の経過時間（分）〔区間の中央時刻〕')
        ax.set_ylabel('集中度（1～7）')
        ax.set_ylim(0.75, 7.25)
        ax.set_yticks(range(1, 8))
        ax.grid(True, alpha=0.25)
        ax.legend(title='評価者', loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.text(0.02, 0.015, '未評価・スキップ区間は線を接続しません。平均の対象人数は評価者・区間によって異なります。', fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for suffix in ('png', 'svg'):
            path = out_dir / f'concentration_video_{vid}.{suffix}'
            fig.savefig(path, dpi=160)
            outputs.append(path)
        if not show:
            plt.close(fig)
    if show:
        plt.show()
        plt.close('all')
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description='各評価者の集中度を名前付き折れ線グラフで可視化します。')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dataset_root', type=Path, help='dataset フォルダ（既定 DB: db/dataset.sqlite）')
    group.add_argument('--db', type=Path, help='評価 DB または複数評価者をまとめた DB')
    parser.add_argument('--raters', nargs='+', help='表示する評価者名。省略時は全評価者。')
    parser.add_argument('--track_id', type=int, help='指定した児童だけ表示。省略時は区間ごとの平均。')
    parser.add_argument('--output_dir', type=Path, help='保存先。既定: DB の隣の ratings_timeseries フォルダ')
    parser.add_argument('--title', help='グラフのタイトル')
    parser.add_argument('--show', action='store_true', help='保存に加えてグラフのウィンドウを表示')
    parser.add_argument('--smooth_window', type=int, default=1,
                        help='中心移動平均の区間数（1以上の奇数）。1=平滑化なし、3=軽め、5=強め。')
    args = parser.parse_args()
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        parser.error('--smooth_window は1以上の奇数を指定してください。')
    app = None
    if args.dataset_root is None and args.db is None:
        from PySide6.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication([])
        chosen, _ = QFileDialog.getOpenFileName(None, '集中度の評価 DB を選択', str(Path(__file__).resolve().parents[2] / 'datasets'), 'SQLite DB (*.sqlite *.sqlite3 *.db);;All files (*)')
        if not chosen:
            return
        args.db = Path(chosen)
        args.show = True
    db = args.db or args.dataset_root / 'db' / 'dataset.sqlite'
    if not db.is_file():
        parser.error(f'DB が見つかりません: {db}')
    try:
        rows, videos = load_series(db, args.raters, args.track_id)
        title = args.title or (args.dataset_root.name if args.dataset_root else db.stem)
        paths = write_outputs(rows, videos, args.output_dir or db.parent / 'ratings_timeseries', title, args.track_id, args.show, args.smooth_window)
    except (ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    for path in paths:
        print('保存:', path.resolve())


if __name__ == '__main__':
    main()
