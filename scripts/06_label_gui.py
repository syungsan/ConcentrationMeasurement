import argparse
import sqlite3
from pathlib import Path
import yaml
import csv
from datetime import datetime
from typing import Any, Optional, Dict, Tuple, List

from PySide6.QtCore import Qt, QUrl, QTimer, Signal, QObject, QEvent
from PySide6.QtGui import QPixmap, QKeyEvent, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QLineEdit, QMessageBox, QGridLayout, QScrollArea,
    QFrame, QCheckBox, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QSplitter, QSizePolicy, QDialog, QHeaderView, QTabWidget, QButtonGroup
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from lib.situation import SITUATIONS
from lib.research_schema import ensure_research_schema, ensure_blind_assignments
from lib.rating_clips import build_rating_clips, sample_animation_paths


# =========================
# Style
# =========================
APP_QSS = """
QWidget { background: #f3f4f6; color: #111827; font-size: 13px; }
QLabel { color: #111827; }

QPushButton {
  background: #e5e7eb; color: #111827;
  border: 1px solid #9ca3af; border-radius: 8px;
  padding: 4px 6px;
}
QPushButton:hover { background: #dbeafe; border: 1px solid #60a5fa; }
QPushButton:pressed { background: #bfdbfe; border: 1px solid #2563eb; }
QPushButton:checked { background: #dbeafe; border: 2px solid #2563eb; }

QPushButton[primary="true"] {
  padding: 7px 10px;
  border: 2px solid #2563eb;
  background: #dbeafe;
  font-weight: 800;
}

QPushButton[compact="true"] {
  padding: 2px 5px;
  border-radius: 6px;
  font-size: 12px;
}

QLineEdit {
  background: #ffffff; color: #111827;
  border: 1px solid #9ca3af; border-radius: 8px;
  padding: 6px;
}
QComboBox {
  background: #ffffff; color: #111827;
  border: 1px solid #9ca3af; border-radius: 8px;
  padding: 4px;
}
QComboBox QAbstractItemView { background: #ffffff; color: #111827; }

QCheckBox { color: #111827; }

QSlider::groove:horizontal { background: #d1d5db; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #2563eb; width: 14px; margin: -6px 0; border-radius: 7px; }

QScrollArea { background: #ffffff; border: 1px solid #d1d5db; border-radius: 8px; }
QScrollArea QWidget { background: #ffffff; }

QMessageBox { background: #ffffff; color: #111827; }

QSplitter::handle:vertical { height: 12px; background: #d1d5db; }
QSplitter::handle:horizontal { width: 12px; background: #d1d5db; }
"""


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def score_from_key(key: int):
    if Qt.Key_1 <= key <= Qt.Key_7:
        return key - Qt.Key_0
    return None


def score_color(score: int | None) -> str:
    if score is None:
        return "#f9fafb"
    if score <= 2:
        return "#fee2e2"
    if score <= 5:
        return "#fef9c3"
    return "#dcfce7"


def flag_to_int(flag) -> int:
    if hasattr(flag, "value"):
        return int(flag.value)
    try:
        return int(flag)
    except TypeError:
        return 0


def norm_str_path(p: str) -> str:
    return str(p).replace("\\", "/").strip()


def anypath(p: str | None, *, base_dir: Path) -> Optional[Path]:
    """
    - absolute はそのまま
    - relative は base_dir から解決
    """
    if not p:
        return None
    p = norm_str_path(p)
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()


# =========================
# Path resolver (display-side robustness)
# =========================
def resolve_thumb_path(
        thumb_path: str | None,
        *,
        project_root: Path,
        dataset_root: Path,
        db_path: Path,
        crop_dir_cfg: str | None,
) -> Optional[Path]:
    """
    DBに入っている path が、
    - 絶対
    - dataset_root からの相対
    - project_root からの相対
    - db近傍からの相対
    - crop_dir配下の相対
    など色々混ざっても「存在するやつ」を探して返す。
    """
    if not thumb_path:
        return None
    s = norm_str_path(thumb_path)
    p = Path(s)

    candidates: list[Path] = []

    # 1) absolute
    if p.is_absolute():
        candidates.append(p)

    # 2) relative from dataset_root / project_root
    candidates.append((dataset_root / p).resolve())
    candidates.append((project_root / p).resolve())

    # 3) relative from db folder
    candidates.append((db_path.parent / p).resolve())
    candidates.append((db_path.parent.parent / p).resolve())
    candidates.append((db_path.parent.parent.parent / p).resolve())

    # 4) crop_dir explicit
    if crop_dir_cfg:
        crop_dir_abs_ds = (dataset_root / norm_str_path(crop_dir_cfg)).resolve()
        crop_dir_abs_pr = (project_root / norm_str_path(crop_dir_cfg)).resolve()
        candidates.append((crop_dir_abs_ds / p).resolve())
        candidates.append((crop_dir_abs_pr / p).resolve())
        # 末尾ファイル名だけで探す
        candidates.append((crop_dir_abs_ds / Path(p.name)).resolve())
        candidates.append((crop_dir_abs_pr / Path(p.name)).resolve())

    # 5) "../" を雑に剥がして project_root / dataset_root で再解決
    ss = s
    while ss.startswith("../"):
        ss = ss[3:]
        candidates.append((dataset_root / ss).resolve())
        candidates.append((project_root / ss).resolve())

    if ss.startswith("./"):
        candidates.append((dataset_root / ss[2:]).resolve())
        candidates.append((project_root / ss[2:]).resolve())

    # 6) 最後に「親フォルダだけ違う」ケース救済（DBが相対化されていても crop_dir 直下の可能性）
    try:
        candidates.append((dataset_root / Path(p.name)).resolve())
        candidates.append((project_root / Path(p.name)).resolve())
    except Exception:
        pass

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            pass
    return None


# =========================
# Table readability helper
# =========================
def setup_table_readability(tbl: QTableWidget, *, wrap: bool = True, tooltips: bool = True):
    tbl.setTextElideMode(Qt.ElideNone)
    tbl.setWordWrap(wrap)
    tbl.setMouseTracking(True)

    tbl.setSelectionBehavior(QAbstractItemView.SelectItems)
    tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)

    try:
        tbl.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    except Exception:
        pass

    try:
        tbl.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        tbl.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    except Exception:
        pass

    tbl.resizeColumnsToContents()
    tbl.resizeRowsToContents()
    _ = tooltips


def _set_item(
        tbl: QTableWidget,
        r: int,
        c: int,
        text: str,
        *,
        align: Qt.AlignmentFlag | Qt.Alignment = Qt.AlignLeft | Qt.AlignVCenter,
        bg: QColor | None = None,
        enabled_only: bool = True,
        tooltip: bool = True
):
    it = QTableWidgetItem(text)
    it.setTextAlignment(align)
    if enabled_only:
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    if tooltip:
        it.setToolTip(text)
    if bg is not None:
        it.setBackground(QBrush(bg))
    tbl.setItem(r, c, it)


# =========================
# Heatmap colors
# =========================
def _lerp(a: int, b: int, t: float) -> int:
    t = max(0.0, min(1.0, t))
    return int(round(a + (b - a) * t))


def heat_color_from_score(avg: Optional[float]) -> QColor:
    if avg is None:
        return QColor(229, 231, 235)
    t = (float(avg) - 1.0) / 6.0
    if t < 0.5:
        tt = t / 0.5
        r = _lerp(254, 254, tt)
        g = _lerp(226, 249, tt)
        b = _lerp(226, 195, tt)
    else:
        tt = (t - 0.5) / 0.5
        r = _lerp(254, 220, tt)
        g = _lerp(249, 252, tt)
        b = _lerp(195, 231, tt)
    return QColor(r, g, b)


# =========================
# Tile
# =========================
class StudentTile(QFrame):
    clicked = Signal(int)
    focused = Signal(int)

    def __init__(
            self,
            index: int,
            seg_id: int,
            track_id: int,
            thumb_path: str | None,
            *,
            project_root: Path,
            dataset_root: Path,
            db_path: Path,
            crop_dir_cfg: str | None,
            tile_w: int = 190,
            tile_h: int = 118,
            debug_missing: bool = False,
    ):
        super().__init__()
        self.index = index
        self.seg_id = seg_id
        self.track_id = track_id
        self.thumb_path = thumb_path

        self.project_root = project_root
        self.dataset_root = dataset_root
        self.db_path = db_path
        self.crop_dir_cfg = crop_dir_cfg

        self.score: int | None = None
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.debug_missing = debug_missing

        self.frame_paths: list[Path] = []
        self.frame_pix: list[QPixmap] = []
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(250)
        self.anim_timer.timeout.connect(self._anim_step)
        self.anim_i = 0

        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self.thumb = QLabel()
        self.thumb.setFixedSize(self.tile_w, self.tile_h)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("QLabel { background: #ffffff; color: #111827; border-radius: 6px; }")

        self.title = QLabel(f"ID {track_id}")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("QLabel { color: #111827; font-weight: 700; }")

        self.score_lbl = QLabel("集中度: -")
        self.score_lbl.setAlignment(Qt.AlignCenter)
        self.score_lbl.setStyleSheet("QLabel { color: #111827; font-weight: 900; }")

        self.miss_lbl = QLabel("")
        self.miss_lbl.setAlignment(Qt.AlignCenter)
        self.miss_lbl.setWordWrap(True)
        self.miss_lbl.setStyleSheet("QLabel { color: #6b7280; font-size: 11px; }")
        self.miss_lbl.setVisible(self.debug_missing)

        lay = QVBoxLayout()
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        lay.addWidget(self.thumb)
        lay.addWidget(self.title)
        lay.addWidget(self.score_lbl)
        lay.addWidget(self.miss_lbl)
        self.setLayout(lay)

        self.update_thumb()
        self.apply_style(selected=False)

    def set_anim_frames(self, paths: list[str | Path]):
        out: list[Path] = []
        for p in paths:
            if isinstance(p, Path):
                out.append(p)
            else:
                rp = resolve_thumb_path(
                    p,
                    project_root=self.project_root,
                    dataset_root=self.dataset_root,
                    db_path=self.db_path,
                    crop_dir_cfg=self.crop_dir_cfg
                )
                if rp:
                    out.append(rp)
        self.frame_paths = out[:60]
        self.frame_pix = []
        self.anim_i = 0

    def start_anim(self):
        if not self.frame_paths:
            return
        if self.anim_timer.isActive():
            return
        self.anim_i = 0
        self.anim_timer.start()

    def stop_anim(self):
        if self.anim_timer.isActive():
            self.anim_timer.stop()
        self.anim_i = 0
        self.frame_pix = []

    def _anim_step(self):
        if not self.frame_paths:
            self.stop_anim()
            return

        if len(self.frame_pix) != len(self.frame_paths):
            self.frame_pix = [QPixmap(str(p)) for p in self.frame_paths]

        pix = self.frame_pix[self.anim_i % len(self.frame_pix)]
        if not pix.isNull():
            self.thumb.setPixmap(pix.scaled(self.tile_w, self.tile_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.anim_i += 1
        if self.anim_i >= len(self.frame_pix):
            self.stop_anim()
            self.update_thumb()

    def update_thumb(self):
        p = resolve_thumb_path(
            self.thumb_path,
            project_root=self.project_root,
            dataset_root=self.dataset_root,
            db_path=self.db_path,
            crop_dir_cfg=self.crop_dir_cfg
        )
        if p and p.exists():
            pix = QPixmap(str(p))
            if not pix.isNull():
                self.thumb.setPixmap(pix.scaled(self.tile_w, self.tile_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.miss_lbl.setText("")
                return
        self.thumb.setText("(画像なし)")
        if self.debug_missing:
            self.miss_lbl.setText(f"missing:\n{self.thumb_path}")

    def apply_style(self, selected: bool):
        bg = score_color(self.score)
        if selected:
            self.setStyleSheet(f"QFrame {{ background: {bg}; border: 3px solid #2563eb; border-radius: 10px; }}")
        else:
            self.setStyleSheet(f"QFrame {{ background: {bg}; border: 1px solid #9ca3af; border-radius: 10px; }}")

    def set_selected(self, selected: bool):
        self.apply_style(selected)

    def set_score(self, score: int | None):
        self.score = score
        self.score_lbl.setText(f"集中度: {score}" if score is not None else "集中度: -")
        self.apply_style(selected=False)

    def mousePressEvent(self, event):
        self.setFocus(Qt.MouseFocusReason)
        self.clicked.emit(self.index)
        super().mousePressEvent(event)

    def focusInEvent(self, event):
        self.focused.emit(self.index)
        super().focusInEvent(event)


# =========================
# Global key catcher
# =========================
class GlobalKeyCatcher(QObject):
    keyPressed = Signal(int, bool, int)

    def __init__(self):
        super().__init__()
        self.handled = {
            Qt.Key_Space,
            Qt.Key_PageUp, Qt.Key_PageDown,
            Qt.Key_L, Qt.Key_G,
            Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
            Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4,
            Qt.Key_5, Qt.Key_6, Qt.Key_7,
            Qt.Key_Delete, Qt.Key_Backspace,
            Qt.Key_W, Qt.Key_R, Qt.Key_E, Qt.Key_H, Qt.Key_Z,
            Qt.Key_N,
        }

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        e: QKeyEvent = event

        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QComboBox)):
            return False

        key = e.key()
        if key in self.handled:
            self.keyPressed.emit(key, e.isAutoRepeat(), flag_to_int(e.modifiers()))
            return True
        return False


# =========================
# Summary dialog (unchanged logic)
# =========================
class SummaryDialog(QDialog):
    def __init__(self, conn: sqlite3.Connection, db_path: Path, rater: str):
        super().__init__()
        self.setWindowTitle("集計（強化）")
        self.conn = conn
        self.cur = conn.cursor()
        self.db_path = db_path
        self.rater = rater

        self.info = QLabel("")
        self.info.setWordWrap(True)

        self.tabs = QTabWidget()

        self.tab_heat = QWidget()
        self.heat_table = QTableWidget()
        self.heat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.heat_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.heat_table.verticalHeader().setVisible(False)
        setup_table_readability(self.heat_table, wrap=True, tooltips=True)
        self.heat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.heat_table.horizontalHeader().setTextElideMode(Qt.ElideNone)

        lay1 = QVBoxLayout()
        lay1.setContentsMargins(6, 6, 6, 6)
        lay1.addWidget(QLabel("時間帯ヒートマップ（分ごとの平均）"))
        lay1.addWidget(self.heat_table, 1)
        self.tab_heat.setLayout(lay1)

        self.tab_person = QWidget()
        self.person_table = QTableWidget()
        self.person_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.person_table.verticalHeader().setVisible(False)
        setup_table_readability(self.person_table, wrap=False, tooltips=True)
        self.person_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.person_table.horizontalHeader().setStretchLastSection(True)
        self.person_table.horizontalHeader().setTextElideMode(Qt.ElideNone)

        lay2 = QVBoxLayout()
        lay2.setContentsMargins(6, 6, 6, 6)
        lay2.addWidget(QLabel("個人別（ID別）平均/件数/分布（1〜7）"))
        lay2.addWidget(self.person_table, 1)
        self.tab_person.setLayout(lay2)

        self.tab_cross = QWidget()
        self.cross_table = QTableWidget()
        self.cross_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cross_table.verticalHeader().setVisible(False)
        setup_table_readability(self.cross_table, wrap=True, tooltips=True)
        self.cross_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.cross_table.horizontalHeader().setStretchLastSection(True)
        self.cross_table.horizontalHeader().setTextElideMode(Qt.ElideNone)

        lay3 = QVBoxLayout()
        lay3.setContentsMargins(6, 6, 6, 6)
        lay3.addWidget(QLabel("状況 × 個人（平均/件数）"))
        lay3.addWidget(self.cross_table, 1)
        self.tab_cross.setLayout(lay3)

        self.tabs.addTab(self.tab_heat, "時間帯")
        self.tabs.addTab(self.tab_person, "個人別")
        self.tabs.addTab(self.tab_cross, "状況×個人")

        self.btn_export = QPushButton("表示中タブをCSV保存")
        self.btn_export.setProperty("primary", True)
        self.btn_export.clicked.connect(self.export_csv_current_tab)

        lay = QVBoxLayout()
        lay.addWidget(self.info)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(self.btn_export)
        self.setLayout(lay)

        self.refresh()

    def refresh(self):
        all_row = self.cur.execute("""
                                   SELECT COUNT(*), AVG(score), MIN(score), MAX(score)
                                   FROM labels
                                   WHERE rater = ?
                                   """, (self.rater,)).fetchone()

        n_all, avg_all, min_all, max_all = all_row if all_row else (0, None, None, None)

        if avg_all is None:
            self.info.setText(f"DB: {self.db_path}\n評価者: {self.rater}\n総ラベル数: {n_all}")
        else:
            self.info.setText(
                f"DB: {self.db_path}\n評価者: {self.rater}\n総ラベル数: {n_all} / 平均: {avg_all:.2f} / 最小: {min_all} / 最大: {max_all}"
            )

        self.build_heatmap()
        self.build_person_stats()
        self.build_cross()

    def build_heatmap(self):
        rows = self.cur.execute("""
                                WITH base AS (
                                    SELECT
                                        CAST(FLOOR(w.t_start / 60.0) AS INT) AS m,
                                        l.score AS score
                                    FROM labels l
                                             JOIN segments s ON s.id = l.segment_id
                                             JOIN windows w ON w.id = s.window_id
                                             LEFT JOIN window_skips ws
                                                       ON ws.rater = l.rater AND ws.window_id = w.id
                                    WHERE l.rater = ? AND ws.id IS NULL
                                )
                                SELECT m, COUNT(*), AVG(score)
                                FROM base
                                GROUP BY m
                                ORDER BY m
                                """, (self.rater,)).fetchall()

        self.heat_table.setSortingEnabled(False)
        self.heat_table.clear()

        if not rows:
            self.heat_table.setRowCount(1)
            self.heat_table.setColumnCount(1)
            self.heat_table.setHorizontalHeaderLabels([""])
            _set_item(self.heat_table, 0, 0, "データがありません（スキップ除外後）",
                      align=Qt.AlignCenter, enabled_only=True, tooltip=True)
            self.heat_table.resizeColumnsToContents()
            self.heat_table.resizeRowsToContents()
            return

        min_m = int(rows[0][0])
        max_m = int(rows[-1][0])
        total_minutes = max_m - min_m + 1

        cols = 10
        rcount = (total_minutes + cols - 1) // cols

        self.heat_table.setRowCount(rcount)
        self.heat_table.setColumnCount(cols)
        self.heat_table.setHorizontalHeaderLabels([f"+{i}分" for i in range(cols)])

        mp: Dict[int, Tuple[int, Optional[float]]] = {}
        for (m, n, avg) in rows:
            mp[int(m)] = (int(n), float(avg) if avg is not None else None)

        for r in range(rcount):
            for c in range(cols):
                m = min_m + r * cols + c
                if m > max_m:
                    _set_item(self.heat_table, r, c, "", align=Qt.AlignCenter, enabled_only=False, tooltip=False)
                    continue

                n, avg = mp.get(m, (0, None))
                label = f"{m:02d}分\nn={n}\n" + (f"{avg:.2f}" if avg is not None else "-")
                _set_item(
                    self.heat_table, r, c, label,
                    align=Qt.AlignCenter,
                    bg=heat_color_from_score(avg),
                    enabled_only=True,
                    tooltip=True
                )

        self.heat_table.resizeRowsToContents()
        self.heat_table.setSortingEnabled(True)

    def build_person_stats(self):
        rows = self.cur.execute("""
                                WITH base AS (
                                    SELECT s.track_id AS track_id, l.score AS score
                                    FROM labels l
                                             JOIN segments s ON s.id = l.segment_id
                                             JOIN windows w ON w.id = s.window_id
                                             LEFT JOIN window_skips ws
                                                       ON ws.rater = l.rater AND ws.window_id = w.id
                                    WHERE l.rater=? AND ws.id IS NULL
                                )
                                SELECT track_id,
                                       COUNT(*) AS n,
                                       AVG(score) AS avg_score,
                                       MIN(score) AS min_score,
                                       MAX(score) AS max_score
                                FROM base
                                GROUP BY track_id
                                ORDER BY n DESC, track_id
                                """, (self.rater,)).fetchall()

        dist_rows = self.cur.execute("""
                                     WITH base AS (
                                         SELECT s.track_id AS track_id, l.score AS score
                                         FROM labels l
                                                  JOIN segments s ON s.id = l.segment_id
                                                  JOIN windows w ON w.id = s.window_id
                                                  LEFT JOIN window_skips ws
                                                            ON ws.rater = l.rater AND ws.window_id = w.id
                                         WHERE l.rater=? AND ws.id IS NULL
                                     )
                                     SELECT track_id, score, COUNT(*) AS n
                                     FROM base
                                     GROUP BY track_id, score
                                     ORDER BY track_id, score
                                     """, (self.rater,)).fetchall()

        dist: Dict[int, Dict[int, int]] = {}
        for tid, sc, n in dist_rows:
            tid = int(tid)
            sc = int(sc)
            dist.setdefault(tid, {})[sc] = int(n)

        self.person_table.setSortingEnabled(False)
        self.person_table.clear()

        headers = ["ID", "件数", "平均", "最小", "最大"] + [str(i) for i in range(1, 8)]
        self.person_table.setColumnCount(len(headers))
        self.person_table.setHorizontalHeaderLabels(headers)
        self.person_table.setRowCount(len(rows))

        for r, (tid, n, avg_s, min_s, max_s) in enumerate(rows):
            tid = int(tid)
            _set_item(self.person_table, r, 0, str(tid), align=Qt.AlignCenter)
            _set_item(self.person_table, r, 1, str(int(n)), align=Qt.AlignCenter)
            _set_item(self.person_table, r, 2, f"{float(avg_s):.2f}" if avg_s is not None else "-", align=Qt.AlignCenter)
            _set_item(self.person_table, r, 3, str(int(min_s)) if min_s is not None else "-", align=Qt.AlignCenter)
            _set_item(self.person_table, r, 4, str(int(max_s)) if max_s is not None else "-", align=Qt.AlignCenter)

            d = dist.get(tid, {})
            for i in range(1, 8):
                _set_item(self.person_table, r, 4 + i, str(d.get(i, 0)), align=Qt.AlignCenter)

        self.person_table.resizeColumnsToContents()
        self.person_table.resizeRowsToContents()
        self.person_table.setSortingEnabled(True)

    def build_cross(self):
        rows = self.cur.execute("""
                                WITH base AS (
                                    SELECT
                                        s.track_id AS track_id,
                                        COALESCE(w.situation, '(未設定)') AS situation,
                                        l.score AS score
                                    FROM labels l
                                             JOIN segments s ON s.id = l.segment_id
                                             JOIN windows w ON w.id = s.window_id
                                             LEFT JOIN window_skips ws
                                                       ON ws.rater = l.rater AND ws.window_id = w.id
                                    WHERE l.rater = ? AND ws.id IS NULL
                                )
                                SELECT situation, track_id, COUNT(*) AS n, AVG(score) AS avg_score
                                FROM base
                                GROUP BY situation, track_id
                                ORDER BY situation, track_id
                                """, (self.rater,)).fetchall()

        self.cross_table.setSortingEnabled(False)
        self.cross_table.clear()

        if not rows:
            self.cross_table.setRowCount(1)
            self.cross_table.setColumnCount(1)
            self.cross_table.setHorizontalHeaderLabels([""])
            _set_item(self.cross_table, 0, 0, "データがありません（スキップ除外後）",
                      align=Qt.AlignCenter, enabled_only=True, tooltip=True)
            self.cross_table.resizeColumnsToContents()
            self.cross_table.resizeRowsToContents()
            return

        situations = sorted({str(s) for (s, _tid, _n, _avg) in rows})
        tids = sorted({int(tid) for (_s, tid, _n, _avg) in rows})

        mp: Dict[Tuple[str, int], Tuple[int, Optional[float]]] = {}
        for s, tid, n, avg in rows:
            mp[(str(s), int(tid))] = (int(n), float(avg) if avg is not None else None)

        headers = ["状況"] + [f"ID{tid}" for tid in tids]
        self.cross_table.setColumnCount(len(headers))
        self.cross_table.setHorizontalHeaderLabels(headers)
        self.cross_table.setRowCount(len(situations))

        for r, sit in enumerate(situations):
            _set_item(self.cross_table, r, 0, sit, align=Qt.AlignLeft | Qt.AlignVCenter)

            for c, tid in enumerate(tids, start=1):
                n, avg = mp.get((sit, tid), (0, None))
                txt = "-" if avg is None else f"{avg:.2f}\n(n={n})"
                _set_item(
                    self.cross_table, r, c, txt,
                    align=Qt.AlignCenter,
                    bg=heat_color_from_score(avg),
                    enabled_only=True,
                    tooltip=True
                )

        self.cross_table.resizeColumnsToContents()
        self.cross_table.resizeRowsToContents()
        self.cross_table.setSortingEnabled(True)

    def export_csv_current_tab(self):
        out_dir = self.db_path.parent / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        idx = self.tabs.currentIndex()

        if idx == 0:
            table = self.heat_table
            name = "heatmap"
        elif idx == 1:
            table = self.person_table
            name = "person"
        else:
            table = self.cross_table
            name = "cross"

        out_path = out_dir / f"summary_{name}_{self.rater}_{ts}.csv"

        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            headers = [
                table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else ""
                for c in range(table.columnCount())
            ]
            w.writerow(headers)
            for r in range(table.rowCount()):
                row = []
                for c in range(table.columnCount()):
                    it = table.item(r, c)
                    row.append(it.text() if it else "")
                w.writerow(row)

        QMessageBox.information(self, "保存", f"CSV保存しました:\n{out_path}")


# =========================
# App
# =========================
class LabelFastApp(QWidget):
    def __init__(self, cfg: dict, *, project_root: Path, dataset_root: Path, overrides: dict):
        super().__init__()
        self.cfg = cfg
        self.project_root = project_root
        self.dataset_root = dataset_root
        self.task = str(overrides.get("task", "rating"))
        self.initial_name = str(overrides.get("name", ""))
        self.blind_order = bool(overrides.get("blind_order", False))
        self.rating_clip_sec = float(overrides.get("rating_clip_sec", 15.0))

        # ---- resolve paths (config -> overrides) ----
        db_path = overrides.get("db_path") or cfg["paths"]["db_path"]
        proxy_video = overrides.get("proxy_video") or cfg["paths"]["proxy_video"]
        crop_dir_cfg = overrides.get("crop_dir") or cfg["paths"].get("crop_dir", None)

        self.db_path = anypath(db_path, base_dir=self.dataset_root)
        self.proxy_video = anypath(proxy_video, base_dir=self.dataset_root)
        self.crop_dir_cfg = crop_dir_cfg

        if not self.db_path or not self.db_path.exists():
            raise FileNotFoundError(f"db not found: {self.db_path}")
        if not self.proxy_video or not self.proxy_video.exists():
            raise FileNotFoundError(f"proxy video not found: {self.proxy_video}")
        self.proxy_video_plain = self.proxy_video
        proxy_ids = self.proxy_video.with_name("proxy_ids.mp4")
        self.using_baked_track_ids = proxy_ids.exists()
        if self.using_baked_track_ids:
            self.proxy_video = proxy_ids

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cur = self.conn.cursor()
        self.ensure_tables()

        self.undo_stack: list[dict[str, Any]] = []

        title = "共有状況設定" if self.task == "situation" else "集中度ラベラー"
        self.setWindowTitle(f"{title}（dataset切替対応）")

        self.current_window: tuple[float, float] | None = None
        self.tiles: list[StudentTile] = []
        self.selected_idx = -1

        self.tile_w = 200
        self.tile_h = 124
        self.cols = 4

        self.windows: list[tuple[float, float]] = []
        self.clip_segment_ids: dict[int, list[int]] = {}
        self.window_idx: int = -1
        # ---- Player ----
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)

        self.video_widget = QVideoWidget(self)
        self.video_widget.setStyleSheet("background: #000000;")
        self.player.setVideoOutput(self.video_widget)
        self.player.setSource(QUrl.fromLocalFile(str(self.proxy_video)))

        self.btn_play = QPushButton("再生/停止（Space）")
        self.btn_play.setProperty("primary", True)
        self.btn_play.clicked.connect(self.toggle_play)

        self.speed_box = QComboBox()
        self.speed_box.addItems(["0.5x", "1.0x", "1.5x", "2.0x", "4.0x"])
        self.speed_box.setCurrentText("1.0x")
        self.speed_box.currentTextChanged.connect(self.change_speed)

        # --- Slider ---
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.sliderMoved.connect(self.seek)
        self.pos_slider.sliderReleased.connect(self.load_window_from_current_time)
        self.player.durationChanged.connect(self.on_duration)
        self.player.positionChanged.connect(self.on_position)

        self.time_now_lbl = QLabel("00:00")
        self.time_total_lbl = QLabel("00:00")
        self.time_now_lbl.setAlignment(Qt.AlignLeft)
        self.time_total_lbl.setAlignment(Qt.AlignRight)

        self.tick0 = QLabel("00:00")
        self.tick25 = QLabel("00:00")
        self.tick50 = QLabel("00:00")
        self.tick75 = QLabel("00:00")
        self.tick100 = QLabel("00:00")
        for t in (self.tick0, self.tick25, self.tick50, self.tick75, self.tick100):
            t.setStyleSheet("QLabel { color: #374151; font-size: 12px; }")
        self.tick0.setAlignment(Qt.AlignLeft)
        self.tick25.setAlignment(Qt.AlignCenter)
        self.tick50.setAlignment(Qt.AlignCenter)
        self.tick75.setAlignment(Qt.AlignCenter)
        self.tick100.setAlignment(Qt.AlignRight)

        # ---- Loop ----
        self.loop_btn = QPushButton("ループ: ON")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setChecked(True)
        self.loop_btn.setProperty("compact", True)
        self.loop_btn.clicked.connect(self.update_loop_label)

        self.loop_timer = QTimer(self)
        self.loop_timer.setInterval(120)
        self.loop_timer.timeout.connect(self.enforce_loop)
        self.loop_timer.start()

        # ---- Right controls ----
        self.rater_edit = QLineEdit()
        self.rater_edit.setPlaceholderText("評価者（例: teacherA）")
        self.rater_edit.setText(self.initial_name)
        self.rater_edit.editingFinished.connect(self.on_rater_changed)

        self.situation_group = QButtonGroup(self)
        self.situation_group.setExclusive(True)
        self.situation_buttons: dict[str, QPushButton] = {}
        for name in SITUATIONS:
            button = QPushButton(name)
            button.setCheckable(True)
            button.setProperty("compact", True)
            self.situation_group.addButton(button)
            self.situation_buttons[name] = button

        self.cb_auto_advance = QCheckBox("入力後に次へ")
        self.cb_auto_advance.setChecked(True)

        self.cb_only_unlabeled = QCheckBox("未完了区間だけ表示")
        self.cb_only_unlabeled.setChecked(False)
        self.cb_only_unlabeled.stateChanged.connect(self.refresh_window_table)

        self.cb_prioritize_current_situation = QCheckBox("この状況の未入力を優先して回る")
        self.cb_prioritize_current_situation.setChecked(True)

        self.cb_hide_skipped = QCheckBox("スキップ区間を隠す")
        self.cb_hide_skipped.setChecked(True)
        self.cb_hide_skipped.stateChanged.connect(self.refresh_window_table)

        self.btn_refresh_list = QPushButton("一覧更新")
        self.btn_refresh_list.setProperty("compact", True)
        self.btn_refresh_list.clicked.connect(self.refresh_window_table)

        self.cb_debug_thumb = QCheckBox("サムネ欠損デバッグ")
        self.cb_debug_thumb.setChecked(False)
        self.cb_debug_thumb.stateChanged.connect(self.reload_current_window)

        self.info = QLabel("ウィンドウ: -")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("QLabel { font-weight: 800; }")

        self.status = QLabel("状態: -")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("QLabel { color: #374151; }")

        # ---- Window table ----
        self.win_table = QTableWidget()
        self.win_table.setColumnCount(7)
        self.win_table.setHorizontalHeaderLabels(["#", "区間", "入力", "平均", "最小", "最大", "状態"])
        self.win_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.win_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.win_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.win_table.setAlternatingRowColors(True)
        self.win_table.setSortingEnabled(True)
        self.win_table.verticalHeader().setVisible(False)
        self.win_table.horizontalHeader().setStretchLastSection(True)
        self.win_table.cellClicked.connect(self.on_window_table_clicked)
        self.win_table.setMinimumHeight(90)
        self.win_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # navigation buttons
        self.btn_prev = QPushButton("前（PgUp）")
        self.btn_prev.setProperty("compact", True)
        self.btn_prev.clicked.connect(self.prev_window_all)

        self.btn_next = QPushButton("次（PgDn）")
        self.btn_next.setProperty("compact", True)
        self.btn_next.clicked.connect(self.next_window_all)

        self.btn_next_unlabeled = QPushButton("次の未完了（N）")
        self.btn_next_unlabeled.setProperty("primary", True)
        self.btn_next_unlabeled.clicked.connect(self.next_window_unlabeled)

        self.btn_load_here = QPushButton("現在時刻の区間（L）")
        self.btn_load_here.setProperty("compact", True)
        self.btn_load_here.clicked.connect(self.load_window_from_current_time)

        self.btn_go_start = QPushButton("区間先頭へ（G）")
        self.btn_go_start.setProperty("compact", True)
        self.btn_go_start.clicked.connect(self.goto_start)

        self.btn_skip_window = QPushButton("この区間は記録しない（スキップ）")
        self.btn_skip_window.setProperty("primary", True)
        self.btn_skip_window.clicked.connect(self.skip_current_window)

        self.btn_unskip_window = QPushButton("スキップ解除")
        self.btn_unskip_window.setProperty("compact", True)
        self.btn_unskip_window.clicked.connect(self.unskip_current_window)

        self.btn_del_one = QPushButton("選択だけ削除（Del）")
        self.btn_del_one.setProperty("compact", True)
        self.btn_del_one.clicked.connect(self.delete_selected_label)

        self.btn_del_window = QPushButton("この区間を削除（W）")
        self.btn_del_window.setProperty("compact", True)
        self.btn_del_window.clicked.connect(self.delete_current_window_labels)

        self.btn_del_rater = QPushButton("評価者を全削除（R）")
        self.btn_del_rater.setProperty("compact", True)
        self.btn_del_rater.clicked.connect(self.delete_all_labels_for_rater)

        self.btn_undo = QPushButton("取り消し（Ctrl+Z）")
        self.btn_undo.setProperty("compact", True)
        self.btn_undo.clicked.connect(self.undo)

        self.btn_export = QPushButton("CSV出力（E）")
        self.btn_export.setProperty("compact", True)
        self.btn_export.clicked.connect(self.export_csv_for_rater)

        self.btn_history = QPushButton("履歴（H）")
        self.btn_history.setProperty("compact", True)
        self.btn_history.clicked.connect(self.show_history_dialog)

        self.btn_summary = QPushButton("集計を見る")
        self.btn_summary.setProperty("primary", True)
        self.btn_summary.clicked.connect(self.open_summary)

        self.btn_apply_situation = QPushButton("現在区間に保存")
        self.btn_apply_situation.setProperty("primary", True)
        self.btn_apply_situation.clicked.connect(self.apply_situation_current)
        self.btn_apply_situation_selected = QPushButton("一覧の選択区間に一括適用")
        self.btn_apply_situation_selected.clicked.connect(self.apply_situation_selected)
        self.btn_next_unset_situation = QPushButton("次の未設定")
        self.btn_next_unset_situation.clicked.connect(self.next_unset_situation)
        self.btn_lock_situations = QPushButton("全区間の設定完了・ロック")
        self.btn_lock_situations.clicked.connect(self.lock_all_situations)
        self.btn_unlock_situations = QPushButton("ロック解除")
        self.btn_unlock_situations.clicked.connect(self.unlock_all_situations)

        self.help = QLabel(
            "【dataset切替】python scripts/06_label_gui.py --dataset datasets/xxx\n"
            "【操作】シークして離す→区間表示→数字キーで集中度\n"
            "【矢印】タイル選択＋追従スクロール＋パラパラ\n"
            "【スキップ】この区間は記録しない / 解除"
        )
        self.help.setStyleSheet("QLabel { color: #374151; }")

        # ---- Tile grid ----
        self.grid_container = QWidget()
        self.grid = QGridLayout()
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(10)
        self.grid_container.setLayout(self.grid)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.grid_container)
        self.scroll.setMinimumHeight(260)
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ---- Left ----
        left = QVBoxLayout()
        left.setContentsMargins(6, 6, 6, 6)
        left.setSpacing(6)
        left.addWidget(self.video_widget, 1)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(6)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(QLabel("速度"))
        ctrl.addWidget(self.speed_box)
        ctrl.addWidget(self.loop_btn)
        ctrl.addStretch(1)
        left.addLayout(ctrl)

        left.addWidget(self.pos_slider)

        time_row = QHBoxLayout()
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.setSpacing(6)
        time_row.addWidget(self.time_now_lbl)
        time_row.addStretch(1)
        time_row.addWidget(self.time_total_lbl)
        left.addLayout(time_row)

        tick_row = QHBoxLayout()
        tick_row.setContentsMargins(0, 0, 0, 0)
        tick_row.setSpacing(6)
        tick_row.addWidget(self.tick0)
        tick_row.addWidget(self.tick25)
        tick_row.addWidget(self.tick50)
        tick_row.addWidget(self.tick75)
        tick_row.addWidget(self.tick100)
        left.addLayout(tick_row)

        left_w = QWidget()
        left_w.setLayout(left)

        # ---- Right top ----
        top_controls = QVBoxLayout()
        top_controls.setContentsMargins(4, 4, 4, 4)
        top_controls.setSpacing(6)

        row_rater = QHBoxLayout()
        row_rater.setContentsMargins(0, 0, 0, 0)
        row_rater.setSpacing(6)
        self.rater_label = QLabel("評価者")
        row_rater.addWidget(self.rater_label)
        row_rater.addWidget(self.rater_edit, 1)
        top_controls.addLayout(row_rater)

        row_sit = QHBoxLayout()
        row_sit.setContentsMargins(0, 0, 0, 0)
        row_sit.setSpacing(6)
        row_sit.addWidget(QLabel("状況（全評価者で共有）"))
        for name in SITUATIONS:
            row_sit.addWidget(self.situation_buttons[name])
        row_sit.addStretch(1)
        top_controls.addLayout(row_sit)

        self.situation_actions = QWidget()
        situation_actions_lay = QHBoxLayout(self.situation_actions)
        situation_actions_lay.setContentsMargins(0, 0, 0, 0)
        situation_actions_lay.setSpacing(6)
        situation_actions_lay.addWidget(self.btn_apply_situation)
        situation_actions_lay.addWidget(self.btn_apply_situation_selected)
        situation_actions_lay.addWidget(self.btn_next_unset_situation)
        situation_actions_lay.addWidget(self.btn_lock_situations)
        situation_actions_lay.addWidget(self.btn_unlock_situations)
        situation_actions_lay.addStretch(1)
        top_controls.addWidget(self.situation_actions)

        row_mode = QHBoxLayout()
        row_mode.setContentsMargins(0, 0, 0, 0)
        row_mode.setSpacing(10)
        row_mode.addWidget(self.cb_auto_advance)
        row_mode.addWidget(self.cb_prioritize_current_situation)
        row_mode.addWidget(self.cb_only_unlabeled)
        row_mode.addWidget(self.cb_hide_skipped)
        row_mode.addWidget(self.btn_refresh_list)
        row_mode.addStretch(1)
        top_controls.addLayout(row_mode)

        row_debug = QHBoxLayout()
        row_debug.setContentsMargins(0, 0, 0, 0)
        row_debug.setSpacing(8)
        row_debug.addWidget(self.cb_debug_thumb)
        row_debug.addStretch(1)
        top_controls.addLayout(row_debug)

        top_controls.addWidget(self.info)
        top_controls.addWidget(self.status)

        nav1 = QHBoxLayout()
        nav1.setContentsMargins(0, 0, 0, 0)
        nav1.setSpacing(6)
        nav1.addWidget(self.btn_prev)
        nav1.addWidget(self.btn_next)
        nav1.addWidget(self.btn_next_unlabeled)
        nav1.addWidget(self.btn_load_here)
        nav1.addWidget(self.btn_go_start)
        nav1.addStretch(1)
        top_controls.addLayout(nav1)

        nav2 = QHBoxLayout()
        nav2.setContentsMargins(0, 0, 0, 0)
        nav2.setSpacing(6)
        nav2.addWidget(self.btn_skip_window)
        nav2.addWidget(self.btn_unskip_window)
        nav2.addWidget(self.btn_summary)
        nav2.addStretch(1)
        top_controls.addLayout(nav2)

        row_tools = QHBoxLayout()
        row_tools.setContentsMargins(0, 0, 0, 0)
        row_tools.setSpacing(6)
        row_tools.addWidget(self.btn_del_one)
        row_tools.addWidget(self.btn_del_window)
        row_tools.addWidget(self.btn_del_rater)
        row_tools.addSpacing(10)
        row_tools.addWidget(self.btn_undo)
        row_tools.addWidget(self.btn_export)
        row_tools.addWidget(self.btn_history)
        row_tools.addStretch(1)
        top_controls.addLayout(row_tools)

        top_controls_w = QWidget()
        top_controls_w.setLayout(top_controls)

        # ---- Right vertical splitter: table vs tiles ----
        table_box = QWidget()
        table_lay = QVBoxLayout()
        table_lay.setContentsMargins(0, 0, 0, 0)
        table_lay.setSpacing(4)
        table_lay.addWidget(QLabel("区間一覧（window）"))
        table_lay.addWidget(self.win_table, 1)
        table_box.setLayout(table_lay)

        tiles_box = QWidget()
        self.tiles_box = tiles_box
        tiles_lay = QVBoxLayout()
        tiles_lay.setContentsMargins(0, 0, 0, 0)
        tiles_lay.setSpacing(4)
        tiles_lay.addWidget(QLabel("生徒（タイル）"))
        tiles_lay.addWidget(self.scroll, 1)
        tiles_lay.addWidget(self.help)
        tiles_box.setLayout(tiles_lay)

        self.right_vsplit = QSplitter(Qt.Vertical)
        self.right_vsplit.setHandleWidth(14)
        self.right_vsplit.setChildrenCollapsible(False)
        self.right_vsplit.addWidget(table_box)
        self.right_vsplit.addWidget(tiles_box)
        self.right_vsplit.setStretchFactor(0, 1)
        self.right_vsplit.setStretchFactor(1, 4)
        self.right_vsplit.setSizes([220, 820])

        right = QVBoxLayout()
        right.setContentsMargins(6, 6, 6, 6)
        right.setSpacing(6)
        right.addWidget(top_controls_w)
        right.addWidget(self.right_vsplit, 1)

        right_w = QWidget()
        right_w.setLayout(right)

        # ---- Main horizontal splitter ----
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(14)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(left_w)
        self.main_splitter.addWidget(right_w)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([1150, 800])

        root_lay = QHBoxLayout()
        root_lay.setContentsMargins(6, 6, 6, 6)
        root_lay.addWidget(self.main_splitter)
        self.setLayout(root_lay)

        # ---- Global key catcher ----
        self.keycatcher = GlobalKeyCatcher()
        QApplication.instance().installEventFilter(self.keycatcher)
        self.keycatcher.keyPressed.connect(self.handle_key)

        self.configure_task_mode()
        self.refresh_windows_cache()
        self.reload_situation_history()

    # ---------------- DB ----------------
    def ensure_tables(self):
        segment_columns = {str(row[1]) for row in self.cur.execute("PRAGMA table_info(segments)")}
        if "window_id" not in segment_columns:
            raise RuntimeError(
                "旧DBスキーマです。scripts/tools/migrate_windows_schema.py --db <DBパス> "
                "を実行してください。"
            )
        window_columns = {str(row[1]) for row in self.cur.execute("PRAGMA table_info(windows)")}
        additions = {
            "situation_set_by": "TEXT",
            "situation_updated_at": "TEXT",
            "situation_locked": "INTEGER NOT NULL DEFAULT 0 CHECK(situation_locked IN (0, 1))",
        }
        for name, definition in additions.items():
            if name not in window_columns:
                self.cur.execute(f"ALTER TABLE windows ADD COLUMN {name} {definition}")
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS label_events (
                                                                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                     created_at TEXT DEFAULT (datetime('now')),
                             rater TEXT NOT NULL,
                             action TEXT NOT NULL,
                             segment_id INTEGER,
                             old_score INTEGER,
                             new_score INTEGER
                             );
                         """)
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS window_skips (
                                                                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                             created_at TEXT DEFAULT (datetime('now')),
                             rater TEXT NOT NULL,
                             window_id INTEGER NOT NULL,
                             UNIQUE(rater, window_id),
                             FOREIGN KEY(window_id) REFERENCES windows(id) ON DELETE CASCADE
                             );
                         """)
        self.conn.commit()
        ensure_research_schema(self.conn)

    def log_event(self, rater: str, action: str, segment_id: int | None, old_score: int | None, new_score: int | None):
        self.cur.execute(
            "INSERT INTO label_events(rater, action, segment_id, old_score, new_score) VALUES(?,?,?,?,?)",
            (rater, action, segment_id, old_score, new_score)
        )
        self.conn.commit()

    def get_window_id(self, t0: float, t1: float) -> int | None:
        row = self.cur.execute(
            "SELECT id FROM windows WHERE t_start=? AND t_end=? ORDER BY id DESC LIMIT 1",
            (float(t0), float(t1)),
        ).fetchone()
        return int(row[0]) if row else None

    def window_ids_for_interval(self, t0: float, t1: float) -> list[int]:
        return [int(row[0]) for row in self.cur.execute("""
            SELECT id FROM windows
            WHERE ((t_start+t_end)/2.0)>=? AND ((t_start+t_end)/2.0)<?
            ORDER BY t_start
        """, (float(t0), float(t1))).fetchall()]

    def is_window_skipped(self, rater: str, t0: float, t1: float) -> bool:
        window_ids = self.window_ids_for_interval(t0, t1)
        if not window_ids:
            return False
        qmarks = ",".join(["?"] * len(window_ids))
        count = int(self.cur.execute(
            f"SELECT COUNT(*) FROM window_skips WHERE rater=? AND window_id IN ({qmarks})",
            [rater] + window_ids,
        ).fetchone()[0])
        return count == len(window_ids)

    def skip_window(self, rater: str, t0: float, t1: float):
        window_ids = self.window_ids_for_interval(t0, t1)
        if not window_ids:
            return
        self.cur.executemany("""
            INSERT OR REPLACE INTO window_skips(rater, window_id)
            VALUES(?,?)
        """, [
            (rater, window_id)
            for window_id in window_ids
        ])
        self.conn.commit()

    def unskip_window_db(self, rater: str, t0: float, t1: float):
        window_ids = self.window_ids_for_interval(t0, t1)
        if not window_ids:
            return
        qmarks = ",".join(["?"] * len(window_ids))
        self.cur.execute(
            f"DELETE FROM window_skips WHERE rater=? AND window_id IN ({qmarks})",
            [rater] + window_ids,
        )
        self.conn.commit()

    # ---------------- helpers ----------------
    def rater(self) -> str | None:
        r = self.rater_edit.text().strip()
        return r if r else None

    def on_rater_changed(self):
        if self.blind_order:
            self.window_idx = -1
            self.refresh_windows_cache()
        else:
            self.refresh_window_table()
        rater = self.rater()
        if rater and self.current_window:
            self.load_window(*self.current_window, rater, keep_play_state=False)
        else:
            self.refresh_info()

    def configure_task_mode(self):
        is_setup = self.task == "situation"
        self.rater_label.setText("代表者" if is_setup else "評価者")
        self.rater_edit.setPlaceholderText(
            "状況設定の担当者名" if is_setup else "評価者（例: teacherA）"
        )
        self.situation_actions.setVisible(is_setup)
        for button in self.situation_buttons.values():
            button.setEnabled(is_setup)
        self.tiles_box.setVisible(not is_setup)

        rating_widgets = [
            self.cb_auto_advance, self.cb_prioritize_current_situation,
            self.cb_only_unlabeled, self.cb_hide_skipped, self.cb_debug_thumb,
            self.btn_next_unlabeled, self.btn_skip_window, self.btn_unskip_window,
            self.btn_summary, self.btn_del_one, self.btn_del_window,
            self.btn_del_rater, self.btn_undo, self.btn_export, self.btn_history,
        ]
        for widget in rating_widgets:
            widget.setVisible(not is_setup)

        if is_setup:
            self.win_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.right_vsplit.setSizes([1000, 0])

    def window_situation_state(self, t0: float, t1: float):
        exact = self.cur.execute("""
            SELECT id, situation, situation_set_by, situation_updated_at, situation_locked
            FROM windows WHERE t_start=? AND t_end=? ORDER BY id DESC LIMIT 1
        """, (float(t0), float(t1))).fetchone()
        if exact is not None:
            return exact
        rows = self.cur.execute("""
            SELECT id, situation, situation_set_by, situation_updated_at, situation_locked
            FROM windows
            WHERE ((t_start+t_end)/2.0)>=? AND ((t_start+t_end)/2.0)<?
            ORDER BY t_start
        """, (float(t0), float(t1))).fetchall()
        if not rows:
            return None
        situations = {row[1] for row in rows}
        setters = {row[2] for row in rows}
        situation = next(iter(situations)) if len(situations) == 1 else None
        setter = next(iter(setters)) if len(setters) == 1 else None
        locked = int(all(bool(row[4]) for row in rows) and situation is not None)
        return rows[0][0], situation, setter, rows[-1][3], locked

    def _apply_situation_to_indices(self, indices: list[int]):
        representative = self.rater()
        situation = self.situation()
        if not representative:
            QMessageBox.information(self, "代表者", "状況設定の担当者名を入力してください。")
            return
        if situation is None:
            QMessageBox.information(self, "状況", "「聞く」「書く」「話し合う」のいずれかを選択してください。")
            return

        window_ids: list[int] = []
        locked = 0
        for index in sorted(set(indices)):
            if not (0 <= index < len(self.windows)):
                continue
            state = self.window_situation_state(*self.windows[index])
            if not state:
                continue
            if int(state[4]):
                locked += 1
            else:
                window_ids.append(int(state[0]))
        if not window_ids:
            QMessageBox.information(self, "保存", "更新できる未ロック区間がありません。")
            return

        with self.conn:
            self.conn.executemany("""
                UPDATE windows
                SET situation=?, situation_set_by=?, situation_updated_at=datetime('now')
                WHERE id=? AND situation_locked=0
            """, [(situation, representative, window_id) for window_id in window_ids])
        self.refresh_window_table()
        if self.current_window:
            self.load_window(*self.current_window, representative, keep_play_state=False)
        message = f"{len(window_ids)}区間に「{situation}」を保存しました。"
        if locked:
            message += f"\nロック済み {locked}区間は変更していません。"
        self.status.setText(f"状態: {message}")

    def apply_situation_current(self):
        if self.window_idx < 0:
            return
        self._apply_situation_to_indices([self.window_idx])

    def apply_situation_selected(self):
        indices: list[int] = []
        for model_index in self.win_table.selectionModel().selectedRows():
            item = self.win_table.item(model_index.row(), 0)
            if item is not None and item.data(Qt.UserRole) is not None:
                indices.append(int(item.data(Qt.UserRole)))
        if not indices:
            QMessageBox.information(self, "一括適用", "区間一覧で適用先の行を選択してください。")
            return
        self._apply_situation_to_indices(indices)

    def next_unset_situation(self):
        if not self.windows:
            return
        order = list(range(self.window_idx + 1, len(self.windows))) + list(range(0, self.window_idx + 1))
        for index in order:
            state = self.window_situation_state(*self.windows[index])
            if state and (state[1] is None or state[2] is None):
                self.window_idx = index
                self.load_window(*self.windows[index], self.rater() or "", keep_play_state=True)
                return
        QMessageBox.information(self, "設定完了", "状況が未設定の区間はありません。")

    def lock_all_situations(self):
        representative = self.rater()
        if not representative:
            QMessageBox.information(self, "代表者", "状況設定の担当者名を入力してください。")
            return
        unset = int(self.cur.execute(
            "SELECT COUNT(*) FROM windows WHERE situation IS NULL OR situation_set_by IS NULL"
        ).fetchone()[0])
        if unset:
            QMessageBox.information(self, "未設定・未確認あり", f"状況が未設定または代表者未確認の区間が {unset}件あります。")
            return
        ret = QMessageBox.question(
            self, "ロック確認",
            "全区間の共有状況を確定し、評価中の変更を防止しますか？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        with self.conn:
            self.conn.execute("UPDATE windows SET situation_locked=1")
        self.refresh_window_table()
        self.refresh_info()

    def unlock_all_situations(self):
        if not self.rater():
            QMessageBox.information(self, "代表者", "状況設定の担当者名を入力してください。")
            return
        ret = QMessageBox.question(
            self, "ロック解除",
            "評価中の状況変更は学習データの整合性に影響します。ロックを解除しますか？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        with self.conn:
            self.conn.execute("UPDATE windows SET situation_locked=0")
        self.refresh_window_table()
        self.refresh_info()

    def situation(self) -> str | None:
        button = self.situation_group.checkedButton()
        return button.text() if button is not None else None

    def set_situation_preset(self, name: str):
        button = self.situation_buttons.get(name)
        if button is not None:
            button.setChecked(True)

    def clear_situation(self):
        self.situation_group.setExclusive(False)
        for button in self.situation_buttons.values():
            button.setChecked(False)
        self.situation_group.setExclusive(True)

    def reload_situation_history(self):
        pass

    def update_cols_by_viewport(self):
        vp_w = self.scroll.viewport().width()
        cell_w = self.tile_w + 22
        self.cols = max(2, min(10, vp_w // max(1, cell_w)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_window and self.rater():
            old_cols = self.cols
            self.update_cols_by_viewport()
            if self.cols != old_cols:
                self.reload_current_window()

    # ---------------- player ----------------
    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def change_speed(self, text: str):
        self.player.setPlaybackRate(float(text.replace("x", "")))

    def on_duration(self, dur_ms):
        self.pos_slider.setRange(0, dur_ms)
        total_sec = dur_ms / 1000.0
        self.time_total_lbl.setText(fmt_time(total_sec))
        self.tick0.setText(fmt_time(0))
        self.tick25.setText(fmt_time(total_sec * 0.25))
        self.tick50.setText(fmt_time(total_sec * 0.50))
        self.tick75.setText(fmt_time(total_sec * 0.75))
        self.tick100.setText(fmt_time(total_sec))

    def on_position(self, pos_ms):
        if not self.pos_slider.isSliderDown():
            self.pos_slider.setValue(pos_ms)
        self.time_now_lbl.setText(fmt_time(pos_ms / 1000.0))

    def seek(self, pos_ms):
        self.player.setPosition(pos_ms)

    def update_loop_label(self):
        self.loop_btn.setText("ループ: ON" if self.loop_btn.isChecked() else "ループ: OFF")

    def enforce_loop(self):
        if not self.loop_btn.isChecked():
            return
        if not self.current_window:
            return
        t_start, t_end = self.current_window
        t = self.player.position() / 1000.0
        if t > t_end:
            self.player.setPosition(int(t_start * 1000))

    def goto_start(self):
        if not self.current_window:
            return
        t_start, _ = self.current_window
        was_playing = (self.player.playbackState() == QMediaPlayer.PlayingState)
        self.player.setPosition(int(t_start * 1000))
        if was_playing:
            self.player.play()
        self.status.setText(f"状態: 区間先頭へ {t_start:.2f}s")

    # ---------------- window ops ----------------
    def ensure_ready(self) -> bool:
        if self.task == "rating" and not self.rater():
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return False
        if not self.windows:
            QMessageBox.information(self, "区間なし", "segments から区間（window）が見つかりません。")
            return False
        return True

    def refresh_windows_cache(self):
        rater = self.rater()
        rows = self.cur.execute("""
            SELECT t_start, t_end, situation, situation_locked
            FROM windows ORDER BY t_start
        """).fetchall()
        base = [
            (float(t0), float(t1), situation, int(locked))
            for t0, t1, situation, locked in rows
        ]
        if self.task == "rating" and self.rating_clip_sec > 0:
            self.windows = build_rating_clips(base, self.rating_clip_sec)
        else:
            self.windows = [(t0, t1) for t0, t1, _s, _locked in base]

        if self.task == "rating" and self.blind_order and rater:
            ensure_blind_assignments(self.conn, rater)
            ordered: list[tuple[int, tuple[float, float]]] = []
            for clip in self.windows:
                row = self.cur.execute("""
                    SELECT MIN(ba.display_order)
                    FROM blind_assignments ba
                    JOIN segments s ON s.id=ba.segment_id
                    JOIN windows w ON w.id=s.window_id
                    WHERE ba.rater=?
                      AND ((w.t_start+w.t_end)/2.0)>=?
                      AND ((w.t_start+w.t_end)/2.0)<?
                """, (rater, clip[0], clip[1])).fetchone()
                if row and row[0] is not None:
                    ordered.append((int(row[0]), clip))
            if not ordered:
                raise RuntimeError(
                    f"blind order could not be created for rater={rater}"
                )
            self.windows = [clip for _order, clip in sorted(ordered)]

        if self.windows and self.window_idx < 0:
            self.window_idx = 0
        self.refresh_window_table()

        if (self.task == "situation" or self.rater()) and self.windows:
            t0, t1 = self.windows[self.window_idx]
            self.load_window(t0, t1, self.rater() or "", keep_play_state=False)

    def count_unlabeled_windows(self, rater: str) -> int:
        if self.task == "rating" and self.rating_clip_sec > 0:
            remaining = 0
            for t0, t1 in self.windows:
                if self.is_window_skipped(rater, t0, t1):
                    continue
                row = self.cur.execute("""
                    SELECT COUNT(s.id), COUNT(l.id)
                    FROM segments s
                    JOIN windows w ON w.id=s.window_id
                    LEFT JOIN labels l ON l.segment_id=s.id AND l.rater=?
                    WHERE ((w.t_start+w.t_end)/2.0)>=?
                      AND ((w.t_start+w.t_end)/2.0)<?
                """, (rater, t0, t1)).fetchone()
                if row and int(row[1]) < int(row[0]):
                    remaining += 1
            return remaining
        row = self.cur.execute("""
                               SELECT COUNT(*)
                               FROM windows w
                               WHERE NOT EXISTS (
                                   SELECT 1 FROM window_skips ws
                                   WHERE ws.rater=? AND ws.window_id=w.id
                               ) AND EXISTS (
                                   SELECT 1 FROM segments s
                                   LEFT JOIN labels l ON l.segment_id=s.id AND l.rater=?
                                   WHERE s.window_id=w.id AND l.id IS NULL
                               )
                               """, (rater, rater)).fetchone()
        return int(row[0]) if row else 0

    def refresh_info(self):
        if not self.current_window:
            self.info.setText("ウィンドウ: -")
            return

        if self.task == "situation":
            t_start, t_end = self.current_window
            state = self.window_situation_state(t_start, t_end)
            situation = str(state[1]) if state and state[1] else "未設定"
            setter = str(state[2]) if state and state[2] else "-"
            locked = bool(state and state[4])
            unset = int(self.cur.execute(
                "SELECT COUNT(*) FROM windows WHERE situation IS NULL OR situation_set_by IS NULL"
            ).fetchone()[0])
            total = int(self.cur.execute("SELECT COUNT(*) FROM windows").fetchone()[0])
            self.info.setText(
                f"ウィンドウ: {t_start:.2f}-{t_end:.2f}s / 状況={situation} / 設定者={setter}"
            )
            self.status.setText(
                f"状態: 未設定・未確認={unset}/{total} / " + ("ロック済み" if locked else "編集可能")
            )
            return

        r = self.rater()
        t_start, t_end = self.current_window
        total = len(self.tiles)
        done = sum(1 for t in self.tiles if t.score is not None)
        idx_txt = f"{self.window_idx + 1}/{len(self.windows)}" if self.window_idx >= 0 and self.windows else "-"
        skip_txt = ""
        if r and self.is_window_skipped(r, t_start, t_end):
            skip_txt = "  [スキップ]"
        self.info.setText(f"ウィンドウ: {t_start:.2f}-{t_end:.2f}s  入力: {done}/{total}  [#{idx_txt}]{skip_txt}")

        if r:
            remaining = self.count_unlabeled_windows(r)
            sit = self.situation() or "-"
            state = self.window_situation_state(t_start, t_end)
            situation_state = "確定済み" if state and int(state[4]) else "未確定"
            self.status.setText(
                f"状態: dataset='{self.dataset_root.name}' / rater='{r}' / "
                f"状況='{sit}'({situation_state}) / 未完了区間={remaining}"
            )
        else:
            self.status.setText("状態: 評価者を入力してください")

    def refresh_window_table(self):
        if self.task == "situation":
            self.refresh_situation_table()
            return
        r = self.rater()
        only_unlabeled = self.cb_only_unlabeled.isChecked()
        hide_skipped = self.cb_hide_skipped.isChecked()

        self.win_table.setSortingEnabled(False)
        self.win_table.clearContents()
        self.win_table.setRowCount(0)

        if not self.windows:
            self.win_table.setSortingEnabled(True)
            return

        stats = {}
        skipset: set[tuple[float, float]] = set()

        if r:
            skip_rows = self.cur.execute("""
                SELECT w.t_start, w.t_end
                FROM window_skips ws JOIN windows w ON w.id=ws.window_id
                WHERE ws.rater=?
            """, (r,)).fetchall()
            skipset = {(float(a), float(b)) for a, b in skip_rows}

            rows = self.cur.execute("""
                                    SELECT w.t_start, w.t_end, COUNT(s.id) AS total,
                                           COUNT(l.id) AS labeled, AVG(l.score), MIN(l.score), MAX(l.score)
                                    FROM windows w
                                    JOIN segments s ON s.window_id=w.id
                                    LEFT JOIN labels l ON l.segment_id=s.id AND l.rater=?
                                    GROUP BY w.id
                                    ORDER BY w.t_start
                                    """, (r,)).fetchall()

            for t0, t1, total, labeled, avg_s, min_s, max_s in rows:
                stats[(float(t0), float(t1))] = {
                    "total": int(total),
                    "labeled": int(labeled),
                    "avg": (float(avg_s) if avg_s is not None else None),
                    "min": (int(min_s) if min_s is not None else None),
                    "max": (int(max_s) if max_s is not None else None),
                }
        else:
            rows = self.cur.execute("""
                                    SELECT t_start, t_end, COUNT(*) AS total
                                    FROM windows w JOIN segments s ON s.window_id=w.id
                                    GROUP BY w.id
                                    ORDER BY w.t_start
                                    """).fetchall()
            for t0, t1, total in rows:
                stats[(float(t0), float(t1))] = {"total": int(total), "labeled": None, "avg": None, "min": None, "max": None}

        display_rows = []
        for i, (t0, t1) in enumerate(self.windows):
            st = stats.get((t0, t1))
            if st is None:
                if r:
                    row = self.cur.execute("""
                        SELECT COUNT(s.id), COUNT(l.id), AVG(l.score),
                               MIN(l.score), MAX(l.score)
                        FROM segments s
                        JOIN windows w ON w.id=s.window_id
                        LEFT JOIN labels l ON l.segment_id=s.id AND l.rater=?
                        WHERE ((w.t_start+w.t_end)/2.0)>=?
                          AND ((w.t_start+w.t_end)/2.0)<?
                    """, (r, t0, t1)).fetchone()
                    st = {
                        "total": int(row[0]), "labeled": int(row[1]),
                        "avg": float(row[2]) if row[2] is not None else None,
                        "min": int(row[3]) if row[3] is not None else None,
                        "max": int(row[4]) if row[4] is not None else None,
                    }
                else:
                    total = int(self.cur.execute("""
                        SELECT COUNT(s.id) FROM segments s
                        JOIN windows w ON w.id=s.window_id
                        WHERE ((w.t_start+w.t_end)/2.0)>=?
                          AND ((w.t_start+w.t_end)/2.0)<?
                    """, (t0, t1)).fetchone()[0])
                    st = {
                        "total": total, "labeled": None, "avg": None,
                        "min": None, "max": None,
                    }
            total = st["total"]
            labeled = st["labeled"]
            is_skipped = bool(r) and self.is_window_skipped(r, t0, t1)

            if hide_skipped and is_skipped:
                continue

            if only_unlabeled and r:
                if is_skipped:
                    continue
                if labeled is None or labeled >= total:
                    continue

            display_rows.append((i, t0, t1, st, is_skipped))

        self.win_table.setRowCount(len(display_rows))

        for row_i, (win_index, t0, t1, st, is_skipped) in enumerate(display_rows):
            total = st["total"]
            labeled = st["labeled"]
            avg_s = st["avg"]
            min_s = st["min"]
            max_s = st["max"]

            item0 = QTableWidgetItem(str(win_index + 1))
            item0.setData(Qt.UserRole, win_index)
            self.win_table.setItem(row_i, 0, item0)
            self.win_table.setItem(row_i, 1, QTableWidgetItem(f"{t0:.2f}-{t1:.2f}"))

            if labeled is None:
                self.win_table.setItem(row_i, 2, QTableWidgetItem(f"-/{total}"))
                self.win_table.setItem(row_i, 3, QTableWidgetItem("-"))
                self.win_table.setItem(row_i, 4, QTableWidgetItem("-"))
                self.win_table.setItem(row_i, 5, QTableWidgetItem("-"))
            else:
                self.win_table.setItem(row_i, 2, QTableWidgetItem(f"{labeled}/{total}"))
                self.win_table.setItem(row_i, 3, QTableWidgetItem(f"{avg_s:.2f}" if avg_s is not None else "-"))
                self.win_table.setItem(row_i, 4, QTableWidgetItem(str(min_s) if min_s is not None else "-"))
                self.win_table.setItem(row_i, 5, QTableWidgetItem(str(max_s) if max_s is not None else "-"))

            st_item = QTableWidgetItem("スキップ" if is_skipped else "通常")
            if is_skipped:
                st_item.setBackground(QBrush(QColor(229, 231, 235)))
            self.win_table.setItem(row_i, 6, st_item)

        self.win_table.resizeColumnsToContents()
        self.win_table.setSortingEnabled(True)

    def refresh_situation_table(self):
        rows = self.cur.execute("""
            SELECT id, t_start, t_end, situation, situation_set_by,
                   situation_updated_at, situation_locked
            FROM windows ORDER BY t_start
        """).fetchall()
        self.win_table.setSortingEnabled(False)
        self.win_table.clear()
        self.win_table.setColumnCount(4)
        self.win_table.setHorizontalHeaderLabels(["#", "区間", "共有状況", "設定者 / 状態"])
        self.win_table.setRowCount(len(rows))
        index_by_window = {window: i for i, window in enumerate(self.windows)}
        for row_index, (_window_id, t0, t1, situation, setter, updated_at, locked) in enumerate(rows):
            key = (float(t0), float(t1))
            window_index = index_by_window.get(key, row_index)
            item0 = QTableWidgetItem(str(window_index + 1))
            item0.setData(Qt.UserRole, window_index)
            self.win_table.setItem(row_index, 0, item0)
            self.win_table.setItem(row_index, 1, QTableWidgetItem(f"{float(t0):.2f}-{float(t1):.2f}"))
            situation_text = str(situation) if situation else "未設定"
            if situation and not setter:
                situation_text += "（代表者未確認）"
            self.win_table.setItem(row_index, 2, QTableWidgetItem(situation_text))
            state = "ロック" if int(locked) else "編集可"
            detail = f"{setter or '-'} / {state}"
            if updated_at:
                detail += f" / {updated_at}"
            self.win_table.setItem(row_index, 3, QTableWidgetItem(detail))
        self.win_table.resizeColumnsToContents()
        self.win_table.horizontalHeader().setStretchLastSection(True)
        self.win_table.setSortingEnabled(True)

    def on_window_table_clicked(self, row: int, col: int):
        it = self.win_table.item(row, 0)
        if not it:
            return
        idx = it.data(Qt.UserRole)
        if idx is None:
            return
        if not self.ensure_ready():
            return
        self.window_idx = max(0, min(int(idx), len(self.windows) - 1))
        t0, t1 = self.windows[self.window_idx]
        self.load_window(t0, t1, self.rater(), keep_play_state=True)

    def prev_window_all(self):
        if not self.ensure_ready():
            return
        self.window_idx = max(0, self.window_idx - 1)
        t0, t1 = self.windows[self.window_idx]
        self.load_window(t0, t1, self.rater(), keep_play_state=True)

    def next_window_all(self):
        if not self.ensure_ready():
            return
        self.window_idx = min(len(self.windows) - 1, self.window_idx + 1)
        t0, t1 = self.windows[self.window_idx]
        self.load_window(t0, t1, self.rater(), keep_play_state=True)

    def _window_has_unlabeled(self, rater: str, t0: float, t1: float) -> bool:
        row = self.cur.execute("""
                               SELECT 1
                               FROM segments s
                               JOIN windows w ON w.id=s.window_id
                                         LEFT JOIN labels l ON l.segment_id = s.id AND l.rater = ?
                               WHERE w.t_start=? AND w.t_end=? AND l.id IS NULL
                                   LIMIT 1
                               """, (rater, float(t0), float(t1))).fetchone()
        return row is not None

    def _window_matches_situation(self, rater: str, t0: float, t1: float, situation: str) -> bool:
        row = self.cur.execute("""
                               SELECT 1
                               FROM windows w
                               WHERE w.t_start=? AND w.t_end=? AND COALESCE(w.situation,'') = ?
                                    LIMIT 1
                               """, (float(t0), float(t1), situation)).fetchone()
        return row is not None

    def find_next_window_index(self, rater: str, start_idx: int, prefer_situation: bool) -> Optional[int]:
        sit = self.situation() if prefer_situation else None

        if sit:
            for i in range(start_idx, len(self.windows)):
                t0, t1 = self.windows[i]
                if self.is_window_skipped(rater, t0, t1):
                    continue
                if not self._window_has_unlabeled(rater, t0, t1):
                    continue
                if self._window_matches_situation(rater, t0, t1, sit):
                    return i

        for i in range(start_idx, len(self.windows)):
            t0, t1 = self.windows[i]
            if self.is_window_skipped(rater, t0, t1):
                continue
            if self._window_has_unlabeled(rater, t0, t1):
                return i

        return None

    def next_window_unlabeled(self):
        if not self.ensure_ready():
            return
        r = self.rater()
        prefer = self.cb_prioritize_current_situation.isChecked() and bool(self.situation())

        start_idx = max(0, self.window_idx) + 1
        found = self.find_next_window_index(r, start_idx, prefer_situation=prefer)
        if found is None:
            found = self.find_next_window_index(r, 0, prefer_situation=prefer)

        if found is None:
            QMessageBox.information(self, "完了", "未完了の区間がありません（スキップ除外後）。")
            return

        self.window_idx = found
        t0, t1 = self.windows[self.window_idx]
        self.load_window(t0, t1, r, keep_play_state=True)

    def skip_current_window(self):
        if not self.ensure_ready() or not self.current_window:
            return
        r = self.rater()
        t0, t1 = self.current_window

        if self.is_window_skipped(r, t0, t1):
            QMessageBox.information(self, "スキップ", "この区間はすでにスキップされています。")
            return

        self.skip_window(r, t0, t1)
        self.status.setText(f"状態: 区間をスキップしました（{t0:.2f}-{t1:.2f}）")
        self.refresh_info()
        self.refresh_window_table()

    def unskip_current_window(self):
        if not self.ensure_ready() or not self.current_window:
            return
        r = self.rater()
        t0, t1 = self.current_window

        if not self.is_window_skipped(r, t0, t1):
            QMessageBox.information(self, "スキップ解除", "この区間はスキップされていません。")
            return

        self.unskip_window_db(r, t0, t1)
        self.status.setText(f"状態: スキップ解除しました（{t0:.2f}-{t1:.2f}）")
        self.refresh_info()
        self.refresh_window_table()

    def unskip_current_window(self):
        if not self.ensure_ready() or not self.current_window:
            return
        r = self.rater()
        t0, t1 = self.current_window

        if not self.is_window_skipped(r, t0, t1):
            QMessageBox.information(self, "スキップ解除", "この区間はスキップされていません。")
            return

        self.unskip_window_db(r, t0, t1)
        self.status.setText(f"状態: スキップ解除しました（{t0:.2f}-{t1:.2f}）")
        self.refresh_info()
        self.refresh_window_table()

    def load_window_from_current_time(self):
        if not self.ensure_ready():
            return
        r = self.rater()
        t = self.player.position() / 1000.0
        if self.task == "rating":
            match = next(
                ((index, window) for index, window in enumerate(self.windows)
                 if window[0] <= t < window[1]),
                None,
            )
            if match is not None:
                self.window_idx, (t0, t1) = match
                self.load_window(t0, t1, r, keep_play_state=True)
                return
        row = self.cur.execute("""
                               SELECT w.t_start, w.t_end
                               FROM windows w
                               WHERE w.t_start <= ? AND w.t_end > ?
                               ORDER BY w.t_start DESC
                                   LIMIT 1
                               """, (float(t), float(t))).fetchone()
        if row is None:
            QMessageBox.information(self, "区間なし", "この時刻を含む区間がDBにありません。")
            return
        t0, t1 = float(row[0]), float(row[1])
        try:
            self.window_idx = self.windows.index((t0, t1))
        except ValueError:
            pass
        self.load_window(t0, t1, r, keep_play_state=True)

    # ---------------- tiles/window rendering ----------------
    def clear_tiles(self):
        for t in self.tiles:
            try:
                t.stop_anim()
            except Exception:
                pass
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self.tiles = []
        self.selected_idx = -1

    def query_anim_frames_for_segment(self, segment_id: int, t_center: float, limit: int = 8) -> list[str]:
        rows = self.cur.execute("""
                                SELECT crop_path
                                FROM segment_frames
                                WHERE segment_id=?
                                ORDER BY ABS(t-?)
                                    LIMIT ?
                                """, (int(segment_id), float(t_center), int(limit))).fetchall()
        return [r[0] for r in rows if r and r[0]]

    def load_window(self, t_start: float, t_end: float, rater: str, keep_play_state: bool):
        was_playing = (self.player.playbackState() == QMediaPlayer.PlayingState) if keep_play_state else False
        self.current_window = (t_start, t_end)
        mid = (t_start + t_end) / 2.0

        if self.task == "situation":
            self.clear_tiles()
            self.clear_situation()
            state = self.window_situation_state(t_start, t_end)
            if state and state[1]:
                self.set_situation_preset(str(state[1]))
            self.player.setPosition(int(t_start * 1000))
            if was_playing:
                self.player.play()
            self.refresh_info()
            return

        all_seg_rows = self.cur.execute("""
                                    SELECT s.id, s.track_id,
                                           ABS(((w.t_start+w.t_end)/2.0)-?) AS distance
                                    FROM segments s
                                    JOIN windows w ON w.id=s.window_id
                                    WHERE ((w.t_start+w.t_end)/2.0)>=?
                                      AND ((w.t_start+w.t_end)/2.0)<?
                                    ORDER BY s.track_id, distance
                                    """, (float(mid), float(t_start), float(t_end))).fetchall()
        grouped_segments: dict[int, list[int]] = {}
        for segment_id, track_id, _distance in all_seg_rows:
            grouped_segments.setdefault(int(track_id), []).append(int(segment_id))
        seg_rows = [
            (segment_ids[0], track_id)
            for track_id, segment_ids in sorted(grouped_segments.items())
        ]
        self.clip_segment_ids = {
            segment_ids[0]: segment_ids
            for segment_ids in grouped_segments.values()
        }

        self.clear_tiles()
        self.update_cols_by_viewport()

        self.clear_situation()
        if seg_rows:
            first_seg = int(seg_rows[0][0])
            meta = self.cur.execute("""
                                    SELECT w.situation
                                    FROM segments s
                                    JOIN windows w ON w.id=s.window_id
                                    WHERE s.id=?
                                        LIMIT 1
                                    """, (first_seg,)).fetchone()
            if meta:
                sit = meta[0]
                if sit:
                    self.set_situation_preset(str(sit))

        for i, (seg_id, track_id) in enumerate(seg_rows):
            seg_id = int(seg_id)
            track_id = int(track_id)

            segment_ids = self.clip_segment_ids.get(seg_id, [seg_id])
            qmarks = ",".join(["?"] * len(segment_ids))
            thumb = self.cur.execute(f"""
                                     SELECT crop_path
                                     FROM segment_frames
                                     WHERE segment_id IN ({qmarks})
                                     ORDER BY ABS(t-?)
                                     LIMIT 1
                                     """, segment_ids + [float(mid)]).fetchone()
            thumb_path = thumb[0] if thumb else None

            tile = StudentTile(
                index=i,
                seg_id=seg_id,
                track_id=track_id,
                thumb_path=thumb_path,
                project_root=self.project_root,
                dataset_root=self.dataset_root,
                db_path=self.db_path,
                crop_dir_cfg=self.crop_dir_cfg,
                tile_w=self.tile_w,
                tile_h=self.tile_h,
                debug_missing=self.cb_debug_thumb.isChecked(),
            )
            tile.clicked.connect(self.on_tile_clicked)
            tile.focused.connect(self.on_tile_focused)

            labels = self.cur.execute(
                f"SELECT score FROM labels WHERE rater=? AND segment_id IN ({qmarks})",
                [rater] + segment_ids,
            ).fetchall()
            scores = [int(row[0]) for row in labels]
            if len(scores) == len(segment_ids) and len(set(scores)) == 1:
                tile.set_score(scores[0])

            r_i = i // self.cols
            c_i = i % self.cols
            self.grid.addWidget(tile, r_i, c_i)
            self.tiles.append(tile)

        self.refresh_info()
        self.refresh_window_table()

        self.player.setPosition(int(t_start * 1000))
        if was_playing:
            self.player.play()

        first = next((i for i, t in enumerate(self.tiles) if t.score is None), 0)
        self.select_tile(first, trigger_anim=True)

    def reload_current_window(self):
        r = self.rater()
        if not r or not self.current_window:
            return
        t0, t1 = self.current_window
        self.load_window(t0, t1, r, keep_play_state=True)

    def on_tile_clicked(self, idx: int):
        self.select_tile(idx, trigger_anim=True)

    def on_tile_focused(self, idx: int):
        self._trigger_tile_anim(idx)

    def _trigger_tile_anim(self, idx: int):
        if idx < 0 or idx >= len(self.tiles) or not self.current_window:
            return
        t_start, t_end = self.current_window
        tile = self.tiles[idx]
        segment_ids = self.clip_segment_ids.get(int(tile.seg_id), [int(tile.seg_id)])
        qmarks = ",".join(["?"] * len(segment_ids))
        rows = self.cur.execute(f"""
            SELECT t, crop_path FROM segment_frames
            WHERE segment_id IN ({qmarks})
              AND t>=? AND t<?
            ORDER BY t
        """, segment_ids + [float(t_start), float(t_end)]).fetchall()
        paths = sample_animation_paths(rows, t_start, t_end, fps=2.0, max_frames=60)
        tile.set_anim_frames(paths)
        tile.start_anim()

    def select_tile(self, idx: int, trigger_anim: bool):
        if not self.tiles:
            self.selected_idx = -1
            return
        idx = max(0, min(idx, len(self.tiles) - 1))
        self.selected_idx = idx

        for i, t in enumerate(self.tiles):
            t.set_selected(i == idx)
            if i != idx:
                t.stop_anim()

        try:
            w = self.tiles[idx]
            self.scroll.ensureWidgetVisible(w, 30, 30)
            w.setFocus(Qt.OtherFocusReason)
        except Exception:
            pass

        if trigger_anim:
            self._trigger_tile_anim(idx)

    def move_selection(self, dx: int, dy: int):
        if not self.tiles:
            return
        if self.selected_idx < 0:
            self.select_tile(0, trigger_anim=True)
            return

        r = self.selected_idx // self.cols
        c = self.selected_idx % self.cols
        r2 = max(0, r + dy)
        c2 = max(0, min(self.cols - 1, c + dx))
        idx2 = r2 * self.cols + c2
        idx2 = max(0, min(idx2, len(self.tiles) - 1))
        self.select_tile(idx2, trigger_anim=True)

        step = 70
        if dx != 0:
            bar = self.scroll.horizontalScrollBar()
            bar.setValue(bar.value() + step * dx)
        if dy != 0:
            bar = self.scroll.verticalScrollBar()
            bar.setValue(bar.value() + step * dy)

    def auto_advance(self):
        if not self.tiles:
            return
        start = self.selected_idx + 1
        for i in range(start, len(self.tiles)):
            if self.tiles[i].score is None:
                self.select_tile(i, trigger_anim=True)
                return
        for i in range(0, len(self.tiles)):
            if self.tiles[i].score is None:
                self.select_tile(i, trigger_anim=True)
                return

    # ---------------- label operations ----------------
    def push_undo(self, action: str, payload: dict[str, Any]):
        self.undo_stack.append({"action": action, "payload": payload})
        if len(self.undo_stack) > 50:
            self.undo_stack = self.undo_stack[-50:]

    def get_existing_score(self, seg_id: int, rater: str) -> int | None:
        row = self.cur.execute(
            "SELECT score FROM labels WHERE segment_id=? AND rater=? LIMIT 1",
            (int(seg_id), rater)
        ).fetchone()
        return int(row[0]) if row else None

    def save_score(self, score: int):
        if not 1 <= score <= 7:
            return
        r = self.rater()
        if not r:
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return
        if self.selected_idx < 0 or self.selected_idx >= len(self.tiles):
            return
        if self.current_window and self.is_window_skipped(r, self.current_window[0], self.current_window[1]):
            QMessageBox.information(self, "スキップ", "この区間はスキップです。解除してから入力してください。")
            return

        tile = self.tiles[self.selected_idx]
        seg_id = int(tile.seg_id)
        segment_ids = self.clip_segment_ids.get(seg_id, [seg_id])
        qmarks = ",".join(["?"] * len(segment_ids))
        old_rows = self.cur.execute(
            f"SELECT segment_id, score FROM labels WHERE rater=? AND segment_id IN ({qmarks})",
            [r] + segment_ids,
        ).fetchall()
        old_by_segment = {
            int(segment_id): int(old_score)
            for segment_id, old_score in old_rows
        }

        state = self.window_situation_state(*self.current_window) if self.current_window else None
        sit = str(state[1]) if state and state[1] else None
        if sit is None:
            QMessageBox.information(self, "共有状況未設定", "代表者がこの区間の共有状況を設定するまで評価できません。")
            return
        if not bool(state[4]):
            QMessageBox.information(self, "共有状況未確定", "代表者が全区間の状況設定を完了し、ロックするまで評価できません。")
            return
        self.push_undo("clip_upsert", {
            "segment_ids": segment_ids,
            "old_by_segment": old_by_segment,
            "new_score": score,
        })

        tile.set_score(score)
        self.cur.executemany("""
            INSERT INTO labels(segment_id, rater, score)
            VALUES(?,?,?)
            ON CONFLICT(segment_id, rater) DO UPDATE SET
                score=excluded.score,
                updated_at=datetime('now')
        """, [
            (target_segment, r, int(score))
            for target_segment in segment_ids
        ])
        self.cur.executemany("""
            INSERT INTO label_observations(
              segment_id, rater, source_clip_start, source_clip_end,
              source_clip_sec, updated_at
            ) VALUES(?,?,?,?,?,datetime('now'))
            ON CONFLICT(segment_id,rater) DO UPDATE SET
              source_clip_start=excluded.source_clip_start,
              source_clip_end=excluded.source_clip_end,
              source_clip_sec=excluded.source_clip_sec,
              updated_at=datetime('now')
        """, [(
            target_segment, r,
            float(self.current_window[0]), float(self.current_window[1]),
            float(self.current_window[1] - self.current_window[0]),
        ) for target_segment in segment_ids])
        self.conn.commit()

        self.reload_situation_history()
        self.log_event(r, "clip_upsert", seg_id, None, score)

        self.refresh_info()
        self.refresh_window_table()
        if self.cb_auto_advance.isChecked():
            self.auto_advance()

    def delete_selected_label(self):
        r = self.rater()
        if not r:
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return
        if self.selected_idx < 0 or self.selected_idx >= len(self.tiles):
            return

        tile = self.tiles[self.selected_idx]
        seg_id = int(tile.seg_id)
        segment_ids = self.clip_segment_ids.get(seg_id, [seg_id])
        qmarks = ",".join(["?"] * len(segment_ids))
        rows = self.cur.execute(
            f"SELECT segment_id,score FROM labels WHERE rater=? AND segment_id IN ({qmarks})",
            [r] + segment_ids,
        ).fetchall()
        if not rows:
            return

        ret = QMessageBox.question(
            self, "確認",
            f"選択中（ID {tile.track_id} / Segment {seg_id}）の集中度を削除しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return

        rows = [(int(a), int(b)) for a, b in rows]
        self.push_undo("delete_window", {"rows": rows})
        self.cur.execute(
            f"DELETE FROM labels WHERE rater=? AND segment_id IN ({qmarks})",
            [r] + segment_ids,
        )
        self.cur.execute(
            f"DELETE FROM label_observations WHERE rater=? AND segment_id IN ({qmarks})",
            [r] + segment_ids,
        )
        self.conn.commit()
        self.log_event(r, "delete", seg_id, None, None)

        tile.set_score(None)
        self.refresh_info()
        self.refresh_window_table()

    def delete_current_window_labels(self):
        r = self.rater()
        if not r:
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return
        if not self.current_window:
            return
        t_start, t_end = self.current_window

        ret = QMessageBox.question(
            self, "確認",
            f"この区間（{t_start:.2f}-{t_end:.2f}s）の全ラベルを削除しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return

        seg_ids = sorted({
            segment_id
            for tile in self.tiles
            for segment_id in self.clip_segment_ids.get(int(tile.seg_id), [int(tile.seg_id)])
        })
        if not seg_ids:
            return

        qmarks = ",".join(["?"] * len(seg_ids))
        rows = self.cur.execute(
            f"SELECT segment_id, score FROM labels WHERE rater=? AND segment_id IN ({qmarks})",
            [r] + seg_ids
        ).fetchall()
        rows = [(int(a), int(b)) for (a, b) in rows]
        self.push_undo("delete_window", {"rows": rows})

        self.cur.execute(f"DELETE FROM labels WHERE rater=? AND segment_id IN ({qmarks})", [r] + seg_ids)
        self.cur.execute(
            f"DELETE FROM label_observations WHERE rater=? AND segment_id IN ({qmarks})",
            [r] + seg_ids,
        )
        self.conn.commit()
        self.log_event(r, "bulk_delete", None, None, None)

        for t in self.tiles:
            t.set_score(None)
        self.refresh_info()
        self.refresh_window_table()

    def delete_all_labels_for_rater(self):
        r = self.rater()
        if not r:
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return

        ret = QMessageBox.question(
            self, "確認",
            f"評価者='{r}' のラベル・メモ・スキップ・操作履歴を全削除しますか？\n"
            "この操作は取り消せません。\n"
            "授業ウィンドウの共有状況は、他の評価者も使うため残ります。",
            QMessageBox.Yes | QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return

        with self.conn:
            self.cur.execute("DELETE FROM labels WHERE rater=?", (r,))
            self.cur.execute("DELETE FROM label_observations WHERE rater=?", (r,))
            self.cur.execute("DELETE FROM window_skips WHERE rater=?", (r,))
            self.cur.execute("DELETE FROM label_events WHERE rater=?", (r,))
        self.undo_stack.clear()

        self.reload_current_window()
        self.refresh_window_table()
        QMessageBox.information(self, "削除完了", f"評価者='{r}' の個人データを全削除しました。")

    def undo(self):
        if not self.undo_stack:
            QMessageBox.information(self, "取り消し", "取り消せる操作がありません。")
            return
        r = self.rater()
        if not r:
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return

        item = self.undo_stack.pop()
        action = item["action"]
        payload = item["payload"]

        if action == "clip_upsert":
            segment_ids = [int(value) for value in payload["segment_ids"]]
            old_by_segment = payload.get("old_by_segment", {})
            with self.conn:
                for segment_id in segment_ids:
                    old = old_by_segment.get(segment_id)
                    if old is None:
                        self.cur.execute(
                            "DELETE FROM labels WHERE segment_id=? AND rater=?",
                            (segment_id, r),
                        )
                    else:
                        old_score = old
                        self.cur.execute("""
                            INSERT INTO labels(segment_id,rater,score)
                            VALUES(?,?,?)
                            ON CONFLICT(segment_id,rater) DO UPDATE SET
                              score=excluded.score,
                              updated_at=datetime('now')
                        """, (segment_id, r, int(old_score)))
            self.log_event(r, "undo", None, payload.get("new_score"), None)
            self.reload_current_window()
            self.refresh_window_table()
            return

        if action == "upsert":
            seg_id = int(payload["segment_id"])
            old_score = payload.get("old_score", None)
            new_score = payload.get("new_score", None)

            if old_score is None:
                self.cur.execute("DELETE FROM labels WHERE segment_id=? AND rater=?", (seg_id, r))
            else:
                self.cur.execute(
                    "INSERT OR REPLACE INTO labels(segment_id, rater, score) VALUES(?,?,?)",
                    (seg_id, r, int(old_score))
                )
            self.conn.commit()
            self.log_event(r, "undo", seg_id, new_score, old_score)
            self.reload_current_window()
            self.refresh_window_table()
            return

        if action == "delete_one":
            seg_id = int(payload["segment_id"])
            old_score = payload.get("old_score", None)
            if old_score is None:
                return
            self.cur.execute(
                "INSERT OR REPLACE INTO labels(segment_id, rater, score) VALUES(?,?,?)",
                (seg_id, r, int(old_score))
            )
            self.conn.commit()
            self.log_event(r, "undo", seg_id, None, old_score)
            self.reload_current_window()
            self.refresh_window_table()
            return

        if action in ("delete_window", "delete_rater"):
            rows = payload.get("rows", [])
            for seg_id, score in rows:
                self.cur.execute(
                    "INSERT OR REPLACE INTO labels(segment_id, rater, score) VALUES(?,?,?)",
                    (int(seg_id), r, int(score))
                )
            self.conn.commit()
            self.log_event(r, "undo", None, None, None)
            self.reload_current_window()
            self.refresh_window_table()
            return

        QMessageBox.information(self, "取り消し", f"未対応のundo: {action}")

    def export_csv_for_rater(self):
        r = self.rater()
        if not r:
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return

        v = self.cur.execute("SELECT id FROM videos ORDER BY id DESC LIMIT 1").fetchone()
        video_id = int(v[0]) if v else None

        rows = self.cur.execute("""
                                SELECT l.segment_id, l.score, l.created_at,
                                       s.track_id, w.t_start, w.t_end,
                                       COALESCE(w.situation, ''),
                                       CASE WHEN ws.id IS NULL THEN 0 ELSE 1 END AS skipped
                                FROM labels l
                                         JOIN segments s ON s.id = l.segment_id
                                         JOIN windows w ON w.id=s.window_id
                                         LEFT JOIN window_skips ws ON ws.rater = l.rater AND ws.window_id=w.id
                                WHERE l.rater = ?
                                ORDER BY w.t_start, s.track_id
                                """, (r,)).fetchall()

        export_dir = self.db_path.parent / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = export_dir / f"labels_{r}_{ts}.csv"

        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rater", "video_id", "segment_id", "track_id", "t_start", "t_end", "score", "situation", "skipped_window", "created_at"])
            for seg_id, score, created_at, track_id, t_start, t_end, sit, skipped in rows:
                w.writerow([r, video_id, int(seg_id), int(track_id), float(t_start), float(t_end), int(score), sit, int(skipped), created_at])

        self.log_event(r, "export", None, None, None)
        QMessageBox.information(self, "CSV出力", f"CSVを書き出しました:\n{out_path}")

    def show_history_dialog(self):
        r = self.rater()
        if not r:
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return

        rows = self.cur.execute("""
                                SELECT created_at, action, segment_id, old_score, new_score
                                FROM label_events
                                WHERE rater=?
                                ORDER BY id DESC
                                    LIMIT 60
                                """, (r,)).fetchall()

        lines = []
        for created_at, action, seg_id, old_s, new_s in rows:
            lines.append(f"{created_at} | {action:10s} | seg={seg_id} | {old_s}->{new_s}")
        text = "\n".join(lines) if lines else "(履歴なし)"
        QMessageBox.information(self, "履歴（最新60件）", text)

    def open_summary(self):
        r = self.rater()
        if not r:
            QMessageBox.information(self, "評価者", "評価者名を入力してください。")
            return
        dlg = SummaryDialog(self.conn, self.db_path, r)
        dlg.resize(980, 620)
        dlg.exec()

    # ---------------- keys ----------------
    def handle_key(self, key: int, is_auto_repeat: bool, modifiers_int: int):
        if is_auto_repeat:
            return

        ctrl_mask = flag_to_int(Qt.ControlModifier)
        if key == Qt.Key_Z and (modifiers_int & ctrl_mask):
            if self.task == "rating":
                self.undo()
            return

        if key == Qt.Key_Space:
            self.toggle_play()
            return

        if key == Qt.Key_PageUp:
            self.prev_window_all()
            return
        if key == Qt.Key_PageDown:
            self.next_window_all()
            return
        if key == Qt.Key_N:
            if self.task == "situation":
                self.next_unset_situation()
            else:
                self.next_window_unlabeled()
            return
        if key == Qt.Key_L:
            self.load_window_from_current_time()
            return
        if key == Qt.Key_G:
            self.goto_start()
            return

        if self.task == "situation":
            return

        if key == Qt.Key_Left:
            self.move_selection(-1, 0)
            return
        if key == Qt.Key_Right:
            self.move_selection(+1, 0)
            return
        if key == Qt.Key_Up:
            self.move_selection(0, -1)
            return
        if key == Qt.Key_Down:
            self.move_selection(0, +1)
            return

        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_label()
            return
        if key == Qt.Key_W:
            self.delete_current_window_labels()
            return
        if key == Qt.Key_R:
            self.delete_all_labels_for_rater()
            return
        if key == Qt.Key_E:
            self.export_csv_for_rater()
            return
        if key == Qt.Key_H:
            self.show_history_dialog()
            return

        s = score_from_key(key)
        if s is not None:
            self.save_score(s)
            return


# =========================
# CLI helpers
# =========================
def guess_paths_from_dataset_dir(dataset_dir: Path) -> dict:
    """
    dataset_dir を渡すだけで動くように「よくある配置」を探索して決める。
    """
    d = dataset_dir.resolve()

    db_candidates = [
        d / "data/db/dataset.sqlite",
        d / "db/dataset.sqlite",
        d / "dataset.sqlite",
        ]
    vid_candidates = [
        d / "data/videos/proxy.mp4",
        d / "videos/proxy.mp4",
        d / "proxy.mp4",
        ]
    crop_candidates = [
        "data/assets/crops",
        "assets/crops",
        "crops",
    ]

    out = {}
    for p in db_candidates:
        if p.exists():
            out["db_path"] = str(p)
            break
    for p in vid_candidates:
        if p.exists():
            out["proxy_video"] = str(p)
            break

    # crop_dir は「存在確認できたら」採用（相対で持つ）
    for rel in crop_candidates:
        if (d / rel).exists():
            out["crop_dir"] = rel
            break

    return out


def main():
    here = Path(__file__).resolve()
    project_root = here.parent.parent  # focus_dataset/

    cfg = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="授業フォルダ（例: datasets/miyazaki）。ここを基準に相対パス解決します。")
    ap.add_argument("--db", type=str, default=None, help="DBパスを直接指定（--dataset より優先）")
    ap.add_argument("--video", type=str, default=None, help="proxy.mp4 を直接指定（--dataset より優先）")
    ap.add_argument("--crop_dir", type=str, default=None, help="crop_dir を上書き（例: data/assets/crops）")
    ap.add_argument("--task", choices=["situation", "rating"], default="rating",
                    help="situation=代表者の共有状況設定 / rating=集中度評価")
    ap.add_argument("--name", default="", help="代表者名または評価者名の初期値")
    ap.add_argument(
        "--blind-order", action="store_true",
        help="評価者ごとに事前生成した盲検ランダム順を使用する",
    )
    ap.add_argument(
        "--rating-clip-sec", type=float, default=15.0,
        help="採点時にまとめて視聴する秒数。内部ラベルは5秒segmentへ展開する",
    )
    args = ap.parse_args()

    # dataset_root を決定（未指定なら project_root を基準に旧運用）
    dataset_root = (project_root / args.dataset).resolve() if args.dataset else project_root

    overrides = {}
    overrides["task"] = args.task
    overrides["name"] = args.name
    overrides["blind_order"] = args.blind_order
    overrides["rating_clip_sec"] = args.rating_clip_sec
    if args.dataset:
        overrides.update(guess_paths_from_dataset_dir(dataset_root))

    if args.db:
        overrides["db_path"] = str((Path(args.db) if Path(args.db).is_absolute() else (dataset_root / args.db)).resolve())
    if args.video:
        overrides["proxy_video"] = str((Path(args.video) if Path(args.video).is_absolute() else (dataset_root / args.video)).resolve())
    if args.crop_dir:
        overrides["crop_dir"] = args.crop_dir

    app = QApplication([])
    app.setStyleSheet(APP_QSS)

    w = LabelFastApp(cfg, project_root=project_root, dataset_root=dataset_root, overrides=overrides)
    w.resize(1880, 1060)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()

# command
# 授業ごとのフォルダを指定
# python scripts/06_label_gui.py --dataset datasets/lesson_001 --task situation
# python scripts/06_label_gui.py --dataset datasets/lesson_001 --task rating

# DB/動画を明示（フォルダ構造がバラバラでもOK）
# python scripts/06_label_gui.py --dataset datasets/lesson_001 --db db/dataset.sqlite --video videos/proxy.mp4
