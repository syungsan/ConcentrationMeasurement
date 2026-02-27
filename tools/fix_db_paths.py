#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Optional, Iterable, Tuple


def norm_db_str(p: str) -> str:
    # DB上は posixっぽく揃える（Windowsの \ 対策）
    return str(p).replace("\\", "/").strip()


def is_abs_like(s: str) -> bool:
    # Path.is_absolute は "C:foo" を false にするので雑に補強
    ss = norm_db_str(s)
    if ss.startswith("/"):
        return True
    if len(ss) >= 3 and ss[1] == ":" and ss[2] in ("/", "\\"):
        return True
    return False


def to_posix_rel(path: Path) -> str:
    return path.as_posix()


def try_relativize(p_str: str, base: Path) -> Optional[str]:
    """
    p_str が base 配下なら base からの相対にして返す。
    """
    s = norm_db_str(p_str)
    try:
        pp = Path(s)
        # pp が絶対ならそのまま resolve できるが、DBに相対が入ってる場合もある
        if not pp.is_absolute():
            return None
        base_r = base.resolve()
        pp_r = pp.resolve()
        try:
            rel = pp_r.relative_to(base_r)
            return to_posix_rel(rel)
        except Exception:
            return None
    except Exception:
        return None


def try_rebase(p_str: str, old_root: Path, new_root: Path) -> Optional[str]:
    """
    p_str が old_root 配下の absolute なら、
    new_root / relative(old_root) に付け替えて absolute を返す。
    """
    s = norm_db_str(p_str)
    if not is_abs_like(s):
        return None
    try:
        pp = Path(s).resolve()
        old_r = old_root.resolve()
        try:
            rel = pp.relative_to(old_r)
        except Exception:
            return None
        new_abs = (new_root.resolve() / rel).resolve()
        return norm_db_str(str(new_abs))
    except Exception:
        return None


def iter_rows(cur: sqlite3.Cursor, sql: str, params=()) -> Iterable[Tuple]:
    for row in cur.execute(sql, params):
        yield row


def update_column(
        conn: sqlite3.Connection,
        table: str,
        id_col: str,
        col: str,
        make_relative_to: Optional[Path],
        old_root: Optional[Path],
        new_root: Optional[Path],
        dry_run: bool,
) -> int:
    cur = conn.cursor()
    rows = list(iter_rows(cur, f"SELECT {id_col}, {col} FROM {table}"))
    n_changed = 0

    for _id, v in rows:
        if v is None:
            continue
        v0 = norm_db_str(str(v))
        v1 = v0

        # 1) rebase (old_root -> new_root) が指定されてれば先に適用
        if old_root and new_root:
            rebased = try_rebase(v1, old_root, new_root)
            if rebased is not None:
                v1 = rebased

        # 2) 相対化
        if make_relative_to:
            rel = try_relativize(v1, make_relative_to)
            if rel is not None:
                v1 = rel

        # 3) 正規化だけは常に
        v1 = norm_db_str(v1)

        if v1 != v0:
            n_changed += 1
            if dry_run:
                print(f"[DRY] {table}.{col} id={_id}:")
                print(f"  - {v0}")
                print(f"  + {v1}")
            else:
                cur.execute(
                    f"UPDATE {table} SET {col}=? WHERE {id_col}=?",
                    (v1, _id)
                )

    if not dry_run:
        conn.commit()
    return n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=str, help="dataset.sqlite のパス")
    ap.add_argument("--dry_run", action="store_true", help="更新せず差分表示だけ")

    ap.add_argument("--old_root", type=str, default=None, help="旧データroot（absolute pathがこの配下の時だけ対象）")
    ap.add_argument("--new_root", type=str, default=None, help="新データroot（付け替え先）")

    ap.add_argument("--make_relative_to", type=str, default=None,
                    help="指定すると absolute path をこのrootからの相対に変換してDBに保存")

    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    old_root = Path(args.old_root).resolve() if args.old_root else None
    new_root = Path(args.new_root).resolve() if args.new_root else None
    make_relative_to = Path(args.make_relative_to).resolve() if args.make_relative_to else None

    if (old_root is None) ^ (new_root is None):
        raise SystemExit("ERROR: --old_root と --new_root は両方指定してください。")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    total = 0
    total += update_column(conn, "videos", "id", "path", make_relative_to, old_root, new_root, args.dry_run)
    total += update_column(conn, "segment_frames", "id", "crop_path", make_relative_to, old_root, new_root, args.dry_run)
    total += update_column(conn, "segment_frames", "id", "pose_path", make_relative_to, old_root, new_root, args.dry_run)

    print(f"DONE. changed rows={total} db={db_path}")
    conn.close()


if __name__ == "__main__":
    main()
