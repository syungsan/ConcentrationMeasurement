from __future__ import annotations

import argparse
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

from merge_rater_databases import rebuild_consensus, require_schema, write_consensus_csv


RATER_COLUMNS = (
    ("labels", "rater"),
    ("label_observations", "rater"),
    ("label_events", "rater"),
    ("window_skips", "rater"),
    ("blind_assignments", "rater"),
    ("excluded_segments", "excluded_by"),
)


def rater_tables(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    tables = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return [(table, column) for table, column in RATER_COLUMNS if table in tables]


def list_raters(conn: sqlite3.Connection) -> None:
    counts: dict[str, dict[str, int]] = {}
    for table, column in rater_tables(conn):
        for name, count in conn.execute(f"""
            SELECT {column}, COUNT(*) FROM {table}
            WHERE {column} IS NOT NULL GROUP BY {column}
        """):
            counts.setdefault(name, {})[table] = count

    if not counts:
        print("No raters found.")
    for name, table_counts in sorted(counts.items()):
        summary = ", ".join(f"{table}={count}" for table, count in table_counts.items())
        print(f"{name!r}: {summary}")


def remove_raters(source: sqlite3.Connection, names: list[str], output: Path) -> None:
    csv_path = output.with_name(f"{output.stem}_consensus.csv")
    with ExitStack() as cleanup:
        for path in (output, csv_path):
            with path.open("xb"):
                cleanup.callback(path.unlink, missing_ok=True)

        with closing(sqlite3.connect(output)) as conn:
            source.backup(conn)
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:
                conn.execute("BEGIN")
                tables = rater_tables(conn)
                deleted_by_name = {}
                for name in dict.fromkeys(names):
                    deleted = {
                        table: conn.execute(
                            f"DELETE FROM {table} WHERE {column} COLLATE BINARY = ?", (name,)
                        ).rowcount
                        for table, column in tables
                    }
                    if not any(deleted.values()):
                        raise ValueError(f"rater not found: {name!r}")
                    deleted_by_name[name] = deleted
                consensus_count = rebuild_consensus(conn)
                if conn.execute("PRAGMA foreign_key_check").fetchone():
                    raise RuntimeError("foreign key check failed after removal")
            write_consensus_csv(conn, output)
        cleanup.pop_all()

    for name, deleted in deleted_by_name.items():
        summary = ", ".join(f"{table}={count}" for table, count in deleted.items() if count)
        print(f"Removed {name!r}: {summary}")
    print(f"Consensus segments: {consensus_count}")
    print(f"Output database: {output}")
    print(f"Consensus CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List or remove a rater from a merged SQLite database.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "remove"):
        subparsers.add_parser(command).add_argument("--db", required=True, type=Path)
    remove = subparsers.choices["remove"]
    remove.add_argument(
        "--rater", required=True, nargs="+", action="extend",
        help="One or more exact rater names, including case",
    )
    remove.add_argument("--output", required=True, type=Path, help="Path to a new database")
    args = parser.parse_args()

    try:
        with closing(sqlite3.connect(f"{args.db.resolve().as_uri()}?mode=ro", uri=True)) as source:
            require_schema(source, args.db)
            if args.command == "list":
                list_raters(source)
            else:
                remove_raters(source, args.rater, args.output.resolve())
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
