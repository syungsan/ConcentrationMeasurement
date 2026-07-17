from __future__ import annotations

import subprocess
import threading
import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent
PYTHON_EXE = ROOT / "WPy64-312101" / "python" / "python.exe"
LABEL_GUI = ROOT / "scripts" / "06_label_gui.py"
MERGE_SCRIPT = ROOT / "scripts" / "tools" / "merge_rater_databases.py"
PACKAGE_BUILDER = ROOT / "scripts" / "tools" / "build_labeling_package.py"
DATASETS_DIR = ROOT / "datasets"
PACKAGE_CONFIG_PATH = ROOT / "package_config.json"


def load_package_config() -> dict:
    if not PACKAGE_CONFIG_PATH.exists():
        return {}
    try:
        value = json.loads(PACKAGE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid package_config.json: {exc}") from exc
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class DatasetEntry:
    root: Path
    db_path: Path
    video_path: Path

    @property
    def ready(self) -> bool:
        return self.db_path.exists() and self.video_path.exists()

    @property
    def display_name(self) -> str:
        return self.root.relative_to(DATASETS_DIR).as_posix()


def discover_datasets() -> list[DatasetEntry]:
    roots: set[Path] = set()
    if DATASETS_DIR.exists():
        for video in DATASETS_DIR.rglob("proxy.mp4"):
            if video.parent.name == "videos":
                roots.add(video.parent.parent)
        for db_path in DATASETS_DIR.rglob("dataset.sqlite"):
            if db_path.parent.name == "db":
                roots.add(db_path.parent.parent)
    return sorted(
        [
            DatasetEntry(
                root=root,
                db_path=root / "db" / "dataset.sqlite",
                video_path=root / "videos" / "proxy.mp4",
            )
            for root in roots
        ],
        key=lambda entry: entry.display_name.lower(),
    )


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("集中度ラベリング ランチャー")
        self.geometry("860x600")
        self.minsize(760, 520)

        self.package_config = load_package_config()
        self.evaluator_only = self.package_config.get("distribution_mode") == "evaluator"
        if self.evaluator_only:
            self.title("集中度採点 評価者用ランチャー")
        self.mode = tk.StringVar(value="evaluator")
        self.name = tk.StringVar(value=str(self.package_config.get("evaluator_name", "")))
        self.status = tk.StringVar(value="datasetを選択してください。")
        self.entries: dict[str, DatasetEntry] = {}

        self._build_menu()
        self._build_ui()
        self.refresh_datasets()
        self.on_mode_changed()

    def _build_menu(self):
        menu = tk.Menu(self)
        if not self.evaluator_only:
            tools = tk.Menu(menu, tearoff=False)
            tools.add_command(label="dataset一覧を更新", command=self.refresh_datasets)
            tools.add_command(label="評価者用パッケージを作成...", command=self.build_evaluator_package)
            tools.add_command(label="評価者DBをマージ...", command=self.merge_databases)
            menu.add_cascade(label="管理", menu=tools)
        menu.add_command(label="終了", command=self.destroy)
        self.config(menu=menu)

    def _build_ui(self):
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        mode_box = ttk.LabelFrame(outer, text="起動モード", padding=12)
        if not self.evaluator_only:
            mode_box.pack(fill=tk.X)
        ttk.Radiobutton(
            mode_box, text="管理者モード（共有状況の設定）",
            variable=self.mode, value="administrator", command=self.on_mode_changed,
        ).pack(side=tk.LEFT, padx=(0, 24))
        ttk.Radiobutton(
            mode_box, text="評価者モード（集中度の採点）",
            variable=self.mode, value="evaluator", command=self.on_mode_changed,
        ).pack(side=tk.LEFT)

        identity = ttk.Frame(outer, padding=(0, 14, 0, 8))
        identity.pack(fill=tk.X)
        self.name_label = ttk.Label(identity, text="評価者名")
        self.name_label.pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(identity, textvariable=self.name, width=36)
        self.name_entry.pack(side=tk.LEFT, padx=10)
        if self.evaluator_only and self.name.get().strip():
            self.name_entry.configure(state="readonly")

        dataset_box = ttk.LabelFrame(outer, text="datasetsフォルダ", padding=10)
        dataset_box.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(
            dataset_box, columns=("dataset", "db", "video", "status"),
            show="headings", selectmode="browse",
        )
        self.tree.heading("dataset", text="dataset")
        self.tree.heading("db", text="DB")
        self.tree.heading("video", text="proxy動画")
        self.tree.heading("status", text="状態")
        self.tree.column("dataset", width=320, anchor=tk.W)
        self.tree.column("db", width=90, anchor=tk.CENTER)
        self.tree.column("video", width=90, anchor=tk.CENTER)
        self.tree.column("status", width=120, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(dataset_box, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<Double-1>", lambda _event: self.launch_label_gui())

        bottom = ttk.Frame(outer, padding=(0, 12, 0, 0))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bottom, text="一覧更新", command=self.refresh_datasets).pack(side=tk.RIGHT, padx=(8, 0))
        self.launch_button = ttk.Button(bottom, text="起動", command=self.launch_label_gui)
        self.launch_button.pack(side=tk.RIGHT)

    def on_mode_changed(self):
        if self.evaluator_only:
            self.mode.set("evaluator")
        is_admin = self.mode.get() == "administrator"
        self.tree.configure(selectmode="extended" if is_admin else "browse")
        self.name_label.configure(text="代表者名" if is_admin else "評価者名")
        self.launch_button.configure(
            text="共有状況設定を起動" if is_admin else "集中度評価を起動"
        )

    def refresh_datasets(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.entries.clear()
        for index, entry in enumerate(discover_datasets()):
            item_id = f"dataset_{index}"
            self.entries[item_id] = entry
            self.tree.insert(
                "", tk.END, iid=item_id,
                values=(
                    entry.display_name,
                    "あり" if entry.db_path.exists() else "なし",
                    "あり" if entry.video_path.exists() else "なし",
                    "準備済み" if entry.ready else "準備未完了",
                ),
            )
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
        self.status.set(f"{len(children)} datasetを検出しました。")

    def selected_dataset(self) -> DatasetEntry | None:
        selected = self.tree.selection()
        return self.entries.get(selected[0]) if selected else None

    def selected_datasets(self) -> list[DatasetEntry]:
        return [self.entries[item] for item in self.tree.selection() if item in self.entries]

    def validate_runtime(self) -> bool:
        if not PYTHON_EXE.exists():
            messagebox.showerror("起動エラー", f"WinPythonが見つかりません:\n{PYTHON_EXE}")
            return False
        if not LABEL_GUI.exists():
            messagebox.showerror("起動エラー", f"GUIが見つかりません:\n{LABEL_GUI}")
            return False
        return True

    def launch_label_gui(self):
        if not self.validate_runtime():
            return
        entry = self.selected_dataset()
        if entry is None:
            messagebox.showinfo("dataset", "起動するdatasetを選択してください。")
            return
        if not entry.ready:
            messagebox.showerror("準備未完了", "dataset.sqlite と videos/proxy.mp4 の両方が必要です。")
            return
        person_name = self.name.get().strip()
        if not person_name:
            label = "代表者名" if self.mode.get() == "administrator" else "評価者名"
            messagebox.showinfo(label, f"{label}を入力してください。")
            return
        if self.evaluator_only and not str(
            self.package_config.get("evaluator_name", "")
        ).strip():
            self.package_config["evaluator_name"] = person_name
            PACKAGE_CONFIG_PATH.write_text(
                json.dumps(self.package_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.name_entry.configure(state="readonly")

        task = "situation" if self.mode.get() == "administrator" else "rating"
        dataset_arg = entry.root.relative_to(ROOT).as_posix()
        command = [
            str(PYTHON_EXE), str(LABEL_GUI),
            "--dataset", dataset_arg,
            "--task", task,
            "--name", person_name,
        ]
        if task == "rating":
            command.extend(["--rating-clip-sec", "15"])
        if self.evaluator_only:
            command.append("--blind-order")
        try:
            subprocess.Popen(command, cwd=str(ROOT))
        except OSError as exc:
            messagebox.showerror("起動エラー", str(exc))
            return
        self.status.set(f"{entry.display_name} を{self.launch_button.cget('text')}しました。")

    def merge_databases(self):
        if self.mode.get() != "administrator":
            messagebox.showinfo("管理者モード", "DBマージは管理者モードで実行してください。")
            return
        if not self.validate_runtime():
            return
        base = self.selected_dataset()
        if base is None or not base.db_path.exists():
            messagebox.showinfo("ベースDB", "準備済みdatasetを一覧で選択してください。")
            return
        inputs = filedialog.askopenfilenames(
            title="回収した評価者DBを選択",
            filetypes=[("SQLite database", "*.sqlite"), ("All files", "*.*")],
        )
        if not inputs:
            return
        default_dir = ROOT / "merged"
        default_dir.mkdir(exist_ok=True)
        output = filedialog.asksaveasfilename(
            title="マージ後DBの保存先",
            initialdir=str(default_dir),
            initialfile=f"{base.root.name}_merged.sqlite",
            defaultextension=".sqlite",
            filetypes=[("SQLite database", "*.sqlite")],
        )
        if not output:
            return
        command = [
            str(PYTHON_EXE), str(MERGE_SCRIPT),
            "--base-db", str(base.db_path),
            "--output", output,
            "--overwrite",
            "--inputs", *inputs,
        ]
        self.status.set("評価者DBをマージ中です...")
        self.launch_button.configure(state=tk.DISABLED)

        def worker():
            result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
            self.after(0, lambda: self._merge_finished(result, Path(output)))

        threading.Thread(target=worker, daemon=True).start()

    def build_evaluator_package(self):
        if self.evaluator_only or self.mode.get() != "administrator":
            messagebox.showinfo("管理者モード", "パッケージ作成は管理者モードで実行してください。")
            return
        entries = self.selected_datasets()
        if not entries:
            messagebox.showinfo("dataset", "収録するdatasetを一覧で選択してください。")
            return
        not_ready = [entry.display_name for entry in entries if not entry.ready]
        if not_ready:
            messagebox.showerror("準備未完了", "次のdatasetはDBまたは動画が不足しています:\n" + "\n".join(not_ready))
            return
        parent = filedialog.askdirectory(title="配布パッケージの保存先フォルダ")
        if not parent:
            return
        output = Path(parent) / "concentration_labeler_package"
        overwrite = False
        if output.exists():
            overwrite = messagebox.askyesno("上書き確認", f"既存フォルダを作り直しますか？\n{output}")
            if not overwrite:
                return
        command = [
            str(PYTHON_EXE), str(PACKAGE_BUILDER),
            "--datasets", *[str(entry.root) for entry in entries],
            "--output", str(output),
        ]
        if overwrite:
            command.append("--overwrite")
        self.status.set("評価者用パッケージを作成中です...")
        self.launch_button.configure(state=tk.DISABLED)

        def worker():
            result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
            self.after(0, lambda: self._package_finished(result, output))

        threading.Thread(target=worker, daemon=True).start()

    def _package_finished(self, result: subprocess.CompletedProcess[str], output: Path):
        self.launch_button.configure(state=tk.NORMAL)
        if result.returncode == 0:
            self.status.set(f"パッケージ作成完了: {output}")
            messagebox.showinfo("作成完了", f"評価者へフォルダごと渡してください。\n{output}")
        else:
            self.status.set("パッケージ作成に失敗しました。")
            messagebox.showerror("作成エラー", (result.stderr or result.stdout).strip())

    def _merge_finished(self, result: subprocess.CompletedProcess[str], output: Path):
        self.launch_button.configure(state=tk.NORMAL)
        if result.returncode == 0:
            self.status.set(f"マージ完了: {output}")
            messagebox.showinfo("マージ完了", result.stdout.strip() or str(output))
        else:
            self.status.set("マージに失敗しました。")
            messagebox.showerror("マージエラー", (result.stderr or result.stdout).strip())


if __name__ == "__main__":
    Launcher().mainloop()
