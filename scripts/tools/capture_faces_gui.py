# capture_faces_gui.py

import multiprocessing as mp
import math
import queue
import subprocess
import traceback
import time
import os
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.devices import face_device, face_providers, face_context_id

import cv2

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACES_DIR = PROJECT_ROOT / "faces"


def configure_wsl_fonts(root: tk.Tk) -> None:
    """WSLでのみ、インストール済みの日本語フォントを設定する。"""
    if sys.platform != "linux":
        return

    is_wsl = (
            "microsoft" in os.uname().release.lower()
            or bool(os.environ.get("WSL_INTEROP"))
            or bool(os.environ.get("WSL_DISTRO_NAME"))
    )
    if not is_wsl:
        return

    available = set(tkfont.families(root=root))

    # フォント名だけ変更し、サイズや太字などの設定は維持する。
    for name in tkfont.names(root=root):
        family = (
            "Noto Sans Mono CJK JP"
            if name == "TkFixedFont"
            else "Noto Sans CJK JP"
        )
        if family in available:
            tkfont.nametofont(name, root=root).configure(family=family)


# InsightFace 1.0.1内のscikit-image旧APIに対する既知の警告だけを抑制する。
warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated.*",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align",
)

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None


def cuda_is_available() -> bool:
    """ONNX Runtime からCUDAを利用できる場合だけTrueを返す。"""
    if ort is not None and torch is not None and hasattr(ort, "preload_dlls"):
        # PyTorch 2.11+cu128に同梱されたCUDA 12.8 DLLを使用する。
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        ort.preload_dlls(directory=str(torch_lib))

    return (
            ort is not None
            and "CUDAExecutionProvider" in ort.get_available_providers()
    )


def ensure_models_use_cuda(app: FaceAnalysis) -> None:
    """CUDA指定時にInsightFaceがCPUへフォールバックしていないか検査する。"""
    cpu_models = []
    for model_name, model in app.models.items():
        session = getattr(model, "session", None)
        providers = session.get_providers() if session is not None else []
        if "CUDAExecutionProvider" not in providers:
            cpu_models.append(model_name)

    if cpu_models:
        names = ", ".join(cpu_models)
        raise RuntimeError(
            "CUDAモデルの初期化に失敗しました。CPUへのフォールバックを停止します。\n"
            f"対象モデル: {names}\n"
            "onnxruntime-gpu用のCUDA/cuDNNランタイムを確認してください。"
        )


def capture_faces(
        cam_index: int,
        out_root: str,
        person_name: str,
        duration: float = 10.0,
        device: str = "auto",
        det_size=(640, 640),
        margin: int = 20,
        min_interval: float = 0.5,
        screen_w: int | None = None,
        screen_h: int | None = None,
        stop_event=None,
        video_path: str | None = None,
):
    """顔画像収集（独立プロセスのメインスレッドで実行する）。"""
    if FaceAnalysis is None:
        raise ImportError(
            "insightface がインポートできません。\n"
            "`pip install -r requirements_for_mac.txt（Mac） / "
            "pip install -r requirements.txt（Windows）` "
            "などでインストールしてください。"
        )

    out_root = Path(out_root)
    person_dir = out_root / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 出力ディレクトリ: {person_dir.resolve()}")
    if video_path is None:
        print(f"[INFO] 収録時間: {duration} 秒")
    print("[INFO] q または ESC で中断できます。")

    device = face_device(device)
    use_cuda = device.startswith("cuda") and cuda_is_available()
    if device.startswith("cuda") and not use_cuda:
        raise RuntimeError(
            "CUDAExecutionProviderが利用できません。\n"
            "CPUで実行する場合はデバイスでCPUを選択してください。"
        )

    providers = face_providers(device)
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=providers,
    )
    if use_cuda:
        ensure_models_use_cuda(app)
        print("[INFO] InsightFace の全モデルをCUDAで初期化しました。")

    ctx_id = face_context_id(device)
    print(f"[INFO] 顔処理デバイス: {device}")
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    source = video_path if video_path is not None else cam_index
    print(f"[INFO] 入力 {source} をオープンします...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"入力を開けません: {source}")

    try:
        start_time = time.time()
        last_save_time = -float("inf")
        saved_count = 0
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS) if video_path is not None else 0.0
        if video_path is not None and (not math.isfinite(fps) or fps <= 0):
            raise RuntimeError(
                "動画のFPSを取得できません。別の動画形式でお試しください。"
            )

        win_name = f"Capture faces: {person_name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # 画面中央に配置（最初のフレームを読んでから位置決め）
        first_frame = True

        while stop_event is None or not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(
                    "[INFO] 動画の読み込みを終了しました。"
                    if video_path is not None
                    else "[WARN] フレーム取得に失敗しました。終了します。"
                )
                break

            if first_frame and screen_w is not None and screen_h is not None:
                fh, fw = frame.shape[:2]
                x = (screen_w - fw) // 2
                y = (screen_h - fh) // 2
                cv2.moveWindow(win_name, max(0, x), max(0, y))
                first_frame = False

            now = time.time()
            elapsed = now - start_time
            if video_path is not None:
                # 処理速度によらず、動画内の時間で保存間隔を判定する。
                elapsed = frame_index / fps
                frame_index += 1

            if video_path is None and elapsed > duration:
                print("[INFO] 指定時間に到達したので終了します。")
                break

            h, w = frame.shape[:2]

            # 顔検出
            faces = app.get(frame)

            # 一番大きな顔を選択
            best_face = None
            best_area = 0.0
            for f in faces:
                x1, y1, x2, y2 = f.bbox
                area = max(0.0, (x2 - x1) * (y2 - y1))
                if area > best_area:
                    best_area = area
                    best_face = f

            if best_face is not None:
                x1, y1, x2, y2 = best_face.bbox
                x1 = int(max(0, x1 - margin))
                y1 = int(max(0, y1 - margin))
                x2 = int(min(w - 1, x2 + margin))
                y2 = int(min(h - 1, y2 + margin))

                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]

                    # 一定間隔ごとに保存（min_interval 秒）
                    if elapsed - last_save_time >= min_interval:
                        save_path = person_dir / f"{saved_count:04d}.jpg"
                        cv2.imwrite(str(save_path), face_crop)
                        saved_count += 1
                        last_save_time = elapsed
                        print(f"[SAVE] {save_path.name} (t={elapsed:.1f}s)")

                    # 画面上に矩形を表示
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )

            # 経過時間も表示
            cv2.putText(
                frame,
                (
                    f"{elapsed:4.1f}s"
                    if video_path is not None
                    else f"{elapsed:4.1f}s / {duration:.1f}s"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("[INFO] キー入力により中断されました。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[DONE] 保存した画像枚数: {saved_count}")

    # OSに合わせて収録フォルダを自動で開く
    try:
        if os.name == "nt":
            os.startfile(str(person_dir.resolve()))
        elif sys.platform == "darwin":
            subprocess.run(
                ["/usr/bin/open", str(person_dir.resolve())],
                check=True,
            )
    except Exception as e:
        print(f"[WARN] フォルダを開く際にエラー: {e}")


def capture_worker(options, stop_event, result_queue):
    """Spawn target: OpenCV Cocoa windows must run on the main thread."""
    try:
        capture_faces(**options, stop_event=stop_event)
    except Exception as exc:
        traceback.print_exc()
        result_queue.put(str(exc))
    else:
        result_queue.put(None)


# ========== GUI 部分 ==========

class FaceCaptureGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Capture (ArcFace)")

        # 画面サイズ取得（OpenCVウィンドウのセンタリングに使う）
        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()

        # ウィンドウ自体も少し小さめで中央に
        win_w, win_h = 600, 410
        x = (self.screen_w - win_w) // 2
        y = (self.screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        self.is_capturing = False
        self.capture_process = None
        self.process_context = mp.get_context("spawn")
        self.stop_event = None
        self.result_queue = None
        self.closing = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        # 各種変数
        self.var_name = tk.StringVar()
        self.var_out_root = tk.StringVar(value=str(DEFAULT_FACES_DIR))
        self.var_duration = tk.DoubleVar(value=30.0)
        self.var_cam = tk.IntVar(value=0)
        self.var_source = tk.StringVar(value="camera")
        self.var_video = tk.StringVar()
        self.var_device = tk.StringVar(value="auto")

        self._build_widgets()

    def _build_widgets(self):
        pad = {"padx": 10, "pady": 5}

        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 名前
        ttk.Label(frame, text="人物名:").grid(
            row=0, column=0, sticky="e", **pad
        )
        ttk.Entry(
            frame, textvariable=self.var_name, width=25
        ).grid(row=0, column=1, sticky="w", **pad)

        # 出力ルート
        ttk.Label(frame, text="出力ルート:").grid(
            row=1, column=0, sticky="e", **pad
        )
        ttk.Entry(
            frame, textvariable=self.var_out_root, width=25
        ).grid(row=1, column=1, sticky="w", **pad)

        # 収録時間
        ttk.Label(frame, text="収録時間(秒):").grid(
            row=2, column=0, sticky="e", **pad
        )
        self.entry_duration = ttk.Entry(
            frame, textvariable=self.var_duration, width=10
        )
        self.entry_duration.grid(
            row=2, column=1, sticky="w", **pad
        )

        # カメラインデックス
        ttk.Label(frame, text="カメラID:").grid(
            row=3, column=0, sticky="e", **pad
        )
        self.entry_cam = ttk.Entry(
            frame, textvariable=self.var_cam, width=10
        )
        self.entry_cam.grid(
            row=3, column=1, sticky="w", **pad
        )

        # デバイス
        ttk.Label(frame, text="デバイス:").grid(
            row=4, column=0, sticky="e", **pad
        )
        dev_frame = ttk.Frame(frame)
        dev_frame.grid(row=4, column=1, sticky="w", **pad)

        ttk.Radiobutton(
            dev_frame,
            text="自動",
            value="auto",
            variable=self.var_device,
        ).pack(side="left")
        ttk.Radiobutton(
            dev_frame,
            text="Mac GPU（CoreML）",
            value="coreml",
            variable=self.var_device,
        ).pack(side="left")
        ttk.Radiobutton(
            dev_frame,
            text="CUDA",
            value="cuda",
            variable=self.var_device,
        ).pack(side="left")
        ttk.Radiobutton(
            dev_frame,
            text="CPU",
            value="cpu",
            variable=self.var_device,
        ).pack(side="left")

        ttk.Label(frame, text="入力:").grid(
            row=5, column=0, sticky="e", **pad
        )
        source_frame = ttk.Frame(frame)
        source_frame.grid(row=5, column=1, sticky="w", **pad)
        for label, value in (
                ("カメラ", "camera"),
                ("動画ファイル", "video"),
        ):
            ttk.Radiobutton(
                source_frame,
                text=label,
                value=value,
                variable=self.var_source,
                command=self.update_source_state,
            ).pack(side="left")

        ttk.Label(frame, text="動画ファイル:").grid(
            row=6, column=0, sticky="e", **pad
        )
        video_frame = ttk.Frame(frame)
        video_frame.grid(row=6, column=1, sticky="ew", **pad)

        self.entry_video = ttk.Entry(
            video_frame, textvariable=self.var_video, width=35
        )
        self.entry_video.pack(side="left", fill="x", expand=True)

        self.btn_video = ttk.Button(
            video_frame, text="参照...", command=self.select_video
        )
        self.btn_video.pack(side="left", padx=5)
        self.update_source_state()

        # ボタン
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=15)

        self.btn_start = ttk.Button(
            btn_frame, text="開始", command=self.on_start
        )
        self.btn_start.pack(side="left", padx=5)

        self.btn_quit = ttk.Button(
            btn_frame, text="終了", command=self.on_quit
        )
        self.btn_quit.pack(side="left", padx=5)

    def update_source_state(self):
        is_video = self.var_source.get() == "video"
        for widget in (self.entry_duration, self.entry_cam):
            widget.config(state="disabled" if is_video else "normal")
        for widget in (self.entry_video, self.btn_video):
            widget.config(state="normal" if is_video else "disabled")

    def select_video(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="入力する動画を選択",
            filetypes=[
                (
                    "動画ファイル",
                    "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm",
                ),
                ("すべてのファイル", "*.*"),
            ],
        )
        if path:
            self.var_video.set(path)

    def on_start(self):
        if self.is_capturing:
            return

        name = self.var_name.get().strip()
        if not name:
            messagebox.showwarning("警告", "人物名を入力してください。")
            return

        out_root = self.var_out_root.get().strip()
        if not out_root:
            messagebox.showwarning(
                "警告", "出力ルートを入力してください。"
            )
            return

        video_path = None
        duration = 30.0
        cam_index = 0

        if self.var_source.get() == "video":
            video_path = self.var_video.get().strip()
            if not video_path or not Path(video_path).is_file():
                messagebox.showwarning(
                    "警告", "存在する動画ファイルを選択してください。"
                )
                return
        else:
            try:
                duration = float(self.var_duration.get())
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError
            except (ValueError, tk.TclError):
                messagebox.showwarning(
                    "警告", "収録時間は正の数で入力してください。"
                )
                return

            try:
                cam_index = int(self.var_cam.get())
                if cam_index < 0:
                    raise ValueError
            except (ValueError, tk.TclError):
                messagebox.showwarning(
                    "警告", "カメラIDは0以上の整数で入力してください。"
                )
                return

        device = self.var_device.get()

        # ボタン無効化
        self.is_capturing = True
        self.btn_start.config(state="disabled")

        self.stop_event = self.process_context.Event()
        self.result_queue = self.process_context.Queue()
        options = dict(
            cam_index=cam_index,
            out_root=out_root,
            person_name=name,
            duration=duration,
            device=device,
            det_size=(640, 640),
            margin=20,
            min_interval=0.5,
            screen_w=self.screen_w,
            screen_h=self.screen_h,
            video_path=video_path,
        )
        self.capture_process = self.process_context.Process(
            target=capture_worker,
            args=(options, self.stop_event, self.result_queue),
        )
        try:
            self.capture_process.start()
        except Exception as exc:
            self.result_queue.close()
            self.is_capturing = False
            self.btn_start.config(state="normal")
            messagebox.showerror("エラー", str(exc))
            return

        self.root.after(100, self.poll_capture)

    def poll_capture(self):
        if self.capture_process.is_alive():
            self.root.after(100, self.poll_capture)
            return

        self.capture_process.join()
        try:
            error = self.result_queue.get(timeout=0.2)
        except queue.Empty:
            error = (
                "キャプチャプロセスが終了しました"
                f"（終了コード: {self.capture_process.exitcode}）。"
            )

        self.result_queue.close()
        self.capture_process.close()
        self.is_capturing = False
        self.btn_start.config(state="normal")

        if self.closing:
            self.root.destroy()
        elif error:
            messagebox.showerror(
                "エラー",
                f"キャプチャ中にエラーが発生しました:\n{error}",
            )

    def on_quit(self):
        if self.is_capturing:
            if not messagebox.askyesno(
                    "確認", "キャプチャ中です。終了してもよろしいですか？"
            ):
                return

            self.closing = True
            self.stop_event.set()
            self.btn_quit.config(state="disabled")
            return

        self.root.destroy()


def main():
    root = tk.Tk()

    # WSLの場合のみ、日本語フォントをウィジェット作成前に設定する。
    configure_wsl_fonts(root)

    app = FaceCaptureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # 子プロセスの復元に必要なcapture_workerの定義後、GUI起動前に分岐する。
    mp.freeze_support()
    main()
