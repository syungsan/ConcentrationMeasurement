# build_face_db.py

import argparse
import warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.devices import face_device, face_providers, face_context_id

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACES_DIR = PROJECT_ROOT / "faces"
DEFAULT_DB_PATH = PROJECT_ROOT / "models" / "face_db.npz"

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
    """PyTorch同梱CUDA 12.8をロードし、ORTから利用可能か確認する。"""
    if ort is not None and torch is not None and hasattr(ort, "preload_dlls"):
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        ort.preload_dlls(directory=str(torch_lib))

    return (
        ort is not None
        and "CUDAExecutionProvider" in ort.get_available_providers()
    )


def ensure_models_use_cuda(app: FaceAnalysis) -> None:
    """InsightFaceの全モデルがCUDAセッションを持つことを確認する。"""
    cpu_models = []
    for model_name, model in app.models.items():
        session = getattr(model, "session", None)
        providers = session.get_providers() if session is not None else []
        if "CUDAExecutionProvider" not in providers:
            cpu_models.append(model_name)

    if cpu_models:
        raise RuntimeError(
            "CUDAモデルの初期化に失敗しました。CPUへのフォールバックを停止します。\n"
            f"対象モデル: {', '.join(cpu_models)}"
        )


def detect_face_with_fallback(app, img, pad=80, target_size=640):
    """
    小さい顔クロップだと検出できないことがあるので、
    1回目: そのまま検出
    2回目: 失敗したら余白を付けてリサイズして再検出
    """
    faces = app.get(img)
    if faces:
        return faces

    h, w = img.shape[:2]

    # 周囲に黒枠パディング
    padded = cv2.copyMakeBorder(
        img,
        pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # 大きめにリサイズ
    padded = cv2.resize(padded, (target_size, target_size))

    faces = app.get(padded)
    return faces


def build_face_db(
        faces_dir: str,
        out_path: str,
        device: str = "auto",
        app_name: str = "buffalo_l",
):
    """
    すでに「顔だけにクロップされた画像」から顔DBを作る。

    faces_dir/
      ai/
        face_ai_xxxx.png
      bob/
        0001.jpg
        ...

    という構成を想定。
    """
    if FaceAnalysis is None:
        raise ImportError(
            "insightface がインポートできません。\n"
            "まず `pip install -r requirements_for_mac.txt（Mac） / pip install -r requirements.txt（Windows）` などでインストールしてください。"
        )

    faces_dir = Path(faces_dir)
    if not faces_dir.is_dir():
        raise RuntimeError(f"faces_dir が存在しません: {faces_dir}")

    print(f"[INFO] Faces root: {faces_dir.resolve()}")
    print(f"[INFO] 出力: {out_path}")
    print(f"[INFO] モデル(app_name): {app_name}, device={device}")

    device = face_device(device)
    use_cuda = device.startswith("cuda") and cuda_is_available()
    if device.startswith("cuda") and not use_cuda:
        raise RuntimeError(
            "CUDAExecutionProviderが利用できません。"
            "CPUで実行する場合は --device cpu を指定してください。"
        )

    providers = face_providers(device)
    app = FaceAnalysis(name=app_name, allowed_modules=["detection", "recognition"], providers=providers)
    if use_cuda:
        ensure_models_use_cuda(app)
        print("[INFO] InsightFaceの全モデルをCUDAで初期化しました。")

    ctx_id = face_context_id(device)
    print(f"[INFO] 顔処理デバイス: {device}")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    person_names = []
    person_embs = []

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        print(f"\n[PERSON] {name}")

        embs = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] 読み込み失敗: {img_path}")
                continue

            # --- ここがポイント: フォールバック付きで顔検出 ---
            faces = detect_face_with_fallback(app, img)
            if not faces:
                print(f"  [WARN] 顔が検出できませんでした: {img_path}")
                continue

            # 一番スコアの高い顔を使用
            face = max(faces, key=lambda f: f.det_score)
            emb = face.normed_embedding.astype("float32")  # L2 正規化済み
            embs.append(emb)
            print(f"  OK: {img_path.name} (det_score={face.det_score:.3f})")

        if not embs:
            print(f"  [WARN] {name} の画像から有効な埋め込みが取れませんでした。スキップします。")
            continue

        embs = np.stack(embs, axis=0)  # (N, D)
        mean_emb = embs.mean(axis=0)
        # 念のためもう一度 L2 正規化
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            mean_emb = mean_emb / norm

        person_names.append(name)
        person_embs.append(mean_emb.astype("float32"))
        print(f"  -> {len(embs)}枚から平均埋め込みを作成")

    if not person_names:
        raise RuntimeError("顔DBが空です。顔画像が正しく配置されているか確認してください。")

    names_arr = np.array(person_names, dtype=object)
    embs_arr = np.stack(person_embs, axis=0)  # (P, D)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, names=names_arr, embs=embs_arr)
    print(
        f"\n[DONE] {len(names_arr)} 人分の顔DBを保存しました: "
        f"{out_path.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces_dir", type=str, default=str(DEFAULT_FACES_DIR),
                        help="顔画像のルートディレクトリ (person_name/xxx.jpg ...)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_DB_PATH),
                        help="出力する npz パス")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, coreml (Mac GPU), cuda, cpu (mps uses CPU for InsightFace)")
    parser.add_argument("--app_name", type=str, default="buffalo_l",
                        help="insightface.app.FaceAnalysis の name")

    args = parser.parse_args()

    build_face_db(
        faces_dir=args.faces_dir,
        out_path=args.out,
        device=args.device,
        app_name=args.app_name,
    )


if __name__ == "__main__":
    main()
