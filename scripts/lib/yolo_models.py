"""Prepare local YOLO weights, downloading missing official models."""
from pathlib import Path


def ensure_yolo_model(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path

    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset

    if model_path.name not in GITHUB_ASSETS_NAMES:
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            "独自モデルは指定先に配置するか、config.yaml の detection_tracking.model を修正してください。"
        )

    print(f"[INFO] 未配置の公式 YOLO モデルを取得します: {model_path}", flush=True)
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = Path(attempt_download_asset(str(model_path)))
        if not downloaded_path.is_file():
            raise FileNotFoundError(f"ダウンロード後もモデルが見つかりません: {downloaded_path}")
        return downloaded_path
    except Exception as exc:
        raise RuntimeError(
            f"YOLO モデルの取得に失敗しました: {model_path}\n"
            "インターネット接続を確認して再実行するか、公式モデルを指定先に配置してください。"
        ) from exc
