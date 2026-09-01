from __future__ import annotations

import warnings
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated.*",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align",
)


def decide_face_exclusion(
    matches: list[str | None], min_matches: int, min_match_ratio: float,
) -> tuple[bool, str | None, int, float]:
    """全照合フレームに対する同一登録者の一致数・割合で判定する。"""
    detected = [name for name in matches if name is not None]
    if not detected:
        return False, None, 0, 0.0
    name, count = Counter(detected).most_common(1)[0]
    ratio = count / len(matches) if matches else 0.0
    excluded = count >= min_matches and ratio >= min_match_ratio
    return excluded, name, count, ratio


class FaceDbMatcher:
    def __init__(
        self,
        db_path: Path,
        *,
        threshold: float,
        device: str = "cuda",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        if device.startswith("cuda"):
            try:
                import torch
                if hasattr(ort, "preload_dlls"):
                    torch_lib = Path(torch.__file__).resolve().parent / "lib"
                    ort.preload_dlls(directory=str(torch_lib))
            except ImportError:
                pass
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("顔DB照合用のCUDAExecutionProviderが利用できません。")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        data = np.load(db_path, allow_pickle=True)
        self.names = [str(name) for name in data["names"].tolist()]
        self.embs = np.asarray(data["embs"], dtype=np.float32)
        if self.embs.ndim != 2 or len(self.names) != len(self.embs):
            raise RuntimeError(f"顔DBの形式が不正です: {db_path}")
        norms = np.linalg.norm(self.embs, axis=1, keepdims=True)
        self.embs = self.embs / np.maximum(norms, 1e-12)
        self.threshold = float(threshold)

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        for model_name, model in self.app.models.items():
            session = getattr(model, "session", None)
            active = session.get_providers() if session is not None else []
            if device.startswith("cuda") and "CUDAExecutionProvider" not in active:
                raise RuntimeError(f"顔認証モデルがCUDAを使用していません: {model_name}")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def match(self, image: np.ndarray) -> tuple[str | None, float]:
        faces = self.app.get(image)
        if not faces:
            padded = cv2.copyMakeBorder(
                image, 80, 80, 80, 80,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            padded = cv2.resize(padded, (640, 640))
            faces = self.app.get(padded)
        if not faces:
            return None, float("-inf")
        face = max(faces, key=lambda item: float(item.det_score))
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        scores = self.embs @ emb
        index = int(np.argmax(scores))
        score = float(scores[index])
        return (self.names[index] if score >= self.threshold else None), score


class OnlineFaceExclusionTracker:
    """追跡IDごとの直近顔照合結果から、一時的な除外状態を管理する。"""

    def __init__(
        self,
        matcher: FaceDbMatcher,
        *,
        min_matches: int,
        min_match_ratio: float,
        max_frames: int,
    ) -> None:
        self.matcher = matcher
        self.min_matches = int(min_matches)
        self.min_match_ratio = float(min_match_ratio)
        self.max_frames = max(1, int(max_frames))
        self.history: dict[int, deque[str | None]] = {}
        self.names: dict[int, str] = {}

    def update(self, track_id: int, image: np.ndarray) -> tuple[bool, str | None]:
        history = self.history.setdefault(
            int(track_id), deque(maxlen=self.max_frames)
        )
        matched_name, _score = self.matcher.match(image)
        history.append(matched_name)
        excluded, name, _count, _ratio = decide_face_exclusion(
            list(history), self.min_matches, self.min_match_ratio,
        )
        if excluded and name is not None:
            self.names[int(track_id)] = name
        elif not excluded:
            self.names.pop(int(track_id), None)
        return excluded, self.names.get(int(track_id))

    def forget(self, track_id: int) -> None:
        self.history.pop(int(track_id), None)
        self.names.pop(int(track_id), None)


def update_auto_exclusion(
    conn,
    segment_id: int,
    *,
    excluded: bool,
    matched_name: str | None,
) -> None:
    """手動除外を保護しつつ、顔DB由来の判定だけを更新する。"""
    row = conn.execute(
        "SELECT excluded_by FROM excluded_segments WHERE segment_id=?",
        (int(segment_id),),
    ).fetchone()
    if excluded:
        if row is None or row[0] == "face_recognition":
            conn.execute("""
                INSERT INTO excluded_segments(segment_id, reason, excluded_by)
                VALUES(?, ?, 'face_recognition')
                ON CONFLICT(segment_id) DO UPDATE SET
                  reason=excluded.reason,
                  excluded_by=excluded.excluded_by
            """, (int(segment_id), f"face_db:{matched_name or 'unknown'}"))
    elif row is not None and row[0] == "face_recognition":
        conn.execute(
            "DELETE FROM excluded_segments WHERE segment_id=?",
            (int(segment_id),),
        )
