"""Device selection shared by training and inference."""
import sys


def default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str = "auto") -> str:
    return default_device() if device == "auto" else device


def face_device(device: str = "auto") -> str:
    """InsightFace uses ONNX Runtime, which cannot use PyTorch's MPS backend."""
    if device == "auto":
        if sys.platform == "darwin":
            return "cpu"
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
        return "cpu"
    if device == "mps":
        return "cpu"
    if device == "coreml":
        import onnxruntime as ort
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CoreMLが利用できません。CPUを選択してください。")
        return "coreml"
    if device == "cpu" or device.startswith("cuda"):
        return device
    raise ValueError(f"Unsupported face device: {device}")


def face_providers(device: str) -> list:
    if device == "coreml":
        return [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
            }),
            "CPUExecutionProvider",
        ]
    if device.startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def face_context_id(device: str) -> int:
    # InsightFace resets all providers to CPU when ctx_id is negative.
    return -1 if device == "cpu" else 0
