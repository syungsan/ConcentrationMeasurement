from __future__ import annotations

from typing import Final, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


SITUATIONS: Final[tuple[str, ...]] = ("聞く", "書く", "話し合う")
SITUATION_TO_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(SITUATIONS)
}


def validate_situation(value: str) -> str:
    value = value.strip()
    if value not in SITUATION_TO_INDEX:
        allowed = ", ".join(SITUATIONS)
        raise ValueError(f"状況は {allowed} のいずれかを指定してください: {value!r}")
    return value


def situation_one_hot(value: str) -> "torch.Tensor":
    import torch

    value = validate_situation(value)
    out = torch.zeros(len(SITUATIONS), dtype=torch.float32)
    out[SITUATION_TO_INDEX[value]] = 1.0
    return out
