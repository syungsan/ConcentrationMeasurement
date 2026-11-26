from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lib.model_defs import Cfg, Regressor  # noqa: E402


def test_disabled_situation_feature_cannot_change_prediction() -> None:
    cfg = Cfg(
        data_roots=[],
        db_paths=[],
        raters=[],
        mode="skeleton",
        temporal="gru",
        T=4,
        K=17,
        pose_feat=8,
        hidden=16,
        layers=1,
        dropout=0.0,
        imagenet_pretrain=False,
        use_situation_feature=False,
    )
    model = Regressor(cfg).eval()
    poses = torch.randn(2, cfg.T, cfg.K, cfg.pose_dim)
    situation_a = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
    situation_b = torch.tensor([[0.0, 1.0, 0.0], [0.9, 0.1, 0.0]])

    with torch.no_grad():
        prediction_a = model(None, poses, situation_a)
        prediction_b = model(None, poses, situation_b)
        logits_a = model.forward_ordinal_logits(None, poses, situation_a)
        logits_b = model.forward_ordinal_logits(None, poses, situation_b)

    torch.testing.assert_close(prediction_a, prediction_b)
    torch.testing.assert_close(logits_a, logits_b)
