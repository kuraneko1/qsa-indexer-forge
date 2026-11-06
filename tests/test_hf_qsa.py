import pytest
import torch

from qsa.hf_qsa import _qsa_forward
from qsa.indexer import QSAIndexer


@pytest.fixture()
def tiny_attention():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
        partial_rotary_factor=0.5,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
        },
    )
    cfg._attn_implementation = "sdpa"
    attn = Qwen3_5Attention(cfg, 0).eval()
    return attn, cfg


def _rope_tables(seq: int, dim: int, device=None):
    angles = torch.rand(seq, dim // 2, device=device) * 6.28
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos()[None], angles.sin()[None]


def test_patched_dense_matches_original(tiny_attention):
    attn, cfg = tiny_attention
    torch.manual_seed(0)
    hidden = torch.randn(1, 8, cfg.hidden_size)
    cos, sin = _rope_tables(8, 8)

    with torch.no_grad():
        want, _ = attn(hidden, (cos, sin), None)

        attn._qsa_block = 4
        attn._qsa_budget = 0
        attn._qsa_indexer = QSAIndexer(cfg.hidden_size, 2, 8, 4, 4).eval()
        got, _ = _qsa_forward(attn, hidden, (cos, sin), None)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_capture_accumulates(tiny_attention):
    attn, cfg = tiny_attention
    torch.manual_seed(0)
    hidden = torch.randn(1, 16, cfg.hidden_size)
    cos, sin = _rope_tables(16, 8)

    attn._qsa_block = 4
    attn._qsa_budget = 0
    attn._qsa_collect = True
    attn._qsa_indexer = QSAIndexer(cfg.hidden_size, 2, 8, 4, 4).eval()
    cap = {"budgets": [4, 8], "sums": {}, "count": 0}
    attn._qsa_capture = cap

    with torch.no_grad():
        _qsa_forward(attn, hidden, (cos, sin), None)

    assert cap["count"] == 13  # queries with at least one complete 4-token block
    assert set(cap["sums"]) == {4, 8}
    for value in cap["sums"].values():
        assert 0 < value / cap["count"] <= 1.0001
    assert attn._qsa_target is None  # discarded when not training
