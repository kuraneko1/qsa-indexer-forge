import os
import tempfile

import pytest
import torch


@pytest.fixture(scope="module")
def tiny_model():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        partial_rotary_factor=0.5,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [3, 3, 2],
        },
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).eval()
    return model, cfg


def test_single_batch_overfits(tiny_model):
    """The KL objective must be able to fit one fixed batch (implementation check)."""
    from qsa.hf_qsa import full_attention_layers, install
    from qsa.train_indexer import distill_step, set_mode

    model, cfg = tiny_model
    install(model, block_size=4, budget=64, indexer_heads=2, indexer_head_dim=16,
            rope_dim=16, dtype=torch.float32)
    model.requires_grad_(False)
    layers = [l for l in full_attention_layers(model) if getattr(l, "_qsa_indexer", None)]
    params = [p for l in layers for p in l._qsa_indexer.parameters()]
    for p in params:
        p.requires_grad_(True)

    train_cfg = {"budget": 64, "block_size": 4}
    set_mode(layers, collect=True, budget=64)
    ids = torch.randint(0, cfg.vocab_size, (1, 256))
    optimizer = torch.optim.AdamW(params, lr=3e-3, betas=(0.9, 0.95))

    history = []
    for _ in range(60):
        loss, capture, _ = distill_step(model, layers, ids, train_cfg, optimizer)
        history.append((loss, capture))

    first_loss, first_capture = history[0]
    final_loss, final_capture = history[-1]
    assert min(l for l, _ in history) < 0.5 * first_loss
    assert final_capture > 0.6, f"capture stuck at {final_capture:.3f}"
    assert final_capture > first_capture
