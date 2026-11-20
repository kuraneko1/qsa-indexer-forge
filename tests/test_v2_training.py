"""CPU checks for the indexer v2 training code (300b recall terms, 64k chunking, kl_rows)."""
from pathlib import Path

import pytest
import torch
import yaml

from qsa import QSAIndexer, block_teacher_from_qk, build_sparse_mask, select_blocks
from qsa.data import combine_iterators, iter_local_sequences
from qsa.sparse_attention import span_sparse_mask
from qsa.teacher import block_teacher_rows

ROOT = Path(__file__).resolve().parents[1]


def rope_tables(t: int, d: int, base: float = 1e7) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    freqs = torch.outer(torch.arange(t, dtype=torch.float64), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def make_indexer(t: int = 37, r: int = 4):
    torch.manual_seed(0)
    idx = QSAIndexer(16, num_heads=2, head_dim=8, rope_dim=4, block_size=r).double()
    x = torch.randn(1, t, 16, dtype=torch.float64)
    cos, sin = rope_tables(t, 4)
    return idx, x, cos, sin


def test_score_spans_match_forward() -> None:
    idx, x, cos, sin = make_indexer()
    want, want_valid = idx(x, cos, sin)
    kb = idx.key_blocks(x, cos, sin)
    parts = [idx.score_span(x[:, s:s + 8], cos[s:s + 8], sin[s:s + 8], kb, s) for s in range(0, x.shape[1], 8)]
    got = torch.cat([p[0] for p in parts], dim=1)
    got_valid = torch.cat([p[1] for p in parts], dim=1)
    assert torch.equal(got_valid, want_valid)
    torch.testing.assert_close(got[got_valid], want[want_valid], rtol=0, atol=1e-12)


def test_score_rows_match_forward_rows() -> None:
    idx, x, cos, sin = make_indexer()
    want, want_valid = idx(x, cos, sin)
    rows = torch.tensor([3, 4, 11, 20, 36])
    kb = idx.key_blocks(x, cos, sin)
    got, got_valid = idx.score_rows(x[:, rows], cos[rows], sin[rows], kb, rows)
    assert torch.equal(got_valid, want_valid[:, rows])
    torch.testing.assert_close(got[got_valid], want[:, rows][want_valid[:, rows]], rtol=0, atol=1e-12)


def test_span_mask_matches_full_mask() -> None:
    idx, x, cos, sin = make_indexer(t=40)
    scores, valid = idx(x, cos, sin)
    sel = select_blocks(scores, valid, token_budget=8, block_size=4)
    full = build_sparse_mask(sel, valid, 40, 4)
    for qs, qe in ((0, 13), (13, 29), (29, 40)):
        got = span_sparse_mask(sel[:, qs:qe], valid[:, qs:qe], qs, qe, 4)
        assert torch.equal(got, full[:, qs:qe, :qe])
        assert not full[:, qs:qe, qe:].any()


def test_teacher_rows_match_full_teacher() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 3, 33, 8, dtype=torch.float64)
    k = torch.randn(1, 3, 33, 8, dtype=torch.float64)
    target, valid = block_teacher_from_qk(q, k, 4)
    rows = torch.tensor([3, 7, 8, 20, 32])
    got, got_valid = block_teacher_rows(q, k, 4, rows, query_chunk=2)
    assert torch.equal(got_valid, valid[rows])
    torch.testing.assert_close(got, target[:, rows].to(got.dtype), rtol=0, atol=1e-6)


def test_chunked_sparse_attention_matches_full(monkeypatch) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextConfig

    from qsa.hf_qsa import _qsa_forward

    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, layer_types=["full_attention"], partial_rotary_factor=0.5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [1, 1, 2]},
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    attn = Qwen3_5Attention(cfg, 0).eval()
    attn._qsa_block = 4
    attn._qsa_budget = 8
    attn._qsa_indexer = QSAIndexer(cfg.hidden_size, 2, 16, 8, 4).eval()  # rope_dim = rotary width of the layer
    hidden = torch.randn(1, 48, cfg.hidden_size)
    angles = torch.rand(48, 4) * 6.28
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos()[None], angles.sin()[None]
    with torch.no_grad():
        monkeypatch.delenv("QSA_QCHUNK", raising=False)
        want, _ = _qsa_forward(attn, hidden, (cos, sin), None)
        monkeypatch.setenv("QSA_QCHUNK", "16")
        got, _ = _qsa_forward(attn, hidden, (cos, sin), None)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_combine_iterators_ratio() -> None:
    def tags(ratio):
        it = combine_iterators(iter(["p"] * 100), iter(["s"] * 100), ratio)
        return [next(it) for _ in range(20)]

    eighty = tags(0.8)
    assert [n for n, x in enumerate(eighty, 1) if x == "s"] == [8, 9, 18, 19]
    assert tags(0.7).count("s") == 6
    assert tags(1.0 - 0.2) == eighty  # the 300b config writes mix_fineweb: 0.2


class CharTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + (ord(c) % 50) for c in text]}


def test_local_sequences_keep_tail_and_labels(tmp_path) -> None:
    doc = "a" * 900 + "RECORD" + "b" * 100
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"doc": "%s", "question": "q?", "fmt": "x", "answer_pos": [900, 905], "decoy_pos": [2]}\n' % doc
        + '{"doc": "%s", "question": "q?", "fmt": "x", "answer_pos": [10], "decoy_pos": []}\n' % doc
    )
    seqs = list(iter_local_sequences(path, CharTokenizer(), seq_len=800, n_blocks=5, block_size=4))
    assert len(seqs) == 1, "the second record lost its answer to the head cut and is skipped"
    s = seqs[0]
    shift = len(doc) - (800 - len("\n\nQuestion: q?\n\n") - 2)
    assert s["q_start"] == len(doc) - shift
    assert s["answer_blocks"] == sorted({(900 - shift) // 4, (905 - shift) // 4})
    assert s["decoy_blocks"] == []  # the decoy fell into the cut head


@pytest.fixture(scope="module")
def tiny_model():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, layer_types=["linear_attention", "full_attention"] * 2,
        linear_conv_kernel_dim=4, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, partial_rotary_factor=0.5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [3, 3, 2]},
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg).eval(), cfg


def _setup(model):
    from qsa.hf_qsa import full_attention_layers, install
    from qsa.train_indexer import set_mode

    install(model, block_size=4, budget=32, indexer_heads=2, indexer_head_dim=16, rope_dim=16, dtype=torch.float32)
    model.requires_grad_(False)
    layers = [l for l in full_attention_layers(model) if getattr(l, "_qsa_indexer", None)]
    torch.manual_seed(1)
    for l in layers:
        for p in l._qsa_indexer.parameters():
            p.requires_grad_(True)
    set_mode(layers, collect=True, budget=32)
    return layers


@pytest.mark.parametrize("extra", [
    {},
    {"restrict_kl": True},
    {"restrict_kl": True, "loss_chunk": 64},
    {"restrict_kl": True, "kl_rows": 40, "loss_chunk": 32, "seed": 0},
])
def test_distill_step_recall_terms(tiny_model, extra) -> None:
    from qsa.train_indexer import distill_step

    model, cfg = tiny_model
    layers = _setup(model)
    params = [p for l in layers for p in l._qsa_indexer.parameters()]
    torch.manual_seed(2)
    ids = torch.randint(1, cfg.vocab_size, (1, 256))
    meta = {"q_start": 224, "answer_blocks": [10, 11, 40], "decoy_blocks": [12, 41], "_step": 0}
    base = {"budget": 32, "block_size": 4, **extra}
    frozen = torch.optim.SGD(params, lr=0.0)

    plain, _, _ = distill_step(model, layers, ids, base, frozen, meta)
    with_terms, _, stats = distill_step(model, layers, ids, {**base, "lambda_answer": 0.3, "mu_margin": 0.3,
                                                             "margin": 2.0}, frozen, meta)
    assert torch.isfinite(torch.tensor([plain, with_terms])).all()
    assert with_terms > plain, "the answer-mass and decoy-margin terms add a positive loss"
    assert 0.0 <= stats["answer_recall"] <= 1.0 and stats["worst_rank"] >= 1

    opt = torch.optim.AdamW(params, lr=3e-3)
    before = [p.detach().clone() for p in params]
    distill_step(model, layers, ids, {**base, "lambda_answer": 0.3, "mu_margin": 0.3}, opt, meta)
    assert any(not torch.equal(a, p) for a, p in zip(before, params)), "the step must update the indexer"
    assert all(torch.isfinite(p).all() for p in params), "rows without a valid block must not give NaN"


def test_v2_configs_are_local_and_pinned() -> None:
    s1 = yaml.safe_load((ROOT / "configs/recall-16k-300b.yaml").read_text())
    s2 = yaml.safe_load((ROOT / "configs/mixup-64k.yaml").read_text())
    for cfg in (s1, s2):
        assert cfg["push_to_hub"] is False
        assert len(cfg["model_revision"]) == 40
        assert cfg["block_size"] == 4 and cfg["budget"] == 2048 and cfg["steps"] == 300
    assert s1["indexer_revision"] == "aedae2c5304698f58d30c5ec709ebf5b37339b1d"
    assert s1["indexer_sha256"] == "e44193e5ef832b6220c8355fff92ac8f07d8c13538f584d2b6231b5dd2f4bc51"
    assert s1["mix_fineweb"] == 0.2 and len(s1["dataset_revision"]) == 40
    assert (s1["lambda_answer"], s1["mu_margin"], s1["seq_len"]) == (0.3, 0.1, 16384)
    assert (s2["lambda_answer"], s2["mu_margin"], s2["seq_len"]) == (0.3, 0.3, 65536)
    assert s2["restrict_kl"] is True and s2["kl_rows"] == 4096 and s2["step_offset"] == 300


def test_continue_from_v2_template_is_pinned() -> None:
    s1 = yaml.safe_load((ROOT / "configs/recall-16k-300b.yaml").read_text())
    c = yaml.safe_load((ROOT / "configs/continue-from-v2.yaml").read_text())
    assert c["indexer_repo"] == "higa2515/Qwen3.8-27B-QSA-BF16" and c["indexer_file"] == "indexers/qsa_indexers.pt"
    assert c["indexer_revision"] == "cf68114a27146533c90224ccda4bd2bd014a5795"   # the commit that put v2 in
    assert c["indexer_sha256"] == "6a6bb0decc88f3a8d411e423069f6634f2a5a69d832853ddf9b6504535d136bc"
    assert c["push_to_hub"] is False and c["step_offset"] == 600
    # the indexer shape and block unit must match the published GGUF; everything else follows stage 1
    for k in ("block_size", "budget", "indexer_heads", "indexer_head_dim", "rope_dim", "model_revision", "seq_len",
              "lambda_answer", "mu_margin", "margin", "mix_fineweb", "dataset_revision"):
        assert c[k] == s1[k], k
    assert set(c) == set(s1)
