import math

import pytest
import torch
import torch.nn.functional as F

from qsa import (
    QSAIndexer,
    block_teacher_from_qk,
    attention_mass_capture,
    build_sparse_mask,
    indexer_kl_loss,
    pool_teacher_distribution,
    qsa_attention,
    select_blocks,
)


def rope_tables(t: int, d: int, base: float = 1e7) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    freqs = torch.outer(torch.arange(t, dtype=torch.float32), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


@pytest.mark.parametrize("t,r", [(32, 4), (17, 4), (12, 3)])
def test_full_budget_matches_dense(t: int, r: int) -> None:
    torch.manual_seed(0)
    b, h, d, hidden = 2, 3, 8, 16
    idx = QSAIndexer(hidden, num_heads=2, head_dim=8, rope_dim=4, block_size=r)
    x = torch.randn(b, t, hidden)
    cos, sin = rope_tables(t, 4)

    scores, valid = idx(x, cos, sin)
    sel = select_blocks(scores, valid, token_budget=t, block_size=r)

    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    got = qsa_attention(q, k, v, sel, valid, block_size=r)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_sparse_mask_excludes_unselected_blocks() -> None:
    t, r = 16, 4
    nb = t // r
    valid = torch.ones(1, t, nb, dtype=torch.bool)
    block_ids = torch.zeros(1, t, 1, dtype=torch.long)  # every query selects block 0

    mask = build_sparse_mask(block_ids, valid, t, r)[0]
    query = 15
    assert mask[query, :4].all(), "selected block 0 must be visible"
    assert not mask[query, 4:12].any(), "blocks 1-2 must be excluded"
    assert mask[query, 12:16].all(), "the incomplete tail block is always visible"
    assert not mask[query, 16:].any() and not mask[:, query + 1:].any()


def test_indexer_selects_salient_block() -> None:
    torch.manual_seed(0)
    t, r, hidden, hd = 16, 4, 8, 4
    idx = QSAIndexer(hidden, num_heads=1, head_dim=hd, rope_dim=0, block_size=r)
    eye = torch.eye(hd, hidden)
    with torch.no_grad():
        idx.q_proj.weight.copy_(eye)
        idx.k_proj.weight.copy_(eye)

    x = torch.zeros(1, t, hidden)
    x[0, 9:12, 0] = 1.0  # block 2 (tokens 8-11) points along e0
    x[0, 15, 0] = 1.0    # query token points along e0
    cos, sin = rope_tables(t, 2)  # rope_dim=0, tables unused

    scores, valid = idx(x, cos, sin)
    sel = select_blocks(scores, valid, token_budget=r, block_size=r)
    assert sel[0, 15, 0].item() == 2


def test_capture_and_kl() -> None:
    t, r = 16, 4
    nb = t // r
    scores = torch.zeros(1, t, nb)
    pos = torch.arange(t)
    # teacher mass concentrated on block 2 for every sufficiently long query
    attn = torch.zeros(1, t, t)
    for i in range(t):
        if i >= 2 * r + r - 1:
            attn[0, i, 8:12] = 0.25
        else:
            attn[0, i, : i + 1] = 1.0 / (i + 1)

    sel = torch.full((1, t, 1), 2, dtype=torch.long)
    cap = attention_mass_capture(attn, sel, block_size=r)
    assert torch.isclose(cap[0, 15], torch.tensor(1.0), atol=1e-6)

    target, valid = pool_teacher_distribution(attn, block_size=r)
    assert torch.allclose(target.sum(-1)[0][valid.any(-1)], torch.ones(int(valid.any(-1).sum())))

    # scores equal to log(target) make the KL exactly zero
    log_target = torch.where(target > 0, target.clamp_min(1e-12).log(), torch.full_like(target, -float("inf")))
    log_target = log_target.masked_fill(~valid[None], -float("inf"))
    loss = indexer_kl_loss(log_target, attn, block_size=r)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)

    # and the KL is differentiable through the indexer
    idx = QSAIndexer(8, num_heads=1, head_dim=4, rope_dim=0, block_size=r)
    x = torch.randn(1, t, 8)
    cos, sin = rope_tables(t, 2)
    s, _ = idx(x, cos, sin)
    loss = indexer_kl_loss(s, attn, block_size=r)
    loss.backward()
    assert idx.q_proj.weight.grad is not None
    assert torch.isfinite(idx.q_proj.weight.grad).all()


def test_block_teacher_matches_naive() -> None:
    torch.manual_seed(0)
    b, h, t, d, r = 2, 3, 24, 8, 4
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)

    target, valid = block_teacher_from_qk(q, k, block_size=r, query_chunk=7)

    logits = (q @ k.transpose(-1, -2)) / (d ** 0.5)
    causal = torch.arange(t)[None, :] <= torch.arange(t)[:, None]
    logits = logits.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(logits, -1).sum(1)
    naive = probs[:, :, : (t // r) * r].reshape(b, t, t // r, r).amax(-1)
    naive = naive.masked_fill(~valid[None], 0.0)
    denom = naive.sum(-1, keepdim=True)
    naive = torch.where(denom > 0, naive / denom.clamp_min(1e-12), naive)

    torch.testing.assert_close(target, naive, atol=1e-6, rtol=1e-5)
    assert valid.shape == (t, t // r)
    assert int(valid.any(-1).sum()) == t - (r - 1)  # early queries see no complete block


def test_per_query_kl_matches_full_kl() -> None:
    from qsa import indexer_kl_from_target, indexer_kl_per_query

    torch.manual_seed(0)
    t, r = 24, 4
    target = torch.zeros(1, t, t // r)
    pos = torch.arange(t)
    valid = (torch.arange(t // r)[None, :] * r + r - 1) <= pos[:, None]
    for i in range(t):
        if valid[i].any():
            target[0, i] = torch.rand(t // r) * valid[i]
            target[0, i] /= target[0, i].sum()
    scores = torch.randn(1, t, t // r).masked_fill(~valid[None], float("-inf"))
    per_q = indexer_kl_per_query(scores, target, valid)[0]
    full = indexer_kl_from_target(scores, target, valid)
    assert abs(per_q[valid.any(-1)].mean().item() - full.item()) < 1e-5
