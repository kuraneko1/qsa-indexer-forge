import torch

from qsa.mm_capture import MMCapture, block_modality, merge_summaries


def _case(target_mass: dict[int, dict[int, float]], t: int, r: int = 4, budget: int = 8):
    nb = t // r
    target = torch.zeros(1, t, nb)
    for i, blocks in target_mass.items():
        for b, value in blocks.items():
            target[0, i, b] = value
    pos = torch.arange(t)
    starts = torch.arange(nb) * r
    valid = (starts[None, :] + r - 1) <= pos[:, None]
    scores = target.clone()
    is_image = torch.zeros(t, dtype=torch.bool)
    is_image[t // 2:] = True
    return target, valid, scores, is_image


def test_block_modality():
    is_image = torch.tensor([False] * 8 + [True] * 8)
    mod = block_modality(is_image, 4, 4)
    assert mod.tolist() == [False, False, True, True]


def test_capture_splits_by_modality():
    t, r, budget = 32, 4, 8
    target, valid, scores, is_image = _case(
        {15: {0: 0.6, 1: 0.4}, 31: {4: 0.7, 5: 0.3}}, t, r, budget)
    cap = MMCapture()
    cap.add(target, valid, scores, is_image, r, budget)
    summary = cap.summary()
    cells = summary["cells"]
    assert cells["text->text"]["learned"] == 1.0
    assert cells["text->text"]["oracle"] == 1.0
    assert cells["image->image"]["learned"] == 1.0
    assert cells["image->image"]["oracle"] == 1.0
    assert "text->image" not in cells  # no teacher mass there for these queries


def test_bad_scores_lower_capture_than_oracle():
    t, r, budget = 32, 4, 8
    target, valid, scores, is_image = _case({31: {4: 0.7, 5: 0.3}}, t, r, budget)
    scores = -target  # selects the lowest-mass blocks
    cap = MMCapture()
    cap.add(target, valid, scores, is_image, r, budget)
    row = cap.summary()["cells"]["image->image"]
    assert row["learned"] < row["oracle"] == 1.0


def test_excludes_fully_visible_queries():
    t, r, budget = 16, 4, 8  # kb = 2; early queries see <= 2 blocks
    target, valid, scores, is_image = _case({15: {0: 1.0}}, t, r, budget)
    cap = MMCapture()
    cap.add(target, valid, scores, is_image, r, budget)
    summary = cap.summary()
    assert summary["queries_excluded_N_le_K"] > 0
    assert summary["queries_seen"] == t


def test_merge_summaries_weights_by_mass():
    a = {"cells": {"text->text": {"queries": 1, "mass": 1.0, "learned": 0.8, "oracle": 1.0}}}
    b = {"cells": {"text->text": {"queries": 1, "mass": 3.0, "learned": 0.4, "oracle": 1.0}}}
    merged = merge_summaries([a, b])["cells"]["text->text"]
    assert abs(merged["learned"] - 0.5) < 1e-9
    assert merged["queries"] == 2


def test_train_capture_handles_3d_scores():
    from qsa.train_mm import _capture

    torch.manual_seed(0)
    t, nb, kb, r = 64, 16, 4, 4
    target = torch.rand(1, t, nb)
    pos = torch.arange(t)
    valid = (torch.arange(nb)[None, :] * r + r - 1) <= pos[:, None]
    target = target * valid[None]
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-9)
    scores = torch.randn(1, t, nb)
    eligible = valid.any(-1)
    value = _capture(scores, target, valid, eligible, kb)
    assert 0.0 <= value <= 1.0
