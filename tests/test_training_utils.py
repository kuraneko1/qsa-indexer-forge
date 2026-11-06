import torch

from qsa.train_mm import _capture


def test_capture_does_not_select_invalid_blocks():
    scores = torch.tensor([[[1.0, 2.0, 100.0]]])
    target = torch.tensor([[[0.25, 0.75, 0.0]]])
    valid = torch.tensor([[True, True, False]])
    eligible = torch.tensor([True])
    assert _capture(scores, target, valid, eligible, kb=1) == 0.75
