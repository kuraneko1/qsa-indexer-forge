"""Modality-aware block capture accounting for MM0.

Capture is the fraction of the teacher's block mass that the selected blocks
carry. MM0 splits it into text->text, text->image, image->text and
image->image, and excludes queries whose whole context is visible (N <= K),
because those do not demonstrate sparsification.
"""

from __future__ import annotations

import math

import torch


def block_modality(is_image: torch.Tensor, block_size: int, n_blocks: int) -> torch.Tensor:
    """``[n_blocks]`` bool: block contains at least one image token."""
    t = is_image.shape[0]
    complete = min(n_blocks, t // block_size)
    if complete == 0:
        return torch.zeros(0, dtype=torch.bool, device=is_image.device)
    blocks = is_image[: complete * block_size].view(complete, block_size)
    return blocks.any(-1)


class MMCapture:
    """Accumulate learned/oracle capture per (query modality, key modality)."""

    def __init__(self) -> None:
        self.sums: dict[tuple[bool, bool, str], float] = {}
        self.mass: dict[tuple[bool, bool], float] = {}
        self.queries: dict[tuple[bool, bool], int] = {}
        self.excluded = 0
        self.seen = 0

    def add(self, target: torch.Tensor, valid: torch.Tensor, scores: torch.Tensor,
            is_image: torch.Tensor, block_size: int, budget: int,
            query_mask: torch.Tensor | None = None) -> None:
        """``target``/``scores``: ``[1, T, nb]``; ``valid``: ``[T, nb]``."""
        nb = target.shape[-1]
        if nb == 0:
            return
        kb = min(nb, math.ceil(budget / block_size))
        n_valid = valid.sum(-1)
        self.seen += int(n_valid.shape[0])
        self.excluded += int((n_valid <= kb).sum())

        blk_mod = block_modality(is_image, block_size, nb)
        learned = torch.topk(scores, kb, dim=-1).indices
        oracle = torch.topk(target, kb, dim=-1).indices
        learned_oh = torch.zeros_like(target, dtype=torch.bool).scatter_(-1, learned, True)
        oracle_oh = torch.zeros_like(target, dtype=torch.bool).scatter_(-1, oracle, True)
        mass = target * valid[None]
        eligible = (n_valid > kb)
        if query_mask is not None:
            eligible = eligible & query_mask

        for query_image in (False, True):
            qmask = (is_image == query_image) & eligible
            for key_image in (False, True):
                bmask = (blk_mod == key_image)
                m = (mass * bmask).sum(-1)[0]
                ok = qmask & (m > 0)
                if not bool(ok.any()):
                    continue
                sel_mass = (mass * bmask * learned_oh).sum(-1)[0]
                ora_mass = (mass * bmask * oracle_oh).sum(-1)[0]
                self.mass[(query_image, key_image)] = self.mass.get((query_image, key_image), 0.0) + m[ok].sum().item()
                self.queries[(query_image, key_image)] = self.queries.get((query_image, key_image), 0) + int(ok.sum())
                for name, values in (("learned", sel_mass), ("oracle", ora_mass)):
                    key = (query_image, key_image, name)
                    self.sums[key] = self.sums.get(key, 0.0) + values[ok].sum().item()

    def summary(self) -> dict:
        out = {
            "queries_seen": self.seen,
            "queries_excluded_N_le_K": self.excluded,
            "cells": {},
        }
        for qm in (False, True):
            for km in (False, True):
                mass = self.mass.get((qm, km), 0.0)
                if mass <= 0:
                    continue
                out["cells"][f"{'image' if qm else 'text'}->{'image' if km else 'text'}"] = {
                    "queries": self.queries.get((qm, km), 0),
                    "mass": mass,
                    "learned": self.sums.get((qm, km, "learned"), 0.0) / mass,
                    "oracle": self.sums.get((qm, km, "oracle"), 0.0) / mass,
                }
        return out


def merge_summaries(summaries: list[dict]) -> dict:
    """Merge per-layer summaries by weighted mass."""
    merged: dict = {"cells": {}}
    for s in summaries:
        for cell, row in s["cells"].items():
            agg = merged["cells"].setdefault(cell, {"queries": 0, "mass": 0.0,
                                                    "_learned": 0.0, "_oracle": 0.0})
            agg["queries"] += row["queries"]
            agg["mass"] += row["mass"]
            agg["_learned"] += row["learned"] * row["mass"]
            agg["_oracle"] += row["oracle"] * row["mass"]
    for cell, agg in merged["cells"].items():
        agg["learned"] = agg.pop("_learned") / agg["mass"]
        agg["oracle"] = agg.pop("_oracle") / agg["mass"]
    return merged
