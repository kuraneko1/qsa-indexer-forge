"""Streaming token blocks for distillation and evaluation."""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import torch


def iter_token_blocks(
    tokenizer,
    seq_len: int,
    n_blocks: int,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    split: str = "train",
    text_key: str = "text",
    skip_docs: int = 0,
    revision: str | None = None,
) -> Iterator[torch.Tensor]:
    """Yield ``n_blocks`` contiguous ``seq_len``-token chunks of streamed text.

    Documents are concatenated with an EOS token between them; the token stream
    is cut into equal blocks, so a block may span documents.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, subset, split=split, streaming=True,
                      revision=revision)
    if skip_docs:
        ds = ds.skip(skip_docs)

    eos = tokenizer.eos_token_id
    buffer: list[int] = []
    produced = 0
    for example in ds:
        ids = tokenizer(example[text_key], add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        buffer.append(eos)
        while len(buffer) >= seq_len and produced < n_blocks:
            yield torch.tensor(buffer[:seq_len], dtype=torch.long)
            del buffer[:seq_len]
            produced += 1
        if produced >= n_blocks:
            return


def iter_local_sequences(path, tokenizer, seq_len, n_blocks, block_size: int = 4, instr: str = ""):
    """Yield question-conditioned sequences from a local jsonl(.gz) for distillation.

    Each record is {"doc", "question", "fmt", "answer_pos"}; answer_pos are token
    offsets of the records that answer the question inside doc. The doc tail is kept
    so that doc + question fits in seq_len.
    """
    import gzip
    import json

    opener = gzip.open if str(path).endswith(".gz") else open
    made = 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if made >= n_blocks:
                return
            if not line.strip():
                continue
            rec = json.loads(line)
            q_text = "\n\nQuestion: " + rec["question"] + "\n\n"
            if instr:
                q_text += instr.format(rec.get("fmt", ""))
            q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]
            doc_ids = tokenizer(rec["doc"], add_special_tokens=False)["input_ids"]
            keep = seq_len - len(q_ids) - 2
            if keep < 512:
                continue
            shift = max(0, len(doc_ids) - keep)
            doc_ids = doc_ids[shift:]
            ids = doc_ids + q_ids
            eos = tokenizer.eos_token_id
            if eos is not None:
                ids = ids + [eos]
            raw = rec.get("answer_pos", [])
            if not raw or any(q - shift < 0 for q in raw):
                continue  # the doc head was cut and an answer record fell outside: skip the sequence
            pos = [q - shift for q in raw]
            blocks = sorted({q // block_size for q in pos})
            dpos = [q - shift for q in rec.get("decoy_pos", []) if q - shift >= 0]
            dblocks = sorted({q // block_size for q in dpos})
            if not blocks:
                continue
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "q_start": len(doc_ids),
                "answer_blocks": blocks,
                "decoy_blocks": dblocks,
            }
            made += 1


def combine_iterators(primary, secondary, primary_ratio: float):
    """Interleave two iterators at a fixed ratio, in tenths (e.g. 0.8 local / 0.2 web).

    Item n (1-based) comes from ``primary`` when ``n % 10 < round(primary_ratio * 10)``. The 300b run used an
    earlier version with ``<=``, which turned its configured 0.7 into 0.8; ``primary_ratio=0.8`` here gives the
    same order (web at n % 10 in {8, 9}).
    """
    it_p, it_s = iter(primary), iter(secondary)
    n = 0
    while True:
        n += 1
        use_primary = (n % 10) < round(primary_ratio * 10)
        try:
            yield next(it_p) if use_primary else next(it_s)
        except StopIteration:
            return
