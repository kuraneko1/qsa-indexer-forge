"""Needle-in-a-haystack retrieval probe for QSA evaluation.

Builds multiple-choice passkey prompts of a requested token length and scores
the four answer letters at the final position. One forward per prompt, so the
probe works with the patched attention (which does not support cached decoding).
"""

from __future__ import annotations

import random

import torch


def _filler_ids(tokenizer, docs, need: int) -> list[int]:
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer(doc, add_special_tokens=False)["input_ids"])
        ids.append(tokenizer.eos_token_id)
        if len(ids) >= need:
            break
    return ids[:need]


def build_examples(tokenizer, docs, lengths, n_per_length: int = 8, seed: int = 0) -> list[dict]:
    """Return prompts with ``length`` target tokens and four lettered options."""
    rng = random.Random(seed)
    option_ids = [tokenizer.encode(" " + letter, add_special_tokens=False)[-1]
                  for letter in "ABCD"]
    examples = []
    for target in lengths:
        for _ in range(n_per_length):
            key = f"{rng.randrange(10000, 99999)}"
            others = set()
            while len(others) < 3:
                cand = f"{rng.randrange(10000, 99999)}"
                if cand != key:
                    others.add(cand)
            options = [key, *others]
            rng.shuffle(options)
            answer = options.index(key)

            # Reserve room for the needle and the question, then split the rest.
            body_tokens = max(target - 120, 256)
            ids = _filler_ids(tokenizer, docs, body_tokens)
            half = len(ids) // 2
            prefix = tokenizer.decode(ids[:half])
            suffix = tokenizer.decode(ids[half:])
            prompt = (
                f"{prefix}\nThe passkey is {key}.\n{suffix}\n\n"
                "Question: What is the passkey?\n"
                f"A) {options[0]}\nB) {options[1]}\n"
                f"C) {options[2]}\nD) {options[3]}\nAnswer:"
            )
            input_ids = tokenizer(prompt, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"]
            examples.append({
                "length": int(input_ids.shape[1]),
                "target": target,
                "input_ids": input_ids,
                "option_ids": option_ids,
                "answer": answer,
                "key": key,
            })
    return examples


@torch.inference_mode()
def retrieval_accuracy(model, examples) -> dict:
    from qsa.hf_qsa import text_model

    per_length: dict[int, list[int]] = {}
    for ex in examples:
        ids = ex["input_ids"].cuda()
        hidden = text_model(model)(input_ids=ids).last_hidden_state
        logits = model.lm_head(hidden[:, -1]).float()[0]
        scores = [logits[tid].item() for tid in ex["option_ids"]]
        pred = int(torch.tensor(scores).argmax())
        per_length.setdefault(ex["length"], []).append(int(pred == ex["answer"]))
        del hidden, logits

    return {
        str(length): sum(hits) / len(hits)
        for length, hits in sorted(per_length.items())
    }
