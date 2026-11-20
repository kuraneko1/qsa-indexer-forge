#!/usr/bin/env python3
"""gen_train_16k.py - stage 1 (300b) training data: recall-grid records in 16k documents.

Each sample is "document + question" with the token positions of the records the question needs (answer_pos, first
token of each record) and of its one-character-off decoy records (decoy_pos). Use a seed different from every
evaluation set. The 300b run used:

  python scripts/recall_grid/gen_train_16k.py --out recall-train-16k.jsonl.gz --docs 200 --doc-tokens 16384 \
      --density 0.6 --tokenizer tokenizer.json --seed 1001

The filler is recall_grid's default (Python standard-library source of the running interpreter), so the output depends
on the Python version. Output lines: {"doc", "question", "fmt", "answer_pos", "decoy_pos", "type"} (qsa/data.py).
"""
import argparse, bisect, gzip, json, os, random, sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rg", os.path.join(HERE, "recall_grid.py"))
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)

TYPES = ["guid", "pointer", "ingredients", "quantities", "revision", "join", "code", "binding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--doc-tokens", type=int, default=8192)
    ap.add_argument("--density", type=float, default=0.6)
    ap.add_argument("--questions-per-doc", type=int, default=3)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--filler", nargs="*", default=None)
    a = ap.parse_args()

    tok = rg.Tok(a.tokenizer)
    filler = rg.load_filler(a.filler)
    fstarts = tok.starts(filler)
    nf = len(fstarts)
    rng = random.Random(a.seed)
    K = rg.Keys(rng)
    items = [rg.make_item(t, b, r, rng, K) for t in TYPES for b in range(rg.NBIN) for r in range(rg.REPS)]
    out = gzip.open(a.out, "wt", encoding="utf-8")
    written = 0
    for _ in range(a.docs):
        L, budget = a.doc_tokens, int(a.doc_tokens * a.density)
        pool = items[:]
        rng.shuffle(pool)
        chosen, used = [], 0
        for it in pool:
            cost = sum(tok.count("\n" + s + "\n\n") for s in it["blocks"] + it["decoys"])
            if used + cost > budget:
                break
            chosen.append(it)
            used += cost
        if not chosen:
            continue
        recs = []
        for it in chosen:
            for s in it["blocks"]:
                recs.append(("\n" + s + "\n\n", it["id"], "rec"))
            for s in it["decoys"]:
                recs.append(("\n" + s + "\n\n", it["id"], "dec"))
        rng.shuffle(recs)

        # spread the filler evenly between the records (and before the first one)
        rec_tokens = sum(tok.count(t) for t, _, _ in recs)
        fill_tokens = max(0, L - rec_tokens)
        slots = len(recs) + 1
        per = max(1, fill_tokens // slots)
        fpos = rng.randrange(0, max(1, nf - per * slots - 1))
        parts = []  # (text, item_id or None, kind or None)
        for i in range(slots):
            if fill_tokens > 0:
                s0, s1 = fstarts[min(fpos, nf - 1)], fstarts[min(fpos + per, nf - 1)]
                if s1 > s0:
                    parts.append((filler[s0:s1], None, None))
                fpos += per
            if i < len(recs):
                parts.append(recs[i])

        text = "".join(t for t, _, _ in parts)
        starts = tok.starts(text)
        rec_pos, dec_pos = {}, {}
        off = 0
        for seg, item_id, kind in parts:
            if item_id is not None:
                target = rec_pos if kind == "rec" else dec_pos
                target.setdefault(item_id, []).append(bisect.bisect_left(starts, off + 1))
            off += len(seg)

        for it in chosen[: a.questions_per_doc]:
            out.write(json.dumps({"doc": text, "question": it["q"], "fmt": it["fmt"],
                                  "answer_pos": rec_pos.get(it["id"], []),
                                  "decoy_pos": dec_pos.get(it["id"], []),
                                  "type": it["type"]}, ensure_ascii=False) + "\n")
            written += 1
    out.close()
    print("wrote", written, "samples")


if __name__ == "__main__":
    main()
