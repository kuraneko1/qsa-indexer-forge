#!/usr/bin/env python3
"""gen_mixup_64k.py - stage 2 (mix64k) training data: 64k documents weighted toward look-alike ("mix-up") errors.

Same generator as the recall-grid evaluation sets (recall_grid.py: make_item, background, the placement of cmd_gen),
with three changes for training:
  1. every run uses its own seed (base + i) and its own slice of filler books, so documents do not repeat
     (cmd_gen picks the filler excerpt from h and L only, not from the seed);
  2. answer_pos / decoy_pos list EVERY token of each record (not only its first token), so all blocks of the needed
     records are supervised and all blocks of the one-character-off decoy records are pushed down;
  3. questions are drawn per document with type weights (default: revision 4, pointer 3, code 2, guid 2, others 1),
     following the error mix of an earlier indexer on an exploration set.

The mix64k run used PG-19 train books (one shard, one .txt per book) and:

  python scripts/recall_grid/gen_mixup_64k.py --out mixup-64k.jsonl.gz --runs 25 --q-per-doc 3 --length 64000 \
      --tokenizer tokenizer.json --filler-dir BOOKS_DIR --seed 2026111700

Output lines: {"doc","question","fmt","answer_pos","decoy_pos","type","id","case","run","hid",...} (qsa/data.py format).
"""
import argparse, bisect, glob, gzip, hashlib, json, math, os, random, sys, importlib.util

RGP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall_grid.py")
spec = importlib.util.spec_from_file_location("rg", RGP); rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
WEIGHTS = {"revision": 4, "pointer": 3, "code": 2, "guid": 2, "binding": 1, "join": 1, "ingredients": 1, "quantities": 1}


def run_docs(seed, L, filler, tok, density):
    """cmd_gen for one length, returning documents with token spans of every block/decoy record."""
    rng = random.Random(seed); K = rg.Keys(rng)
    items = [rg.make_item(t, b, r, rng, K) for t in rg.TYPES for b in range(rg.NBIN) for r in range(rg.REPS)]
    for it in items:
        it["btok"] = [tok.count("\n" + s + "\n\n") for s in it["blocks"]]; it["dtok"] = [tok.count("\n" + s + "\n\n") for s in it["decoys"]]
    fo = tok.starts(filler); N = len(fo)
    total_rec = sum(sum(it["btok"]) + sum(it["dtok"]) for it in items)
    Ldoc = L - rg.OVERHEAD; H = max(1, math.ceil(total_rec / ((density - 0.01) * Ldoc)))
    groups, load = [[] for _ in range(H)], [0] * H
    for it in sorted(items, key=lambda it: (-sum(it["btok"]) - sum(it["dtok"]), it["id"])):
        h = min(range(H), key=lambda h: (load[h], len(groups[h]), h)); groups[h].append(it); load[h] += sum(it["btok"]) + sum(it["dtok"])
    rl = random.Random(seed + L); docs = []
    for h, grp in enumerate(groups):
        rec_tok = sum(sum(it["btok"]) + sum(it["dtok"]) for it in grp)
        KB = rg.Keys(rl); KB.used = set(K.used); bg, bg_tok = [], 0
        while rec_tok + bg_tok < density * Ldoc:
            blk = rg.background(len(bg), rl, KB); bg.append((blk, tok.count("\n" + blk + "\n\n"))); bg_tok += bg[-1][1]
        K.used |= KB.used; F = Ldoc - rec_tok - bg_tok
        if N - F - 2 <= 0: raise SystemExit(f"filler too short: {N} tokens for F={F}")
        s = rl.randrange(0, N - F - 2); c0 = filler.find("\n", fo[s]) + 1; s = bisect.bisect_left(fo, c0)
        ce = fo[s + F] if s + F < N else len(filler)
        place = []
        for i, it in enumerate(grp):
            for j, blk in enumerate(it["blocks"]): place.append((rl.uniform(it["bin"] / rg.NBIN, (it["bin"] + 1) / rg.NBIN), blk, i, "b", it["btok"][j]))
            for j, blk in enumerate(it["decoys"]): place.append((rl.uniform(0, 1), blk, i, "d", it["dtok"][j]))
        for blk, bt in bg: place.append((rl.uniform(0, 1), blk, -1, "g", bt))
        place.sort(key=lambda p: p[0])
        pieces, cum, prev_f, cprev, doc_len, spans = [], 0, 0, fo[s], 0, []
        for frac, blk, i, role, bt in place:
            f = max(prev_f, min(F, int(round(frac * Ldoc - cum))))
            c = fo[s + f] if s + f < N else ce
            nl = filler.find("\n", c, ce); c = ce if nl < 0 else nl + 1; c = max(c, cprev)
            pieces.append(filler[cprev:c]); doc_len += c - cprev
            spans.append((doc_len, doc_len + len(blk) + 3, i, role)); pieces.append("\n" + blk + "\n\n"); doc_len += len(blk) + 3
            cprev, prev_f, cum = c, f, cum + bt
        pieces.append(filler[cprev:ce]); doc = "".join(pieces)
        do = tok.starts(doc); pos = {}
        for c_start, c_end, i, role in spans:
            t0 = bisect.bisect_right(do, c_start + 1) - 1; t1 = bisect.bisect_left(do, c_end)
            pos.setdefault((i, role), []).extend(range(t0, max(t0 + 1, t1)))
        docs.append({"hid": f"L{L}-h{h}", "doc": doc, "doc_tokens": len(do),
                     "items": [dict(id=it["id"], type=it["type"], case=it.get("case"), q=it["q"], fmt=it["fmt"],
                                    answer_pos=sorted(set(pos[(i, "b")])), decoy_pos=sorted(set(pos.get((i, "d"), []))))
                               for i, it in enumerate(grp)]})
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--q-per-doc", type=int, default=3); ap.add_argument("--length", type=int, default=64000)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--filler-dir", required=True)
    ap.add_argument("--seed", type=int, default=2026111700); ap.add_argument("--density", type=float, default=rg.DENSITY)
    a = ap.parse_args()
    tok = rg.Tok(a.tokenizer, 3.5)
    books = sorted(glob.glob(os.path.join(a.filler_dir, "*.txt")))
    if len(books) < a.runs: raise SystemExit(f"need at least {a.runs} books, found {len(books)}")
    rq = random.Random(a.seed); written = 0; types = {}
    out = gzip.open(a.out, "wt", encoding="utf-8")
    for run in range(a.runs):
        mine = books[run::a.runs]                               # disjoint book slice per run
        filler = rg.load_filler(mine, a.filler_dir)
        docs = run_docs(a.seed + run, a.length, filler, tok, a.density)
        for d in docs:
            its = d["items"]; w = [WEIGHTS.get(it["type"], 1) for it in its]; chosen = []
            while len(chosen) < min(a.q_per_doc, len(its)):
                it = rq.choices(its, weights=w)[0]
                if it not in chosen: chosen.append(it)
            for it in chosen:
                out.write(json.dumps({"doc": d["doc"], "question": it["q"], "fmt": it["fmt"], "answer_pos": it["answer_pos"],
                                      "decoy_pos": it["decoy_pos"], "type": it["type"], "id": it["id"], "case": it["case"],
                                      "run": run, "hid": d["hid"], "doc_tokens": d["doc_tokens"],
                                      "filler_sha256": hashlib.sha256(filler.encode()).hexdigest()[:16]}, ensure_ascii=False) + "\n")
                written += 1; types[it["type"]] = types.get(it["type"], 0) + 1
        print(f"run {run}: {len(docs)} docs, {len(mine)} books, doc tokens {[d['doc_tokens'] for d in docs]}", flush=True)
    out.close()
    print(f"wrote {written} samples to {a.out}; types {dict(sorted(types.items()))}")


if __name__ == "__main__":
    main()
