#!/usr/bin/env python3
"""recall_grid.py - a structured long-context recall benchmark: context length x depth, eight task types.

Plants the records of 240 lookup items (8 types x 10 depth bins x 3) at known depths in filler text, at several context
lengths, asks one question per item and scores the exact answer. Task types: GUID lookup, pointer chain, recipe
ingredients, recipe quantities, latest revision, table join, code trace, bindings. Every lookup has a hard negative (a key
one character off), and unasked look-alike records top every document up to 15% records, so distractor density is the
same at every length. The same 240 items are planted at every length, so columns differ only in length.

Works with any OpenAI-compatible chat endpoint (llama.cpp server, vLLM, SGLang, hosted APIs). Standard library only;
the optional `tokenizers` package gives exact token positions.

  python recall_grid.py gen    --out grid.jsonl.gz [--tokenizer tokenizer.json] [--lengths 8000,32000,...] [--filler PATH ...]
  python recall_grid.py check  --data grid.jsonl.gz
  python recall_grid.py run    --data grid.jsonl.gz --base-url http://127.0.0.1:8080/v1 --label mymodel [--results results.jsonl]
  python recall_grid.py report --results results.jsonl [--data grid.jsonl.gz] [--html report.html] [--name mymodel="My model"]

See README.md for details.
"""
import argparse, bisect, collections, glob, gzip, hashlib, html, json, math, os, random, re, sys, sysconfig, time, urllib.error, urllib.request

LENGTHS = [8000, 16000, 32000, 64000, 128000, 192000, 255000]
TYPES = ["guid", "pointer", "ingredients", "quantities", "revision", "join", "code", "binding"]
TNAME = {"guid": "GUID lookup", "pointer": "Pointer chain", "ingredients": "Recipe ingredients", "quantities": "Recipe quantities",
         "revision": "Latest revision", "join": "Table join", "code": "Code trace", "binding": "Bindings"}
NBIN, REPS, DENSITY, OVERHEAD, SEED = 10, 3, 0.15, 400, 20261108
VOCAB = ("flour sugar butter cinnamon eggplant honey ginger garlic onion carrot celery lentil rice barley almond walnut raisin date fig "
         "apricot lemon lime mango coconut vanilla nutmeg clove cardamom saffron paprika cumin turmeric fennel basil thyme rosemary sage "
         "mint parsley dill chive leek shallot spinach kale cabbage beet turnip").split()
VSET = set(VOCAB)
HEX = "0123456789abcdef"
SYSTEM = "You answer questions about the attached document, which has data records mixed into it. Use only what the records say."
FMT = {"guid": "the value only", "pointer": "the value only", "ingredients": "ingredient names separated by commas",
       "quantities": "ingredient=amount pairs separated by commas", "revision": "the value, or null", "join": "the final value only",
       "code": "the returned string, without quotes", "binding": "the host, or null"}
INSTR = "If it helps, work it out in a few short lines first. Then end with one line of the form: ANSWER: <{}>"
TEXT_EXT = {".py", ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".rs", ".go", ".java", ".js", ".ts", ".md", ".rst", ".txt"}


# ================================================================ items
class Keys:
    def __init__(self, rng): self.rng, self.used = rng, set()
    def hx(self, n): return "".join(self.rng.choice(HEX) for _ in range(n))
    def new(self, prefix, n):
        while True:
            k = prefix + self.hx(n)
            if k not in self.used: self.used.add(k); return k
    def near(self, key, last=False):
        """Same key with one hex character changed in its second half (the last one if last=True)."""
        body = re.search(r"[0-9a-f]+$", key).group(0); head = key[:len(key) - len(body)]
        while True:
            p = len(body) - 1 if last else self.rng.randrange(len(body) // 2, len(body))
            k = head + body[:p] + self.rng.choice([c for c in HEX if c != body[p]]) + body[p + 1:]
            if k not in self.used: self.used.add(k); return k
    def uuid(self): return "-".join(self.hx(n) for n in (8, 4, 4, 4, 12))


def make_item(t, b, r, rng, K):
    idx = b * REPS + r
    it = {"id": f"{t}-b{b}-r{r}", "type": t, "bin": b, "rep": r, "fmt": FMT[t], "blocks": [], "decoys": []}
    if t == "guid":
        key, val = K.new("node_", 12), K.uuid()
        it["blocks"] = [f"[node] {key} value={val}"]; it["decoys"] = [f"[node] {K.near(key)} value={K.uuid()}"]
        it.update(case="lookup", q=f"What is the value of {key}?", want=val, args={"start": key})
    elif t == "pointer":
        h = 2 + idx % 3; nodes = [K.new("node_", 12) for _ in range(h)]; val = K.uuid()
        it["blocks"] = [f"[node] {nodes[i]} next={nodes[i + 1]}" for i in range(h - 1)] + [f"[node] {nodes[-1]} value={val}"]
        it["decoys"] = [f"[node] {K.near(nodes[1])} value={K.uuid()}"]
        it.update(case=f"{h} hops", want=val, args={"start": nodes[0]},
                  q=f"Start at {nodes[0]} and follow the next= links until you reach a node that has a value. What is that value?")
    elif t in ("ingredients", "quantities"):
        deep = idx % 2 == 0
        root, A, B, C = (K.new("rcp_", 6) for _ in range(4)); n = rng.sample(VOCAB, 7)
        amt = lambda: rng.randint(1, 3); bat = lambda: rng.randint(1, 2)
        comps = {root: [(amt(), n[0]), (amt(), n[1]), (bat(), A), (bat(), B)],
                 A: [(amt(), n[2]), (amt(), n[3])] + ([(bat(), C)] if deep else [(amt(), n[5])]),
                 B: [(amt(), n[4]), (amt(), n[0]), (amt(), n[6])]}
        if deep: comps[C] = [(amt(), n[5]), (amt(), n[2])]
        tot = {}
        def expand(rid, m):
            for a, x in comps[rid]:
                if x in comps: expand(x, m * a)
                else: tot[x] = tot.get(x, 0) + m * a
        expand(root, 1)
        def rec(rid, parts):
            parts = parts[:]; rng.shuffle(parts)
            return f"[recipe] {rid} = " + " + ".join(f"{a} {x}" if x in VSET else f"{a} {'batch' if a == 1 else 'batches'} of {x}" for a, x in parts)
        it["blocks"] = [rec(rid, p) for rid, p in comps.items()]
        it["decoys"] = [rec(K.near(A), [(amt(), x) for x in rng.sample([v for v in VOCAB if v not in n], 2)])]
        it["case"] = "deep" if deep else "shallow"; it["args"] = {"start": root}
        if t == "ingredients":
            it.update(want=sorted(tot), q=f"List every base ingredient needed to make {root}, expanding the sub-recipes it uses (and theirs). Name each ingredient once.")
        else:
            it.update(want=dict(sorted(tot.items())), q=f"How many units of each base ingredient does one batch of {root} need in total? Multiply amounts through the sub-recipes and add up repeated ingredients.")
    elif t == "revision":
        key = K.new("item_", 8); case = ["latest value", "latest deleted", "deleted then restored"][idx % 3]
        revs = sorted(rng.sample(range(1, 40), 3)); vals = [K.hx(8) for _ in revs]
        dele = {"latest value": -1, "latest deleted": 2, "deleted then restored": 1}[case]
        it["blocks"] = [f"[rev] {key} rev {rv} deleted" if i == dele else f"[rev] {key} rev {rv} value={v}" for i, (rv, v) in enumerate(zip(revs, vals))]
        it["decoys"] = [f"[rev] {K.near(key)} rev {revs[-1] + rng.randint(1, 9)} value={K.hx(8)}"]
        it.update(case=case, want=None if dele == 2 else vals[-1], args={"start": key},
                  q=f"What is the current value of {key}? The record with the highest rev number wins; if that record says deleted, the answer is null.")
    elif t == "join":
        m = 2 + idx % 3; ks = [K.new("k_", 6) for _ in range(m + 1)]; tabs = [K.new("tbl_", 5) for _ in range(m)]
        for j in range(m):
            rows = [(ks[j], ks[j + 1]), (K.near(ks[j]), K.new("k_", 6))]
            if j >= 1: rows.append((ks[j - 1], K.new("k_", 6)))  # the previous key, in the wrong table
            while len(rows) < 5: rows.append((K.new("k_", 6), K.new("k_", 6)))
            rng.shuffle(rows)
            it["blocks"].append(f"[table {tabs[j]}] key -> value\n" + "\n".join(f"{a} -> {v}" for a, v in rows))
        it.update(case=f"{m} tables", want=ks[m], args={"start": ks[0], "tables": tabs},
                  q=f"Start with key {ks[0]} and look it up in table {tabs[0]}. Use each value you get as the key in the next table, in this order: {', '.join(tabs)}. What is the value from the last table?")
    elif t == "code":
        d = 1 + idx % 3; fs = [K.new("f_", 8) for _ in range(d)]; lit = K.hx(8)
        for i, f in enumerate(fs):
            if i < d - 1:
                style, call = rng.choice(["direct", "var", "dict"]), f"{fs[i + 1]}()"
                body = {"direct": f"    return {call}", "var": f"    result = {call}\n    return result",
                        "dict": f"    options = {{'primary': {call}, 'fallback': '{K.hx(8)}'}}\n    return options['primary']"}[style]
            else:
                body = rng.choice([f"    return '{lit}'", f"    options = {{'primary': '{lit}', 'fallback': '{K.hx(8)}'}}\n    return options['primary']"])
            it["blocks"].append(f"def {f}():\n{body}")
        it["decoys"] = [f"def {K.near(fs[0])}():\n    return '{K.hx(8)}'"]
        it.update(case=f"depth {d}", want=lit, args={"start": fs[0]}, q=f"What string does calling {fs[0]}() return?")
    elif t == "binding":
        case = ["verified", "verified + wrong key", "unverified only", "wrong key only"][idx % 4]
        a = K.new("asset_", 12); na = K.near(a, last=True); h1, h2 = K.new("host-", 6), K.new("host-", 6)
        recs = {"verified": [(a, h1, "verified"), (a, h2, "unverified")], "verified + wrong key": [(a, h1, "verified"), (na, h2, "verified")],
                "unverified only": [(a, h1, "unverified"), (na, h2, "verified")], "wrong key only": [(na, h1, "verified")]}[case]
        it["blocks"] = [f"[binding] {x} -> {h} ({v})" for x, h, v in recs]
        it.update(case=case, want=h1 if case.startswith("verified") else None, args={"start": a},
                  q=f"Which host is {a} bound to? Count only records for exactly this asset id that are marked verified. If there is none, the answer is null.")
    return it


def background(n, rng, K):
    """An unrelated record of one of the planted kinds (cycling through them), never asked about."""
    k = n % 7
    if k == 0: return f"[node] {K.new('node_', 12)} value={K.uuid()}"
    if k == 1: return f"[node] {K.new('node_', 12)} next={K.new('node_', 12)}"
    if k == 2: return f"[recipe] {K.new('rcp_', 6)} = " + " + ".join(f"{rng.randint(1, 3)} {x}" for x in rng.sample(VOCAB, rng.randint(2, 4)))
    if k == 3: return f"[rev] {K.new('item_', 8)} rev {rng.randint(1, 40)} " + (f"value={K.hx(8)}" if rng.random() < 0.8 else "deleted")
    if k == 4: return f"[table {K.new('tbl_', 5)}] key -> value\n" + "\n".join(f"{K.new('k_', 6)} -> {K.new('k_', 6)}" for _ in range(rng.randint(3, 5)))
    if k == 5: return f"def {K.new('f_', 8)}():\n    return '{K.hx(8)}'"
    return f"[binding] {K.new('asset_', 12)} -> {K.new('host-', 6)} ({rng.choice(['verified', 'unverified'])})"


# ================================================================ tokens, filler, data files
class Tok:
    """Exact token counts and offsets from a Hugging Face tokenizer, or a characters-per-token estimate."""
    def __init__(self, spec=None, chars_per_token=3.5):
        self.tk, self.cpt, self.desc = None, chars_per_token, f"estimate, {chars_per_token} characters per token"
        if spec:
            from tokenizers import Tokenizer
            self.tk = Tokenizer.from_file(spec) if os.path.exists(spec) else Tokenizer.from_pretrained(spec)
            self.desc = os.path.basename(spec) if os.path.exists(spec) else spec
    def count(self, s):
        return len(self.tk.encode(s, add_special_tokens=False).ids) if self.tk else max(1, math.ceil(len(s) / self.cpt))
    def starts(self, s):
        """Character offset at which each token starts."""
        if self.tk: return [a for a, b in self.tk.encode(s, add_special_tokens=False).offsets]
        return [int(i * self.cpt) for i in range(math.ceil(len(s) / self.cpt))]


def load_filler(paths, root=None):
    """Concatenate text files (dirs are searched for source and text files), largest first, each under a '--- name ---' header."""
    if not paths:
        paths = [sysconfig.get_paths()["stdlib"]]; root = root or paths[0]
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True) if os.path.isfile(f) and os.path.splitext(f)[1] in TEXT_EXT]
        else:
            files += [f for f in (glob.glob(p) or [p]) if os.path.isfile(f)]
    files = sorted(set(files), key=lambda f: (-os.path.getsize(f), f))
    if not files: raise SystemExit("no filler files found")
    root = root or os.path.commonpath([os.path.dirname(os.path.abspath(f)) for f in files])
    parts = []
    for f in files:
        try: parts.append(f"\n--- {os.path.relpath(f, root)} ---\n" + open(f, errors="replace").read())
        except OSError: pass
    return "".join(parts)


def open_text(path, mode="rt"):
    return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def read_data(path):
    """Returns (meta, iterator over haystacks)."""
    f = open_text(path)
    first = json.loads(f.readline())
    meta = first.get("meta", {})
    def haystacks():
        with f:
            for line in f:
                if line.strip(): yield json.loads(line)
    return meta, haystacks()


# ================================================================ gen
def cmd_gen(a):
    tok = Tok(a.tokenizer, a.chars_per_token)
    rng = random.Random(a.seed); K = Keys(rng)
    items = [make_item(t, b, r, rng, K) for t in TYPES for b in range(NBIN) for r in range(REPS)]
    for it in items:
        it["btok"] = [tok.count("\n" + s + "\n\n") for s in it["blocks"]]; it["dtok"] = [tok.count("\n" + s + "\n\n") for s in it["decoys"]]
    filler = load_filler(a.filler, a.filler_root)
    fo = tok.starts(filler); N = len(fo)
    lengths = [int(x) for x in a.lengths.split(",")]
    total_rec = sum(sum(it["btok"]) + sum(it["dtok"]) for it in items)
    if N < max(lengths) * 1.2: raise SystemExit(f"filler has {N} tokens, need well over {max(lengths)}: add more --filler")
    meta = {"format": "recall-grid/1", "seed": a.seed, "lengths": lengths, "bins": NBIN, "reps": REPS, "density": a.density,
            "tokenizer": tok.desc, "filler": a.filler or [f"Python {sys.version.split()[0]} standard library"],
            "filler_sha256": hashlib.sha256(filler.encode()).hexdigest(), "filler_tokens": N, "system": a.system, "instruction": INSTR,
            "created": time.strftime("%Y-%m-%d")}
    print(f"filler: {len(filler):,} characters, {N:,} tokens ({tok.desc}), {len(items)} items")
    out = open_text(a.out, "wt"); out.write(json.dumps({"meta": meta}) + "\n")
    for L in lengths:
        Ldoc = L - OVERHEAD; H = max(1, math.ceil(total_rec / ((a.density - 0.01) * Ldoc)))
        groups, load = [[] for _ in range(H)], [0] * H  # largest items first, each to the lightest document
        for it in sorted(items, key=lambda it: (-sum(it["btok"]) - sum(it["dtok"]), it["id"])):
            h = min(range(H), key=lambda h: (load[h], len(groups[h]), h)); groups[h].append(it); load[h] += sum(it["btok"]) + sum(it["dtok"])
        rl = random.Random(a.seed + L)
        for h, grp in enumerate(groups):
            rec_tok = sum(sum(it["btok"]) + sum(it["dtok"]) for it in grp)
            KB = Keys(rl); KB.used = set(K.used); bg, bg_tok = [], 0
            while rec_tok + bg_tok < a.density * Ldoc:
                blk = background(len(bg), rl, KB); bg.append((blk, tok.count("\n" + blk + "\n\n"))); bg_tok += bg[-1][1]
            K.used |= KB.used; F = Ldoc - rec_tok - bg_tok
            s = (h * 104729 * 37 + L * 13) % (N - F - 2); c0 = filler.find("\n", fo[s]) + 1; s = bisect.bisect_left(fo, c0)
            ce = fo[s + F] if s + F < N else len(filler)
            place = []
            for i, it in enumerate(grp):
                for j, blk in enumerate(it["blocks"]): place.append((rl.uniform(it["bin"] / NBIN, (it["bin"] + 1) / NBIN), blk, i, "b", it["btok"][j]))
                for j, blk in enumerate(it["decoys"]): place.append((rl.uniform(0, 1), blk, i, "d", it["dtok"][j]))
            for blk, bt in bg: place.append((rl.uniform(0, 1), blk, -1, "g", bt))
            rec_tok += bg_tok
            place.sort(key=lambda p: p[0])
            pieces, cum, prev_f, cprev, doc_len, starts = [], 0, 0, fo[s], 0, []
            for frac, blk, i, role, bt in place:
                f = max(prev_f, min(F, int(round(frac * Ldoc - cum))))
                c = fo[s + f] if s + f < N else ce
                nl = filler.find("\n", c, ce); c = ce if nl < 0 else nl + 1; c = max(c, cprev)
                pieces.append(filler[cprev:c]); doc_len += c - cprev
                starts.append((doc_len, i, role)); pieces.append("\n" + blk + "\n\n"); doc_len += len(blk) + 3
                cprev, prev_f, cum = c, f, cum + bt
            pieces.append(filler[cprev:ce]); doc = "".join(pieces)
            do = tok.starts(doc); T = len(do); pos = {}
            for ch, i, role in starts: pos.setdefault((i, role), []).append(bisect.bisect_right(do, ch + 1) - 1)
            recs = []
            for i, it in enumerate(grp):
                r = {k: it[k] for k in ("id", "type", "bin", "rep", "case", "q", "fmt", "want", "args")}
                r["pos"] = sorted(pos[(i, "b")]); r["decoy_pos"] = sorted(pos.get((i, "d"), []))
                recs.append(r)
            rl.shuffle(recs)
            out.write(json.dumps({"L": L, "hid": f"L{L}-h{h}", "doc_tokens": T, "rec_tokens": rec_tok, "doc": doc, "items": recs}) + "\n")
        print(f"L={L}: {H} documents")
    out.close()
    print(f"wrote {a.out}; run `check` before using it")


# ================================================================ reference solver (no code execution)
def solve_code(doc, start):
    funcs = {}
    for m in re.finditer(r"^def (f_[0-9a-f]{8})\(\):\n((?:    .*\n?)+)", doc, re.M):
        if m.group(1) in funcs: raise ValueError(f"duplicate function {m.group(1)}")
        funcs[m.group(1)] = m.group(2).strip()
    def ev(name, depth=0):
        if depth > 16: raise ValueError("call depth")
        b = funcs[name]
        m = re.fullmatch(r"return '([0-9a-f]{8})'", b)
        if m: return m.group(1)
        m = re.fullmatch(r"return (f_[0-9a-f]{8})\(\)", b) or re.fullmatch(r"result = (f_[0-9a-f]{8})\(\)\n\s*return result", b)
        if m: return ev(m.group(1), depth + 1)
        m = re.fullmatch(r"options = \{'primary': (?:'([0-9a-f]{8})'|(f_[0-9a-f]{8})\(\)), 'fallback': '[0-9a-f]{8}'\}\n\s*return options\['primary'\]", b)
        if m: return m.group(1) or ev(m.group(2), depth + 1)
        raise ValueError(f"unknown function form in {name}")
    return ev(start)


def solve(doc, it):
    t, a = it["type"], it["args"]
    if t in ("guid", "pointer"):
        nodes = {}
        for k, kind, v in re.findall(r"^\[node\] (node_[0-9a-f]{12}) (value|next)=(\S+)$", doc, re.M):
            if k in nodes: raise ValueError(f"duplicate node {k}")
            nodes[k] = (kind, v)
        k, hops = a["start"], 0
        while nodes[k][0] == "next":
            k = nodes[k][1]; hops += 1
            if hops > 64: raise ValueError("pointer loop")
        return nodes[k][1]
    if t in ("ingredients", "quantities"):
        R = {}
        for rid, body in re.findall(r"^\[recipe\] (rcp_[0-9a-f]{6}) = (.*)$", doc, re.M):
            if rid in R: raise ValueError(f"duplicate recipe {rid}")
            R[rid] = [(int(x), y) for x, y in re.findall(r"(\d+) (?:batch(?:es)? of )?(\S+)", body)]
        tot = {}
        def ex(rid, m, depth=0):
            if depth > 16: raise ValueError("recipe depth")
            for x, y in R[rid]:
                if y in R: ex(y, m * x, depth + 1)
                else: tot[y] = tot.get(y, 0) + m * x
        ex(a["start"], 1)
        return sorted(tot) if t == "ingredients" else dict(sorted(tot.items()))
    if t == "revision":
        rs = [(int(r), v or None) for r, v in re.findall(rf"^\[rev\] {re.escape(a['start'])} rev (\d+) (?:value=([0-9a-f]{{8}})|deleted)$", doc, re.M)]
        return max(rs)[1]
    if t == "join":
        tabs = {}
        for name, body in re.findall(r"^\[table (tbl_[0-9a-f]{5})\] key -> value\n((?:k_[0-9a-f]{6} -> k_[0-9a-f]{6}\n?)+)", doc, re.M):
            if name in tabs: raise ValueError(f"duplicate table {name}")
            tabs[name] = dict(re.findall(r"(k_[0-9a-f]{6}) -> (k_[0-9a-f]{6})", body))
        k = a["start"]
        for name in a["tables"]: k = tabs[name][k]
        return k
    if t == "code": return solve_code(doc, a["start"])
    if t == "binding":
        v = [h for x, h, s in re.findall(r"^\[binding\] (asset_[0-9a-f]{12}) -> (host-[0-9a-f]{6}) \((verified|unverified)\)$", doc, re.M) if x == a["start"] and s == "verified"]
        if len(v) > 1: raise ValueError("two verified bindings")
        return v[0] if v else None


def cmd_check(a):
    meta, hs = read_data(a.data); bad = n = 0
    for h in hs:
        for it in h["items"]:
            n += 1
            try: got = solve(h["doc"], it)
            except Exception as e: got = f"error {e!r}"
            if got != it["want"]: bad += 1; print("MISMATCH", h["hid"], it["id"], it["want"], got)
            lo, hi = it["pos"][0] / h["doc_tokens"], it["pos"][-1] / h["doc_tokens"]
            if lo < it["bin"] / NBIN - 0.02 or hi > (it["bin"] + 1) / NBIN + 0.02: print(f"off-bin {h['hid']} {it['id']} {lo:.3f}-{hi:.3f}")
    print(f"check: {n - bad}/{n} items solved from the document text alone ({meta.get('tokenizer')}, filler {meta.get('filler_sha256', '')[:12]})")
    return 1 if bad else 0


# ================================================================ scoring
PAT = {"guid": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "pointer": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
       "revision": r"(?<![\w-])[0-9a-f]{8}(?![\w-])", "code": r"(?<![\w-])[0-9a-f]{8}(?![\w-])", "join": r"\bk_[0-9a-f]{6}\b", "binding": r"host-[0-9a-f]{6}"}


def answer_part(text):
    m = list(re.finditer(r"ANSWER\s*:", text, re.I))
    if m: return text[m[-1].end():].strip(), True
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return (lines[-1].strip() if lines else ""), False


def ingr(x):
    x = re.sub(r"\b\d+\b|\bx\b|\bunits?\b|\bof\b|[^a-z ]", " ", x.lower()); x = " ".join(x.split())
    return x[:-1] if x.endswith("s") and x[:-1] in VSET else x


def score(it, text):
    """Returns (correct, partial credit, answer text, had an ANSWER line)."""
    ans, tagged = answer_part(text); a = ans.lower().replace("*", "").replace("`", ""); t, want = it["type"], it["want"]
    if t in PAT:
        found = set(re.findall(PAT[t], a))
        ok = (bool(re.search(r"\bnull\b|\bnone\b", a)) and not found) if want is None else found == {want}
        return ok, float(ok), ans, tagged
    if t == "ingredients":
        got = {ingr(x) for x in re.split(r",|;|\n|\band\b", a)} - {""}; w = set(want)
        return got == w, len(got & w) / len(got | w), ans, tagged
    got = {}
    for name, v in re.findall(r"([a-z][a-z ]*?)\s*[=:]\s*(\d+)", a): got[ingr(name)] = int(v)
    right = sum(got.get(k) == v for k, v in want.items())
    return got == want, right / max(len(want), len(got)), ans, tagged


# ================================================================ run
def chat(a, messages):
    body = {"model": a.model, "messages": messages, "max_tokens": a.max_tokens, "temperature": a.temperature}
    if a.seed is not None: body["seed"] = a.seed
    if a.no_think: body["chat_template_kwargs"] = {"enable_thinking": False}
    if a.llamacpp: body["cache_prompt"] = True
    body.update(json.loads(a.extra_body) if a.extra_body else {})
    headers = {"Content-Type": "application/json"}
    key = a.api_key or os.environ.get("OPENAI_API_KEY")
    if key: headers["Authorization"] = f"Bearer {key}"
    url = a.base_url.rstrip("/") + "/chat/completions"
    for attempt in range(a.retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, json.dumps(body).encode(), headers)
            d = json.load(urllib.request.urlopen(req, timeout=a.timeout))
            return d, time.time() - t0
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            code = getattr(e, "code", None)
            if attempt == a.retries or (code is not None and code < 500 and code != 429): raise
            time.sleep(min(60, 5 * 2 ** attempt))


def cmd_run(a):
    meta, hs = read_data(a.data)
    lengths = {int(x) for x in a.lengths.split(",")} if a.lengths else None
    done = set()
    if os.path.exists(a.results):
        for l in open(a.results, encoding="utf-8"):
            r = json.loads(l)
            if r["label"] == a.label: done.add((r["L"], r["id"]))
    system = meta.get("system", SYSTEM); instr = meta.get("instruction", INSTR)
    out = open(a.results, "a", encoding="utf-8"); cur, acc = None, []
    for h in hs:
        if lengths and h["L"] not in lengths: continue
        todo = [it for it in h["items"] if (h["L"], it["id"]) not in done]
        if a.limit: todo = todo[:a.limit]
        if not todo: continue
        if cur != h["L"]:
            if acc: print(f"  L={cur}: {sum(acc)}/{len(acc)}", flush=True)
            cur, acc = h["L"], []
        t0, ok_n = time.time(), 0
        for it in todo:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": h["doc"] + "\n\nQuestion: " + it["q"] + "\n\n" + instr.format(it["fmt"])}]
            d, secs = chat(a, msgs)
            ch = d["choices"][0]; text = (ch.get("message") or {}).get("content") or ""
            ok, part, ans, tagged = score(it, text); usage = d.get("usage") or {}; tm = d.get("timings") or {}
            rec = {"label": a.label, "L": h["L"], "hid": h["hid"], "id": it["id"], "type": it["type"], "bin": it["bin"], "case": it["case"],
                   "lo": round(it["pos"][0] / h["doc_tokens"], 4), "hi": round(it["pos"][-1] / h["doc_tokens"], 4), "ok": ok, "partial": round(part, 3),
                   "want": it["want"], "got": ans[:200], "tagged": tagged, "finish": ch.get("finish_reason"), "max_tokens": a.max_tokens,
                   "prompt_tokens": usage.get("prompt_tokens", tm.get("prompt_n")), "completion_tokens": usage.get("completion_tokens", tm.get("predicted_n")),
                   "seconds": round(secs, 2), "out": text[-800:]}
            out.write(json.dumps(rec) + "\n"); out.flush(); ok_n += ok; acc.append(ok)
        print(f"{time.strftime('%H:%M:%S')} {a.label} {h['hid']}: {ok_n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    if acc: print(f"  L={cur}: {sum(acc)}/{len(acc)}", flush=True)


# ================================================================ miss classes
KEYPAT = r"(?:node|item|asset|f|k|rcp)_[0-9a-f]+"


def _edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def classify(r, doc, item):
    """Why an answer was wrong, checked against the document the model read."""
    if r["ok"]: return None
    if r["finish"] == "length":
        return "cut off: listing records" if len(set(re.findall(KEYPAT, r["out"]))) >= 12 else "cut off: long working"
    t, a = r["type"], r["got"].lower().replace("*", "").replace("`", "")
    if t in ("ingredients", "quantities"):
        names = {ingr(x) for x in re.split(r",|;|\n|\band\b", re.sub(r"\s*[=:]\s*\d+", "", a))} - {""}
        return "wrong ingredient set" if names != set(r["want"]) else "right set, wrong amounts"
    found = set(re.findall(PAT[t], a))
    if r["want"] is None: return "value instead of null" if found else "no clear answer"
    if r["want"] not in found and any(0 < _edits(x, r["want"]) <= 2 for x in re.findall(r"[\w-]+", a) if abs(len(x) - len(r["want"])) <= 2):
        return "right value, miscopied"
    if not found:
        if re.search(r"not (?:defined|found|present|listed|given|exist|in the)|does not (?:appear|exist)|no such|cannot be determined|undefined|no (?:record|entry|definition)", a):
            return "says it is not in the document"
        return "null instead of value" if re.search(r"\bnull\b|\bnone\b", a) else "no clear answer"
    if len(found) > 1: return "several values"
    v = found.pop()
    lines = [l for l in doc.splitlines() if v in l]
    if not lines: return "invented (not in document)"
    keys = set(re.findall(KEYPAT, " ".join(lines)))
    for blk in re.split(r"\n\n+", doc):
        if v in blk: keys |= set(re.findall(KEYPAT, blk.split("\n")[0]))
    asked = set(re.findall(KEYPAT, item["q"])) | {item["args"]["start"]}
    one_off = lambda x, y: len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1
    if any(one_off(k, s) for k in keys for s in asked): return "decoy (key one character off)"
    return "value of another record"


def analysis(rows, data_path):
    need = {r["hid"] for r in rows if not r["ok"]}; docs = {}
    _, hs = read_data(data_path)
    for h in hs:
        if h["hid"] in need: docs[h["hid"]] = (h["doc"], {it["id"]: it for it in h["items"]})
    by_type, examples = collections.defaultdict(collections.Counter), collections.defaultdict(list)
    for r in rows:
        if r["ok"] or r["hid"] not in docs: continue
        doc, items = docs[r["hid"]]; c = classify(r, doc, items[r["id"]])
        by_type[r["type"]][c] += 1
        if len(examples[c]) < 4: examples[c].append({"L": r["L"], "id": r["id"], "case": r["case"], "want": r["want"], "got": r["got"][:120]})
    classes = sorted({c for v in by_type.values() for c in v}, key=lambda c: -sum(v[c] for v in by_type.values()))
    return {"classes": classes, "by_type": by_type, "examples": examples}


# ================================================================ report
def aggregate(rs):
    Ls = sorted({r["L"] for r in rs})
    cell = lambda f: [sum(r["ok"] for r in rs if f(r)), sum(1 for r in rs if f(r))]
    return {"lengths": Ls, "n": len(rs), "ok": sum(r["ok"] for r in rs),
            "trunc": {t: sum(r["finish"] == "length" for r in rs if r["type"] == t) for t in TYPES},
            "max_tokens": max((r.get("max_tokens") or 0) for r in rs),
            "grid": [[cell(lambda r, b=b, L=L: r["bin"] == b and r["L"] == L) for L in Ls] for b in range(NBIN)],
            "types": {t: [cell(lambda r, t=t, L=L: r["type"] == t and r["L"] == L) for L in Ls] for t in TYPES},
            "type_grid": {t: [[cell(lambda r, t=t, b=b, L=L: r["type"] == t and r["bin"] == b and r["L"] == L) for L in Ls] for b in range(NBIN)] for t in TYPES},
            "cases": {t: {c: cell(lambda r, t=t, c=c: r["type"] == t and r["case"] == c) for c in sorted({r["case"] for r in rs if r["type"] == t})} for t in TYPES}}


def print_report(lab, ag, an):
    pct = lambda c: f"{100 * c[0] / c[1]:5.0f}" if c[1] else "    -"
    kL = lambda L: f"{L // 1000}K"; Ls = ag["lengths"]
    add = lambda cs: [sum(c[0] for c in cs), sum(c[1] for c in cs)]
    print(f"== {lab}: {ag['ok']}/{ag['n']} correct; {sum(ag['trunc'].values())} cut off at the token cap (counted wrong)")
    print("depth \\ length " + "".join(f"{kL(L):>6}" for L in Ls) + "   all")
    for b in range(NBIN): print(f"  {b * 10:3d}-{b * 10 + 10:3d}%     " + "".join(f"{pct(c):>6}" for c in ag["grid"][b]) + f"{pct(add(ag['grid'][b])):>6}")
    print("  all          " + "".join(f"{pct(add([ag['grid'][b][j] for b in range(NBIN)])):>6}" for j in range(len(Ls))) + f"{pct([ag['ok'], ag['n']]):>6}")
    print("type \\ length  " + "".join(f"{kL(L):>6}" for L in Ls) + "   all")
    for t in TYPES: print(f"  {t:12s}" + "".join(f"{pct(c):>6}" for c in ag["types"][t]) + f"{pct(add(ag['types'][t])):>6}")
    if an:
        print("why answers were wrong:")
        for c in an["classes"]: print(f"  {c:32s} {sum(v[c] for v in an['by_type'].values()):4d} | " + ", ".join(f"{t} {v[c]}" for t, v in an["by_type"].items() if v[c]))


def cmd_report(a):
    rows = [json.loads(l) for l in open(a.results, encoding="utf-8")]
    labels = a.labels.split(",") if a.labels else list(dict.fromkeys(r["label"] for r in rows))
    names = dict(x.split("=", 1) for x in a.name)
    meta = read_data(a.data)[0] if a.data else {}
    data, an = {}, {}
    for lab in labels:
        rs = [r for r in rows if r["label"] == lab]
        if not rs: print(f"== {lab}: no rows"); continue
        data[lab] = aggregate(rs); an[lab] = analysis(rs, a.data) if a.data else None
        print_report(names.get(lab, lab), data[lab], an[lab])
    if a.html:
        write_html(a.html, data, {k: v for k, v in an.items() if v}, names, meta, a.title, a.subtitle, open(a.notes).read() if a.notes else "")
        print("wrote", a.html)


# ================================================================ HTML page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--empty:#efeee9;--mid:#f0efec;--neg:#e34948;--pos:#2a78d6;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--empty:#242423;--mid:#383835;--neg:#e66767;--pos:#3987e5;}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--empty:#242423;--mid:#383835;--neg:#e66767;--pos:#3987e5;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:40px 0 4px}
p.lede,p.note{color:var(--ink2);margin:4px 0 14px;max-width:760px}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0 6px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:150px}
.tile .v{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums} .tile .k{color:var(--ink2);font-size:13px}
.row{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;flex:1 1 340px;min-width:0}
.panel h3{font-size:14px;margin:0 0 8px;font-weight:600}
.hm{display:grid;gap:2px;font-variant-numeric:tabular-nums}
.hm .c{border-radius:3px;text-align:center;font-size:12px;line-height:26px;height:26px;cursor:default;outline-offset:-2px}
.hm .c:hover,.hm .c:focus{outline:2px solid var(--ink)}
.hm .rl{font-size:12px;color:var(--ink2);text-align:right;padding-right:6px;line-height:26px;white-space:nowrap}
.hm .cl{font-size:12px;color:var(--ink2);text-align:center} .hm .tot{font-weight:600}
.small .hm .c{font-size:10px;line-height:18px;height:18px} .small .hm .rl{font-size:10px;line-height:18px} .small .hm .cl{font-size:10px}
.small .panel{flex:1 1 250px;padding:10px}
.scale{display:flex;flex-wrap:wrap;align-items:center;gap:8px;color:var(--ink2);font-size:12px;margin:10px 0 0}
.scale .bar{height:10px;width:180px;border-radius:3px}
.seg{display:inline-flex;flex-wrap:wrap;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:6px 0 12px}
.seg button{font:inherit;font-size:13px;background:var(--surface);color:var(--ink2);border:0;padding:6px 14px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--page)}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;width:100%;max-width:760px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left} td.n,th.n{text-align:right} th{color:var(--ink2);font-weight:600}
.tablewrap{overflow-x:auto} table.why{max-width:none} table.why td:first-child,table.why th:first-child{min-width:190px}
ul{max-width:820px;color:var(--ink2);padding-left:20px} li{margin:4px 0} code{font-size:13px}
.findings{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:6px 20px 10px;margin:18px 0;max-width:900px}
.findings ul{color:var(--ink)} .ex{font-family:ui-monospace,monospace;font-size:12px;color:var(--ink2)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);font-size:12px;padding:6px 9px;border-radius:6px;opacity:0;transition:opacity .08s;max-width:260px;z-index:9}
</style></head>
<body><main>
<h1 id="title"></h1>
<p class="lede" id="lede"></p>
<div class="tiles" id="tiles"></div>
__NOTES__
<h2>Depth × context length</h2>
<p class="note">Share of answers fully correct. Rows are the tenth of the document holding every record of an item.</p>
<div class="row" id="depth"></div>
<div class="scale" id="scale"></div>
<h2>Task type × context length</h2>
<p class="note">Columns are context length in thousands of tokens.</p>
<div class="row" id="types"></div>
<h2>Each task type: depth × context length</h2>
<p class="note">Columns are context length in thousands of tokens. Few items per cell, so single cells are coarse; look for patterns across a panel.</p>
<div class="seg" id="seg" role="group" aria-label="Run"></div>
<div class="row small" id="multi"></div>
<h2>By case</h2>
<p class="note">Each type cycles through variants, all lengths pooled.</p>
<div class="tablewrap"><table id="cases"></table></div>
<div id="whywrap"><h2>Why answers were wrong</h2>
<p class="note">Wrong answers sorted by what the model wrote, checked against the document it read. A decoy is the value of a key one character off the asked one.</p>
<div id="whylist"></div></div>
<h2>How the test works</h2>
<ul>
<li><b>Tasks:</b> GUID lookup; pointer chain (2-4 hops); recipe ingredients and recipe quantities (sub-recipes 1-2 levels deep, amounts multiplied down the tree, repeats added); latest revision (latest value, latest deleted = null, deleted then restored); table join (2-4 tables, with the previous key planted in the wrong table); code trace (Python, 1-3 calls deep, some through a dict with a decoy literal); bindings (only exact-key records marked verified count; otherwise null).</li>
<li><b>Hard negatives:</b> every lookup type also has a record whose key differs by one character with a different value, placed anywhere in the same document. Unasked records of every kind top each document up to 15% records, so distractor density is the same at every length.</li>
<li><b>Same items at every length:</b> 240 items (8 types × 10 depth bins × 3) with identical keys and answers; only the length and the filler change. Short lengths spread the items over several documents.</li>
<li><b>Scoring:</b> the model may reason briefly, then writes <code>ANSWER: ...</code>. Correct means the answer line holds exactly the right value (or null), the exact ingredient set, or every ingredient with the right amount. Answers cut off by the token cap count as wrong.</li>
<li id="datainfo"></li>
</ul>
</main>
<div id="tip" role="tooltip"></div>
<script>
const DATA = __DATA__, NAMES = __NAMES__, AN = __ANALYSIS__, META = __META__, HEAD = __HEAD__;
const TYPES = ["guid","pointer","ingredients","quantities","revision","join","code","binding"];
const TNAME = {guid:"GUID lookup",pointer:"Pointer chain",ingredients:"Recipe ingredients",quantities:"Recipe quantities",revision:"Latest revision",join:"Table join",code:"Code trace",binding:"Bindings"};
const RAMP = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"];
const labs = Object.keys(DATA); const nm = l => NAMES[l] || l;
document.getElementById("title").textContent = HEAD.title; document.getElementById("lede").textContent = HEAD.subtitle;
document.getElementById("datainfo").textContent = `Data: seed ${META.seed ?? "?"}, tokenizer ${META.tokenizer ?? "?"}, filler ${(META.filler || []).join(", ")} (sha256 ${String(META.filler_sha256 || "").slice(0, 12)}). Generated by recall_grid.py.`;
const dark = () => getComputedStyle(document.documentElement).getPropertyValue("--page").trim() === "#0d0d0d";
const hex = h => [1,3,5].map(i => parseInt(h.slice(i,i+2),16));
const mix = (a,b,t) => "rgb(" + hex(a).map((x,i) => Math.round(x + (hex(b)[i]-x)*t)).join(",") + ")";
function seq(v){ const r = dark() ? RAMP.slice().reverse() : RAMP; const p = v*(r.length-1), i = Math.min(r.length-2, Math.floor(p)); return mix(r[i], r[i+1], p-i); }
function div(d){ const cs = getComputedStyle(document.documentElement); return mix(cs.getPropertyValue("--mid").trim(), cs.getPropertyValue(d >= 0 ? "--pos" : "--neg").trim(), Math.min(1, Math.abs(d)/0.3)); }
function inkOn(rgb){ const [r,g,b] = rgb.match(/\d+/g).map(Number).map(x => { x/=255; return x <= .03928 ? x/12.92 : ((x+.055)/1.055)**2.4; }); return (0.2126*r + 0.7152*g + 0.0722*b) > 0.3 ? "#0b0b0b" : "#f0efec"; }
const tip = document.getElementById("tip");
function hook(el, text){ el.tabIndex = 0;
  const show = () => { tip.textContent = text; tip.style.opacity = 1; const r = el.getBoundingClientRect(); tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, r.left) + "px"; tip.style.top = (r.top - tip.offsetHeight - 6) + "px"; };
  el.addEventListener("pointerenter", show); el.addEventListener("focus", show); el.addEventListener("pointerleave", () => tip.style.opacity = 0); el.addEventListener("blur", () => tip.style.opacity = 0); }
const kL = L => Math.round(L/1000) + "K"; const pc = c => c[1] ? Math.round(100*c[0]/c[1]) : null;
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
function heat(title, cols, rows, opts = {}){
  const p = el("div", "panel"); p.appendChild(el("h3", "", title)); const g = el("div", "hm");
  g.style.gridTemplateColumns = `${opts.rw || 64}px repeat(${cols.length}, minmax(0,1fr))` + (opts.total ? " minmax(0,1fr)" : "");
  g.appendChild(el("div")); for (const c of cols) g.appendChild(el("div", "cl", opts.short ? String(Math.round(c/1000)) : kL(c))); if (opts.total) g.appendChild(el("div", "cl", "all"));
  for (const r of rows) { g.appendChild(el("div", "rl", r.label));
    const cells = opts.total ? r.cells.concat([r.cells.reduce((a,c) => [a[0]+c[0], a[1]+c[1]], [0,0])]) : r.cells;
    cells.forEach((c, j) => { const d = el("div", "c" + (opts.total && j === cols.length ? " tot" : "")); const col = j < cols.length ? kL(cols[j]) : "all lengths";
      if (opts.diff) { if (c === null) d.style.background = "var(--empty)"; else { const bg = div(c); d.style.background = bg; d.style.color = inkOn(bg); d.textContent = (c > 0 ? "+" : "") + Math.round(100*c); hook(d, `${r.label}, ${col}: ${opts.diff} ${c >= 0 ? "+" : ""}${Math.round(100*c)} points`); } }
      else if (!c[1]) d.style.background = "var(--empty)";
      else { const bg = seq(c[0]/c[1]); d.style.background = bg; d.style.color = inkOn(bg); d.textContent = pc(c); hook(d, `${r.label}, ${col}: ${c[0]}/${c[1]} correct (${pc(c)}%)`); }
      g.appendChild(d); }); }
  p.appendChild(g); return p; }
const depthRows = grid => grid.map((cells, b) => ({label: `${b*10}-${b*10+10}%`, cells}));
const diffCells = (a, b) => a.map((c, j) => (c[1] && b[j][1]) ? c[0]/c[1] - b[j][0]/b[j][1] : null);
let current = labs[0];
function multi(l){ const m = document.getElementById("multi"); m.replaceChildren(); for (const t of TYPES) m.appendChild(heat(TNAME[t], DATA[l].lengths, depthRows(DATA[l].type_grid[t]), {rw: 52, short: true})); for (const b of document.querySelectorAll("#seg button")) b.setAttribute("aria-pressed", b.dataset.l === l); }
function render(){
  const tiles = document.getElementById("tiles"); tiles.replaceChildren();
  for (const l of labs) { const t = el("div", "tile"); t.append(el("div", "v", Math.round(100*DATA[l].ok/DATA[l].n) + "%"), el("div", "k", `${nm(l)}: ${DATA[l].ok}/${DATA[l].n} correct`)); tiles.appendChild(t); }
  const dep = document.getElementById("depth"), typ = document.getElementById("types"); dep.replaceChildren(); typ.replaceChildren();
  for (const l of labs) { dep.appendChild(heat(nm(l), DATA[l].lengths, depthRows(DATA[l].grid), {total: true})); typ.appendChild(heat(nm(l), DATA[l].lengths, TYPES.map(t => ({label: TNAME[t], cells: DATA[l].types[t]})), {total: true, rw: 112, short: true})); }
  if (labs.length === 2 && JSON.stringify(DATA[labs[0]].lengths) === JSON.stringify(DATA[labs[1]].lengths)) { const [a, b] = labs, title = `${nm(a)} minus ${nm(b)}, points`, d = `${nm(a)} against ${nm(b)}:`;
    dep.appendChild(heat(title, DATA[a].lengths, DATA[a].grid.map((cells, i) => ({label: `${i*10}-${i*10+10}%`, cells: diffCells(cells, DATA[b].grid[i])})), {diff: d}));
    typ.appendChild(heat(title, DATA[a].lengths, TYPES.map(t => ({label: TNAME[t], cells: diffCells(DATA[a].types[t], DATA[b].types[t])})), {diff: d, rw: 112, short: true})); }
  const sc = document.getElementById("scale"); sc.replaceChildren(); const bar = el("span", "bar"); bar.style.background = `linear-gradient(90deg, ${[0,.25,.5,.75,1].map(seq).join(",")})`;
  sc.append(el("span", "", "0%"), bar, el("span", "", "100% correct"));
  if (labs.length === 2) sc.appendChild(el("span", "", `Difference panel: blue = ${nm(labs[0])} better, red = ${nm(labs[1])} better, gray = equal.`));
  multi(current);
  const tb = document.getElementById("cases"); tb.replaceChildren(); const hr = el("tr");
  for (const h of ["Type", "Case"].concat(labs.map(nm))) hr.appendChild(el("th", ["Type","Case"].includes(h) ? "" : "n", h)); tb.appendChild(hr);
  for (const t of TYPES) for (const c of Object.keys(DATA[labs[0]].cases[t])) { const tr = el("tr"); tr.append(el("td", "", TNAME[t]), el("td", "", c));
    for (const l of labs) { const x = (DATA[l].cases[t] || {})[c]; tr.appendChild(el("td", "n", x ? `${x[0]}/${x[1]} (${pc(x)}%)` : "-")); } tb.appendChild(tr); }
  const tr = el("tr"); tr.append(el("td", "", "Cut off at the token cap"), el("td", "", "counted wrong")); for (const l of labs) tr.appendChild(el("td", "n", String(Object.values(DATA[l].trunc).reduce((x, y) => x + y, 0)))); tb.appendChild(tr); }
const seg = document.getElementById("seg"); for (const l of labs) { const b = el("button", "", nm(l)); b.dataset.l = l; b.addEventListener("click", () => { current = l; multi(l); }); seg.appendChild(b); }
function why(){ const an = Object.keys(AN); if (!an.length) { document.getElementById("whywrap").remove(); return; }
  const box = document.getElementById("whylist"); box.replaceChildren();
  for (const l of an) { const A = AN[l]; const h = el("h3", "", `${nm(l)}: ${TYPES.reduce((s, t) => s + Object.values(A.by_type[t] || {}).reduce((x, y) => x + y, 0), 0)} wrong`); h.style.margin = "18px 0 8px"; box.appendChild(h);
    const wrap = el("div", "tablewrap"), tb = el("table", "why"); wrap.appendChild(tb); box.appendChild(wrap); const hr = el("tr");
    for (const x of ["Class", "All"].concat(TYPES.map(t => TNAME[t]))) hr.appendChild(el("th", x === "Class" ? "" : "n", x)); tb.appendChild(hr);
    for (const c of A.classes) { const tr = el("tr"); tr.appendChild(el("td", "", c)); const all = el("td", "n", String(TYPES.reduce((s, t) => s + ((A.by_type[t] || {})[c] || 0), 0))); all.style.fontWeight = 600; tr.appendChild(all);
      for (const t of TYPES) tr.appendChild(el("td", "n", String(((A.by_type[t] || {})[c]) || ""))); tb.appendChild(tr); }
    const ul = el("ul"); for (const c of A.classes) { const e = (A.examples[c] || [])[0]; if (!e) continue; const li = el("li"); li.append(el("b", "", c + ": "), el("span", "ex", `${e.id} at ${Math.round(e.L/1000)}K (${e.case}); wanted ${JSON.stringify(e.want)}, answer line ${JSON.stringify(e.got)}`)); ul.appendChild(li); }
    box.appendChild(ul); } }
render(); why(); matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body></html>
"""


def write_html(path, data, an, names, meta, title, subtitle, notes):
    js = lambda x: json.dumps(x).replace("</", "<\\/")  # keep model output and labels from closing the script tag
    head = {"title": title, "subtitle": subtitle}
    page = PAGE.replace("__TITLE__", html.escape(title))
    page = page.replace("__NOTES__", f'<section class="findings">{notes}</section>' if notes else "")
    page = page.replace("__DATA__", js(data)).replace("__NAMES__", js(names)).replace("__ANALYSIS__", js(an)).replace("__META__", js(meta)).replace("__HEAD__", js(head))
    open(path, "w", encoding="utf-8").write(page)


# ================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Structured long-context recall benchmark (context length x depth).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="generate a benchmark data file")
    g.add_argument("--out", required=True, help="data file to write (.jsonl or .jsonl.gz)")
    g.add_argument("--lengths", default=",".join(map(str, LENGTHS)), help="comma-separated context lengths in tokens")
    g.add_argument("--tokenizer", help="tokenizer.json path or Hugging Face repo id (needs the `tokenizers` package); omit to estimate")
    g.add_argument("--chars-per-token", type=float, default=3.5, help="estimate used without --tokenizer")
    g.add_argument("--filler", nargs="*", help="files, globs or directories of filler text (default: this Python's standard library)")
    g.add_argument("--filler-root", help="names in the filler headers are relative to this directory")
    g.add_argument("--system", default=SYSTEM, help="system prompt stored with the data")
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--density", type=float, default=DENSITY, help="fraction of tokens that are records (default 0.15)")
    c = sub.add_parser("check", help="solve every item from its document alone (catches generator problems)")
    c.add_argument("--data", required=True)
    r = sub.add_parser("run", help="ask the questions through an OpenAI-compatible chat endpoint")
    r.add_argument("--data", required=True); r.add_argument("--label", required=True, help="name of this run in the results file")
    r.add_argument("--results", default="results.jsonl"); r.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    r.add_argument("--model", default="default"); r.add_argument("--api-key", help="default: $OPENAI_API_KEY")
    r.add_argument("--lengths", help="only these lengths, comma-separated"); r.add_argument("--limit", type=int, help="at most N new items per document per call (smoke test)")
    r.add_argument("--max-tokens", type=int, default=768); r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=42, help="request seed (use --seed -1 to omit)")
    r.add_argument("--no-think", action="store_true", help="send chat_template_kwargs enable_thinking=false (Qwen-style templates)")
    r.add_argument("--llamacpp", action="store_true", help="send cache_prompt=true (llama.cpp server)")
    r.add_argument("--extra-body", help="JSON merged into every request body")
    r.add_argument("--timeout", type=float, default=3600); r.add_argument("--retries", type=int, default=3)
    p = sub.add_parser("report", help="text grids and an optional HTML page")
    p.add_argument("--results", required=True); p.add_argument("--data", help="data file, for the why-wrong breakdown")
    p.add_argument("--labels", help="comma-separated labels to include (default: all, in order of appearance)")
    p.add_argument("--name", action="append", default=[], help='display name, label="Name" (repeatable)')
    p.add_argument("--html", help="write a self-contained HTML report here"); p.add_argument("--title", default="Recall grid")
    p.add_argument("--subtitle", default="Structured long-context recall: eight lookup task types at known depths and several context lengths.")
    p.add_argument("--notes", help="HTML fragment to show as findings (trusted input)")
    a = ap.parse_args()
    if a.cmd == "run" and a.seed == -1: a.seed = None
    sys.exit({"gen": cmd_gen, "check": cmd_check, "run": cmd_run, "report": cmd_report}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
