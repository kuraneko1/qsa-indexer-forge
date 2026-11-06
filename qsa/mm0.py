"""MM0: diagnose the frozen step-600 indexer on multimodal inputs.

No training. For each synthetic image+text case:

1. run the unpatched dense model (reference);
2. patch attention and verify the patched dense forward reproduces the
   reference logits (functional parity, real image tokens and M-RoPE);
3. run the frozen indexers with sparse selection and measure block capture
   split by query/key modality, plus the multiple-choice answer.

Writes a local JSON report; it does not upload anything.
"""

from __future__ import annotations

import hashlib
import json

import torch


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _letters(tokenizer) -> list[int]:
    return [tokenizer.encode(" " + letter, add_special_tokens=False)[-1]
            for letter in "ABCD"]


def run_mm0(cfg: dict) -> None:
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from qsa.hf_qsa import full_attention_layers, install
    from qsa.mm_capture import MMCapture, merge_summaries
    from qsa.mm_data import GENERATOR_VERSION, build_inputs, load_filler_text, make_specs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("MM0 requires a CUDA GPU")

    processor = AutoProcessor.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    tokenizer = processor.tokenizer
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["model"], revision=cfg["model_revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda",
    ).eval()

    specs = make_specs(cfg.get("seed", 0))
    cases = []
    for spec in specs[: cfg.get("n_short", 2)]:
        inputs, prompt = build_inputs(processor, spec)
        cases.append({"spec": spec, "inputs": inputs, "filler_tokens": 0, "prompt": prompt})
    for i in range(cfg.get("n_long", 6)):
        spec = specs[i % len(specs)]
        filler = load_filler_text(
            tokenizer, cfg["long_tokens"], cfg["dataset"], cfg["dataset_revision"],
            cfg.get("subset", "sample-10BT"), cfg.get("skip_docs", 0) + 1000 * i,
        )
        inputs, prompt = build_inputs(processor, spec, filler,
                                      image_position=cfg.get("image_position", "late"))
        cases.append({"spec": spec, "inputs": inputs, "filler_tokens": cfg["long_tokens"],
                      "prompt": prompt})

    image_token_id = int(cfg["image_token_id"])

    def moved(inputs):
        return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}

    def is_image(case):
        return (case["inputs"]["input_ids"][0] == image_token_id).cuda()

    def forward(case):
        with torch.inference_mode():
            return model(**moved(case["inputs"]), logits_to_keep=1).logits.float().cpu()

    # 1) unpatched dense reference
    refs = [forward(case) for case in cases]
    print("[mm0] dense reference done", flush=True)

    # 2) patch and load the frozen indexer weights
    install(model, block_size=cfg["block_size"], budget=cfg["budget"])
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(cfg["indexer_repo"], cfg["indexer_file"],
                           revision=cfg["indexer_revision"], repo_type="model")
    sha = _sha256(path)
    if cfg.get("indexer_sha256") and sha != cfg["indexer_sha256"]:
        raise SystemExit(f"indexer SHA mismatch: {sha}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    layers = full_attention_layers(model)
    for layer in layers:
        layer._qsa_indexer.load_state_dict(state[layer.layer_idx])
    print(f"[mm0] loaded indexers sha={sha[:12]} layers={len(layers)}", flush=True)

    def configure(budget: int, collect: bool, capture: MMCapture | None, image):
        for layer in layers:
            layer._qsa_budget = budget
            layer._qsa_collect = collect
            layer._qsa_keep_target = False
            layer._qsa_mm_capture = capture
            layer._qsa_is_image = image
            layer._qsa_capture = None
            layer._qsa_target = None

    # 3) patched dense parity
    parity = []
    for case, ref in zip(cases, refs):
        configure(0, False, None, is_image(case))
        logits = forward(case)
        parity.append({
            "id": case["spec"]["id"],
            "max_abs": (logits - ref).abs().max().item(),
            "mean_abs": (logits - ref).abs().mean().item(),
            "top1_match": int((logits.argmax(-1) == ref.argmax(-1)).all().item()),
        })
    print("[mm0] parity done", flush=True)

    # 4) frozen indexer capture + sparse answers
    letters = _letters(tokenizer)
    layer_summaries: dict[int, list[dict]] = {layer.layer_idx: [] for layer in layers}
    question_summaries: dict[int, list[dict]] = {layer.layer_idx: [] for layer in layers}
    sparse_rows = []
    for case, ref in zip(cases, refs):
        image = is_image(case)
        caps = {layer.layer_idx: MMCapture() for layer in layers}
        qcaps = {layer.layer_idx: MMCapture() for layer in layers}
        configure(cfg["budget"], True, None, image)
        qmask = torch.zeros_like(image)
        qmask[max(0, image.shape[0] - cfg.get("question_span", 96)):] = True
        for layer in layers:
            layer._qsa_mm_capture = caps[layer.layer_idx]
        logits = forward(case)
        for layer in layers:
            target = layer._qsa_mm_target
            valid = layer._qsa_mm_valid
            sc = layer._qsa_mm_scores
            valid2 = valid[0] if valid.dim() == 3 else valid
            caps[layer.layer_idx].add(target, valid2, sc, image,
                                      cfg["block_size"], cfg["budget"])
            qcaps[layer.layer_idx].add(target, valid2, sc, image,
                                       cfg["block_size"], cfg["budget"], query_mask=qmask)
            layer._qsa_mm_target = layer._qsa_mm_valid = layer._qsa_mm_scores = None
            layer_summaries[layer.layer_idx].append(caps[layer.layer_idx].summary())
            question_summaries[layer.layer_idx].append(qcaps[layer.layer_idx].summary())

        scores = [logits[0, 0, tid].item() for tid in letters]
        ref_scores = [ref[0, 0, tid].item() for tid in letters]
        sparse_rows.append({
            "id": case["spec"]["id"],
            "kind": case["spec"]["kind"],
            "filler_tokens": case["filler_tokens"],
            "tokens": int(case["inputs"]["input_ids"].shape[1]),
            "image_tokens": int(image.sum()),
            "dense_ok": int(max(range(4), key=lambda i: ref_scores[i]) == case["spec"]["answer"]),
            "sparse_ok": int(max(range(4), key=lambda i: scores[i]) == case["spec"]["answer"]),
            "sparse_letter_scores": [round(s, 3) for s in scores],
        })
        print(f"[mm0] case {case['spec']['id']} tokens={case['inputs']['input_ids'].shape[1]} "
              f"dense_ok={sparse_rows[-1]['dense_ok']} sparse_ok={sparse_rows[-1]['sparse_ok']}",
              flush=True)

    merged = {idx: merge_summaries(rows) for idx, rows in layer_summaries.items()}
    overall = merge_summaries(list(merged.values()))
    qmerged = {idx: merge_summaries(rows) for idx, rows in question_summaries.items()}
    qoverall = merge_summaries(list(qmerged.values()))
    seen = sum(r["queries_seen"] for rows in layer_summaries.values() for r in rows)
    excluded = sum(r["queries_excluded_N_le_K"] for rows in layer_summaries.values() for r in rows)

    result = {
        "stage": "mm0",
        "generator": GENERATOR_VERSION,
        "model": cfg["model"],
        "model_revision": cfg["model_revision"],
        "indexer": {"repo": cfg["indexer_repo"], "file": cfg["indexer_file"],
                     "revision": cfg["indexer_revision"], "sha256": sha},
        "budget": cfg["budget"],
        "block_size": cfg["block_size"],
        "cases": [{k: v for k, v in row.items() if k != "sparse_letter_scores"}
                  for row in sparse_rows],
        "parity": parity,
        "capture": {"overall": overall,
                     "question_region": qoverall,
                     "per_layer": {str(k): v for k, v in merged.items()}},
        "evidence": {
            "sparse_layers": len(layers),
            "queries_seen": seen,
            "queries_excluded_N_le_K": excluded,
            "case_tokens": [row["tokens"] for row in sparse_rows],
        },
        "qa": {
            "dense_correct": sum(r["dense_ok"] for r in sparse_rows),
            "sparse_correct": sum(r["sparse_ok"] for r in sparse_rows),
            "n": len(sparse_rows),
        },
    }
    with open("mm0_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT " + json.dumps({
        "cases": len(sparse_rows),
        "parity_max_abs": max(p["max_abs"] for p in parity),
        "parity_top1": sum(p["top1_match"] for p in parity),
        "capture_overall": overall["cells"],
        "capture_question": qoverall["cells"],
        "qa": result["qa"],
        "queries_excluded_N_le_K": excluded,
        "sparse_layers": len(layers),
    }), flush=True)
