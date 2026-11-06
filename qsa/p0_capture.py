"""P0: dense teacher attention statistics and oracle block-capture sweep.

Runs the dense 27B model without training, records the block-level attention
mass (r=4 token blocks, max-pooled and L1-normalized per query), and reports how
much of that mass a *perfect* top-k block selector would capture for several
token budgets. Also reports dense perplexity on held-out blocks.

Usage:
    python -m qsa.p0_capture --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
import yaml

from qsa.compat import patch_httpx_decoders

patch_httpx_decoders()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_capture(model, budgets: list[int]) -> None:
    from qsa.hf_qsa import full_attention_layers

    for layer in full_attention_layers(model):
        layer._qsa_collect = True
        layer._qsa_capture = {"budgets": budgets, "sums": {}, "count": 0}


@torch.inference_mode()
def dense_perplexity(model, blocks, chunk: int = 512) -> tuple[float, int]:
    total_nll, total_tokens = 0.0, 0
    for ids in blocks:
        ids = ids.cuda(non_blocking=True)[None]
        hidden = model.model.language_model(input_ids=ids).last_hidden_state
        length = ids.shape[1]
        for start in range(0, length - 1, chunk):
            end = min(start + chunk, length - 1)
            logits = model.lm_head(hidden[:, start:end + 1]).float()
            nll = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                ids[:, start + 1:end + 1].reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += (end - start)
        del hidden
    return math.exp(total_nll / max(total_tokens, 1)), total_tokens


@torch.inference_mode()
def capture_sweep(model, blocks, budgets: list[int]) -> dict:
    started = time.time()
    from qsa.hf_qsa import full_attention_layers

    for i, ids in enumerate(blocks):
        model.model.language_model(input_ids=ids.cuda(non_blocking=True)[None])
        print(f"[p0] capture seq {i + 1}/{len(blocks)} {time.time() - started:.1f}s", flush=True)

    per_layer = {}
    layer_sums: dict[int, dict[int, float]] = {}
    layer_counts: dict[int, int] = {}
    for layer in full_attention_layers(model):
        cap = getattr(layer, "_qsa_capture", None)
        if cap is None:
            continue
        idx = getattr(layer, "layer_idx", -1)
        layer_sums[idx] = {k: v for k, v in cap["sums"].items()}
        layer_counts[idx] = cap["count"]
        per_layer[str(idx)] = {
            str(k): (v / max(cap["count"], 1)) for k, v in cap["sums"].items()
        }

    overall = {}
    for budget in budgets:
        num = sum(layer_sums[i][budget] for i in layer_sums)
        den = sum(layer_counts[i] for i in layer_sums)
        overall[str(budget)] = num / max(den, 1)
    return {"per_layer": per_layer, "overall": overall, "seconds": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="p0_results.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("stage") == "mm1":
        from qsa.train_mm import run_mm1

        run_mm1(cfg)
        return
    if cfg.get("stage") == "mm0":
        from qsa.mm0 import run_mm0

        run_mm0(cfg)
        return
    if cfg.get("stage", "capture") == "distill":
        from qsa.train_indexer import run_distillation

        run_distillation(cfg)
        return

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from qsa.data import iter_token_blocks
    from qsa.hf_qsa import full_attention_layers, install, set_collect

    print(f"[p0] config {json.dumps(cfg, ensure_ascii=False)}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("P0 requires a CUDA GPU")
    print(f"[p0] gpu {torch.cuda.get_device_name()} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}GiB", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda",
    ).eval()

    layers = install(model, block_size=cfg["block_size"])
    print(f"[p0] qsa indexers attached to layers {layers}", flush=True)

    budgets = [int(b) for b in cfg["budgets"]]
    set_capture(model, budgets)

    blocks = iter_token_blocks(
        tokenizer, cfg["seq_len"], cfg["n_seqs"],
        dataset=cfg["dataset"], subset=cfg.get("subset", "sample-10BT"),
        skip_docs=cfg.get("skip_docs", 0),
    )
    capture = capture_sweep(model, list(blocks), budgets)

    set_collect(model, False)
    for layer in full_attention_layers(model):
        layer._qsa_capture = None

    ppl_blocks = iter_token_blocks(
        tokenizer, cfg["ppl_len"], cfg["ppl_seqs"],
        dataset=cfg["dataset"], subset=cfg.get("subset", "sample-10BT"),
        skip_docs=cfg.get("ppl_skip_docs", 10_000),
    )
    ppl, tokens = dense_perplexity(model, list(ppl_blocks))

    result = {
        "model": cfg["model"],
        "block_size": cfg["block_size"],
        "seq_len": cfg["seq_len"],
        "n_seqs": cfg["n_seqs"],
        "capture": capture,
        "ppl": {"value": ppl, "tokens": tokens, "seq_len": cfg["ppl_len"]},
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT " + json.dumps({
        "capture_overall": capture["overall"],
        "ppl": result["ppl"],
        "seconds": {"capture": round(capture["seconds"], 1)},
    }), flush=True)


if __name__ == "__main__":
    main()
