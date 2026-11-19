"""Stage-1 indexer distillation for QSA on Qwen3.8-27B.

The backbone stays frozen; only the indexers train, against the dense attention
distribution of the same forward pass. Each step

1. runs the dense model once under ``no_grad`` and records, per full-attention
   layer, the block teacher target (fp16) and the layer input hidden states;
2. recomputes each indexer with gradients on those stored inputs and applies
   the block-level KL from the paper (Eq. 18);
3. reports the mass of the teacher distribution captured by the indexer's own
   top-k selection.

``overfit: true`` reuses a single fixed sequence to test whether the objective
can fit at all before spending tokens on a real run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F


@torch.inference_mode()
def perplexity(model, blocks, chunk: int = 512) -> tuple[float, int]:
    from qsa.hf_qsa import text_model

    total_nll, total_tokens = 0.0, 0
    for ids in blocks:
        ids = ids[None].cuda()
        hidden = text_model(model)(input_ids=ids).last_hidden_state
        length = ids.shape[1]
        for start in range(0, length - 1, chunk):
            end = min(start + chunk, length - 1)
            logits = model.lm_head(hidden[:, start:end + 1]).float()
            total_nll += F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                ids[:, start + 1:end + 1].reshape(-1),
                reduction="sum",
            ).item()
            total_tokens += end - start
        del hidden
    return math.exp(total_nll / max(total_tokens, 1)), total_tokens


def set_mode(layers, collect: bool, budget: int) -> None:
    for layer in layers:
        layer._qsa_collect = collect
        layer._qsa_budget = budget
        layer._qsa_keep_target = collect
        layer._qsa_target = None
        layer._qsa_hidden = None
        layer._qsa_rope = None


def distill_step(model, layers, ids, cfg, optimizer, meta=None) -> tuple[float, float, dict]:
    """One optimizer step on a single sequence; returns (loss, capture)."""
    from qsa.hf_qsa import text_model
    from qsa.losses import indexer_kl_from_target
    from qsa.sparse_attention import select_blocks

    torch.cuda.empty_cache()  # 64k leaves the allocator fragmented between steps
    import os as _os
    _prof = _os.environ.get("QSA_PROFILE") == "1"
    if _prof:
        from qsa.hf_qsa import PROF as _P
        _P["teacher"] = _P["sparse"] = 0.0
        torch.cuda.synchronize(); _tf = time.time()
    # kl_rows: teacher and KL only at every question token plus kl_rows random document
    # positions (seeded by seed and global step); the answer-mass and decoy-margin terms still use all question tokens.
    rows, qst = None, None
    kl_rows = int(cfg.get("kl_rows", 0) or 0)
    if kl_rows and meta is not None:
        from qsa.hf_qsa import set_kl_rows
        T_ = ids.shape[1]; qst = int(meta.get("q_start", 0)); r_ = int(cfg["block_size"])
        g_ = torch.Generator().manual_seed(int(cfg.get("seed", 0) or 0) * 1000003 + int(meta.get("_step", 0)))
        cand = torch.arange(r_ - 1, qst)
        pick = cand[torch.randperm(cand.numel(), generator=g_)[:kl_rows]].sort().values
        rows = torch.cat([pick, torch.arange(qst, T_)]).to(ids.device)
        set_kl_rows(rows)
    try:
        with torch.no_grad():
            text_model(model)(input_ids=ids)
    finally:
        if rows is not None:
            set_kl_rows(None)
    if _prof:
        torch.cuda.synchronize(); _tl = time.time()

    params = [p for layer in layers for p in layer._qsa_indexer.parameters()]
    total_loss, cap_sum, cap_count, used = 0.0, 0.0, 0, 0
    tok_used = 0
    ent_sum, std_sum = 0.0, 0.0
    rec_sum, rec_cnt, worst_max = 0.0, 0, 0
    for layer in layers:
        target = layer._qsa_target
        hidden = getattr(layer, "_qsa_hidden", None)
        rope = getattr(layer, "_qsa_rope", None)
        layer._qsa_target = None
        layer._qsa_hidden = None
        layer._qsa_rope = None
        if target is None or hidden is None or rope is None:
            if target is None and hidden is None and rope is None:
                pass
            else:
                print("[p1] incomplete capture on layer "
                      f"{getattr(layer, 'layer_idx', '?')}: target={target is not None} "
                      f"hidden={hidden is not None} rope={rope is not None}", flush=True)
            continue
        cos, sin = rope
        dev = next(layer._qsa_indexer.parameters()).device
        if hidden.device != dev:
            hidden = hidden.to(dev, non_blocking=True)
        if cos.device != dev:
            cos, sin = cos.to(dev), sin.to(dev)
        seq = hidden.shape[1]
        _chunk = int(cfg.get("loss_chunk", 0) or 0)
        if rows is not None:
            # target is [1, len(rows), nb]; document rows in chunks, then all question rows as one span
            nd = int((rows < qst).sum())
            step_ = _chunk if _chunk else max(nd, 1)
            spans = [(s, min(s + step_, nd)) for s in range(0, nd, step_)] + [(nd, rows.numel())]
            hidden_f = hidden.float()
        else:
            spans = ([(s, min(s + _chunk, seq)) for s in range(0, seq, _chunk)]
                     if _chunk and seq > _chunk else [(0, seq)])
            # With several spans, the key blocks come from the WHOLE sequence (the indexer's
            # forward() on a slice would only see keys inside the slice); queries are scored one span at a time.
            hidden_f = hidden.float() if len(spans) > 1 else None
        for qs0, qe0 in spans:
            if rows is not None:
                r_idx = rows[qs0:qe0]
                h = hidden[:, r_idx]
                c0 = cos[:, r_idx]
                s0 = sin[:, r_idx]
            else:
                h = hidden[:, qs0:qe0]
                c0 = cos[:, qs0:qe0]
                s0 = sin[:, qs0:qe0]
            tgt = target[:, qs0:qe0]
            if tgt.device != dev:
                tgt = tgt.to(dev, non_blocking=True)
            if rows is not None:
                kb_all = layer._qsa_indexer.key_blocks(hidden_f, cos[0].float(), sin[0].float())
                scores, valid = layer._qsa_indexer.score_rows(h.float(), c0[0].float(), s0[0].float(), kb_all, r_idx)
            elif hidden_f is not None:
                kb_all = layer._qsa_indexer.key_blocks(hidden_f, cos[0].float(), sin[0].float())
                scores, valid = layer._qsa_indexer.score_span(h.float(), c0[0].float(), s0[0].float(), kb_all, qs0)
                tgt = tgt[..., : scores.shape[-1]]
            else:
                scores, valid = layer._qsa_indexer(h.float(), c0[0].float(), s0[0].float())
            target32 = tgt.float()
            if cfg.get("restrict_kl"):
                # restrict_kl (QSA paper, Eq. 20): renormalize the teacher inside the indexer's top-K blocks and take
                # the KL there only. The comparison stays sharp when the teacher is spread thin over a long context.
                kb = max(1, min(scores.shape[-1],
                                int(math.ceil(cfg["budget"] / cfg["block_size"]))))
                masked = scores.masked_fill(~valid, float("-inf"))
                top = torch.topk(masked, kb, dim=-1).indices
                sel = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top, True) & valid
                tsel = torch.where(sel, target32, torch.zeros_like(target32))
                denom = tsel.sum(-1, keepdim=True)
                ok = denom.squeeze(-1) > 0
                logp = torch.log_softmax(masked.masked_fill(~sel, float("-inf")), dim=-1)
                safe_logp = torch.where(sel, logp, torch.zeros_like(logp))
                tprob = tsel / denom.clamp_min(1e-12)
                terms = tprob * (tprob.clamp_min(1e-12).log() - safe_logp)
                terms = torch.where(sel, terms, torch.zeros_like(terms))
                kl = terms.sum(-1)
                loss = kl[ok].mean() if bool(ok.any()) else scores.sum() * 0.0
            else:
                loss = indexer_kl_from_target(scores, target32, valid)
            lam = float(cfg.get("lambda_answer", 0.0) or 0.0)
            mu = float(cfg.get("mu_margin", 0.0) or 0.0)
            margin = float(cfg.get("margin", 2.0))
            if meta is not None and (lam > 0.0 or mu > 0.0):
                qs = (0 if qs0 == nd else -1) if rows is not None else int(meta.get("q_start", 0)) - qs0
                ab = [b for b in meta.get("answer_blocks", []) if 0 <= b < scores.shape[-1]]
                db = [b for b in meta.get("decoy_blocks", []) if 0 <= b < scores.shape[-1]]
                if 0 <= qs < scores.shape[1] and ab:
                    qscores = scores[0, qs:, :]
                    idx = torch.tensor(ab, device=scores.device, dtype=torch.long)
                    if lam > 0.0:
                        logp = torch.log_softmax(qscores, dim=-1)
                        mass = logp.index_select(-1, idx).exp().sum(-1).mean()
                        loss = loss + lam * (1.0 - mass)
                    if mu > 0.0 and db:
                        didx = torch.tensor(db, device=scores.device, dtype=torch.long)
                        s_ans = qscores.index_select(-1, idx).max(-1).values
                        s_dec = qscores.index_select(-1, didx).max(-1).values
                        loss = loss + mu * torch.relu(margin - (s_ans - s_dec)).mean()
                    with torch.no_grad():
                        sd = qscores.detach()
                        s_best = sd.index_select(-1, idx).max(-1).values
                        rank = (sd > s_best.unsqueeze(-1)).sum(-1) + 1
                        kb = max(1, min(sd.shape[-1], int(math.ceil(cfg["budget"] / cfg["block_size"]))))
                        rec_sum += (rank <= kb).float().mean().item()
                        rec_cnt += 1
                        worst_max = max(worst_max, int(rank.max().item()))
            # Each span's mean loss gets weight 1/len(spans). With equal spans this is the mean over all rows; with
            # kl_rows the question span (and its answer/decoy terms) weighs the same as one document span.
            (loss / len(spans)).backward()
            total_loss += loss.item() * (qe0 - qs0)
            tok_used += qe0 - qs0
            used += 1

            with torch.no_grad():
                probs = target32 / target32.sum(-1, keepdim=True).clamp_min(1e-12)
                ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
                ent_sum += ent[valid.any(-1)].mean().item()
                sc = scores.detach().masked_fill(~valid, 0.0)
                cnt = valid.sum(-1).clamp_min(1)
                mean = sc.sum(-1) / cnt
                var = (((sc - mean.unsqueeze(-1)) ** 2) * valid).sum(-1) / cnt
                std_sum += var.clamp_min(0).sqrt().mean().item()
                selected = select_blocks(scores.detach(), valid, cfg["budget"], cfg["block_size"])
                mass = target32.gather(-1, selected).sum(-1)
                mask = valid.any(-1).expand_as(mass)
                cap_sum += mass[mask].sum().item()
                cap_count += int(mask.sum())
            del scores, loss, target32

    if _prof:
        torch.cuda.synchronize()
        print("PROF " + json.dumps({"forward_total": round(_tl - _tf, 1), "teacher": round(_P["teacher"], 1),
                                    "sparse_attn": round(_P["sparse"], 1), "loss_backward": round(time.time() - _tl, 1)}), flush=True)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    stats = {
        "teacher_entropy": ent_sum / max(used, 1),
        "score_std": std_sum / max(used, 1),
        "answer_recall": rec_sum / max(rec_cnt, 1),
        "worst_rank": worst_max,
    }
    return total_loss / max(tok_used, 1), cap_sum / max(cap_count, 1), stats


class _CPUEmbed(torch.nn.Module):
    """Keep only the input embedding on the CPU (offload_embed: true).

    Moves 248k x 5120 fp16 = 2.5 GiB off the GPU. Each step adds a host-to-device copy of
    seq_len x 5120 x 2 bytes (168 MB at 16k), well under 0.05 s.
    """

    def __init__(self, weight: torch.Tensor, out_device) -> None:
        super().__init__()
        self.weight = weight
        self.out_device = out_device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.embedding(ids.to("cpu"), self.weight).to(self.out_device, non_blocking=True)


def save_indexers(layers, step: int | None = None) -> str:
    suffix = f"_step{step}" if step is not None else ""
    path = f"qsa_indexers{suffix}.pt"
    state = {
        getattr(layer, "layer_idx", i): layer._qsa_indexer.state_dict()
        for i, layer in enumerate(layers)
    }
    torch.save(state, path)
    print(f"[p1] saved {path} ({os.path.getsize(path) / 2**20:.1f} MiB)", flush=True)
    return path


def check_sha256(path: str, want: str | None) -> None:
    """Stop when a pinned warm-start file does not have the expected SHA256."""
    if not want:
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        raise SystemExit(f"[p1] {path}: sha256 {h.hexdigest()} != expected {want}")


def check_hub(cfg: dict) -> tuple[str, object] | None:
    """Fail fast when the run is asked to persist to the Hub but cannot."""
    if not cfg.get("push_to_hub", False):
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("[p1] HF_TOKEN is not set in the runtime environment")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as exc:
        raise SystemExit(f"[p1] HF authentication rejected: {type(exc).__name__}: {exc}")
    private = bool(cfg.get("hub_private", False))
    repo = cfg.get("hf_repo") or f"{user}/qwen38-27b-qsa-indexers"
    api.create_repo(repo_id=repo, private=private, exist_ok=True, repo_type="model")
    if not private:
        try:
            api.update_repo_settings(repo_id=repo, private=False, repo_type="model")
        except Exception as exc:
            print(f"[p1] could not force public visibility: {exc}", flush=True)
    print(f"[p1] hub ready user={user} repo={repo} private={private}", flush=True)
    return repo, api


def upload_files(api, repo: str, paths: list[str], prefix: str = "pilot") -> None:
    from huggingface_hub import HfApi

    for path in paths:
        if not os.path.exists(path):
            continue
        api.upload_file(path_or_fileobj=path, path_in_repo=f"{prefix}/{path}",
                        repo_id=repo, repo_type="model")
        print(f"[p1] uploaded {path} -> {repo}/{prefix}/{path}", flush=True)



@torch.inference_mode()
def local_rank_eval(model, layers, cfg, tokenizer, path: str, n: int = 20) -> tuple[float, int]:
    """Answer-block recall@K and worst rank on held-out sequences (mean over layers)."""
    from qsa.data import iter_local_sequences
    from qsa.hf_qsa import text_model

    seqs = list(iter_local_sequences(path, tokenizer, cfg["seq_len"], n,
                                     block_size=cfg["block_size"],
                                     instr=cfg.get("local_instruction", "")))
    recs, worst = [], 0
    kb = None
    for s in seqs:
        set_mode(layers, collect=True, budget=cfg["budget"])
        text_model(model)(input_ids=s["input_ids"][None].cuda())
        qs = s["q_start"]
        for layer in layers:
            target, hidden, rope = layer._qsa_target, layer._qsa_hidden, layer._qsa_rope
            layer._qsa_target = layer._qsa_hidden = layer._qsa_rope = None
            if target is None:
                continue
            cos, sin = rope
            dev = next(layer._qsa_indexer.parameters()).device
            hidden = hidden.to(dev)
            cos, sin = cos.to(dev), sin.to(dev)
            scores, valid = layer._qsa_indexer(hidden.float(), cos[0].float(), sin[0].float())
            ab = [b for b in s["answer_blocks"] if b < scores.shape[-1]]
            if not ab:
                continue
            qscores = scores[0, qs:, :]
            idx = torch.tensor(ab, device=scores.device, dtype=torch.long)
            s_best = qscores.index_select(-1, idx).max(-1).values
            rank = (qscores > s_best.unsqueeze(-1)).sum(-1) + 1
            kb = max(1, min(qscores.shape[-1], int(math.ceil(cfg["budget"] / cfg["block_size"]))))
            recs.append((rank <= kb).float().mean().item())
            worst = max(worst, int(rank.max().item()))
    return (sum(recs) / max(1, len(recs)), worst)


def run_distillation(cfg: dict) -> None:
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from qsa.data import iter_token_blocks
    from qsa.hf_qsa import full_attention_layers, install, set_spill

    seed = cfg.get("seed")
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_spill(bool(cfg.get("spill_capture", False)))

    hub = check_hub(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("distillation requires a CUDA GPU")

    revision = cfg.get("model_revision")
    tok_path = cfg.get("tokenizer_path") or cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, revision=None if cfg.get("tokenizer_path") else revision)
    dtype_name = str(cfg.get("dtype", "auto"))
    if dtype_name == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = getattr(torch, dtype_name)
    device_map = cfg.get("device_map", "cuda")
    if device_map in ("balanced", "auto") and torch.cuda.device_count() > 1:
        device_map = "balanced"
    load_kwargs: dict = {}
    if cfg.get("load_in_8bit"):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["dtype"] = torch.float16
        print("[p1] loading backbone in int8 (bitsandbytes)", flush=True)
    if cfg.get("config_path"):
        load_kwargs["config"] = Qwen3_5Config.from_pretrained(cfg["config_path"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["model"], revision=revision,
        **({"dtype": dtype} if "dtype" not in load_kwargs else {}),
        attn_implementation="sdpa", device_map=device_map, **load_kwargs,
    ).eval()
    print(f"[p1] model={cfg['model']} dtype={dtype} device_map={device_map}", flush=True)

    install(
        model,
        block_size=cfg["block_size"],
        budget=cfg["budget"],
        indexer_heads=cfg["indexer_heads"],
        indexer_head_dim=cfg["indexer_head_dim"],
        rope_dim=cfg["rope_dim"],
    )
    model.requires_grad_(False)
    layers = [layer for layer in full_attention_layers(model)
              if getattr(layer, "_qsa_indexer", None) is not None]
    params = [p for layer in layers for p in layer._qsa_indexer.parameters()]
    for p in params:
        p.requires_grad_(True)
    if cfg.get("indexer_file") or cfg.get("init_from_hub"):
        from huggingface_hub import hf_hub_download

        # indexer_repo / indexer_revision / indexer_file / indexer_sha256 pin the warm start (as in train_mm);
        # init_from_hub + hf_repo is the older unpinned form.
        repo = cfg.get("indexer_repo") or cfg.get("hf_repo")
        file = cfg.get("indexer_file") or cfg["init_from_hub"]
        if not repo:
            raise SystemExit("indexer_file / init_from_hub requires indexer_repo (or hf_repo)")
        path = hf_hub_download(repo, file, revision=cfg.get("indexer_revision"), repo_type="model")
        check_sha256(path, cfg.get("indexer_sha256"))
        state = torch.load(path, map_location="cpu", weights_only=True)
        for layer in layers:
            layer._qsa_indexer.load_state_dict(state[layer.layer_idx])
        print(f"[p1] warm start {repo}/{file}@{cfg.get('indexer_revision') or 'main'} "
              f"({len(state)} layers)", flush=True)

    if cfg.get("offload_embed"):
        from accelerate.hooks import remove_hook_from_module

        from qsa.hf_qsa import text_model as _text_model

        text = _text_model(model)
        emb = getattr(text, "embed_tokens", None)
        if emb is not None:
            weight = emb.weight.detach().to("cpu")
            freed = weight.numel() * weight.element_size() / 2**30
            dev0 = next(text.layers[0].parameters()).device
            remove_hook_from_module(emb, recurse=True)
            text.embed_tokens = _CPUEmbed(weight, dev0)
            torch.cuda.empty_cache()
            print(f"[p1] embed_tokens -> CPU ({freed:.1f} GiB freed, out {dev0})", flush=True)

    if cfg.get("init_local"):
        check_sha256(cfg["init_local"], cfg.get("init_sha256"))
        state = torch.load(cfg["init_local"], map_location="cpu", weights_only=True)
        for layer in layers:
            layer._qsa_indexer.load_state_dict(state[layer.layer_idx])
        print(f"[p1] warm start local {cfg['init_local']} ({len(state)} layers)", flush=True)

    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.0)
    print(f"[p1] layers {len(layers)} params {sum(p.numel() for p in params)/1e6:.1f}M",
          flush=True)

    overfit = bool(cfg.get("overfit", False))
    if not cfg.get("skip_eval", False):
        eval_blocks = list(iter_token_blocks(
            tokenizer, cfg["seq_len"], cfg["eval_seqs"],
            dataset=cfg["dataset"], subset=cfg.get("subset", "sample-10BT"),
            skip_docs=cfg.get("eval_skip_docs", 50_000),
            revision=cfg.get("dataset_revision"),
        ))
        set_mode(layers, collect=False, budget=0)
        dense_ppl, dense_tokens = perplexity(model, eval_blocks)
        print(f"[p1] dense ppl {dense_ppl:.4f} ({dense_tokens} tokens)", flush=True)

    if overfit:
        fixed = next(iter(iter_token_blocks(
            tokenizer, cfg["seq_len"], 1,
            dataset=cfg["dataset"], subset=cfg.get("subset", "sample-10BT"),
            skip_docs=cfg.get("skip_docs", 0),
            revision=cfg.get("dataset_revision"),
        )))[None].cuda()
        train_iter = iter([fixed] * cfg["steps"])
    elif cfg.get("local_data"):
        from qsa.data import combine_iterators, iter_local_sequences
        local_it = iter_local_sequences(
            cfg["local_data"], tokenizer, cfg["seq_len"], cfg["steps"],
            block_size=cfg["block_size"], instr=cfg.get("local_instruction", ""))
        if float(cfg.get("mix_fineweb", 0.0) or 0.0) > 0.0:
            web_it = iter_token_blocks(
                tokenizer, cfg["seq_len"], cfg["steps"],
                dataset=cfg["dataset"], subset=cfg.get("subset", "sample-10BT"),
                skip_docs=cfg.get("skip_docs", 0),
                revision=cfg.get("dataset_revision"))
            train_iter = iter(combine_iterators(local_it, web_it, 1.0 - float(cfg["mix_fineweb"])))
        else:
            train_iter = iter(local_it)
    else:
        train_iter = iter(iter_token_blocks(
            tokenizer, cfg["seq_len"], cfg["steps"],
            dataset=cfg["dataset"], subset=cfg.get("subset", "sample-10BT"),
            skip_docs=cfg.get("skip_docs", 0),
            revision=cfg.get("dataset_revision"),
        ))

    set_mode(layers, collect=True, budget=cfg["budget"])
    started = time.time()
    history = []
    step = 0
    ema_recall, ema_worst = None, None
    offset = int(cfg.get("step_offset", 0))
    prefix = cfg.get("upload_prefix", "pilot")
    for item in train_iter:
        if step >= cfg["steps"]:
            break
        meta = None
        if isinstance(item, dict):
            meta = item
            ids = item["input_ids"][None].cuda()
        elif overfit:
            ids = item
        else:
            ids = item[None].cuda()
        if meta is not None:
            meta["_step"] = step + offset
        loss, capture, stats = distill_step(model, layers, ids, cfg, optimizer, meta)
        step += 1
        a = 0.1
        ema_recall = stats.get("answer_recall", 0.0) if ema_recall is None else (1 - a) * ema_recall + a * stats.get("answer_recall", 0.0)
        ema_worst = stats.get("worst_rank", 0) if ema_worst is None else (1 - a) * ema_worst + a * stats.get("worst_rank", 0)
        save_every = cfg.get("save_every", 0)
        if save_every and step % save_every == 0:
            checkpoint = save_indexers(layers, step + offset)
            if hub is not None:
                repo, api = hub
                upload_files(api, repo, [checkpoint], prefix)

        if step % cfg.get("log_every", 10) == 0 or step == 1:
            record = {
                "step": step,
                "global_step": step + offset,
                "loss": loss,
                "capture": capture,
                "teacher_entropy": round(stats["teacher_entropy"], 3),
                "score_std": round(stats["score_std"], 3),
                "answer_recall": round(stats.get("answer_recall", 0.0), 4),
                "worst_rank": stats.get("worst_rank", 0),
                "ema_recall": round(ema_recall, 4),
                "ema_worst": round(ema_worst, 1),
                "seconds": round(time.time() - started, 1),
                "tokens": (step + offset) * cfg["seq_len"],
            }
            history.append(record)
            print("TRAIN " + json.dumps(record), flush=True)

    if cfg.get("local_eval_data"):
        er, ew = local_rank_eval(model, layers, cfg, tokenizer, cfg["local_eval_data"],
                                 int(cfg.get("local_eval_n", 20)))
        print(f"[p1] heldout answer_recall {er:.4f} worst_rank {ew}", flush=True)

    checkpoint = save_indexers(layers)
    result = {
        "overfit": overfit,
        "steps": step,
        "step_offset": offset,
        "tokens": (step + offset) * cfg["seq_len"],
        "final_loss": history[-1]["loss"] if history else None,
        "final_capture": history[-1]["capture"] if history else None,
        "history": history,
        "eval_answer_recall": er if cfg.get("local_eval_data") else None,
        "eval_worst_rank": ew if cfg.get("local_eval_data") else None,
        "checkpoint": checkpoint,
    }

    if not cfg.get("skip_eval", False):
        set_mode(layers, collect=False, budget=cfg["budget"])
        for layer in layers:
            layer._qsa_indexer.eval()
        sparse_ppl, sparse_tokens = perplexity(model, eval_blocks)
        result.update({
            "dense_ppl": dense_ppl,
            "sparse_ppl": sparse_ppl,
            "eval_tokens": sparse_tokens,
            "ppl_delta_pct": 100 * (sparse_ppl / dense_ppl - 1),
        })

    if cfg.get("retrieval_lengths"):
        from datasets import load_dataset

        from qsa.retrieval import build_examples, retrieval_accuracy

        stream = load_dataset(cfg["dataset"], cfg.get("subset", "sample-10BT"),
                              split="train", streaming=True,
                              revision=cfg.get("dataset_revision"))
        docs = [row["text"] for row in stream.take(cfg.get("retrieval_docs", 200))]
        examples = build_examples(tokenizer, docs, cfg["retrieval_lengths"],
                                  cfg.get("retrieval_n", 8))
        set_mode(layers, collect=False, budget=0)
        dense_ret = retrieval_accuracy(model, examples)
        print(f"[p1] retrieval dense {dense_ret}", flush=True)
        set_mode(layers, collect=False, budget=cfg["budget"])
        sparse_ret = retrieval_accuracy(model, examples)
        print(f"[p1] retrieval sparse {sparse_ret}", flush=True)
        result["retrieval"] = {"dense": dense_ret, "sparse": sparse_ret}

    with open("p1_results.json", "w") as f:
        json.dump(result, f, indent=2)

    if hub is not None:
        repo, api = hub
        paths = [checkpoint]
        if os.path.exists("p1_results.json"):
            paths.append("p1_results.json")
        try:
            upload_files(api, repo, paths, prefix)
        except Exception as exc:
            print(f"[p1] final upload failed: {type(exc).__name__}: {exc}", flush=True)
            raise

    print("RESULT " + json.dumps({
        key: result[key] for key in
        ("overfit", "steps", "tokens", "final_loss", "final_capture",
         "dense_ppl", "sparse_ppl", "ppl_delta_pct", "retrieval")
        if key in result
    }), flush=True)
