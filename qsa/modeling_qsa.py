"""Reloadable QSA inference model, preserving Qwen's image/video dense path.

This is a correctness reference, not a sparse CUDA kernel. Only text decoder
attention instances are changed; the Transformers attention class is untouched.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForConditionalGeneration,
    apply_rotary_pos_emb,
)

from .indexer import QSAIndexer, apply_partial_rope


class QSAQwen3_5Config(Qwen3_5Config):
    # A distinct type prevents AutoModel from silently loading the dense class
    # when the caller forgets to enable the packaged QSA implementation.
    model_type = "qwen3_5_qsa"


class QSADynamicCache(DynamicCache):
    """Keep indexer blocks and the multimodal routing decision with the KV cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qsa_states = {}
        self.qsa_dense = False

    def _map_qsa(self, fn):
        for state in self.qsa_states.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = fn(value)

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self._map_qsa(lambda x: x.index_select(0, beam_idx.to(x.device)))

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        self._map_qsa(lambda x: x.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self._map_qsa(lambda x: x.index_select(0, indices.to(x.device)))

    def reset(self):
        super().reset()
        self.qsa_states.clear()
        self.qsa_dense = False

    def crop(self, tokens_to_remove):
        if tokens_to_remove:
            raise NotImplementedError("QSA reference cache does not support speculative rollback")
        super().crop(tokens_to_remove)


def _indexer_scores(indexer, hidden, cos, sin, state, offset):
    """Incrementally pool keys, including blocks spanning prefill/decode calls."""
    batch, length, _ = hidden.shape
    dtype = indexer.q_proj.weight.dtype
    hidden, cos, sin = hidden.to(dtype), cos.to(dtype), sin.to(dtype)
    q = indexer.q_proj(hidden).view(batch, length, indexer.num_heads, indexer.head_dim)
    q = indexer._norm(q, indexer.q_norm)
    q = apply_partial_rope(q, cos[:, :, None], sin[:, :, None], indexer.rope_dim)
    keys = indexer._norm(indexer.k_proj(hidden), indexer.k_norm)
    if state is not None:
        keys = torch.cat((state["tail"], keys), dim=1)
        cos = torch.cat((state["cos"], cos), dim=1)
        sin = torch.cat((state["sin"], sin), dim=1)
    r = indexer.block_size
    complete = keys.shape[1] // r
    pooled = keys[:, :complete * r].reshape(batch, complete, r, indexer.head_dim).mean(2)
    pooled = indexer._norm(pooled, indexer.k_block_norm)
    pooled = apply_partial_rope(
        pooled, cos[:, :complete * r:r], sin[:, :complete * r:r], indexer.rope_dim
    )
    blocks = torch.cat((state["blocks"], pooled), dim=1) if state is not None else pooled
    next_state = {
        "blocks": blocks.detach(),
        "tail": keys[:, complete * r:].detach().clone(),
        "cos": cos[:, complete * r:].detach().clone(),
        "sin": sin[:, complete * r:].detach().clone(),
        "length": offset + length,
    }
    logits = torch.einsum("bqhd,bnd->bhqn", q, blocks)
    # Match the FP32 indexer's temperature used by the completed pilot.
    scores = logits.relu().sum(1) * indexer.head_dim ** -0.5
    query_pos = torch.arange(offset, offset + length, device=hidden.device)
    block_ends = torch.arange(blocks.shape[1], device=hidden.device) * r + r - 1
    valid = (block_ends[None] <= query_pos[:, None])[None].expand(batch, -1, -1)
    return scores.masked_fill(~valid, -float("inf")), valid, next_state


class QSAQwenAttention(Qwen3_5Attention):
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, qsa_use_sparse=True, **kwargs):
        if not qsa_use_sparse or self._qsa_budget == 0:
            return super().forward(hidden_states, position_embeddings, attention_mask,
                                   past_key_values=past_key_values, **kwargs)
        if past_key_values is not None and not isinstance(past_key_values, QSADynamicCache):
            raise TypeError("QSA text inference requires QSADynamicCache")
        batch, length, _ = hidden_states.shape
        q, gate = self.q_proj(hidden_states).view(batch, length, -1, self.head_dim * 2).chunk(2, -1)
        query = self.q_norm(q).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch, length, -1, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        offset, state = 0, None
        if past_key_values is not None:
            offset = past_key_values.get_seq_length(self.layer_idx)
            state = past_key_values.qsa_states.get(self.layer_idx)
            if offset and (state is None or state["length"] != offset):
                raise ValueError("Indexer cache and attention KV cache are out of sync")
            key, value = past_key_values.update(key, value, self.layer_idx)
        scores, valid, next_state = _indexer_scores(
            self._qsa_indexer, hidden_states, cos, sin, state, offset
        )
        if past_key_values is not None:
            past_key_values.qsa_states[self.layer_idx] = next_state
        r = self._qsa_block
        nb = scores.shape[-1]
        kb = min(nb, math.ceil(self._qsa_budget / r))
        chosen = scores.topk(kb, dim=-1).indices
        selected = torch.zeros_like(valid).scatter_(-1, chosen, True) & valid
        key_pos = torch.arange(key.shape[-2], device=key.device)
        query_pos = torch.arange(offset, offset + length, device=key.device)
        block_ids = key_pos // r
        if nb:
            keep = selected[:, :, block_ids.clamp(max=nb - 1)] & (block_ids < nb)[None, None]
        else:
            keep = torch.zeros(batch, length, key.shape[-2], device=key.device, dtype=torch.bool)
        tail = block_ids[None] == (query_pos // r)[:, None]
        mask = ((keep | tail[None]) & (key_pos[None] <= query_pos[:, None])[None])[:, None]
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                mask = mask & attention_mask
            else:
                mask = attention_mask.masked_fill(~mask, -float("inf"))
        # Explicit GQA expansion matches the experimental implementation and
        # avoids an SDPA backend that materializes the full per-head score table.
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                               dropout_p=0.0, scale=self.scaling)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        output = output * gate.reshape(batch, length, -1).sigmoid()
        return self.o_proj(output), None


class QSAQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Text uses QSA; image/video requests retain original dense computation."""

    config_class = QSAQwen3_5Config
    _keep_in_fp32_modules = [
        *(getattr(Qwen3_5ForConditionalGeneration, "_keep_in_fp32_modules", None) or []),
        "_qsa_indexer",
    ]

    def __init__(self, config):
        super().__init__(config)
        qsa = getattr(config, "qsa", {})
        if qsa.get("multimodal_policy", "dense") != "dense":
            raise ValueError("Only the validated dense multimodal policy is supported")
        self.qsa_config = qsa
        for layer in self.model.language_model.layers:
            attn = getattr(layer, "self_attn", None)
            if not isinstance(attn, Qwen3_5Attention):
                continue
            attn.__class__ = QSAQwenAttention
            attn._qsa_block = qsa.get("block_size", 4)
            attn._qsa_budget = qsa.get("budget", 2048)
            attn._qsa_indexer = QSAIndexer(
                hidden_size=config.text_config.hidden_size,
                num_heads=qsa.get("indexer_heads", 4),
                head_dim=qsa.get("indexer_head_dim", 128),
                rope_dim=qsa.get("rope_dim", 64),
                block_size=attn._qsa_block,
            ).to(device=attn.q_proj.weight.device, dtype=torch.float32)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                video_grid_thw=None, mm_token_type_ids=None, logits_to_keep=0,
                mm_encoder_outputs=None, use_cache=None, **kwargs):
        use_cache = self.config.text_config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, QSADynamicCache):
            if type(past_key_values) is not DynamicCache or past_key_values.get_seq_length():
                raise TypeError("Use the default dynamic cache with this QSA reference model")
            past_key_values = QSADynamicCache(config=self.config.text_config)
        elif past_key_values is None and use_cache:
            past_key_values = QSADynamicCache(config=self.config.text_config)
        multimodal = any(x is not None for x in (
            pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw,
        )) or bool(mm_encoder_outputs)
        if input_ids is not None:
            multimodal = multimodal or bool(
                ((input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)).any()
            )
        if mm_token_type_ids is not None:
            multimodal = multimodal or bool(mm_token_type_ids.any())
        # Unknown embedded inputs and padded requests conservatively use dense.
        dense = multimodal or (inputs_embeds is not None and input_ids is None)
        dense = dense or (attention_mask is not None and attention_mask.ndim == 2
                          and bool((attention_mask == 0).any()))
        if past_key_values is not None:
            if dense and past_key_values.qsa_states:
                raise ValueError("Start a fresh cache when switching a text-only request to multimodal/padded input")
            past_key_values.qsa_dense = past_key_values.qsa_dense or dense
            dense = past_key_values.qsa_dense
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids, logits_to_keep=logits_to_keep,
            mm_encoder_outputs=mm_encoder_outputs, use_cache=use_cache,
            qsa_use_sparse=not dense, **kwargs,
        )


QSAQwen3_5Config.register_for_auto_class()
QSAQwen3_5ForConditionalGeneration.register_for_auto_class("AutoModelForImageTextToText")
