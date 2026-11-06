"""Exercise generation and multimodal routing, not just artifact filenames."""

import json

import pytest
import torch
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from qsa.export_model import export_model
from qsa.validate_export import validate_export
from qsa.modeling_qsa import QSAQwen3_5ForConditionalGeneration, QSADynamicCache


def config(budget=4):
    return Qwen3_5Config(
        image_token_id=120, video_token_id=121,
        vision_start_token_id=122, vision_end_token_id=123,
        text_config={
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 4, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 32,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "linear_conv_kernel_dim": 4, "linear_num_key_heads": 2,
            "linear_num_value_heads": 4, "linear_key_head_dim": 16,
            "linear_value_head_dim": 16, "pad_token_id": 0,
            "eos_token_id": 127,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.5, "mrope_section": [3, 3, 2]},
        },
        vision_config={
            "depth": 2, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 4, "out_hidden_size": 64, "patch_size": 2,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
            "num_position_embeddings": 16,
        },
        qsa={"budget": budget, "block_size": 4, "indexer_heads": 2,
             "indexer_head_dim": 16, "rope_dim": 16, "multimodal_policy": "dense"},
    )


@pytest.fixture()
def models():
    torch.manual_seed(13)
    cfg = config()
    cfg._attn_implementation = "sdpa"
    dense = Qwen3_5ForConditionalGeneration(cfg).eval()
    converted = QSAQwen3_5ForConditionalGeneration(cfg).eval()
    missing, unexpected = converted.load_state_dict(dense.state_dict(), strict=False)
    assert missing and all("._qsa_indexer." in key for key in missing)
    assert not unexpected
    return dense, converted


@torch.inference_mode()
def test_runtime_prefill_matches_research_reference(models):
    from qsa.hf_qsa import _qsa_forward
    _, model = models
    attn = model.model.language_model.layers[1].self_attn
    hidden = torch.randn(1, 17, 64)
    positions = torch.arange(17)[None, None].expand(3, 1, -1)
    rope = model.model.language_model.rotary_emb(hidden, positions)
    want, _ = _qsa_forward(attn, hidden, rope)
    got, _ = attn(hidden, rope)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)


@torch.inference_mode()
def test_text_sparse_cache_matches_full_sequence(models):
    _, model = models
    ids = torch.randint(1, 100, (1, 15))
    # Cross both an incomplete block and a complete-block boundary on decode.
    cache = None
    for start, end in [(0, 7), (7, 9), (9, 10), (10, 15)]:
        got = model(input_ids=ids[:, start:end], past_key_values=cache, use_cache=True)
        cache = got.past_key_values
        want = model(input_ids=ids[:, :end], use_cache=False)
        torch.testing.assert_close(got.logits, want.logits[:, start:end], atol=2e-5, rtol=2e-4)
    assert isinstance(cache, QSADynamicCache)
    assert not cache.qsa_dense
    assert len(cache.qsa_states) == 2
    assert all(state["length"] == 15 for state in cache.qsa_states.values())


@pytest.mark.parametrize("modality", ["image", "video"])
@torch.inference_mode()
def test_multimodal_dense_parity_and_cached_continuation(models, modality):
    dense, model = models
    ids = torch.randint(1, 100, (1, 16))
    ids[0, 1:4] = torch.tensor([122, 120 if modality == "image" else 121, 123])
    types = torch.zeros_like(ids)
    types[0, 2] = 1 if modality == "image" else 2
    pixels = torch.randn(4, 3 * 2 * 2 * 2)
    grid = torch.tensor([[1, 2, 2]])
    kwargs = {"mm_token_type_ids": types}
    if modality == "image":
        kwargs.update(pixel_values=pixels, image_grid_thw=grid)
    else:
        kwargs.update(pixel_values_videos=pixels, video_grid_thw=grid)
    original = dense(input_ids=ids, use_cache=True, **kwargs)
    converted = model(input_ids=ids, use_cache=True, **kwargs)
    torch.testing.assert_close(converted.logits, original.logits, atol=1e-6, rtol=1e-5)
    assert converted.past_key_values.qsa_dense
    assert not converted.past_key_values.qsa_states
    token = torch.tensor([[15]])
    original_next = dense(input_ids=token, past_key_values=original.past_key_values, use_cache=True)
    converted_next = model(input_ids=token, past_key_values=converted.past_key_values, use_cache=True)
    torch.testing.assert_close(converted_next.logits, original_next.logits, atol=1e-6, rtol=1e-5)
    original_generated = dense.generate(ids, max_new_tokens=3, do_sample=False,
                                        pad_token_id=0, **kwargs)
    converted_generated = model.generate(ids, max_new_tokens=3, do_sample=False,
                                         pad_token_id=0, **kwargs)
    assert torch.equal(converted_generated, original_generated)


@torch.inference_mode()
def test_full_budget_and_padding_preserve_dense(models):
    dense, model = models
    ids = torch.randint(1, 100, (2, 12))
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    want = dense(input_ids=ids, attention_mask=mask, use_cache=False)
    got = model(input_ids=ids, attention_mask=mask, use_cache=True)
    assert got.past_key_values.qsa_dense
    torch.testing.assert_close(got.logits, want.logits, atol=1e-6, rtol=1e-5)
    for layer in model.model.language_model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn._qsa_budget = 64
    want = dense(input_ids=ids, use_cache=False)
    got = model(input_ids=ids, use_cache=False)
    torch.testing.assert_close(got.logits, want.logits, atol=1e-6, rtol=1e-5)


@torch.inference_mode()
def test_generation_and_beam_reordering(models):
    _, model = models
    ids = torch.randint(1, 100, (1, 9))
    for beams in [1, 2]:
        cached = model.generate(ids, max_new_tokens=4, do_sample=False, num_beams=beams,
                                use_cache=True, pad_token_id=0)
        fresh = model.generate(ids, max_new_tokens=4, do_sample=False, num_beams=beams,
                               use_cache=False, pad_token_id=0)
        assert torch.equal(cached, fresh)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_export_is_standalone_and_auto_reloadable(models, tmp_path, dtype):
    dense, trained = models
    dense.to(dtype)
    trained.load_state_dict(dense.state_dict(), strict=False, assign=True)
    source = tmp_path / "base"
    dense.save_pretrained(source, max_shard_size="30KB")
    (source / "preprocessor_config.json").write_text('{"preserved": true}')
    (source / "chat_template.jinja").write_text("test template")
    indexers = {i: layer.self_attn._qsa_indexer.state_dict()
                for i, layer in enumerate(trained.model.language_model.layers)
                if hasattr(layer, "self_attn")}
    checkpoint = tmp_path / "indexers.pt"
    torch.save(indexers, checkpoint)
    output = tmp_path / "converted"
    manifest = export_model(source, checkpoint, output, qsa=trained.config.qsa)
    assert validate_export(output)["tensor_layout_validated"]
    assert manifest["vision_tensor_count"] > 0
    assert (output / "chat_template.jinja").read_text() == "test template"
    # Hard links are standalone files: deleting source directory entries is safe.
    import shutil
    shutil.rmtree(source)
    restored = AutoModelForImageTextToText.from_pretrained(
        output, trust_remote_code=True, attn_implementation="sdpa", dtype="auto",
    ).eval()
    for name, parameter in restored.named_parameters():
        assert parameter.dtype == (torch.float32 if "._qsa_indexer." in name else dtype)
    ids = torch.randint(1, 100, (1, 13))
    torch.testing.assert_close(restored(ids, use_cache=False).logits,
                               trained(ids, use_cache=False).logits, atol=1e-6, rtol=1e-5)
    assert restored.config.qsa["multimodal_policy"] == "dense"
    assert set(restored.state_dict()) == set(trained.state_dict())
    for name, weight in dense.state_dict().items():
        assert torch.equal(weight, restored.state_dict()[name])
    config_saved = json.loads((output / "config.json").read_text())
    assert "AutoModelForMultimodalLM" in config_saved["auto_map"]
    with pytest.raises(ValueError):
        AutoConfig.from_pretrained(output, trust_remote_code=False)
