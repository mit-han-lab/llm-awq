"""RoPE initialization follows the model's configured context and cache device."""

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


@pytest.mark.parametrize("model_limit, cache_limit", [(4096, 64), (64, 4096)])
def test_llama_rope_uses_configured_length_and_cache_device(
    monkeypatch, model_limit, cache_limit
):
    # Construction does not call either optional inference extension.
    monkeypatch.setitem(
        sys.modules, "awq_inference_engine", types.ModuleType("awq_inference_engine")
    )
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_func = None
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    from tinychat.utils import constants

    constants.init()
    path = Path(__file__).resolve().parents[1] / "tinychat/models/llama.py"
    spec = importlib.util.spec_from_file_location("llama_rope_cache", path)
    llama = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llama)
    monkeypatch.setattr(llama, "max_seq_len", cache_limit)
    monkeypatch.setattr(llama, "max_batch_size", 1)
    config = SimpleNamespace(
        num_attention_heads=2,
        hidden_size=16,
        num_key_value_heads=2,
        max_position_embeddings=model_limit,
        rope_theta=10000.0,
        rope_scaling=None,
    )

    attention = llama.LlamaAttentionFused(config)

    assert isinstance(attention.rotary_emb, LlamaRotaryEmbedding)
    assert attention.kv_max_seq_len == min(model_limit, cache_limit)
    assert attention.rotary_emb.max_seq_len_cached == model_limit
    assert attention.rotary_emb.inv_freq.device == attention.cache_k.device
