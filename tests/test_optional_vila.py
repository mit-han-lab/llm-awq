import importlib
import sys
import types

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, OPTConfig, OPTForCausalLM


def test_text_model_embeddings_without_vila(monkeypatch):
    # Simulate a text-only installation, regardless of optional local packages.
    monkeypatch.setitem(sys.modules, "tinychat.models", None)
    monkeypatch.setitem(sys.modules, "awq_inference_engine", types.ModuleType("awq_inference_engine"))
    from awq.quantize import pre_quant
    importlib.reload(pre_quant)

    neox = GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4,
    ))
    assert pre_quant.get_blocks(neox) is neox.gpt_neox.layers
    pre_quant.move_embed(neox, "meta")
    assert neox.gpt_neox.embed_in.weight.device == torch.device("meta")
    assert neox.embed_out.weight.device == torch.device("meta")

    opt = OPTForCausalLM(OPTConfig(
        vocab_size=32, hidden_size=16, ffn_dim=32,
        num_hidden_layers=1, num_attention_heads=4,
    ))
    pre_quant.move_embed(opt, "meta")
    assert opt.model.decoder.embed_tokens.weight.device == torch.device("meta")
    assert opt.model.decoder.embed_positions.weight.device == torch.device("meta")
