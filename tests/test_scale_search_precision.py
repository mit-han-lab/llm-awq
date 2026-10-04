import importlib
import sys
import types

import pytest
import torch
from torch import nn


class CheckedLinear(nn.Linear):
    def forward(self, x):
        assert self.weight.dtype == x.dtype
        assert torch.isfinite(self.weight).all()
        result = super().forward(x)
        assert torch.isfinite(result).all()
        return result


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_value = CheckedLinear(4, 4, bias=False)

    def forward(self, x):
        return self.query_key_value(x)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_h_to_4h = CheckedLinear(4, 8, bias=False)
        self.act = nn.GELU()
        self.dense_4h_to_h = CheckedLinear(8, 4, bias=False)

    def forward(self, x):
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class GPTNeoXTestLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.attention = Attention()
        self.mlp = MLP()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("activation", [0.0, 1e-6, 1.0])
def test_scale_search_keeps_weights_finite(monkeypatch, dtype, activation):
    # Scale search uses PyTorch operations only; no CUDA extension is called.
    monkeypatch.setitem(sys.modules, "awq_inference_engine", types.ModuleType("awq_inference_engine"))
    auto_scale = importlib.import_module("awq.quantize.auto_scale")
    torch.manual_seed(0)
    layer = GPTNeoXTestLayer().to(dtype)
    features = {
        name: torch.full((1, 4, module.in_features), activation, dtype=dtype)
        for name, module in layer.named_modules()
        if isinstance(module, nn.Linear)
    }
    scales = auto_scale.auto_scale_block(
        layer, {}, w_bit=4,
        q_config={"zero_point": True, "q_group_size": 4}, input_feat=features,
    )
    assert len(scales) == 3
    for _, _, scale in scales:
        assert scale.dtype == dtype
        assert torch.isfinite(scale).all()
        assert (scale > 0).all()
