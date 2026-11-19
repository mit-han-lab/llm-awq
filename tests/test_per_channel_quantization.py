"""CPU regressions for per-channel AWQ search and checkpoint packing."""
import sys
from unittest.mock import Mock, patch

import pytest
import torch

# These tests exercise packing, not the optional CUDA inference extension.
with patch.dict(sys.modules, {"awq_inference_engine": Mock()}):
    from awq.quantize.auto_clip import auto_clip_layer
    from awq.quantize.qmodule import WQLinear
    from awq.quantize.quantizer import pseudo_quantize_tensor


def test_per_channel_clipping_matches_explicit_group():
    torch.manual_seed(0)
    weights = torch.randn(64, 128)
    inputs = torch.randn(32, 128)
    results = [
        auto_clip_layer(
            weights, inputs, 4, {"zero_point": True, "q_group_size": group_size},
            n_sample_token=32,
        )
        for group_size in (-1, 128)
    ]
    assert results[0].shape == (64, 1, 1)
    torch.testing.assert_close(*results)


@pytest.mark.parametrize("bias", [False, True])
def test_per_channel_packing_matches_explicit_group(bias):
    torch.manual_seed(0)
    linear = torch.nn.Linear(128, 64, bias=bias)
    linear.weight.data, scales, zeros = pseudo_quantize_tensor(
        linear.weight.data, n_bit=4, q_group_size=-1, get_scale_zp=True
    )
    implicit = WQLinear.from_linear(linear, 4, -1, scales=scales, zeros=zeros)
    explicit = WQLinear.from_linear(linear, 4, 128, scales=scales, zeros=zeros)
    assert implicit.group_size == 128
    for name, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name])
