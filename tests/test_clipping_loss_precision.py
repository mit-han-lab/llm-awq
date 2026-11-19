"""CPU regressions for clipping-search reconstruction loss precision."""

import sys
from unittest.mock import Mock, patch

import pytest
import torch

# Clipping search uses the real tensor quantizer, not the CUDA extension.
with patch.dict(sys.modules, {"awq_inference_engine": Mock()}):
    from awq.quantize.auto_clip import auto_clip_layer
    from awq.quantize.quantizer import pseudo_quantize_tensor


@pytest.mark.parametrize("group_size", [8, -1])
@pytest.mark.parametrize(
    "dtype, activation_scale",
    [
        (torch.float16, 1),
        (torch.float16, 1000),
        (torch.float16, 10000),
        (torch.float32, 1),
        (torch.float32, 1000),
        (torch.float32, 1000000),
    ],
)
def test_clipping_matches_float32_loss_oracle(dtype, activation_scale, group_size):
    generator = torch.Generator().manual_seed(0)
    row = torch.randn(1, 8, generator=generator).to(dtype)
    weights = row.repeat(64, 1)
    inputs = torch.randn(8, 8, generator=generator).to(dtype) * activation_scale
    config = {"zero_point": True, "q_group_size": group_size}

    # Retain the original forward and quantization dtype. Only the reference
    # subtraction, square, and reduction use float32, as in the production fix.
    original = (inputs * row).sum(dim=-1)
    assert original.dtype == dtype
    assert torch.isfinite(original).all()
    thresholds, errors, low_precision_errors = [], [], []
    for shrink in range(10):
        threshold = row.abs().max() * (1 - shrink / 20)
        quantized = pseudo_quantize_tensor(
            row.clamp(-threshold, threshold), n_bit=4, **config
        )
        output = (inputs * quantized).sum(dim=-1)
        assert output.dtype == dtype
        assert torch.isfinite(output).all()
        thresholds.append(threshold)
        errors.append((output.float() - original.float()).square().mean())
        low_precision_errors.append((output - original).square().mean())
    errors = torch.stack(errors)
    assert torch.isfinite(errors).all()
    expected = thresholds[errors.argmin().item()]
    # The fixture must distinguish a real search from returning the uncut bound.
    assert expected < row.abs().max()
    if dtype == torch.float16 and activation_scale > 1:
        assert torch.isinf(torch.stack(low_precision_errors)).any()
    if dtype == torch.float16 and activation_scale == 10000:
        assert errors.min() > torch.finfo(dtype).max
    if dtype == torch.float32 and activation_scale == 1000000:
        assert errors.min() > 1e9

    actual = auto_clip_layer(
        weights, inputs, 4, config, n_sample_token=inputs.shape[0]
    )
    assert actual.shape == (64, 1, 1)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0)
