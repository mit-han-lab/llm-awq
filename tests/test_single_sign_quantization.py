"""CPU regression tests for affine quantization ranges."""
import sys
from unittest.mock import Mock, patch

import pytest
import torch

# Fake quantization does not call the optional CUDA extension.
with patch.dict(sys.modules, {"awq_inference_engine": Mock()}):
    from awq.quantize.quantizer import pseudo_quantize_tensor


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("group_size", [-1, 4, 8])
def test_single_sign_groups_have_bounded_rounding_error(inplace, group_size):
    positive = torch.arange(10.5, 14.5, 0.5)
    weights = torch.stack((positive, -positive, torch.ones(8), -torch.ones(8), torch.zeros(8)))
    original = weights.clone()
    quantized, scales, zeros = pseudo_quantize_tensor(
        weights, n_bit=4, q_group_size=group_size, inplace=inplace, get_scale_zp=True
    )
    width = 8 if group_size == -1 else group_size
    error = (quantized - original).abs().reshape(-1, width)
    assert torch.all(error <= scales.reshape(-1, 1) / 2 + 1e-6)
    assert torch.all((zeros >= 0) & (zeros <= 15))
    assert torch.isfinite(quantized).all()
    if inplace:
        assert quantized.data_ptr() == weights.data_ptr()
    else:
        torch.testing.assert_close(weights, original)


def test_mixed_sign_range_is_unchanged():
    weights = torch.tensor([[-2., -1., 1., 3.]])
    quantized, scales, zeros = pseudo_quantize_tensor(weights, n_bit=4, get_scale_zp=True)
    expected_scale = torch.tensor([[5. / 15.]])
    expected_zero = torch.tensor([[6.]])
    expected = ((weights / expected_scale).round() + expected_zero).clamp(0, 15)
    expected = (expected - expected_zero) * expected_scale
    torch.testing.assert_close(scales, expected_scale)
    torch.testing.assert_close(zeros, expected_zero)
    torch.testing.assert_close(quantized, expected)
