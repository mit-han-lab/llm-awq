import importlib.util
from pathlib import Path

import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "smooth", Path(__file__).parents[1] / "awq/quantize/smooth.py"
)
smooth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smooth)


class TransposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, x):
        return self.proj(x.transpose(1, 2))


def test_activation_scales_accept_noncontiguous_linear_inputs():
    model = TransposedModel()
    data = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
    actual = smooth.get_act_scales(model, data)["proj"]
    expected = data.transpose(1, 2).reshape(-1, 4).abs().amax(dim=0)
    torch.testing.assert_close(actual, expected)
