import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


spec = importlib.util.spec_from_file_location(
    "smooth", Path(__file__).parents[1] / "awq/quantize/smooth.py"
)
smooth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smooth)


class CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=0)
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        if x.shape[0] == 0:
            raise RuntimeError("empty calibration batch")
        return self.proj(x)


def test_static_calibration_visits_every_sample_not_every_token():
    model = CalibrationModel()
    data = torch.ones(3, 2, 1)
    data[2] = 7
    _, stats = smooth.get_static_decoder_layer_scales(model, data)
    assert model.batch_sizes == [1, 1, 1]
    assert stats["proj"]["input"] == 7


def test_static_calibration_does_not_run_empty_batches():
    model = CalibrationModel()
    smooth.get_static_decoder_layer_scales(model, torch.ones(2, 4, 1))
    assert model.batch_sizes == [1, 1]
