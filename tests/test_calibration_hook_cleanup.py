import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "smooth", Path(__file__).parents[1] / "awq/quantize/smooth.py"
)
smooth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smooth)


class FailingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def forward(self, x):
        self.proj(x)
        raise RuntimeError("calibration failed")


@pytest.mark.parametrize(
    "calibrate", [smooth.get_act_scales, smooth.get_static_decoder_layer_scales]
)
def test_failed_calibration_does_not_leave_hooks_on_model(calibrate):
    model = FailingModel()
    observed = []
    original_hook = model.proj.register_forward_hook(lambda *args: observed.append(1))
    with pytest.raises(RuntimeError, match="calibration failed"):
        calibrate(model, torch.ones(2, 3, 2))
    assert list(model.proj._forward_hooks) == [original_hook.id]
    model.proj(torch.ones(1, 2))
    assert observed == [1, 1]
    original_hook.remove()
