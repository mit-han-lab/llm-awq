import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from transformers.models.llama.modeling_llama import LlamaMLP

import awq_inference_engine


class QuantLlamaMLP(nn.Module):
    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj,
    ):
        super().__init__()
        self.register_buffer("gate_proj_qweight", gate_proj.qweight)
        self.register_buffer("gate_proj_scales", gate_proj.scales)
        self.register_buffer("gate_proj_qzeros", gate_proj.qzeros)
        self.register_buffer("up_proj_qweight", up_proj.qweight)
        self.register_buffer("up_proj_scales", up_proj.scales)
        self.register_buffer("up_proj_qzeros", up_proj.qzeros)

        self.in_features = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features
        self.out_features = down_proj.out_features
        self.w_bit = gate_proj.w_bit
        self.down_proj = down_proj
        self.split_k_iters = down_proj.split_k_iters

    def forward(self, x):
        return self.down_proj(self.our_llama_mlp(x))

    def our_llama_mlp(self, x):
        out_shape = x.shape[:-1] + (self.intermediate_size,)
        x = x.reshape(-1, x.shape[-1])

        if x.shape[0] <= 8:
            gate_output = awq_inference_engine.gemv_forward_cuda(
                x,
                self.gate_proj_qweight,
                self.gate_proj_scales,
                self.gate_proj_qzeros,
                self.down_proj.group_size,
            )
            gate_output = F.silu(gate_output)
            up_output = awq_inference_engine.gemv_forward_cuda(
                x,
                self.up_proj_qweight,
                self.up_proj_scales,
                self.up_proj_qzeros,
                self.down_proj.group_size,
            )
        else:
            gate_output = awq_inference_engine.gemm_forward_cuda(
                x,
                self.gate_proj_qweight,
                self.gate_proj_scales,
                self.gate_proj_qzeros,
                self.down_proj.group_size,
                self.split_k_iters,
            )
            gate_output = F.silu(gate_output)
            up_output = awq_inference_engine.gemm_forward_cuda(
                x,
                self.up_proj_qweight,
                self.up_proj_scales,
                self.up_proj_qzeros,
                self.down_proj.group_size,
                self.split_k_iters,
            )

        c = gate_output * up_output
        c = c.reshape(out_shape)
        return c


def make_fused_mlp(m, parent_name=""):
    if not hasattr(make_fused_mlp, "called"):
        # print("[Warning] Calling a fake MLP fusion. But still faster than Huggingface Implimentation.")
        make_fused_mlp.called = True
    """
    Replace all LlamaMLP modules with QuantLlamaMLP modules, which fuses many of the operations.
    """
    if m.__class__.__name__ in ["LlamaMLP"]:
        return QuantLlamaMLP(m.gate_proj, m.down_proj, m.up_proj)

    for name, child in m.named_children():
        child = make_fused_mlp(child, parent_name=f"{parent_name}.{name}")

        if isinstance(child, QuantLlamaMLP):
            setattr(m, name, child)
    return m
