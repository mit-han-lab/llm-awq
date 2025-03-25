import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from awq.quantize import W8A8OF16LinearDynamicInputScale
from llava.model.multimodal_encoder.siglip.modeling_siglip import (
    SiglipMLP,
    SiglipEncoder,
    SiglipAttention,
    SiglipEncoderLayer,
)
from tinychat.utils.input_metadata import ActivationBuffer
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from flash_attn import flash_attn_func
import time

CLIP_RANGE = 5


import awq_inference_engine


class QuantSiglipEncoder(nn.Module):
    def __init__(self, module: SiglipEncoder, bsz=64, seqlen=1024):
        super().__init__()
        self.config = module.config
        self.layers = [QuantSiglipEncoderLayer(layer) for layer in module.layers]
        self.buffer = ActivationBuffer(module)
        self.bsz = bsz
        self.seqlen = seqlen
        self.buffer.allocate_activation_buffer(self.bsz * self.seqlen)

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,  # dummy
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # TODO Find why this code is necessary
        # torch.sum(inputs_embeds!=inputs_embeds)
        bsz, seqlen, _ = inputs_embeds.shape
        if self.bsz != bsz or self.seqlen != seqlen:
            self.buffer.allocate_activation_buffer(bsz * seqlen)
            self.bsz = bsz
            self.seqlen = seqlen

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (
                    hidden_states.reshape(bsz, seqlen, -1),
                )
            hidden_states = encoder_layer(
                hidden_states, self.buffer, attention_mask, bsz, seqlen
            )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states.reshape(bsz, seqlen, -1),)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states.reshape(bsz, seqlen, -1),
            hidden_states=encoder_states,
            attentions=None,
        )


class QuantSiglipMLP(nn.Module):
    def __init__(self, siglipmlp, init_only=False):
        super().__init__()
        self.config = siglipmlp.config
        self.activation_fn = siglipmlp.activation_fn
        self.fc1 = W8A8OF16LinearDynamicInputScale.from_linear(
            siglipmlp.fc1, init_only=init_only, fc1=False
        )
        self.fc2 = W8A8OF16LinearDynamicInputScale.from_linear(
            siglipmlp.fc2, init_only=init_only
        )
        self.invoke_quant = self.invoke_quant_mlp

    def invoke_quant_mlp(self, buffer, actfn_output):
        awq_inference_engine.invoke_quant(
            buffer.quantized_mlp_act_buffer,
            actfn_output,
            buffer.quantized_scale_buffer,
        )

    def forward(self, buffer: ActivationBuffer) -> torch.Tensor:
        # INT8 in, FP16 out
        self.fc1(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.fc1_buffer,
        )
        # Act & quantization
        awq_inference_engine.gelu_and_quant(
            buffer.quantized_mlp_act_buffer,
            buffer.fc1_buffer,
            buffer.quantized_scale_buffer,
            buffer.tmp,
        )
        # INT8 in, FP16 out
        self.fc2(
            buffer.quantized_mlp_act_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )


class QuantSiglipFlashAttention2(nn.Module):
    def __init__(
        self,
        module: SiglipAttention,
        init_only=False,
    ):
        super().__init__()
        self.config = module.config
        self.embed_dim = module.embed_dim
        self.num_heads = module.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = W8A8OF16LinearDynamicInputScale.from_qkv(
            module.q_proj, module.k_proj, module.v_proj, init_only=init_only
        )
        self.out_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            module.out_proj, init_only=init_only
        )
        self.invoke_quant = self.invoke_quant_wo

    def invoke_quant_wo(self, buffer, attn_output):
        awq_inference_engine.invoke_quant(
            buffer.quantized_hidden_states_buffer,
            attn_output,
            buffer.quantized_scale_buffer,
        )

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self, buffer: ActivationBuffer, bsz=64, seqlen=1024
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # qkv
        self.qkv_proj(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.qkv_proj_act_buffer,
        )
        q, k, v = buffer.qkv_proj_act_buffer.split(
            [self.embed_dim, self.embed_dim, self.embed_dim], dim=-1
        )
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        attn_output = flash_attn_func(q, k, v, softmax_scale=None, causal=False)
        attn_output = attn_output.reshape(bsz * seqlen, -1)
        # FP16 -> int8
        self.invoke_quant(buffer, attn_output)
        # INT8 in, FP16 out
        self.out_proj(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )


class QuantSiglipEncoderLayer(nn.Module):
    def __init__(self, module: SiglipEncoderLayer):
        super().__init__()
        self.embed_dim = module.embed_dim
        self.self_attn = QuantSiglipFlashAttention2(module.self_attn)
        self.layer_norm1 = RMSNormGeneral(
            module.layer_norm1.weight.data,
            module.layer_norm1.bias.data,
            module.layer_norm1.eps,
            True,
        ).cuda()
        self.mlp = QuantSiglipMLP(module.mlp)
        self.layer_norm2 = RMSNormGeneral(
            module.layer_norm2.weight.data,
            module.layer_norm2.bias.data,
            module.layer_norm2.eps,
            True,
        ).cuda()
        self.quant = self.invoke_quant_norm

    def invoke_quant_norm(self, buffer, normfn_output):
        awq_inference_engine.invoke_quant(
            buffer.quantized_hidden_states_buffer,
            normfn_output,
            buffer.quantized_scale_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        buffer: ActivationBuffer,
        attention_mask,
        bsz,
        seqlen,
    ) -> Tuple[torch.FloatTensor]:
        # Attention block
        # FP16 in int8 out, layernorm & quantization
        residual = hidden_states
        self.layer_norm1(
            hidden_states.reshape(-1, self.embed_dim),
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
        )

        # INT8 -> FP16
        self.self_attn(buffer, bsz, seqlen)
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        # Fully Connected
        residual = hidden_states
        # FP16 in int8 out, layernorm & quantization
        self.layer_norm2(
            hidden_states.reshape(-1, self.embed_dim),
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
        )

        # INT8 -> FP16
        self.mlp(buffer)
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        return hidden_states


class RMSNormGeneral(nn.Module):
    """Root mean square normalization (w/ per-token or per-tensor quant).

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        weight: torch.tensor,
        bias: torch.tensor,
        eps: float = 1e-6,
        use_per_token_quant: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.variance_epsilon = eps
        self.use_per_token_quant = use_per_token_quant

    def forward(
        self,
        x: torch.Tensor,
        quantized_hidden_states_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor = None,
    ) -> torch.Tensor:
        # quantized_sum_buffer is not used, only to keep the consistency of the interface
        awq_inference_engine.rms_norm_general(
            quantized_hidden_states_buffer,
            x,
            self.weight.data,
            self.bias.data,
            quantized_scale_buffer,
            self.variance_epsilon,
            self.use_per_token_quant,
        )
