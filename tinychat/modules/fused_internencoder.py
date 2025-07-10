from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.layers import DropPath
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from awq.quantize import W8A8OF16LinearDynamicInputScale
import awq_inference_engine

from tinychat.models.internvl.internvit import (FlashAttention,
                                                InternRMSNorm,
                                                InternVisionEmbeddings,
                                                InternAttention,
                                                InternMLP,
                                                InternVisionEncoderLayer,
                                                InternVisionEncoder)
from tinychat.models.internvl.configuration_internvl import InternVisionConfig

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import \
        flash_attn_varlen_qkvpacked_func
    has_flash_attn = True
except:
    print('FlashAttention2 is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)


class QuantInternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, module: InternVisionEncoder, bsz=64, seqlen=1024):
        super().__init__()
        self.config = module.config
        # stochastic depth decay rule
        self.layers = nn.ModuleList([QuantInternVisionEncoderLayer(layer, self.config) for layer in module.layers])
        self.gradient_checkpointing = True
        self.bsz = bsz
        self.seqlen = seqlen

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )

class QuantInternRMSNorm(nn.Module):
    def __init__(self, module: nn.Module, use_per_token_quant=True):
        super().__init__()
        self.weight = nn.Parameter(module.weight.data, requires_grad=False)
        self.bias = nn.Parameter(module.bias.data, requires_grad=False)
        self.variance_epsilon = module.eps
        self.use_per_token_quant = use_per_token_quant

    def forward(self, hidden_states):
        bsz, seqlen, hidden_size = hidden_states.shape
        output = torch.empty((bsz * seqlen), hidden_size, device=hidden_states.device, dtype=torch.int8)
        scale = torch.empty((bsz * seqlen), device=hidden_states.device, dtype=hidden_states.dtype)
        awq_inference_engine.rms_norm_general(
            output,
            hidden_states,
            self.weight,
            self.bias,
            scale,
            self.variance_epsilon,
            self.use_per_token_quant,
        )
        return output, scale

class QuantInternAttention(nn.Module):
    def __init__(self, module: InternAttention, config: InternVisionConfig, init_only=False):
        super().__init__()
        self.config = config
        self.embed_dim = module.embed_dim
        self.num_heads = module.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = module.scale
        self.use_flash_attn = config.use_flash_attn

        self.qkv = W8A8OF16LinearDynamicInputScale.from_linear(module.qkv, init_only=init_only)
        self.proj = W8A8OF16LinearDynamicInputScale.from_linear(module.proj, init_only=init_only)

        self.qk_normalization = module.qk_normalization
        if self.qk_normalization:
            self.q_norm = QuantInternRMSNorm(module.q_norm)
            self.k_norm = QuantInternRMSNorm(module.k_norm)

        if self.use_flash_attn:
            from tinychat.models.internvl.internvit import FlashAttention
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)

    def forward(self, hidden_states: torch.Tensor, scale_in: torch.Tensor):
        bsz, seqlen, hidden_size = hidden_states.shape

        qkv_out = torch.empty(bsz * seqlen, 3 * hidden_size, dtype=torch.float16, device=hidden_states.device)
        self.qkv(hidden_states.reshape(-1, hidden_size), scale_in, qkv_out)

        qkv = rearrange(qkv_out.view(bsz, seqlen, -1), 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q, _ = self.q_norm(q.flatten(-2, -1)); q = q.view_as(q)
            k, _ = self.k_norm(k.flatten(-2, -1)); k = k.view_as(k)
            qkv = torch.stack([q, k, v], dim=2)

        attn_out, _ = self.inner_attn(qkv, need_weights=False, causal=False)
        attn_out = rearrange(attn_out, 'b s h d -> (b s) (h d)')

        quant_out = torch.empty_like(attn_out, dtype=torch.int8)
        scale_proj_in = torch.empty(bsz * seqlen, device=hidden_states.device, dtype=torch.float16)
        awq_inference_engine.invoke_quant(quant_out, attn_out, scale_proj_in)

        proj_out = torch.empty_like(attn_out)
        self.proj(quant_out, scale_proj_in, proj_out)

        return proj_out

class QuantInternMLP(nn.Module):
    def __init__(self, module: InternMLP, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = module.act
        self.fc1 = W8A8OF16LinearDynamicInputScale.from_linear(module.fc1)
        self.fc2 = W8A8OF16LinearDynamicInputScale.from_linear(module.fc2)

    def forward(self, hidden_states: torch.Tensor, scale_in: torch.Tensor):
        bsz, seqlen, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        fc1_out = torch.empty((bsz * seqlen), self.config.intermediate_size, dtype=torch.float16, device=device)
        self.fc1(hidden_states.reshape(-1, hidden_size), scale_in, fc1_out)
        
        tmp = torch.empty(
            ((bsz * seqlen) * self.config.intermediate_size),
            device=device,
            dtype=torch.float16,
        )
        act_out = torch.empty_like(fc1_out, dtype=torch.int8)
        scale_act = torch.empty(bsz * seqlen, device=device, dtype=torch.float16)
        awq_inference_engine.gelu_and_quant(act_out, fc1_out, scale_act, tmp)

        fc2_out = torch.empty((bsz * seqlen), hidden_size, dtype=torch.float16, device=device)
        self.fc2(act_out, scale_act, fc2_out)

        return fc2_out

class QuantInternVisionEncoderLayer(nn.Module):
    def __init__(self, module: InternVisionEncoderLayer, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.attn = QuantInternAttention(module.attn, config)
        self.mlp = QuantInternMLP(module.mlp, config)

        self.norm1 = QuantInternRMSNorm(module.norm1)
        self.norm2 = QuantInternRMSNorm(module.norm2)

        self.ls1 = module.ls1
        self.ls2 = module.ls2

    def forward(self, hidden_states: torch.Tensor):
        bsz, seqlen, hidden_size = hidden_states.shape

        residual = hidden_states
        norm1_out, scale1 = self.norm1(hidden_states)
        attn_out = self.attn(norm1_out.view(bsz, seqlen, hidden_size), scale1)
        hidden_states = residual + attn_out.view(bsz, seqlen, hidden_size) * self.ls1

        residual = hidden_states
        norm2_out, scale2 = self.norm2(hidden_states)
        mlp_out = self.mlp(norm2_out.view(bsz, seqlen, hidden_size), scale2)
        hidden_states = residual + mlp_out.view(bsz, seqlen, hidden_size) * self.ls2

        return hidden_states


