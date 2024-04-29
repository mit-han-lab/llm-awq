# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
import awq_inference_engine
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# from flash_attn.flash_attn_interface import flash_attn_unpadded_func

import tinychat.utils.constants

max_batch_size = tinychat.utils.constants.max_batch_size
global_max_seq_len = tinychat.utils.constants.max_seq_len


def gen_slopes(n_heads, alibi_bias_max=8):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(
    n_heads, seq_len, full=False, alibi_bias_max=8, dtype=torch.float32
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32).view(1, 1, 1, seq_len)
    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32).view(
            1, 1, seq_len, 1
        )
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max)
    alibi_bias = alibi_bias * slopes
    slopes = slopes.squeeze(0).squeeze(-1).squeeze(-1)
    return slopes.to(dtype=dtype), alibi_bias.to(dtype=dtype)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class LPLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_bias = (
            _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        )
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


class SharedEmbedding(nn.Embedding):
    def forward(self, input: torch.Tensor, unembed: bool = False) -> torch.Tensor:
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)


class MPTAttentionFused(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_heads
        self.head_dim = args.d_model // args.n_heads
        args.max_seq_len = min(args.max_seq_len, global_max_seq_len)

        self.Wqkv = nn.Linear(
            args.d_model,
            args.n_heads * self.head_dim * 3,
            bias=False,
        )

        self.out_proj = nn.Linear(
            args.n_heads * self.head_dim,
            args.d_model,
            bias=False,
        )

        # following fastertransformer definition

        self.cache_v = (
            torch.zeros(
                (
                    max_batch_size,
                    self.n_local_heads,
                    args.max_seq_len,
                    self.head_dim,
                )
            )
            .cuda()
            .half()
        )  # added to half
        # 8: pack 8 fp16 in FT, if fp32 then use 4
        self.cache_k = (
            torch.zeros(
                (
                    max_batch_size,
                    self.n_local_heads,
                    self.head_dim // 8,
                    args.max_seq_len,
                    8,
                )
            )
            .cuda()
            .half()
        )  # added to half

        alibi_slopes, alibi_bias = build_alibi_bias(
            self.n_local_heads, args.max_seq_len
        )
        # TODO (Haotian): fix device
        self.alibi_slopes = alibi_slopes.float().to("cuda:0")
        self.alibi_bias = alibi_bias.to("cuda:0")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xqkv = self.Wqkv(x)
        xqkv = xqkv.view(bsz, seqlen, -1, self.n_local_heads, self.head_dim)
        xq = xqkv[:, :, 0]
        xk = xqkv[:, :, 1]
        xv = xqkv[:, :, 2]

        if seqlen > 1:
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape(bsz, seqlen, self.n_local_heads, self.head_dim // 8, 8)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )

            self.cache_v[:bsz, :, start_pos : start_pos + seqlen, :] = values_store
            self.cache_k[:bsz, :, :, start_pos : start_pos + seqlen, :] = keys_store

            keys = xk
            values = xv

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores += self.alibi_bias[..., :seqlen]
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
            # xq = xq[:, 0, :, :]
            # xk = xk[:, 0, :, :]
            # xv = xv[:, 0, :, :]
            xq = xq.view(bsz, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, self.n_local_heads, self.head_dim)
            output = awq_inference_engine.single_query_attention(
                xq,
                xk,
                xv,
                self.cache_k,
                self.cache_v,
                None,
                # with alibi embedding
                self.alibi_slopes.float(),
                start_pos,
                # rotary embed dim = 0 => no rotary embedding
                0,
                10000,
                1.0,
                True,
            )
            output = output.reshape(bsz, 1, -1)

        return self.out_proj(output)


class MPTMLP(nn.Module):
    def __init__(self, d_model: int, expansion_ratio: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, expansion_ratio * d_model, bias=False)
        self.act = nn.GELU(approximate="none")
        self.down_proj = nn.Linear(expansion_ratio * d_model, d_model, bias=False)
        self.down_proj._is_residual = True

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class MPTBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.d_model
        self.head_dim = args.d_model // args.n_heads
        self.attn = MPTAttentionFused(args)
        self.ffn = MPTMLP(d_model=args.d_model, expansion_ratio=4)
        self.layer_id = layer_id
        self.norm_1 = LPLayerNorm(args.d_model, eps=1e-6)
        self.norm_2 = LPLayerNorm(args.d_model, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attn.forward(self.norm_1(x), start_pos, mask)
        out = h + self.ffn.forward(self.norm_2(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.wte = SharedEmbedding(params.vocab_size, params.d_model)

        self.blocks = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.blocks.append(MPTBlock(layer_id, params))

        self.norm_f = LPLayerNorm(params.d_model, eps=1e-6)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.wte(tokens)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.blocks:
            h = layer(h, start_pos, mask)
        h = self.norm_f(h)
        return h


class MPTForCausalLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.config = params
        self.transformer = Transformer(params)
        if params.no_bias:
            for module in self.modules():
                if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                    module.register_parameter("bias", None)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        h = self.transformer(tokens, start_pos)
        output = self.transformer.wte(h, unembed=True)  # only compute last logits
        return output.float()
