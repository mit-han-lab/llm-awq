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
multiple_of = tinychat.utils.constants.llama_multiple_of
max_seq_len = tinychat.utils.constants.max_seq_len


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = torch.empty_like(x)
        awq_inference_engine.layernorm_forward_cuda(x, self.weight, output, self.eps)
        return output


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # k_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_ = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
    )
    xk_ = torch.view_as_complex(
        xk.float().reshape(*xk.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-2, -1).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-2, -1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaAttentionFused(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_local_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = args.max_position_embeddings
        # self.rope_theta = args.rope_theta

        kv_max_seq_len = min(max_seq_len, self.max_position_embeddings)

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # following fastertransformer definition
        self.cache_v = (
            torch.zeros(
                (
                    max_batch_size,
                    self.num_key_value_heads,
                    # args.max_position_embeddings,
                    kv_max_seq_len,
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
                    self.num_key_value_heads,
                    self.head_dim // 8,
                    # args.max_position_embeddings,
                    kv_max_seq_len,
                    8,
                )
            )
            .cuda()
            .half()
        )  # added to half

        # dummy
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=2048, device="cuda:0"
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        # xqkv = self.qkv_proj(x)
        # xqkv = xqkv.view(bsz, seqlen, -1, self.n_local_heads, self.head_dim)
        # xq = xqkv[:, :, 0]
        # xk = xqkv[:, :, 1]
        # xv = xqkv[:, :, 2]

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if seqlen > 1:
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape(bsz, seqlen, self.num_key_value_heads, self.head_dim // 8, 8)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )

            self.cache_v[:bsz, :, start_pos : start_pos + seqlen, :] = values_store
            self.cache_k[:bsz, :, :, start_pos : start_pos + seqlen, :] = keys_store

            keys = xk
            values = xv

            keys = torch.repeat_interleave(
                keys, dim=2, repeats=self.num_key_value_groups
            )
            values = torch.repeat_interleave(
                values, dim=2, repeats=self.num_key_value_groups
            )

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
            xq = xq.view(bsz, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, self.num_key_value_heads, self.head_dim)
            xv = xv.view(bsz, self.num_key_value_heads, self.head_dim)

            output = awq_inference_engine.single_query_attention(
                xq,
                xk,
                xv,
                self.cache_k,
                self.cache_v,
                None,
                # alibi position encodings
                None,
                start_pos,
                self.head_dim,
                10000,
                True,
            )
            output = output.reshape(bsz, 1, -1)

        return self.o_proj(output)


class LlamaMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.self_attn = LlamaAttentionFused(args)
        self.mlp = LlamaMLP(args)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.self_attn.forward(
            self.input_layernorm(x), start_pos, freqs_cis, mask
        )
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_hidden_layers

        self.embed_tokens = nn.Embedding(params.vocab_size, params.hidden_size)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.params.hidden_size // self.params.num_attention_heads,
            self.params.max_position_embeddings * 2,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        return h


class LlamaForCausalLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = Transformer(params)
        self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        h = self.model(tokens, start_pos)
        output = self.lm_head(h)  # only compute last logits
        return output.float()
