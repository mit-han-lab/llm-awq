# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
import awq_inference_engine

import tinychat.utils.constants

max_batch_size = tinychat.utils.constants.max_batch_size
max_seq_len = tinychat.utils.constants.max_seq_len


# rotary pos emb helpers (torch.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


class RotaryEmbedding(nn.Module):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is design to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """

    def __init__(
        self,
        head_dim: int,
        base=10000,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def cos_sin(
        self,
        seq_len: int,
        device="cuda",
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, _q, _k):
        batch, seq_len, num_heads, head_dim = _q.shape
        q = _q.permute(0, 2, 1, 3).contiguous().reshape(-1, seq_len, head_dim)
        k = _k.permute(0, 2, 1, 3).contiguous().reshape(-1, seq_len, head_dim)
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class FalconAttentionFused(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_head
        self.head_dim = args.hidden_size // args.n_head

        self.query_key_value = nn.Linear(
            args.hidden_size,
            args.n_head * self.head_dim + 2 * self.head_dim,
            bias=False,
        )

        self.dense = nn.Linear(
            args.n_head * self.head_dim,
            args.hidden_size,
            bias=False,
        )

        # following fastertransformer definition

        self.cache_v = (
            torch.zeros(
                (
                    max_batch_size,
                    1,
                    max_seq_len,
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
                    1,
                    self.head_dim // 8,
                    max_seq_len,
                    8,
                )
            )
            .cuda()
            .half()
        )  # added to half

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape

        xqkv = self.query_key_value(x)
        xqkv = xqkv.view(bsz, seqlen, self.n_local_heads + 2, self.head_dim)
        xq = xqkv[:, :, :-2]
        xk = xqkv[:, :, [-2]]
        xv = xqkv[:, :, [-1]]

        if seqlen > 1:
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, 1, self.head_dim)
            xv = xv.view(bsz, seqlen, 1, self.head_dim)

            xq, xk = self.rotary_emb(xq, xk)
            xq = (
                xq.reshape(bsz, self.n_local_heads, seqlen, self.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            xk = (
                xk.reshape(bsz, 1, seqlen, self.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape(bsz, seqlen, 1, self.head_dim // 8, 8)
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
            xk = xk.view(bsz, 1, self.head_dim)
            xv = xv.view(bsz, 1, self.head_dim)

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

        return self.dense(output)


class FalconMLP(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(dim, 4 * dim, bias=False)
        self.act = nn.GELU()
        self.dense_4h_to_h = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.n_head
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.n_head
        self.self_attention = FalconAttentionFused(args)
        self.mlp = FalconMLP(dim=args.hidden_size)
        self.layer_id = layer_id
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_epsilon
        )
        # self.post_attention_layernorm = nn.LayerNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        layernorm_output = self.input_layernorm(x)
        h_attn = x + self.self_attention.forward(layernorm_output, start_pos, mask)
        h_mlp = self.mlp(layernorm_output)
        out = h_attn + h_mlp
        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layer

        self.word_embeddings = nn.Embedding(params.vocab_size, params.hidden_size)

        self.h = torch.nn.ModuleList()
        for layer_id in range(params.n_layer):
            self.h.append(TransformerBlock(layer_id, params))

        self.ln_f = nn.LayerNorm(params.hidden_size, eps=params.layer_norm_epsilon)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.word_embeddings(tokens)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.h:
            h = layer(h, start_pos, mask)
        h = self.ln_f(h)
        return h


class FalconForCausalLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.config = params
        self.transformer = Transformer(params)
        self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        h = self.transformer(tokens, start_pos)
        output = self.lm_head(h)  # only compute last logits
        return output.float()
