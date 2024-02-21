"""
@misc{https://doi.org/10.48550/arxiv.2204.06745,
  doi = {10.48550/ARXIV.2204.06745},
  url = {https://arxiv.org/abs/2204.06745},
  author = {Black, Sid and Biderman, Stella and Hallahan, Eric and Anthony, Quentin and Gao, Leo and Golding, Laurence and He, Horace and Leahy, Connor and McDonell, Kyle and Phang, Jason and Pieler, Michael and Prashanth, USVSN Sai and Purohit, Shivanshu and Reynolds, Laria and Tow, Jonathan and Wang, Ben and Weinbach, Samuel},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {GPT-NeoX-20B: An Open-Source Autoregressive Language Model},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
import awq_inference_engine
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding

# from flash_attn.flash_attn_interface import flash_attn_unpadded_func

import tinychat.utils.constants

max_batch_size = tinychat.utils.constants.max_batch_size
multiple_of = tinychat.utils.constants.llama_multiple_of
max_seq_len = tinychat.utils.constants.max_seq_len


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


class GPTNeoXAttentionFused(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_local_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.rotary_ndim = int(args.rotary_pct * self.head_dim)

        self.max_position_embeddings = args.max_position_embeddings
        # self.rope_theta = args.rope_theta

        kv_max_seq_len = min(max_seq_len, self.max_position_embeddings)

        self.query_key_value = nn.Linear(args.hidden_size, 3 * args.hidden_size)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)

        # following fastertransformer definition
        self.cache_v = (
            torch.zeros(
                (
                    max_batch_size,
                    self.num_heads,
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
                    self.num_heads,
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
        self.rotary_emb = GPTNeoXRotaryEmbedding(
            self.rotary_ndim,
            max_position_embeddings=args.max_position_embeddings,
            device="cuda:0",
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

        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(x)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.n_local_heads, 3, self.head_dim)
        qkv = qkv.view(*new_qkv_shape)
        xq = qkv[:, :, :, 0].contiguous()
        xk = qkv[:, :, :, 1].contiguous()
        xv = qkv[:, :, :, 2].contiguous()

        if seqlen > 1:
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xq_rot = xq[..., : self.rotary_ndim].contiguous()
            xk_rot = xk[..., : self.rotary_ndim].contiguous()
            xq_pass = xq[..., self.rotary_ndim :].contiguous()
            xk_pass = xk[..., self.rotary_ndim :].contiguous()
            xq_rot, xk_rot = apply_rotary_emb(xq_rot, xk_rot, freqs_cis=freqs_cis)
            xq = torch.cat([xq_rot, xq_pass], -1).contiguous()
            xk = torch.cat([xk_rot, xk_pass], -1).contiguous()

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
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
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
                # alibi position encodings
                None,
                start_pos,
                self.rotary_ndim,
                10000,
                True,
            )
            output = output.reshape(bsz, 1, -1)

        return self.dense(output)


class GPTNeoXMLP(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.dense_4h_to_h = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.attention = GPTNeoXAttentionFused(args)
        self.mlp = GPTNeoXMLP(dim=args.hidden_size)
        self.layer_id = layer_id
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h_attn = self.attention.forward(
            self.input_layernorm(x), start_pos, freqs_cis, mask
        )
        h_mlp = self.mlp(self.post_attention_layernorm(x))
        out = x + h_attn + h_mlp
        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_hidden_layers

        self.embed_in = nn.Embedding(params.vocab_size, params.hidden_size)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.final_layer_norm = nn.LayerNorm(
            params.hidden_size, eps=params.layer_norm_eps
        )

        self.freqs_cis = precompute_freqs_cis(
            int(
                self.params.hidden_size
                // self.params.num_attention_heads
                * self.params.rotary_pct
            ),
            self.params.max_position_embeddings * 2,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.embed_in(tokens)
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
        h = self.final_layer_norm(h)
        return h


class GPTNeoXForCausalLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.config = params
        self.gpt_neox = Transformer(params)
        self.embed_out = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        h = self.gpt_neox(tokens, start_pos)
        output = self.embed_out(h)  # only compute last logits
        return output.float()
