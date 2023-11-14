"""
@article{li2023starcoder,
  title={StarCoder: may the source be with you!},
  author={Li, Raymond and Allal, Loubna Ben and Zi, Yangtian and Muennighoff, Niklas and Kocetkov, Denis and Mou, Chenghao and Marone, Marc and Akiki, Christopher and Li, Jia and Chim, Jenny and others},
  journal={arXiv preprint arXiv:2305.06161},
  year={2023}
}
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import functools
import torch
from torch import nn
import torch.nn.functional as F
import awq_inference_engine
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# from flash_attn.flash_attn_interface import flash_attn_unpadded_func
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
except ImportError:
    print ("FlashAttention not found. Install it if you need to train models.")

import tinychat.utils.constants

max_batch_size = tinychat.utils.constants.max_batch_size
multiple_of = tinychat.utils.constants.llama_multiple_of
max_seq_len = tinychat.utils.constants.max_seq_len

@functools.cache
def get_cuseqlen(bsz, seqlen, device="cuda:0"):
    return torch.Tensor([seqlen * i for i in range(bsz + 1)]).int().to(device)

class GPTBigCodeAttentionFused(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        self.num_key_value_groups = self.num_heads // self.kv_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.c_attn = nn.Linear(
            self.embed_dim,
            self.embed_dim + 2 * self.kv_heads * self.head_dim,
            bias=True,
        )
        self.c_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=True,
        )

        # following fastertransformer definition
        self.cache_v = (
            torch.zeros(
                (
                    max_batch_size,
                    self.kv_heads,
                    # args.max_position_embeddings,
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
                    self.kv_heads,
                    self.head_dim // 8,
                    # args.max_position_embeddings,
                    max_seq_len,
                    8,
                )
            )
            .cuda()
            .half()
        )  # added to half


    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        # xqkv = self.qkv_proj(x)
        # xqkv = xqkv.view(bsz, seqlen, -1, self.num_heads, self.head_dim)
        # xq = xqkv[:, :, 0]
        # xk = xqkv[:, :, 1]
        # xv = xqkv[:, :, 2]

        xqkv = self.c_attn(x)
        xqkv = xqkv.view(bsz, seqlen, self.num_heads + 2 * self.kv_heads, self.head_dim)
        xq = xqkv[:, :, : -2 * self.kv_heads]
        xk = xqkv[:, :, -2 * self.kv_heads : -self.kv_heads]
        xv = xqkv[:, :, -self.kv_heads:]

        if seqlen > 1:
            xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.kv_heads, self.head_dim)


            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape(bsz, seqlen, self.kv_heads, self.head_dim // 8, 8)
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
            """
            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            """
            cu_seqlens = get_cuseqlen(bsz, seqlen, keys.device)
            output = flash_attn_varlen_func(
                q=xq.view(-1, self.num_heads, self.head_dim), k=keys.view(-1, self.num_heads, self.head_dim), v=values.view(-1, self.num_heads, self.head_dim),
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seqlen, max_seqlen_k=seqlen,
                dropout_p=0.0, causal=True)
            output = output.reshape(bsz, seqlen, -1)
            
        else:
            xq = xq.view(bsz, self.num_heads, self.head_dim)
            xk = xk.view(bsz, self.kv_heads, self.head_dim)
            xv = xv.view(bsz, self.kv_heads, self.head_dim)

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
                # No RoPE
                0, #self.head_dim,
                10000,
                True,
            )
            output = output.reshape(bsz, 1, -1)

        return self.c_proj(output)


class GPTBigCodeMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTBigCodeAttentionFused(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPTBigCodeMLP(self.inner_dim, config)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attn.forward(
            self.ln_1(x), start_pos, mask
        )
        out = h + self.mlp.forward(self.ln_2(h))
        return out


class GPTBigCodeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.h = nn.ModuleList([GPTBigCodeBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias", torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)), persistent=False
        )


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.wte(tokens)
        pe = self.wpe.weight[start_pos : start_pos + seqlen].unsqueeze(0)
        h = h + pe

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


class GPTBigCodeForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPTBigCodeModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        h = self.transformer(tokens, start_pos)
        output = self.lm_head(h)  # only compute last logits
        return output.float()
