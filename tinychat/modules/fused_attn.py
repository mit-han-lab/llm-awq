import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from typing import Optional
from awq.quantize.qmodule import WQLinear
import awq_inference_engine
from tinychat.models.llama import apply_rotary_emb
import gc

import tinychat.utils.constants

max_batch_size = tinychat.utils.constants.max_batch_size
max_seq_len = tinychat.utils.constants.max_seq_len


class QuantLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)

        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        # self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("cos_sin_cache", cache.half(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ):
        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        # print(positions.shape, query.shape, key.shape, self.cos_sin_cache.shape)
        query = query.contiguous()
        key = key.contiguous()
        awq_inference_engine.rotary_embedding_neox(
            positions,
            query,
            key,
            self.dim,
            self.cos_sin_cache,
        )
        return query, key


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_heads, qkv_proj, o_proj, dev):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.rotary_emb = QuantLlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=2048, device=dev
        )

    def forward(
        self,
        hidden_states,
        past_key_value=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        use_cache=False,
    ):
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.view(bsz, q_len, 3, self.num_heads, self.head_dim)

        # This updates the query and key states in-place, saving VRAM.
        query_states, key_states, value_states = torch.split(qkv_states, 1, dim=2)
        query_states, key_states = self.rotary_emb(
            query_states, key_states, position_ids
        )

        del qkv_states
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        is_causal = past_key_value is None

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        value_states = value_states.to("cuda:0")

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            # Since qkv_proj is fused, query_states etc will hold a reference to the original qkv_states tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            query_states = query_states.contiguous()

        past_key_value = (key_states, value_states) if use_cache else None

        # with torch.backends.cuda.sdp_kernel(enable_math=False):
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=is_causal
        )
        del query_states, key_states, value_states

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class QuantLlamaAttentionFused(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_layer, o_proj, dev, args):
        super().__init__()

        self.args = args
        self.n_local_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.rope_theta = args.rope_theta
        self.rope_scaling = args.rope_scaling
        if self.rope_scaling is None:
            self.rope_scaling = 1.0

        self.qkv_proj = qkv_layer
        self.o_proj = o_proj

        kv_max_seq_len = min(max_seq_len, args.max_position_embeddings)

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
            .to(dev)
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
            .to(dev)
            .half()
        )  # added to half

        # dummy
        self.rotary_emb = QuantLlamaRotaryEmbedding(
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
        xqkv = self.qkv_proj(x)
        xqkv = xqkv.view(
            bsz,
            seqlen,
            self.n_local_heads + self.num_key_value_heads * 2,
            self.head_dim,
        )
        xq = xqkv[:, :, 0 : self.n_local_heads]
        xk = xqkv[
            :, :, self.n_local_heads : (self.n_local_heads + self.num_key_value_heads)
        ]
        xv = xqkv[:, :, -self.num_key_value_heads :]

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
                None,
                start_pos,
                self.head_dim,
                self.rope_theta,
                self.rope_scaling,
                True,
            )
            output = output.reshape(bsz, 1, -1)

        return self.o_proj(output)


def make_quant_attn(model, dev):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """
    model = model.cpu()
    for name, m in model.named_modules():
        if not m.__class__.__name__ in ["LlamaAttention", "LlamaAttentionFused"]:
            continue

        q_proj = m.q_proj
        k_proj = m.k_proj
        v_proj = m.v_proj

        qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=0)
        scaled_zeros = torch.cat(
            [q_proj.scaled_zeros, k_proj.scaled_zeros, v_proj.scaled_zeros], dim=1
        ).contiguous()
        scales = torch.cat(
            [q_proj.scales, k_proj.scales, v_proj.scales], dim=1
        ).contiguous()
        # g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)
        g_idx = None
        bias = (
            torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
            if q_proj.bias is not None
            else None
        )

        qkv_layer = WQLinear(
            q_proj.w_bit,
            q_proj.group_size,
            q_proj.in_features,
            q_proj.out_features + k_proj.out_features + v_proj.out_features,
            q_proj.bias is not None,
            q_proj.qweight.device,
        )
        qkv_layer.qweight = qweights
        qkv_layer.scaled_zeros = scaled_zeros
        qkv_layer.scales = scales

        qkv_layer.bias = bias
        qkv_layer.split_k_iters = q_proj.split_k_iters
        # We're dropping the rotary embedding layer m.rotary_emb here. We don't need it in the triton branch.
        if isinstance(m, LlamaAttention):
            attn = QuantLlamaAttention(
                m.hidden_size, m.num_heads, qkv_layer, m.o_proj, dev
            )
        else:
            attn = QuantLlamaAttentionFused(
                m.args.hidden_size,
                m.args.num_attention_heads,
                qkv_layer,
                m.o_proj,
                dev,
                m.args,
            )
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            child_name = name[len(parent_name) + 1 :]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ""
            parent = model
            child_name = name

        # print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")
        setattr(parent, child_name, attn)
        gc.collect()
        torch.cuda.empty_cache()
    model = model.to(dev)
