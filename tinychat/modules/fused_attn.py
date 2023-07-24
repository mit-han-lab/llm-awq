import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, apply_rotary_pos_emb 

from awq.quantize.qmodule import WQLinear
import awq_inference_engine


class QuantLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
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

    def __init__(
        self,
        hidden_size,
        num_heads,
        qkv_proj,
        o_proj,
        dev
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {num_heads}).")
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.rotary_emb = QuantLlamaRotaryEmbedding(self.head_dim, max_position_embeddings=2048, device = dev)

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False):
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.view(bsz, q_len, 3, self.num_heads, self.head_dim)

        # This updates the query and key states in-place, saving VRAM.
        query_states, key_states, value_states = torch.split(qkv_states, 1, dim=2)
        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        del qkv_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

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
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=is_causal)
        del query_states, key_states, value_states

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


def make_quant_attn(model, dev):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """

    for name, m in model.named_modules():
        if not isinstance(m, LlamaAttention):
            continue

        q_proj = m.q_proj
        k_proj = m.k_proj
        v_proj = m.v_proj

        qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
        qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
        scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
        # g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)
        g_idx = None
        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

        qkv_layer = WQLinear(q_proj.w_bit, q_proj.group_size, q_proj.in_features, q_proj.out_features + k_proj.out_features + v_proj.out_features, q_proj.bias is not None, q_proj.qweight.device)
        qkv_layer.qweight = qweights
        qkv_layer.qzeros = qzeros
        qkv_layer.scales = scales

        qkv_layer.bias = bias
        # We're dropping the rotary embedding layer m.rotary_emb here. We don't need it in the triton branch.

        attn = QuantLlamaAttention(m.hidden_size, m.num_heads, qkv_layer, m.o_proj, dev)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        #print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, attn)
