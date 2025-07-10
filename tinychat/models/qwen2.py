# Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
"""PyTorch Qwen2 model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import awq_inference_engine
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.activations import ACT2FN
import tinychat
import torch.nn.functional as F
import time
from tqdm import tqdm
from transformers import GenerationMixin
from transformers.models.qwen2 import Qwen2ForCausalLM
from flash_attn import flash_attn_func

max_batch_size = tinychat.utils.constants.max_batch_size
max_seq_len = tinychat.utils.constants.max_seq_len


class Qwen2RMSNorm(nn.Module):
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


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, scale: float = 1.0
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t * scale, freqs).float()  # type: ignore

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def precompute_freqs(
    dim: int, end: int, theta: float = 10000.0, scale: float = 1.0, device=None
):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float().to(device) / dim))
    seq = torch.arange(end, dtype=inv_freq.dtype, device=device)
    freqs = torch.einsum("i , j -> i j", seq, inv_freq)
    freqs = freqs.reshape(freqs.shape[0], 1, 1, -1)
    return torch.cat((freqs, freqs), dim=-1)


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


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2AttentionFused(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.args = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        if self.rope_scaling is None:
            self.rope_scaling = 1.0
        elif isinstance(self.rope_scaling, dict):
            self.rope_scaling = self.rope_scaling.get("factor", 1.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.kv_max_seq_len = min(max_seq_len, self.max_position_embeddings)
        # following fastertransformer definition
        self.cache_v = (
            torch.zeros(
                (
                    max_batch_size,
                    self.num_key_value_heads,
                    # args.max_position_embeddings,
                    self.kv_max_seq_len,
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
                    self.kv_max_seq_len,
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
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor],
        chunk_prefilling: bool = False,
    ):
        bsz, seqlen, _ = x.shape

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        if seqlen > 1:
            xq = query_states.view(bsz, seqlen, self.num_heads, self.head_dim)
            xk = key_states.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
            xv = value_states.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs)

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
            if chunk_prefilling:
                keys = self.cache_k[:, :, :, 0 : start_pos + seqlen, :]
                keys = (
                    keys.permute(0, 3, 1, 2, 4)
                    .reshape(
                        bsz, start_pos + seqlen, self.num_key_value_heads, self.head_dim
                    )
                    .contiguous()
                )
                values = self.cache_v[:, :, 0 : start_pos + seqlen, :]
                values = (
                    values.transpose(2, 1)
                    .reshape(
                        bsz, start_pos + seqlen, self.num_key_value_heads, self.head_dim
                    )
                    .contiguous()
                )
            else:
                keys = xk
                values = xv
            output = flash_attn_func(
                q=xq,
                k=keys,
                v=values,
                causal=True,
            )
            output = output.contiguous().view(bsz, seqlen, -1)
        else:
            xq = query_states.view(bsz, self.num_heads, self.head_dim)
            xk = key_states.view(bsz, self.num_key_value_heads, self.head_dim)
            xv = value_states.view(bsz, self.num_key_value_heads, self.head_dim)
            
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
                self.rope_theta,
                self.rope_scaling,
                True,
            )
            output = output.reshape(bsz, 1, -1)

        return self.o_proj(output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2AttentionFused(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor],
        chunk_prefilling: bool = False,
    ):
        residual = x
        x = self.input_layernorm(x)

        # Self Attention
        x = self.self_attn(
            x=x,
            start_pos=start_pos,
            freqs=freqs,
            mask=mask,
            chunk_prefilling=chunk_prefilling,
        )
        x = residual + x

        # Fully Connected
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Note (Haotian): rope_theta has to be defined here, otherwise context stage is wrong.
        rope_scale = config.rope_scaling
        if rope_scale is None:
            rope_scale = 1.0
        else:
            rope_scale = 1.0 / rope_scale["factor"]
        self.freqs = precompute_freqs(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings * 2,
            config.rope_theta,
            rope_scale,
        )
        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings * 2,
            config.rope_theta,
            rope_scale,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        start_pos: Optional[int] = 0,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        chunk_prefilling: bool = False,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        seqlen = inputs_embeds.shape[1]

        self.freqs = self.freqs.to(inputs_embeds.device)
        freqs = self.freqs[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=inputs_embeds.device
            )
            mask = torch.triu(mask, diagonal=1).type_as(inputs_embeds)
            if chunk_prefilling:
                mask_history = torch.zeros(
                    (1, 1, seqlen, start_pos),
                    dtype=torch.float16,
                    device=inputs_embeds.device,
                ).type_as(inputs_embeds)
                mask = torch.cat((mask_history, mask), dim=-1)
        x = inputs_embeds

        for decoder_layer in self.layers:
            x = decoder_layer(x, start_pos, freqs, mask, chunk_prefilling)
        x = x[:, -1:, :]
        x = self.norm(x)

        return x

    def forwardfp16(
        self,
        input_ids: torch.LongTensor = None,
        start_pos: Optional[int] = 0,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        chunk_prefilling: bool = False,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        seqlen = inputs_embeds.shape[1]

        self.freqs_cis = self.freqs_cis.to(inputs_embeds.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=inputs_embeds.device
            )
            mask = torch.triu(mask, diagonal=1).type_as(inputs_embeds)
            if chunk_prefilling:
                mask_history = torch.zeros(
                    (1, 1, seqlen, start_pos),
                    dtype=torch.float16,
                    device=inputs_embeds.device,
                ).type_as(inputs_embeds)
                mask = torch.cat((mask_history, mask), dim=-1)
        x = inputs_embeds

        for decoder_layer in self.layers:
            x = decoder_layer(x, start_pos, freqs_cis, mask, chunk_prefilling)
        x = x[:, -1:, :]
        x = self.norm(x)

        return x


class Qwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.kaiming_normal_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        from transformers import modeling_utils

        modeling_utils._init_weights = False

        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        inputs_embeds: torch.Tensor = None,
        chunk_prefilling: bool = False,
        quant=True,
    ):
        if quant:
            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                start_pos=start_pos,
                chunk_prefilling=chunk_prefilling,
            )
        else:
            outputs = self.model.forwardfp16(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                start_pos=start_pos,
                chunk_prefilling=chunk_prefilling,
            )
        logits = self.lm_head(outputs)
        return logits

    def benchmark(self, inputs_embeds, attention_mask, max_output=128, quant_llm=True):
        output_list = []
        start_pos = 0
        for i in range(10):
            torch.cuda.synchronize()
            tst = time.time()
            token = self.forward(None, start_pos, inputs_embeds, quant=quant_llm)
            torch.cuda.synchronize()
            ted = time.time()
        print(
            "LLM TTFT: {:.6f} s for {} tokens".format(
                (ted - tst), inputs_embeds.shape[1]
            )
        )
        start_pos = inputs_embeds.shape[1]
        token = torch.argmax(token, keepdim=True)[0]
        output_list.append(token)

        torch.cuda.synchronize()
        tst = time.time()
        for _ in range(max_output):
            token = self.forward(token, start_pos)
            token = torch.argmax(token, keepdim=True)[
                0
            ]  # Only fixed-length eager decoding is supported now
            output_list.append(token)
            start_pos += 1
        torch.cuda.synchronize()
        ted = time.time()
        print("Decoding througput: {:.6f} tokens/s".format(max_output / (ted - tst)))

        return torch.cat(output_list, dim=1)
