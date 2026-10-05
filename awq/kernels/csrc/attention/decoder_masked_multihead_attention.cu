/*
 * ROCm-LLM-AWQ — Ported from mit-han-lab/llm-awq
 *
 * AMD PORTING DECISION (AMDable KB: ptx_asm_to_hip_builtins, aiter_llm_operator_library_amd):
 *   Original: decoder_masked_multihead_attention.cu from FasterTransformer v5.2.1
 *   Uses heavy PTX: add.f16, mul.f16, fma.rn.f16x2, cvt.rn.f16.f32, mov.b32, etc.
 *   (decoder_masked_multihead_attention_utils.h has 13+ PTX asm blocks)
 *
 *   AMD STRATEGY: Scope-guard NVIDIA kernel with !__HIP_PLATFORM_AMD__.
 *   AMD path: aiter paged attention OR standard PyTorch scaled_dot_product_attention.
 *   aiter is AMD's production LLM inference library (vLLM/SGLang default on ROCm).
 *   See KB entry: aiter_llm_operator_library_amd
 *
 *   The ft_attention.cpp wrapper dispatches to single_query_attention() —
 *   on AMD this routes to the torch SDPA fallback, which uses MIOpen under the hood.
 *   For production: replace with aiter.single_query_cached_kv_attention() from
 *   the Python layer (awq/modules/fused_attn.py).
 */

#include "hip_compat.h"
#include "decoder_masked_multihead_attention.h"
#include "ft_attention.h"

#if !defined(__HIP_PLATFORM_AMD__)
// ============================================================
// ORIGINAL NVIDIA CUDA KERNEL (FasterTransformer)
// Heavy PTX: cvt.f16, add.f16x2, fma.rn.f16x2, mul.f16, mov.b32
// Full source: https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/attention/decoder_masked_multihead_attention.cu
// ============================================================
#include <cuda_fp16.h>
#include "decoder_masked_multihead_attention_template.hpp"
// [Original kernel instantiations — CUDA only]

template void masked_multihead_attention(
    const Masked_multihead_attention_params<float>& params, const cudaStream_t& stream);
template void masked_multihead_attention(
    const Masked_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
template void masked_multihead_attention(
    const Masked_multihead_attention_params<__nv_bfloat16>& params, const cudaStream_t& stream);
#endif

#else // __HIP_PLATFORM_AMD__
// ============================================================
// AMD HIP PATH - stub implementations to satisfy the linker.
// ft_attention.cpp dispatches to PyTorch SDPA on AMD (never calls these).
// ============================================================
#include <hip/hip_runtime.h>
#include "decoder_masked_multihead_attention.h"

void masked_multihead_attention(
    const Masked_multihead_attention_params<float>& params,
    const hipStream_t& stream)
{ (void)params; (void)stream; }

void masked_multihead_attention(
    const Masked_multihead_attention_params<uint16_t>& params,
    const hipStream_t& stream)
{ (void)params; (void)stream; }

#ifdef ENABLE_BF16
void masked_multihead_attention(
    const Masked_multihead_attention_params<__hip_bfloat16>& params,
    const hipStream_t& stream)
{ (void)params; (void)stream; }
#endif

void cross_multihead_attention(
    const Cross_multihead_attention_params<float>& params,
    const hipStream_t& stream)
{ (void)params; (void)stream; }

void cross_multihead_attention(
    const Cross_multihead_attention_params<uint16_t>& params,
    const hipStream_t& stream)
{ (void)params; (void)stream; }

#ifdef ENABLE_BF16
void cross_multihead_attention(
    const Cross_multihead_attention_params<__hip_bfloat16>& params,
    const hipStream_t& stream)
{ (void)params; (void)stream; }
#endif

#endif // __HIP_PLATFORM_AMD__
