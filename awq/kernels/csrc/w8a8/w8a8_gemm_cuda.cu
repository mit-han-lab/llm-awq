/*
 * ROCm-LLM-AWQ — Ported from mit-han-lab/llm-awq
 *
 * AMD PORTING DECISION (AMDable KB: ptx_asm_to_hip_builtins, scope_guard):
 *   AMD gfx942 equivalent: hipBLASLt with HIPBLAS_R_8I / HIPBLAS_COMPUTE_32I.
 *   AMD STRATEGY: Scope-guard NVIDIA kernel; AMD path uses hipBLASLt INT8 GEMM.
 *
 *   Signature fix (v10): match header void(Tensor x5).
 */

#include "hip_compat.h"
#include "w8a8_gemm_cuda.h"
#include <torch/extension.h>

#if !defined(__HIP_PLATFORM_AMD__)
// ============================================================
// ORIGINAL NVIDIA CUDA KERNEL (Ampere only — mma.sync s8 PTX)
// ============================================================
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
// [Original kernel body omitted — CUDA only]

#else // __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

void w8a8_gemm_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _wscales,
    torch::Tensor _ascales,
    torch::Tensor _out_feats)
{
    TORCH_CHECK(_in_feats.scalar_type() == at::ScalarType::Char,
        "w8a8_gemm (AMD): _in_feats must be INT8");
    TORCH_CHECK(_kernel.scalar_type() == at::ScalarType::Char,
        "w8a8_gemm (AMD): _kernel must be INT8");
    TORCH_CHECK(_in_feats.device().type() == at::kCUDA,
        "w8a8_gemm (AMD): tensors must be on GPU");
    TORCH_CHECK(_in_feats.dim() == 2, "w8a8_gemm (AMD): _in_feats must be 2D");
    TORCH_CHECK(_kernel.dim() == 2, "w8a8_gemm (AMD): _kernel must be 2D");

    int m = _in_feats.size(0);
    int k = _in_feats.size(1);
    int n = _kernel.size(0);

    auto options_i32 = torch::TensorOptions()
        .dtype(torch::kInt32).device(_in_feats.device());
    auto output_i32 = torch::zeros({m, n}, options_i32);

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    const int8_t* A = _in_feats.data_ptr<int8_t>();
    const int8_t* B = _kernel.data_ptr<int8_t>();
    int32_t* C = output_i32.data_ptr<int32_t>();

    int32_t alpha_i32 = 1, beta_i32 = 0;
    hipblasGemmEx(handle,
        HIPBLAS_OP_T, HIPBLAS_OP_N,
        n, m, k,
        &alpha_i32,
        B, HIPBLAS_R_8I, k,
        A, HIPBLAS_R_8I, k,
        &beta_i32,
        C, HIPBLAS_R_32I, n,
        HIPBLAS_COMPUTE_32I,
        HIPBLAS_GEMM_DEFAULT);

    hipblasDestroy(handle);

    auto output_fp = output_i32.to(torch::kFloat32);
    auto sr = _ascales.to(torch::kFloat32).view({m, 1});
    auto sc = _wscales.to(torch::kFloat32).view({1, n});
    auto result_fp16 = (output_fp * sr * sc).to(torch::kFloat16).contiguous();
    _out_feats.copy_(result_fp16);
}

void w8a8_gemm_fuse_bias_forward_cuda(
    torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _wscales, torch::Tensor _ascales,
    torch::Tensor _out_feats, torch::Tensor _bias)
{
    w8a8_gemm_forward_cuda(_in_feats, _kernel, _wscales, _ascales, _out_feats);
    _out_feats.add_(_bias.to(torch::kFloat16).view({1, -1}));
}

#endif // __HIP_PLATFORM_AMD__
