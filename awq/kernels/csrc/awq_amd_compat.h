/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * HIP build compatibility shim for the AWQ inference engine.
 *
 * \author Jeff Daily <jeff.daily@amd.com>
 *
 * Force-included on the ROCm build (see setup.py). It supplies the CUDA
 * spellings the kernels use that torch's hipify does not translate:
 *   - the unsuffixed nv_bfloat16 / nv_bfloat162 typedefs (hipify maps only the
 *     __nv_-prefixed forms; cuda_bf16.h provides the unsuffixed aliases on
 *     NVIDIA, but no such alias exists after hipify),
 *   - implicit half/bfloat16 <-> float conversions and operators, which HIP
 *     gates behind __HIP_NO_HALF_*; CUDA enables them via -U__CUDA_NO_HALF_*.
 */
#pragma once

#if defined(__HIP_PLATFORM_AMD__)

// Enable implicit half/bf16 operators and conversions (CUDA's -U__CUDA_NO_*).
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_BFLOAT16_OPERATORS__
#undef __HIP_NO_BFLOAT16_CONVERSIONS__
#undef __HIP_NO_BFLOAT162_OPERATORS__
#undef __HIP_NO_BFLOAT162_CONVERSIONS__

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// hipify renames the __nv_-prefixed bf16 types but leaves the unsuffixed
// CUDA aliases (nv_bfloat16, nv_bfloat162) untranslated.
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;

#include <hip/hip_runtime.h>

// HIP's __ldg lacks overloads for the half/bf16 (vector) types these kernels
// read through it. Provide a plain-load fallback; the read-only cache hint is
// advisory and HIP often lowers __ldg to the same load anyway.
template <typename T>
__device__ __forceinline__ T awq_ldg(const T *p) { return *p; }
#define __ldg(ptr) awq_ldg(ptr)

#endif // __HIP_PLATFORM_AMD__
