/*
 * ROCm-LLM-AWQ -- AMD HIP compatibility header
 *
 * KB entries applied:
 *   warp_size_32_vs_64_amd, shfl_sync_to_shfl_amd, syncwarp_noop_on_amd,
 *   hip_platform_guard_pattern, ptx_asm_to_hip_builtins,
 *   awq_hip_bfloat16_type_version_split, awq_pytorch_hipify_header_renaming
 *
 * bfloat16 FINAL resolution (2026-06-13, ROCm 7.2.4 compute-rocm-rel-7.2):
 *   ROCm 7.2.4: include BOTH hip_bfloat16.h AND amd_hip_bf16.h.
 *   Both are compatible in ROCm 7.2.4 (clang 22) -- amd_hip_bf16.h provides
 *   __hip_bfloat16 + __hip_bfloat162 structs; hip_bfloat16.h provides hip_bfloat16.
 *   DO NOT define your own __hip_bfloat162 struct -- it conflicts with the header.
 *   Just add the nv_ CUDA-compatibility aliases after the includes.
 */
#pragma once

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_bf16.h>

/* Warp size: AMD CDNA wavefront = 64 (KB: warp_size_32_vs_64_amd) */
#ifdef WARP_SIZE
#undef WARP_SIZE
#endif
#define WARP_SIZE 64
#define CUDA_WARP_SIZE 32
#define WARP_ID_FROM_TIDX(tidx) ((tidx) >> 6)
#define LANE_ID_FROM_TIDX(tidx) ((tidx) & 0x3f)

/* __shfl_sync -> __shfl (KB: shfl_sync_to_shfl_amd) */
#define __shfl_sync(mask, val, src, ...)        __shfl(val, src, ##__VA_ARGS__)
#define __shfl_down_sync(mask, val, delta, ...) __shfl_down(val, delta, ##__VA_ARGS__)
#define __shfl_up_sync(mask, val, delta, ...)   __shfl_up(val, delta, ##__VA_ARGS__)
#define __shfl_xor_sync(mask, val, lane, ...)   __shfl_xor(val, lane, ##__VA_ARGS__)

/* __syncwarp -> no-op (KB: syncwarp_noop_on_amd) */
#define __syncwarp(...)

/* L2_CACHEHINT PTX -> no-op (KB: ptx_asm_to_hip_builtins) */
#ifdef L2_CACHEHINT
#undef L2_CACHEHINT
#endif
#define L2_CACHEHINT(size)

#define AMD_BFLOAT16_AVAILABLE 1

/*
 * CUDA nv_ bfloat16 aliases (KB: awq_hip_bfloat16_type_version_split)
 * After including both headers, __hip_bfloat16 and __hip_bfloat162 are defined.
 * Only add the CUDA nv_ compatibility aliases here.
 */
#ifndef nv_bfloat16
  typedef __hip_bfloat16  nv_bfloat16;
  /* hipify-opaque alias for __hip_bfloat16 -- use in .cu to avoid hipify token mangling */
  typedef __hip_bfloat16  awq_bf16_t;
#endif
#ifndef __nv_bfloat16
  typedef __hip_bfloat16  __nv_bfloat16;
#endif
#ifndef __nv_bfloat162
  typedef __hip_bfloat162 __nv_bfloat162;
#endif
#ifndef nv_bfloat162
  typedef __hip_bfloat162 nv_bfloat162;
#endif

/* __ldg: __half and half2 already defined in amd_hip_fp16.h (ROCm 7.2+) */
__device__ inline __hip_bfloat16  __ldg(const __hip_bfloat16* ptr)  { return *ptr; }
__device__ inline __hip_bfloat162 __ldg(const __hip_bfloat162* ptr) { return *ptr; }

/* PyTorch ROCm maps c10/cuda, AT_CUDA_CHECK, getCurrentCUDAStream (KB: torch_cuda_to_hip_device) */

#else  /* NVIDIA CUDA */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define CUDA_WARP_SIZE 32
#define WARP_ID_FROM_TIDX(tidx) ((tidx) >> 5)
#define LANE_ID_FROM_TIDX(tidx) ((tidx) & 0x1f)

#endif  /* __HIP_PLATFORM_AMD__ */
