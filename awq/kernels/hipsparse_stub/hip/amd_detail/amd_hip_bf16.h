/* AWQ amd_hip_bf16.h stub: intercept for GCC host compilation ONLY
 * KB: float2_device_type_host_compilation, amd_hip_bf16_header_dual_include_pattern
 */
#pragma once
#if !defined(__HIPCC__) && !defined(__HIP_DEVICE_COMPILE__)
/* Pure GCC host (not hipcc): provide minimal struct stubs */
#if !defined(_AWQ_HIP_BF16_STUB)
#define _AWQ_HIP_BF16_STUB
struct __attribute__((aligned(2))) __hip_bfloat16 { unsigned short data; };
struct __attribute__((aligned(4))) __hip_bfloat162 {
    struct __hip_bfloat16 x;
    struct __hip_bfloat16 y;
};
#endif
#else
/* hipcc (any pass): use the real ROCm header */
#include_next <hip/amd_detail/amd_hip_bf16.h>
#endif
