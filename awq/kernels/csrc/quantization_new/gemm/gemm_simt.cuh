/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Portable HIP path for the W4A16 GEMM (gemm_forward_cuda_new).
 *
 * \author Jeff Daily <jeff.daily@amd.com>
 *
 * The upstream CUDA GEMM is built on NVIDIA tensor-core PTX
 * (ldmatrix.sync + mma.sync.m16n8k16 + cp.async), none of which exists on
 * AMD clang/HIP. Rather than reverse-engineer the MMA fragment layout, this
 * file provides a correctness-first SIMT kernel that reuses the SAME packed
 * int4 weight layout and dequantization (dequantize_s4_to_fp16x2) that the
 * GEMV path decodes, generalized over an arbitrary number of rows M. The
 * per-output-block weight unpacking is identical to gemv_cuda.cu's gemv_kernel
 * (the authoritative SIMT decoder of this format); only the activation tiling
 * differs (a runtime M loop instead of a compile-time Batch).
 *
 * License of the derived layout/dequant: Apache-2.0 (TRT-LLM), see the GEMV
 * and dequantize headers.
 */
#pragma once

#include <cuda_fp16.h>
#include "../dequantize.cuh"

// Mirrors the GEMV decoder constants (quantization_new/gemv/gemv_cuda.cu).
#ifndef AWQ_HIP_PACK_FACTOR
#define AWQ_HIP_PACK_FACTOR 8
#endif

// One thread block produces NPerBlock*kInterleave output columns for ROWS rows.
// Block layout and weight indexing match gemv_kernel exactly; the only change is
// that we loop over ROWS activation rows (the GEMV's Batch) at runtime.
template <int NPerBlock, int BlockSize, int GroupSize, typename T>
__global__ void gemm_w4a16_hip(
    const T *__restrict__ inputs, const uint32_t *__restrict__ weight,
    const T *__restrict__ scales, const T *__restrict__ zeros,
    T *__restrict__ outputs, const int IC, const int OC, const int M)
{
    constexpr int kInterleave = 4;
    constexpr int MEM_ACCESS_SIZE = 128;
    constexpr int kStride = 64;
    constexpr int kElemsPerThread = MEM_ACCESS_SIZE / 4; // 32
    constexpr int kThreadsNumPerTile = kStride / kElemsPerThread; // 2

    using T2 = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;

    constexpr int kShuffleBasicTile = 2;
    constexpr int kShuffleContinous = 4;
    constexpr int kShuffleStrided = 4;

    // psum: NPerBlock outputs per row, one row at a time.
    const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
    const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
    const int act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride
                             + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    const int group_offset = act_k_offset / GroupSize;

    const uint32_t *blk_weight_ptr = weight + blk_row_offset * IC / AWQ_HIP_PACK_FACTOR;
    const T *scale_base = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    const T *zeros_base = zeros + blk_row_offset + thd_row_offset + group_offset * OC;

    const int act_forward_step = BlockSize * kElemsPerThread / kInterleave;
    const int scale_forward_step = act_forward_step / GroupSize * OC;

    // Shared accumulator: one logical-32 subgroup contributes a partial sum per
    // (row, oc). out_smem[subgroup][row * NPerBlock*kInterleave + oc].
    constexpr int kWarpSize = 32;
    __shared__ float out_smem[BlockSize / kWarpSize][NPerBlock * kInterleave];

    // Process the M rows in batches that fit the register budget.
    constexpr int kBatch = 4;
    for (int m0 = 0; m0 < M; m0 += kBatch)
    {
        const int rows = (M - m0 < kBatch) ? (M - m0) : kBatch;

        T psum[kBatch * NPerBlock];
        #pragma unroll
        for (int i = 0; i < kBatch * NPerBlock; ++i)
            psum[i] = static_cast<T>(0.f);

        const uint32_t *w_ptr = blk_weight_ptr;
        const T *scale_ptr = scale_base;
        const T *zeros_ptr = zeros_base;
        const T *inputs_ptr = inputs + (m0 * IC) + act_k_offset;

        for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
        {
            T dequantized_weight[kElemsPerThread * NPerBlock];
            #pragma unroll
            for (int idx = 0; idx < NPerBlock; ++idx)
            {
                uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
                T half_weight_buffer[kElemsPerThread];
                *((float4 *)(local_qweights)) =
                    *((float4 *)(w_ptr + (idx * kInterleave * IC + kk) / AWQ_HIP_PACK_FACTOR));
                T local_scale = *(scale_ptr + idx * kInterleave);
                T local_scaled_zero = *(zeros_ptr + idx * kInterleave);

                #pragma unroll
                for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
                {
                    dequantize_s4_to_fp16x2<T>(*reinterpret_cast<half2 *>(local_qweights + i),
                                               reinterpret_cast<uint4 *>(half_weight_buffer + i * AWQ_HIP_PACK_FACTOR));
                }

                #pragma unroll
                for (int i = 0; i < kShuffleContinous; ++i)
                {
                    #pragma unroll
                    for (int j = 0; j < kShuffleStrided; ++j)
                    {
                        T2 w = *reinterpret_cast<T2 *>(
                            half_weight_buffer + (i + j * kShuffleContinous) * kShuffleBasicTile);
                        if constexpr (std::is_same<T, half>::value)
                            w = __hfma2(w, __half2half2(local_scale), __half2half2(local_scaled_zero));
                        else
                            w = __hfma2(w, __bfloat162bfloat162(local_scale), __bfloat162bfloat162(local_scaled_zero));
                        dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0) * NPerBlock + idx] = w.x;
                        dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1) * NPerBlock + idx] = w.y;
                    }
                }
            }

            for (int b = 0; b < rows; ++b)
            {
                T local_inputs[kElemsPerThread];
                const T *local_inputs_ptr = inputs_ptr + b * IC;
                #pragma unroll
                for (int idx = 0; idx < kElemsPerThread / 8; ++idx)
                    *((float4 *)(local_inputs + idx * 8)) = *((float4 *)(local_inputs_ptr + idx * 8));

                #pragma unroll
                for (int x = 0; x < NPerBlock / 2; ++x)
                {
                    #pragma unroll
                    for (int y = 0; y < kElemsPerThread; ++y)
                    {
                        if constexpr (std::is_same<T, half>::value)
                            *reinterpret_cast<half2 *>(psum + b * NPerBlock + x * 2) =
                                __hfma2(*reinterpret_cast<half2 *>(dequantized_weight + y * NPerBlock + x * 2),
                                        __half2half2(local_inputs[y]),
                                        *reinterpret_cast<half2 *>(psum + b * NPerBlock + x * 2));
                        else
                            *reinterpret_cast<nv_bfloat162 *>(psum + b * NPerBlock + x * 2) =
                                __hfma2(*reinterpret_cast<nv_bfloat162 *>(dequantized_weight + y * NPerBlock + x * 2),
                                        __bfloat162bfloat162(local_inputs[y]),
                                        *reinterpret_cast<nv_bfloat162 *>(psum + b * NPerBlock + x * 2));
                    }
                }
            }
            inputs_ptr += act_forward_step;
            scale_ptr += scale_forward_step;
            zeros_ptr += scale_forward_step;
        }

        // Reduce each row's NPerBlock partials across the logical-32 subgroup,
        // matching gemv_kernel's warp_reduce (offsets 16,8,1 sum the kInterleave
        // partners), then accumulate the subgroups in shared memory.
        const int subgroup = threadIdx.x / kWarpSize;
        const int lane = threadIdx.x % kWarpSize;
        for (int b = 0; b < rows; ++b)
        {
            float fpsum[NPerBlock];
            #pragma unroll
            for (int i = 0; i < NPerBlock; ++i)
                fpsum[i] = static_cast<float>(psum[b * NPerBlock + i]);
            #pragma unroll
            for (int i = 0; i < NPerBlock; ++i)
            {
                fpsum[i] += __shfl_xor(fpsum[i], 16, kWarpSize);
                fpsum[i] += __shfl_xor(fpsum[i], 8, kWarpSize);
                fpsum[i] += __shfl_xor(fpsum[i], 1, kWarpSize);
            }
            __syncthreads();
            if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
            {
                #pragma unroll
                for (int i = 0; i < NPerBlock; ++i)
                    out_smem[subgroup][i * kInterleave + lane / 2] = fpsum[i];
            }
            __syncthreads();

            for (int i = threadIdx.x; i < NPerBlock * kInterleave; i += BlockSize)
            {
                float acc = 0.f;
                #pragma unroll
                for (int j = 0; j < BlockSize / kWarpSize; ++j)
                    acc += out_smem[j][i];
                outputs[(m0 + b) * OC + blk_row_offset + i] = static_cast<T>(acc);
            }
            __syncthreads();
        }
    }
}
