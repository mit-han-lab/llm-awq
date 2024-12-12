// Modified from https://github.com/NVIDIA/TransformerEngine
// Modified by Shang Yang.

/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "fused_rope_with_pos.h"
// #include <transformer_engine/fused_rope.h>

// #include "../common.h"
// #include "../util/logging.h"
// #include "../utils.cuh"
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define THREADS_PER_WARP 32

template <typename scalar_t>
__device__ void fused_rope_with_pos_block_forward(
    const scalar_t *src, const float *freqs, scalar_t *dst,
    const int offset_block, const int offset_block_dst, const int h,
    const int d, const int d2, const int stride_h, const int stride_d,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x;
  int s = gridDim.x;
  int b_id = blockIdx.y;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos, v_sin;
    sincosf(freqs[(b_id * s + s_id) * d2 + d_id], &v_sin, &v_cos);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate =
          (d_id + d2 / 2 < d2)
              ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
              : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

// template <typename scalar_t>
// __device__ void fused_rope_block_backward(const scalar_t *src, const float
// *freqs, scalar_t *dst,
//                                           const int offset_block, const int
//                                           offset_block_dst, const int h,
//                                           const int d, const int d2, const
//                                           int stride_h, const int stride_d,
//                                           const int o_stride_h, const int
//                                           o_stride_d) {
//   int s_id = blockIdx.x;
// #pragma unroll
//   for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
//     float v_cos = cosf(freqs[s_id * d2 + d_id]);
//     float v_sin = (d_id + d2 / 2 < d2) ? sinf(freqs[s_id * d2 + d_id + d2 /
//     2])
//                                        : -sinf(freqs[s_id * d2 + d_id + d2 /
//                                        2 - d2]);
// #pragma unroll
//     for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
//       int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
//       int offset_dst = offset_block_dst + h_id * o_stride_h + d_id *
//       o_stride_d; float v_src = src[offset_src]; float v_src_rotate = (d_id +
//       d2 / 2 < d2) ? src[offset_src + (d2 / 2) * stride_d]
//                                                 : src[offset_src + (d2 / 2 -
//                                                 d2) * stride_d];
//       dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
//     }
//   }

//   // handle the tail
//   if (d > d2) {
// #pragma unroll
//     for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
//       int offset_head = offset_block + h_id * stride_h;
//       int offset_head_dst = offset_block_dst + h_id * o_stride_h;
// #pragma unroll
//       for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
//         dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id *
//         stride_d];
//       }
//     }
//   }
// }

template <typename scalar_t>
__global__ void fused_rope_with_pos_forward_kernel(
    const scalar_t *src, const float *freqs, scalar_t *dst, const int h,
    const int d, const int d2, const int stride_s, const int stride_b,
    const int stride_h, const int stride_d, const int o_stride_s,
    const int o_stride_b, const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_with_pos_block_forward<scalar_t>(
      src, freqs, dst, offset_block, offset_block_dst, h, d, d2, stride_h,
      stride_d, o_stride_h, o_stride_d);
}

// template <typename scalar_t>
// __global__ void fused_rope_backward_kernel(const scalar_t *src, const float
// *freqs, scalar_t *dst,
//                                            const int h, const int d, const
//                                            int d2, const int stride_s, const
//                                            int stride_b, const int stride_h,
//                                            const int stride_d, const int
//                                            o_stride_s, const int o_stride_b,
//                                            const int o_stride_h, const int
//                                            o_stride_d) {
//   int s_id = blockIdx.x, b_id = blockIdx.y;
//   int offset_block = s_id * stride_s + b_id * stride_b;
//   int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
//   fused_rope_block_backward<scalar_t>(src, freqs, dst, offset_block,
//   offset_block_dst, h, d, d2, stride_h,
//                             stride_d, o_stride_h, o_stride_d);
// }

template <typename scalar_t>
void fused_rope_with_pos_forward_launcher(
    const scalar_t *input, const float *freqs, scalar_t *output, const int s,
    const int b, const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_with_pos_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      input, freqs, output, h, d, d2, stride_s, stride_b, stride_h, stride_d,
      o_stride_s, o_stride_b, o_stride_h, o_stride_d);
  // NVTE_CHECK_CUDA(cudaGetLastError());
}

// template <typename scalar_t>
// void fused_rope_backward_launcher(const scalar_t *output_grads, const float
// *freqs,
//                                   scalar_t *input_grads, const int s, const
//                                   int b, const int h, const int d, const int
//                                   d2, const int stride_s, const int stride_b,
//                                   const int stride_h, const int stride_d,
//                                   const int o_stride_s, const int o_stride_b,
//                                   const int o_stride_h, const int o_stride_d,
//                                   cudaStream_t stream) {
//   int warps_per_block = h < 16 ? 4 : 8;
//   dim3 blocks(s, b);
//   dim3 threads(THREADS_PER_WARP, warps_per_block);

//   fused_rope_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
//       output_grads, freqs, input_grads, h, d, d2, stride_s, stride_b,
//       stride_h, stride_d, o_stride_s, o_stride_b, o_stride_h, o_stride_d);
//   // NVTE_CHECK_CUDA(cudaGetLastError());
// }

template <typename scalar_t>
void fused_rope_with_pos_forward(const at::Tensor &input,
                                 const at::Tensor &freqs, at::Tensor &output,
                                 const int s, const int b, const int h,
                                 const int d, const int d2, const int stride_s,
                                 const int stride_b, const int stride_h,
                                 const int stride_d, const int o_stride_s,
                                 const int o_stride_b, const int o_stride_h,
                                 const int o_stride_d, cudaStream_t stream) {
  // TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
  //     input.data.dtype, scalar_t,
  fused_rope_with_pos_forward_launcher<scalar_t>(
      reinterpret_cast<const scalar_t *>(input.data_ptr()),
      reinterpret_cast<const float *>(freqs.data_ptr()),
      reinterpret_cast<scalar_t *>(output.data_ptr()), s, b, h, d, d2, stride_s,
      stride_b, stride_h, stride_d, o_stride_s, o_stride_b, o_stride_h,
      o_stride_d, stream);
  // );
}

// template <typename scalar_t>
// void fused_rope_backward(const at::Tensor &output_grads, const at::Tensor
// &freqs, at::Tensor &input_grads,
//                          const int s, const int b, const int h, const int d,
//                          const int d2, const int stride_s, const int
//                          stride_b, const int stride_h, const int stride_d,
//                          const int o_stride_s, const int o_stride_b, const
//                          int o_stride_h, const int o_stride_d, cudaStream_t
//                          stream) {
//   // TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
//   //     output_grads.data.dtype, scalar_t,
//       fused_rope_backward_launcher<scalar_t>(reinterpret_cast<const scalar_t
//       *>(output_grads.data_ptr()),
//                                    reinterpret_cast<const float
//                                    *>(freqs.data_ptr()),
//                                    reinterpret_cast<scalar_t
//                                    *>(input_grads.data_ptr()), s, b, h, d,
//                                    d2, stride_s, stride_b, stride_h,
//                                    stride_d, o_stride_s, o_stride_b,
//                                    o_stride_h, o_stride_d, stream);
//   // );
// }

template <typename scalar_t>
void nvte_fused_rope_with_pos_forward(
    const at::Tensor input, const at::Tensor freqs, at::Tensor output,
    const int s, const int b, const int h, const int d, const int d2,
    const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d, cudaStream_t stream) {
  // NVTE_API_CALL(nvte_fused_rope_forward);
  // using namespace transformer_engine;
  fused_rope_with_pos_forward<scalar_t>(
      input, freqs, output, s, b, h, d, d2, stride_s, stride_b, stride_h,
      stride_d, o_stride_s, o_stride_b, o_stride_h, o_stride_d, stream);
}

// template <typename scalar_t>
// void nvte_fused_rope_backward(const at::Tensor output_grads, const at::Tensor
// freqs,
//                               at::Tensor input_grads, const int s, const int
//                               b, const int h, const int d, const int d2,
//                               const int stride_s, const int stride_b, const
//                               int stride_h, const int stride_d, const int
//                               o_stride_s, const int o_stride_b, const int
//                               o_stride_h, const int o_stride_d, cudaStream_t
//                               stream) {
//   // NVTE_API_CALL(nvte_fused_rope_backward);
//   // using namespace transformer_engine;
//   fused_rope_backward<scalar_t>(output_grads, freqs, input_grads, s, b, h, d,
//   d2, stride_s, stride_b,
//                       stride_h, stride_d, o_stride_s, o_stride_b, o_stride_h,
//                       o_stride_d, stream);
// }

// Interface for Python
at::Tensor fused_rope_with_pos_forward_func(
    const at::Tensor &input, const at::Tensor &freqs,
    const bool transpose_output_memory) {
  // using namespace transformer_engine;
  // TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  // TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  // TORCH_CHECK(input.size(0) <= freqs.size(0),
  //             "expected freqs tensor has a longer sequence length than
  //             input");
  // TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
  //             "expected the second and third dims of the freqs tensor equal
  //             1");
  // TORCH_CHECK(input.size(3) >= freqs.size(3),
  //             "expected the last dim of the input tensor equals or is "
  //             "greater than the freqs tensor");
  // TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
  //             "Dtype of the freqs tensor must be float");

  // input sizes: (s, b, h, d)
  // s: sequence length
  // b: batch size
  // h: head num
  // d: dim of each head
  const int s = input.size(0);
  const int b = input.size(1);
  const int h = input.size(2);
  const int d = input.size(3);
  // input strides
  const int stride_s = input.stride(0);
  const int stride_b = input.stride(1);
  const int stride_h = input.stride(2);
  const int stride_d = input.stride(3);
  // freqs' shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  // freqs' shape is now (B, S, D)
  const int d2 = freqs.size(-1);

  // output
  auto act_options = input.options().requires_grad(false);
  at::Tensor output;
  if (transpose_output_memory) {
    output = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
  } else {
    output = torch::empty({s, b, h, d}, act_options);
  }
  // output strides
  const int o_stride_s = output.stride(0);
  const int o_stride_b = output.stride(1);
  const int o_stride_h = output.stride(2);
  const int o_stride_d = output.stride(3);

  auto input_cu = input;
  auto freqs_cu = freqs;
  auto output_cu = output;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "nvte_fused_rope_forward", [&] {
        nvte_fused_rope_with_pos_forward<scalar_t>(
            input_cu.data(), freqs_cu.data(), output_cu.data(), s, b, h, d, d2,
            stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
            o_stride_h, o_stride_d, at::cuda::getCurrentCUDAStream());
      });

  // nvte_fused_rope_forward<input.scalar_type()>(input_cu.data(),
  // freqs_cu.data(), output_cu.data(), s, b, h, d, d2,
  //                         stride_s, stride_b, stride_h, stride_d, o_stride_s,
  //                         o_stride_b, o_stride_h, o_stride_d,
  //                         at::cuda::getCurrentCUDAStream());

  return output;
}

// // Interface for Python
// at::Tensor fused_rope_backward_func(const at::Tensor &output_grads, const
// at::Tensor &freqs,
//                                const bool transpose_output_memory) {
//   // using namespace transformer_engine;
//   // TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
//   // TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
//   // TORCH_CHECK(output_grads.size(0) <= freqs.size(0),
//   //             "expected freqs tensor has a longer sequence length than
//   output_grads");
//   // TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
//   //             "expected the second and third dims of the freqs tensor
//   equal 1");
//   // TORCH_CHECK(output_grads.size(3) >= freqs.size(3),
//   //             "expected the last dim of the output_grads tensor equals or
//   is "
//   //             "greater than the freqs tensor");
//   // TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
//   //             "Dtype of the freqs tensor must be float");

//   // output_grads sizes: (s, b, h, d)
//   // s: sequence length
//   // b: batch size
//   // h: head num
//   // d: dim of each head
//   const int s = output_grads.size(0);
//   const int b = output_grads.size(1);
//   const int h = output_grads.size(2);
//   const int d = output_grads.size(3);
//   // output_grads strides
//   const int stride_s = output_grads.stride(0);
//   const int stride_b = output_grads.stride(1);
//   const int stride_h = output_grads.stride(2);
//   const int stride_d = output_grads.stride(3);
//   // freqs' shape is always (s, 1, 1, d2), so the strides are same under
//   // different memory formats
//   const int d2 = freqs.size(3);

//   auto act_options = output_grads.options().requires_grad(false);
//   at::Tensor input_grads;
//   if (transpose_output_memory) {
//     input_grads = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
//   } else {
//     input_grads = torch::empty({s, b, h, d}, act_options);
//   }
//   const int o_stride_s = input_grads.stride(0);
//   const int o_stride_b = input_grads.stride(1);
//   const int o_stride_h = input_grads.stride(2);
//   const int o_stride_d = input_grads.stride(3);

//   auto output_grads_cu = output_grads;
//   auto freqs_cu = freqs;
//   auto input_grads_cu = input_grads;

//   VLLM_DISPATCH_FLOATING_TYPES(
//         output_grads.scalar_type(), "nvte_fused_rope_forward", [&] {
//           nvte_fused_rope_backward<scalar_t>(output_grads_cu.data(),
//           freqs_cu.data(), input_grads_cu.data(), s, b, h,
//                                   d, d2, stride_s, stride_b, stride_h,
//                                   stride_d, o_stride_s, o_stride_b,
//                                   o_stride_h, o_stride_d,
//                                   at::cuda::getCurrentCUDAStream());
//         });

//   // nvte_fused_rope_backward<float>(output_grads_cu.data(), freqs_cu.data(),
//   input_grads_cu.data(), s, b, h,
//   //                          d, d2, stride_s, stride_b, stride_h, stride_d,
//   o_stride_s, o_stride_b,
//   //                          o_stride_h, o_stride_d,
//   at::cuda::getCurrentCUDAStream());

//   return input_grads;
// }
