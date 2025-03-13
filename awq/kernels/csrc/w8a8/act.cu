#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

#include "dispatch_utils.h"
#include "utils.cuh"
#include "reduction_utils.cuh"

namespace vllm {

template <typename T> __device__ __forceinline__ T silu(const T &x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T> __device__ __forceinline__ T gelu_new(const T &x) {
  const half x3 = (half)(x * x * x);
  const T t = (T)tanhf((T)((T)0.79788456f * (half)(x + (T)((T)0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast(const T &x) {
  const half f = (half)x;
  const T t =
      (T)tanhf(((T)(f * (T)0.79788456f)) * (((T)1.0) + (T)((T)0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

  

// dequant int32 input, apply silu and mul, then per token quant to int8
template <typename scale_type, bool use_per_token_quant>
__global__ void gelu_and_quant_kernel(
    int8_t *__restrict__ out,          // [..., d]
    half *__restrict__ input, // [..., d]
    const int d,
    scale_type * scale_out,                  // [num_tokens]
    half *__restrict__ tmp = nullptr // [num_tokens, d]
) {
  const int token_idx = blockIdx.x;
  const float max_value= 127.0f;
  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const half zero = 0.0001f;

    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const half x =
          (half)__ldg(&input[token_idx * d + idx]);
      half t = gelu_fast(x);
      tmp[token_idx * d + idx] = t;
      t = t > zero ? t : -t;
      if ((float)t > amax_val)
        amax_val = (float)t;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (threadIdx.x == 0) {
      s_amax = block_amax_val;
      scale_out[token_idx] = half(block_amax_val / max_value);
    }
    __syncthreads();
    
    float tmp_scale = max_value / s_amax;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      out[token_idx * d + idx] =
          float_to_int8_rn((half)tmp_scale * tmp[token_idx * d + idx]);
    }
  } else {
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const float x =
          (float)__ldg(&input[token_idx * d + idx]);
      out[token_idx * d + idx] = float_to_int8_rn((half)gelu_fast(x)  / scale_out[0]);
    }
  }
}
} // namespace vllm



void gelu_and_quant(
    torch::Tensor &out,   // [..., d]
    torch::Tensor &input, // [..., d]
    torch::Tensor &scale_out, // [...]
    torch::Tensor &tmp // [num_tokens, d]
    ) {
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1);
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 128));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::gelu_and_quant_kernel<half, true><<<grid, block, 0, stream>>>(
      out.data_ptr<int8_t>(), reinterpret_cast<half *>(input.data_ptr<at::Half>()), d, reinterpret_cast<half *>(scale_out.data_ptr<at::Half>()),reinterpret_cast<half *>(tmp.data_ptr<at::Half>()));
}



namespace vllm {
  
template<typename scalar_t>
__global__ void silu_and_mul_kernel(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2 * d]
  const int d) {

  const int token_idx = blockIdx.x;
  const int64_t token_idx_d = token_idx * int64_t(d);
  const int64_t token_idx_2d = token_idx_d * 2;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx_2d + idx]);
    const scalar_t y = __ldg(&input[token_idx_2d + d + idx]);
    out[token_idx_d + idx] = silu(x) * y;
  }
}
} // namespace vllm



torch::Tensor silu_and_mul(
  torch::Tensor& input)    // [..., 2 * d]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  std::vector<int64_t> output_shape = input.sizes().vec();
  output_shape[output_shape.size() - 1]=d;
  auto options =
      torch::TensorOptions().dtype(input.dtype()).device(input.device());
  at::Tensor output = torch::empty(output_shape, options);


  dim3 grid(num_tokens);
  dim3 block(std::min(d, 256));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_kernel", [&] {
    vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
  });
  return output;
}