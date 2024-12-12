// Inspired by vLLM-SmoothQuant: https://github.com/vllm-project/vllm/pull/1112 and TensorRT-LLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024awq,
//   title={AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration},
//   author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
//   journal={Proceedings of Machine Learning and Systems},
//   volume={6},
//   pages={87--100},
//   year={2024}
// }
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "utils.cuh"
#include <cuda_fp16.h>
#include <cassert>
#include "quantization.h"

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}

/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;  // in-warp idx
    int                 wid  = threadIdx.x >> 5;    // warp idx
    val = warpReduceMax(val);  // get maxx in each warp
    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;
    __syncthreads();
    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);
    return val;
}



namespace vllm {
template <typename T, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel(const T *__restrict__ input,
                             int8_t *__restrict__ output, scale_type scale,
                             int num_tokens, int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const float zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = (float)input[token_idx * hidden_size + i];
      val = val > zero ? val : -val;
      if (val > amax_val)
        amax_val = val;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (tid == 0) {
      s_amax = block_amax_val;
      scale[token_idx] = __float2half_rn(block_amax_val / 127.0f);
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) / __half2float(scale));
    }
  }
}
}



void invoke_quant(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  torch::Tensor &scale) { // [num_tokens]
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel", [&] {
    vllm::quant_kernel<scalar_t, at::Half *, true><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
        scale.data_ptr<at::Half>(), num_tokens, hidden_size);
  });
}

