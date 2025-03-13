// Inspired by QServe https://github.com/mit-han-lab/qserve/tree/main.
// Modified by Yuming Lou.
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
#include "dispatch_utils.h"
#include "utils.cuh"
#include "reduction_utils.cuh"


namespace vllm {

// from TRTLLM
template <typename Tf, typename T>
__inline__ __device__ Tf compute_layernorm(Tf val, float s_mean, float s_variance, const T* gamma, const T* beta, int i)
{
    Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gamma[i]);
    if (beta != nullptr)
    {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

// from TRTLLM
/* Computes the layernorm https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 * normed_output <- ( (input - E[input]) / Sqrt(Var[input] + eps) ) * gamma + beta
 * input is [tokens, hidden_dim]. Mean and Variance are per-row (i.e. per-token)
 *
 * One CTA handles one row.
 *
 * with USE_DIFF_OF_SQUARES set to false:
 * First pass (loop) computes the mean.
 * Second computes the variance via Var[x] = E[(x - E[x])²].
 * Third pass computes and writes normed_output
 * For better speedup, we set USE_DIFF_OF_SQUARES to true (may be faster but less accurate):
 * It turns out the accuracy dosen't drop.
 * First pass (loop) computes the mean and variance via Var[x] = E[x²] - E[x]²
 * Second pass computes and writes normed_output
 * 
 *
 * use_shmem controls if we cache input values into shared memory
 *
 * Optional: with dynamic scaling, the last pass doesn't write immediately but finds the
 *           amax per row. A final pass scales to int8 accordingly, and writes output to
 *           normed_output_quant.
 */
template <typename T, typename scale_type, bool USE_DIFF_OF_SQUARES = true>
__global__ void generalLayerNorm(const T* input, const T* gamma, const T* beta, T* normed_output, const float eps,
    int tokens, int hidden_dim, const scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;
    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }
        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }
    //Compute mean
    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();


    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val); // - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    // Compute LN and Quantize
    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
        }
    }
}


} // namespace vllm

void rms_norm_general(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &bias, // [hidden_size]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon,
              bool use_per_token_quant = true) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 128));//Reduce the idle probability of threads
  block.x = 32 * ((block.x + 31) / 32);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generalLayerNorm", [&] {
    using T = typename FloatTypeConverter<scalar_t>::Type;
    if (use_per_token_quant) {
      // per-token
      vllm::generalLayerNorm<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(bias.data_ptr<scalar_t>()),
        nullptr, epsilon, num_tokens, hidden_size, nullptr, scaling.data_ptr<at::Half>(),
        out.data_ptr<int8_t>(), false
      );
      // input, gamma, beta, normed_output, eps, tokens, hidden_dim, per_tensor_scale, per_token_scale
      // normed_output_quant, use_shmem
        // out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
        // weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    } else {
      // per-tensor
      vllm::generalLayerNorm<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, scaling.data_ptr<at::Half>(), nullptr,
        out.data_ptr<int8_t>(), false
      );
    }
  });
}