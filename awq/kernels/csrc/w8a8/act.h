// Inspired by TRT-LLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024awq,
//   title={AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration},
//   author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
//   journal={Proceedings of Machine Learning and Systems},
//   volume={6},
//   pages={87--100},
//   year={2024}
// }

#include <torch/extension.h>
#include <cuda_fp16.h>
// Inspired by vLLM-SmoothQuant: https://github.com/vllm-project/vllm/pull/1112.
#include <torch/extension.h>


void gelu_and_quant(torch::Tensor &out,   // [..., d]
                                torch::Tensor &input, // [..., d]
                                torch::Tensor &scale_out, // [num_tokens]
                                torch::Tensor &tmp // [num_tokens, d]
);

torch::Tensor silu_and_mul(torch::Tensor &input  // [..., 2 * d]
);


        

