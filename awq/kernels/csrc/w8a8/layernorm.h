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
void rms_norm_general(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &bias, // [hidden_size]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon,
              bool use_per_token_quant);

