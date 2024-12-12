#include <torch/extension.h>
void invoke_quant(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  torch::Tensor &scale);  // [num_tokens]