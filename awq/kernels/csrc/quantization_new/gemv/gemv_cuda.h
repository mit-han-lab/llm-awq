#pragma once
#include <torch/extension.h>

torch::Tensor gemv_forward_cuda_new(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int m,
    int n,
    int k,
    int group_size);
