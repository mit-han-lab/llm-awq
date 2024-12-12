#include <torch/extension.h>

void w8a8_gemm_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _wscales, torch::Tensor _ascales, torch::Tensor _out_feats);
void w8a8_gemm_fuse_bias_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _wscales, torch::Tensor _ascales, torch::Tensor _out_feats, torch::Tensor _bias);