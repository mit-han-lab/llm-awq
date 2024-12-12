#include <torch/extension.h>

at::Tensor fused_rope_with_pos_forward_func(const at::Tensor &input,
                                            const at::Tensor &freqs,
                                            const bool transpose_output_memory);
