#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attention/ft_attention.h"
#include "layernorm/layernorm.h"
#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"
#include "position_embedding/pos_encoding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm_forward_cuda", &layernorm_forward_cuda, "FasterTransformer layernorm kernel");
    m.def("gemm_forward_cuda", &gemm_forward_cuda, "Quantized GEMM kernel.");
    m.def("gemv_forward_cuda", &gemv_forward_cuda, "Quantized GEMV kernel.");
    m.def("rotary_embedding_neox", &rotary_embedding_neox, "Apply GPT-NeoX style rotary embedding to query and key");
    m.def("single_query_attention", &single_query_attention, "Attention with a single query",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("length_per_sample_"), py::arg("alibi_slopes_"), py::arg("timestep"), py::arg("rotary_embedding_dim")=0,
          py::arg("rotary_base")=10000.0f, py::arg("neox_rotary_style")=true);
}
