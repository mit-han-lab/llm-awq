/*
 * ROCm-LLM-AWQ — Ported from mit-han-lab/llm-awq
 * pybind.cpp — PyTorch extension entry point
 * No kernel changes needed here; function signatures are identical.
 * AMD path is selected at compile time via __HIP_PLATFORM_AMD__ in each .cu file.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attention/ft_attention.h"
#include "layernorm/layernorm.h"
#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"
#include "position_embedding/pos_encoding.h"
#include "rope_new/fused_rope_with_pos.h"
#include "w8a8/w8a8_gemm_cuda.h"
#include "w8a8/quantization.h"
#include "w8a8/layernorm.h"
#include "w8a8/act.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm_forward_cuda", &layernorm_forward_cuda,
          "FasterTransformer layernorm kernel (AMD: HIP port)");
    m.def("gemm_forward_cuda", &gemm_forward_cuda,
          "INT4 quantized GEMM (AMD: hipBLASLt path)");
    m.def("gemv_forward_cuda", &gemv_forward_cuda,
          "INT4 quantized GEMV (AMD: HIP port)");
    m.def("gemm_forward_cuda_new", &gemm_forward_cuda_new,
          "New INT4 quantized GEMM (AMD: hipBLASLt path)");
    m.def("gemv_forward_cuda_new", &gemv_forward_cuda_new,
          "New INT4 quantized GEMV (AMD: HIP port)");
    m.def("rotary_embedding_neox", &rotary_embedding_neox,
          "Apply GPT-NeoX style rotary embedding to query and key");
    m.def("single_query_attention", &single_query_attention,
          "Attention with a single query (AMD: torch SDPA / aiter fallback)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("k_cache"), py::arg("v_cache"),
          py::arg("length_per_sample_"), py::arg("alibi_slopes_"),
          py::arg("timestep"),
          py::arg("rotary_embedding_dim")=0,
          py::arg("rotary_base")=10000.0f,
          py::arg("rotary_scale")=1.0f,
          py::arg("neox_rotary_style")=true);
    m.def("fused_rope_with_pos_forward_func", &fused_rope_with_pos_forward_func,
          "Fused rope forward function with B,S,D embedding");
    m.def("w8a8_gemm_forward_cuda", &w8a8_gemm_forward_cuda,
          "W8A8 INT8 GEMM (AMD: hipBLASLt INT8 path)");
    m.def("w8a8_gemm_fuse_bias_forward_cuda", &w8a8_gemm_fuse_bias_forward_cuda,
          "W8A8 INT8 GEMM fused bias");
    m.def("invoke_quant", &invoke_quant,
          "FP16 -> INT8 activation quantization");
    m.def("rms_norm_general", &rms_norm_general,
          py::arg("out"), py::arg("input"), py::arg("weight"),
          py::arg("bias"), py::arg("scaling"), py::arg("epsilon"),
          py::arg("use_per_token_quant")=true,
          "RMS Normalization (AMD: HIP port)");
    m.def("silu_and_mul", &silu_and_mul,
          "SiLU activation + multiply (AMD: HIP port)");
    m.def("gelu_and_quant", &gelu_and_quant,
          "GELU activation + INT8 quantization (AMD: HIP port)");
}
