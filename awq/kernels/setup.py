import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def is_rocm():
    try:
        import torch
        return hasattr(torch.version, "hip") and torch.version.hip is not None
    except ImportError:
        return False

ROCM_BUILD = is_rocm()

SOURCES = [
    "csrc/pybind.cpp",
    "csrc/quantization/gemm_cuda_gen.cu",
    "csrc/quantization/gemv_cuda.cu",
    "csrc/quantization_new/gemv/gemv_cuda.cu",
    "csrc/quantization_new/gemm/gemm_cuda.cu",
    "csrc/layernorm/layernorm.cu",
    "csrc/position_embedding/pos_encoding_kernels.cu",
    "csrc/attention/ft_attention.cpp",
    "csrc/attention/decoder_masked_multihead_attention.cu",
    "csrc/rope_new/fused_rope_with_pos.cu",
    "csrc/w8a8/w8a8_gemm_cuda.cu",
    "csrc/w8a8/quantization.cu",
    "csrc/w8a8/act.cu",
    "csrc/w8a8/layernorm.cu",
]

if ROCM_BUILD:
    import glob as _glob
    rocm_home = os.path.realpath(os.environ.get("ROCM_HOME", "/opt/rocm"))
    hip_include = os.path.join(rocm_home, "include")
    hipblas_include = os.path.join(rocm_home, "include/hipblas")
    _here = os.path.dirname(os.path.abspath(__file__))
    _stub = os.path.join(_here, "hipsparse_stub")
    # Do NOT add hip_include -- hipcc finds ROCm headers via its own defaults.
    # Adding it explicitly duplicates the path that torch's THH/THC also adds,
    # causing amd_hip_bf16.h to be included twice (#pragma once dedup fails).
    # KB: rocm_pragma_once_fails_with_duplicate_isystem_paths
    extra_includes = [hipblas_include, "csrc"]
    if os.path.isdir(_stub):
        extra_includes.insert(0, _stub)
    # Note: do NOT add /opt/rocm-<ver>/include separately -- it's the same as
    # /opt/rocm/include (symlink) but breaks #pragma once deduplication in clang.
    extra_compile_args = {
        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16", "-D__HIP_PLATFORM_AMD__"],
        "nvcc": ["-O3", "-std=c++17", "-DENABLE_BF16", "-D__HIP_PLATFORM_AMD__", "-ffast-math", "--offload-arch=gfx90a,gfx942", "-Wno-unused-result", "-Wno-narrowing", "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__", "-mllvm", "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false"],
    }
    ext = CUDAExtension(
        name="awq_inference_engine",
        sources=SOURCES,
        include_dirs=extra_includes,
        extra_compile_args=extra_compile_args,
        libraries=["hipblas", "hipblaslt"],
        library_dirs=["/opt/rocm-7.2.4/lib", "/opt/rocm/lib"],
        runtime_library_dirs=["/opt/rocm-7.2.4/lib", "/opt/rocm/lib"],
    )
else:
    extra_compile_args = {
        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
        "nvcc": ["-O3", "-std=c++17", "-DENABLE_BF16",
                 "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                 "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                 "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                 "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "--threads=8"],
    }
    ext = CUDAExtension(name="awq_inference_engine", sources=SOURCES, extra_compile_args=extra_compile_args)

setup(
    name="awq_inference_engine",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
