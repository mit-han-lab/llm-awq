import os
import shutil
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

IS_ROCM = torch.version.hip is not None
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if IS_ROCM:
    # On ROCm, torch hipifies the sources at build time and compiles them with
    # hipcc/amdclang. The NVIDIA-only nvcc flags (--use_fast_math, --threads,
    # --expt-*) do not apply; -ffast-math is the clang equivalent. The
    # -U__HIP_NO_HALF_* flags mirror CUDA's -U__CUDA_NO_HALF_* (implicit
    # half/bf16 conversions), and the force-included compat shim supplies the
    # bf16 typedefs hipify leaves untranslated.
    compat_header = os.path.join(THIS_DIR, "csrc", "awq_amd_compat.h")
    nvcc_args = [
        "-O3",
        "-std=c++20",
        "-DENABLE_BF16",
        "-ffast-math",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT16_OPERATORS__",
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT162_OPERATORS__",
        "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        "-include", compat_header,
    ]
else:
    nvcc_args = [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8",
    ]

cxx_args = ["-g", "-O3", "-std=c++17", "-DENABLE_BF16"]
if not IS_ROCM:
    # -fopenmp/-lgomp are Linux GCC flags; on ROCm every .cpp source is
    # routed through hipcc (amdclang), which does not need them.
    cxx_args += ["-fopenmp", "-lgomp"]

extra_compile_args = {
    "cxx": cxx_args,
    "nvcc": nvcc_args,
}

sources = [
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

# The W8A8 GEMM calls hipBLASLt directly on ROCm.
libraries = ["hipblaslt"] if IS_ROCM else []
extra_link_args = []

if IS_ROCM and os.name == "nt":
    # Some Windows ROCm SDKs ship the hipBLASLt import library only as
    # libhipblaslt.dll.a, which link.exe does not find as "hipblaslt".
    # Pass it by path when hipblaslt.lib is absent.
    from torch.utils.cpp_extension import ROCM_HOME

    rocm_lib_dir = os.path.join(ROCM_HOME, "lib")
    hipblaslt_dll_a = os.path.join(rocm_lib_dir, "libhipblaslt.dll.a")
    if not os.path.exists(os.path.join(rocm_lib_dir, "hipblaslt.lib")) and os.path.exists(hipblaslt_dll_a):
        libraries = []
        extra_link_args = [hipblaslt_dll_a]

if IS_ROCM:
    # On ROCm, torch compiles .cpp files with the host compiler (MSVC on
    # Windows, which cannot parse the GCC __attribute__ syntax in the HIP
    # runtime headers; g++ on Linux, which some ROCm releases' HIP bf16
    # header fails under). Copy each .cpp source to a *_cpp_shim.cu shim so
    # the .cu extension routes it through hipcc (amdclang) instead.
    # The shim is transient; do NOT commit it (it is gitignored).
    new_sources = []
    for s in sources:
        if s.endswith(".cpp"):
            shim = s[:-4] + "_cpp_shim.cu"
            shim_abs = os.path.join(THIS_DIR, shim)
            src_abs = os.path.join(THIS_DIR, s)
            if not os.path.exists(shim_abs) or os.path.getmtime(src_abs) > os.path.getmtime(shim_abs):
                shutil.copy2(src_abs, shim_abs)
            new_sources.append(shim)
        else:
            new_sources.append(s)
    sources = new_sources

setup(
    name="awq_inference_engine",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="awq_inference_engine",
            sources=sources,
            extra_compile_args=extra_compile_args,
            libraries=libraries,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
