import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get environment variables
build_cuda_extension = os.environ.get('BUILD_CUDA_EXT', '1') == '1'
torch_is_prebuilt = os.environ.get('TORCH_IS_PREBUILT', '0') == '1'

# Define dependencies
dependencies = [
    "accelerate", "sentencepiece", "tokenizers>=0.12.1",
    "transformers>=4.32.0", 
    "lm_eval", "texttable",
    "toml", "attributedict",
    "protobuf"
]

if not torch_is_prebuilt:
    dependencies.extend(["torch>=2.0.0", "torchvision"])

# Setup CUDA extension
ext_modules = []

if build_cuda_extension:
    ext_modules.append(
        CUDAExtension(
            name="awq_inference_engine",
            sources=[
                "awq/kernels/csrc/pybind.cpp",
                "awq/kernels/csrc/quantization/gemm_cuda_gen.cu",
                "awq/kernels/csrc/layernorm/layernorm.cu",
                "awq/kernels/csrc/position_embedding/pos_encoding_kernels.cu"
            ],
            extra_compile_args={
                "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"]
            },
        )
    )

setup(
    name="awq",
    version="0.1.0",
    description="An efficient and accurate low-bit weight quantization(INT3/4) method for LLMs.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=dependencies,
    packages=find_packages(exclude=["results*", "scripts*", "examples*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
