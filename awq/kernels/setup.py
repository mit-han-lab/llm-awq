from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
    "nvcc": ["-O3", "-std=c++17", "-keep"],
}

setup(
    name="f16s4_gemm",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="f16s4_gemm",
            sources=["pybind.cpp", "gemm_cuda_gen.cu"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)