from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('_C', [
            'deformable/deform_conv.cpp',
            'deformable/deform_conv_cuda.cu',
            'deformable/deform_conv_cuda_kernel.cu'],
              include_dirs=["deformable"],
              define_macros = [("WITH_CUDA", None)],
              extra_compile_args={"nvcc":[
                    "-O3",
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__"],
                    "cxx":[],}
              )
    ],
    cmdclass={'build_ext': BuildExtension})