import os
import torch
# from torch.utils.ffi import create_extension

from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

sources = ['correlation_package/src/corr.c']
headers = ['correlation_package/src/corr.h']

sources += ['correlation_package/src/corr1d.c']
headers += ['correlation_package/src/corr1d.h']

defines = []
with_cuda = False

# if torch.cuda.is_available():
#     print('Including CUDA code.')
#     sources += ['correlation_package/src/corr_cuda.cpp']
#     headers += ['correlation_package/src/corr_cuda.h']
#
#     sources += ['correlation_package/src/corr1d_cuda.cpp']
#     headers += ['correlation_package/src/corr1d_cuda.h']
#
#     defines += [('WITH_CUDA', None)]
#     with_cuda = True
#
# this_file = os.path.dirname(os.path.realpath(__file__))
# extra_objects = ['correlation_package/src/corr_cuda_kernel.cu.o']
# extra_objects += ['correlation_package/src/corr1d_cuda_kernel.cu.o']
# extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
#
# ffi = create_extension(
#     'correlation_package._ext.corr',
#     package=True,
#     headers=headers,
#     sources=sources,
#     define_macros=defines,
#     relative_to=__file__,
#     with_cuda=with_cuda,
#     extra_objects=extra_objects,
# )

if __name__ == '__main__':
    # ffi.build()
    setup(
        name='correlation_package',
        ext_modules=[
            CUDAExtension('correlation_package',
                          [
                              'correlation_package/src/entry.cpp',
                              # 'correlation_package/src/corr.c',
                              # 'correlation_package/src/corr1d.c',
                              # 'correlation_package/src/corr_cuda.cpp',
                              # 'correlation_package/src/corr1d_cuda.cpp',
                              'correlation_package/src/corr_cuda_kernel.cu',
                              'correlation_package/src/corr1d_cuda_kernel.cu',
                          ],
                          ),
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
