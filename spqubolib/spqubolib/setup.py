from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

compile_args = ['-std=c++11', '-O3'] # for Linux
# compile_args = ['-std=c++11', '--stdlib=libc++', '-O3'] # for MacOS

ext = Extension("interaction.cxx_spin_mapping",
                sources=["interaction/cxx/cxx_spin_mapping.pyx",
                         "interaction/cxx/_cxx_spin_mapping.cpp",
                         ],
                include_dirs=["interaction/cxx", numpy.get_include()],
                extra_compile_args=compile_args,
                language="c++",
                )

setup(ext_modules=cythonize([ext]))
