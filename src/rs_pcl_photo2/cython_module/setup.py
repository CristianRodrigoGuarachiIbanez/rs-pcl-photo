from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np



setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(Extension("image_reconstruction",
                                    sources=["image_reconstruction.pyx"],
                                    language="c++",
                                    include_dirs=[np.get_include()],
                                    )
                          )
)