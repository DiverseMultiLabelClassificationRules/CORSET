import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
# enable line profiling
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True
extensions = [
    Extension(
        'dmrs.samplers.dfs', ["dmrs/samplers/dfs.pyx"],
        define_macros=[('CYTHON_TRACE', '1')],
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11", "-O3"],
        extra_link_args=["-std=c++11"]
        # annotate=True,
    )
]
setup(
    name='dmrs',
    ext_modules=cythonize(extensions),
    # zip_safe=False,

)
