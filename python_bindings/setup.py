import os
import sys
import platform

import numpy as np
import pybind11
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.1.0'
os.environ["CXX"] = "g++"

include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

# compatibility when run in python_bindings
bindings_dir = 'python_bindings'
if bindings_dir in os.path.basename(os.getcwd()):
    source_files = ['./bindings.cpp']
    include_dirs.extend(['../oqglib/'])
else:
    source_files = ['./python_bindings/bindings.cpp']
    include_dirs.extend(['./oqglib/'])


libraries = ["z"]
extra_objects = []


ext_modules = [
    Extension(
        'oqglib',
        source_files,
        include_dirs=include_dirs,
        libraries=libraries,
        language='c++',
        extra_objects=extra_objects,
    ),
]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    if has_flag(compiler, '-std=c++23'):
        return '-std=c++23'
    elif has_flag(compiler, '-std=c++20'):
        return '-std=c++20'
    elif has_flag(compiler, '-std=c++17'):
        return '-std=c++17'
    elif has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


def _gcc_file(name):
    try:
        p = subprocess.check_output(["g++", "-print-file-name=" + name], text=True).strip()
        return None if (not p or p == name) else p
    except Exception:
        return None


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'unix': ['-Ofast', '-lopenblas', '-fopenmp', '-march=native', '-ffast-math', '-mfma', '-funroll-loops', '-flto', '-frename-registers'],  # , '-w'
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    c_opts['unix'].append("-fopenmp")
    c_opts['unix'].append("-DUSE_AVX512")
    link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = BuildExt.c_opts.get(ct, [])
        if ct == 'unix':

            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            print(f"Use c++ {cpp_flag(self.compiler)}")
            opts.append(cpp_flag(self.compiler))

        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(BuildExt.link_opts.get(ct, []))

        build_ext.build_extensions(self)


setup(
    name='oqglib',
    version=__version__,
    description='oqglib',
    author='Xiao Luo',
    url='https://github.com/TheDatumOrg/OQG',
    long_description="""oqglib""",
    ext_modules=ext_modules,
    install_requires=['numpy', 'faiss-cpu'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
