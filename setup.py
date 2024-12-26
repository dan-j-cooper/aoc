from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="parser",
        ext_modules=cythonize("src/parser.pyx"),
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(name="parser", ext_modules=cythonize(extensions))
