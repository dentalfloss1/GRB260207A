from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "lcmodel",
    ["src/forwardshock.cpp"],
    cxx_std=17,
    extra_compile_args=["-O3", "-march=native"],
)

setup(name="lcmodel", ext_modules=[ext], cmdclass={"build_ext": build_ext})
