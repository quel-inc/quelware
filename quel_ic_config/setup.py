from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from glob import glob

ext_modules = [
    Pybind11Extension(
        "adi_ad9081_v106",
        sources=glob("adi_ad9081_v106/*.c") + glob("adi_ad9081_v106/*.cpp"),
    )
]

setup(
    ext_modules=ext_modules,
)
