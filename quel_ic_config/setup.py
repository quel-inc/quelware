from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "adi_ad9081_v106",
        ["v106/ad9081_wrapper.cpp", "v106/adi_ad9081_adc.c", "v106/adi_ad9081_dac.c", "v106/adi_ad9081_device.c", "v106/adi_ad9081_hal.c", "v106/adi_ad9081_jesd.c"],
    )
]

setup(
    ext_modules=ext_modules,
)
