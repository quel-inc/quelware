from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from glob import glob

ext_modules = [
    Pybind11Extension(
        "adi_ad9081_v106",
        sources=glob("adi_ad9081_v106/*.c") + glob("adi_ad9081_v106/*.cpp"),
        include_dirs=["adi_ad9081_v106"],
    ),
    Pybind11Extension(
        "adi_ad9082_v161",
        sources=glob("adi_ad9082_v161/*.cpp") + glob("adi_ad9082_v161/ad9082_api/ad9082/src/*.c") + glob("adi_ad9082_v161/ad9082_api/adi_utils/src/*.c"),
        include_dirs=["adi_ad9082_v161/ad9082_api/ad9082/inc", "adi_ad9082_v161/ad9082_api/adi_inc", "adi_ad9082_v161/ad9082_api/adi_utils/inc"],
    )
]

setup(
    ext_modules=ext_modules,
)
