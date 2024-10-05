#!/bin/bash

set -eu

clang-format --style=file -i adi_ad9081_v106/ad9082_wrapper.cpp
python -m build
pip install --force-reinstall -r requirements_simplemulti_standard.txt
pybind11-stubgen --output-dir="./generated" adi_ad9081_v106
cp ./generated/adi_ad9081_v106.pyi src/adi_ad9081_v106/__init__.pyi
isort src/adi_ad9081_v106
black src/adi_ad9081_v106
pflake8 src/adi_ad9081_v106
