#!/bin/bash

set -eu

clang-format --style=file -i adi_ad9082_v161/ad9082_wrapper.cpp
python -m build
pip install --force-reinstall -r requirements_simplemulti_standard.txt
pybind11-stubgen --output-dir="./generated" adi_ad9082_v161
cp ./generated/adi_ad9082_v161.pyi src/adi_ad9082_v161/__init__.pyi
isort src/adi_ad9082_v161
black src/adi_ad9082_v161
pflake8 src/adi_ad9082_v161
