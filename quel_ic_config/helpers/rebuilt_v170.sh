#!/bin/bash

set -eu

clang-format --style=file -i adi_ad9082_v170/ad9082_wrapper.cpp
python -m build
pip install --force-reinstall -r requirements_dev.txt
pybind11-stubgen --output-dir="./generated" adi_ad9082_v170
cp ./generated/adi_ad9082_v170.pyi src/adi_ad9082_v170/__init__.pyi
isort src/adi_ad9082_v170
black src/adi_ad9082_v170
pflake8 src/adi_ad9082_v170
