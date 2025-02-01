#!/bin/bash

set -eu

echo "[jq]"
find src/quel_ic_config/settings -name "*.json" -print -exec jq "." {} \; > /dev/null
find src/quel_ic_config_cli/settings -name "*.json" -print -exec jq "." {} \; > /dev/null

echo "[clang-format]"
clang-format -i adi_ad9082_v170/ad9082_wrapper.cpp

echo "[isort]"
isort src tests testlibs scripts scripts_internal
echo "[black]"
black src tests testlibs scripts scripts_internal
echo "[pflake8]"
pflake8 src tests testlibs scripts scripts_internal
echo "[mypy]"
mypy --check-untyped-defs src/quel_ic_config src/quel_ic_config_utils src/quel_ic_config_cli tests scripts scripts_internal
