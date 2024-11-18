#!/bin/bash

set -eu

echo "[jq]"
find src/quel_ic_config/settings -name "*.json" -print -exec jq "." {} \; > /dev/null

echo "[clang-format]"
clang-format -i adi_ad9081_v106/ad9081_wrapper.cpp

echo "[isort]"
isort src tests testlibs scripts scripts_internal
echo "[black]"
black src tests testlibs scripts scripts_internal
echo "[pflake8]"
pflake8 src tests testlibs scripts scripts_internal
echo "[mypy]"
mypy --check-untyped-defs tests/without_devices src/quel_ic_config
echo "[mypy]"
# Notes: for avoiding the problem related to importing e7awgsw.
mypy --ignore-missing-imports --check-untyped-defs  src/quel_ic_config_utils src/quel_ic_config_cli tests scripts scripts_internal
