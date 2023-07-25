#!/bin/bash

set -eu

echo "[jq]"
find settings -name "*.json" -print -exec jq "." {} \; > /dev/null

echo "[clang-format]"
clang-format -i v106/ad9081_wrapper.cpp

echo "[isort]"
isort quel_ic_config tests tests_with_devices testlibs scripts
echo "[black]"
black quel_ic_config tests tests_with_devices testlibs scripts
echo "[pflake8]"
pflake8 quel_ic_config tests tests_with_devices testlibs scripts
echo "[mypy]"
MYPYPATH=stubs mypy --check-untyped-defs tests quel_ic_config
echo "[mypy]"
# Notes: for avoiding the problem related to importing qubelsi and e7awgsw.
MYPYPATH=stubs mypy --ignore-missing-imports --check-untyped-defs tests_with_devices scripts
