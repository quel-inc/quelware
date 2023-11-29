#!/bin/bash

set -eu

echo "[jq]"
find quel_ic_config/settings -name "*.json" -print -exec jq "." {} \; > /dev/null

echo "[clang-format]"
clang-format -i adi_ad9081_v106/ad9081_wrapper.cpp

echo "[isort]"
isort quel_ic_config quel_ic_config_utils quel_ic_config_cli tests tests_with_devices testlibs scripts
echo "[black]"
black quel_ic_config quel_ic_config_utils quel_ic_config_cli tests tests_with_devices testlibs scripts
echo "[pflake8]"
pflake8 quel_ic_config quel_ic_config_utils quel_ic_config_cli tests tests_with_devices testlibs scripts
echo "[mypy]"
mypy --check-untyped-defs tests quel_ic_config
echo "[mypy]"
# Notes: for avoiding the problem related to importing e7awgsw.
mypy --ignore-missing-imports --check-untyped-defs  quel_ic_config_utils quel_ic_config_cli tests_with_devices scripts
