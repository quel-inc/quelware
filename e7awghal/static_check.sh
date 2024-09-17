#!/bin/bash

set -eu

echo "[isort]"
isort src tests scripts
echo "[black]"
black src tests scripts
echo "[pflake8]"
pflake8 src tests scripts
echo "[mypy]"
MYPYPATH=src:. mypy --check-untyped-defs --explicit-package-bases src/e7awghal tests scripts
