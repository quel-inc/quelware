#!/bin/bash

set -eu

echo "[isort]"
isort src tests scripts testlibs
echo "[black]"
black src tests scripts testlibs
echo "[pflake8]"
pflake8 src tests scripts testlibs
echo "[mypy]"
MYPYPATH=src:. mypy --check-untyped-defs --explicit-package-bases src/e7awghal tests scripts testlibs
