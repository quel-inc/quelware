#!/bin/bash

set -eu

echo "[isort]"
isort quel_inst_tool quel_inst_tool_consoleapps tests tests_with_devices tests_with_web
echo "[black]"
black quel_inst_tool quel_inst_tool_consoleapps tests tests_with_devices tests_with_web
echo "[pflake8]"
pflake8 quel_inst_tool quel_inst_tool_consoleapps tests tests_with_devices tests_with_web
echo "[mypy]"
mypy --check-untyped-defs quel_inst_tool quel_inst_tool_consoleapps
# tests are checked separately to avoid the conflict in the resolution of conftest.py
mypy --check-untyped-defs tests
mypy --check-untyped-defs tests_with_devices
mypy --check-untyped-defs tests_with_web
