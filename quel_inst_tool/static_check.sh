#!/bin/bash

set -eu

echo "[isort]"
isort quel_inst_tool quel_inst_tool_consoleapps tests tests_with_devices
echo "[black]"
black quel_inst_tool quel_inst_tool_consoleapps tests tests_with_devices
echo "[pflake8]"
pflake8 quel_inst_tool quel_inst_tool_consoleapps tests tests_with_devices
echo "[mypy]"
mypy quel_inst_tool quel_inst_tool_consoleapps tests tests_with_devices
