#!/bin/bash

set -eu

echo "[isort]"
isort quel_staging_tool quel_staging_tool_consoleapps tests_with_device
echo "[black]"
black quel_staging_tool quel_staging_tool_consoleapps tests_with_device
echo "[pflake8]"
pflake8 quel_staging_tool quel_staging_tool_consoleapps tests_with_device
echo "[mypy]"
mypy --check-untyped-defs quel_staging_tool quel_staging_tool_consoleapps tests_with_device
