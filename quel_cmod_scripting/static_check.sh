#!/bin/bash

set -eu

echo "[isort]"
isort quel_cmod_consoleapps quel_cmod_scripting tests_with_devices scripts
echo "[black]"
black quel_cmod_consoleapps quel_cmod_scripting tests_with_devices scripts
echo "[pflake8]"
pflake8 quel_cmod_consoleapps quel_cmod_scripting tests_with_devices scripts
echo "[mypy]"
mypy --check-untyped-defs quel_cmod_consoleapps quel_cmod_scripting tests_with_devices scripts
