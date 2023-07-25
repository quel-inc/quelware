#!/bin/bash

set -eu

echo "[isort]"
isort quel_clock_master consoleapps tests tests_with_devices
echo "[black]"
black quel_clock_master consoleapps tests tests_with_devices
echo "[pflake8]"
pflake8 quel_clock_master consoleapps tests tests_with_devices
echo "[mypy]"
mypy --check-untyped-defs quel_clock_master consoleapps tests tests_with_devices


# TODO: examples are temporarily excluded. Don't forget add it again.
