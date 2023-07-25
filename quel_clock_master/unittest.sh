#!/bin/bash

set -eu

if [ $# -ge 1 ] && [ "${1}" = "device" ]; then
  PYTHONPATH=. pytest --cov=quel_clock_master --cov-branch --cov-report=html --ignore=examples
else
  PYTHONPATH=. pytest --cov=quel_clock_master --cov-branch --cov-report=html --ignore=examples --ignore=tests_with_devices
fi
