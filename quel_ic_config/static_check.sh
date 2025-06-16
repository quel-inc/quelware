#!/bin/bash

set -eu

echo "validate-json..."
make validate-json

echo "format check..."
make format-cpp-check
make format-py-check

echo "lint..."
make lint

echo "typecheck..."
make typecheck
