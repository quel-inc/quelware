#!/bin/bash

set -eu

echo "format check..."
make format-check

echo "lint..."
make lint

echo "typecheck..."
make typecheck
