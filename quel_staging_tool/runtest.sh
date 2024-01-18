#!/bin/bash

set -eu

PYTHONPATH=. pytest --log-cli-level=INFO --cov=quel_staging_tool --cov-branch --cov-report=html --ignore=deactivated
