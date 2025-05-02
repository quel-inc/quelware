#!/bin/bash

set -eu

PYTHONPATH=. pytest --log-cli-level=INFO --cov=quel_cmod_scripting --cov-branch --cov-report=html --ignore=deactivated
