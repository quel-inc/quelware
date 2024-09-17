#!/bin/bash

PYTHONPATH=src pytest --log-cli-level=INFO --cov=e7awghal --cov-branch --cov-report=html
