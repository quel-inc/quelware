#!/bin/bash

isort quel_pyxsdb
black quel_pyxsdb
mypy --check-untyped-defs quel_pyxsdb
pflake8 quel_pyxsdb
