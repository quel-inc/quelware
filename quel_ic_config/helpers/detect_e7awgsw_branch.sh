#!/bin/bash

set -eu

python << EOS
from quel_ic_config_utils import detect_branch_of_library
print(str(detect_branch_of_library()).split(".")[1])
EOS
