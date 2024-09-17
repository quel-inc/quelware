#!/bin/bash

set -eu

python << EOS
import quel_ic_config
from packaging.version import Version
if Version(quel_ic_config.__version__) > Version("0.9.6"):
    print("HAL")
else:
    from quel_ic_config import detect_branch_of_library
    print(str(detect_branch_of_library()).split(".")[1])
EOS
