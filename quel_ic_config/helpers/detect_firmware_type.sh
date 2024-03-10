#!/bin/bash

set -eu

python << EOS
from quel_ic_config import resolve_hw_type
import e7awgsw

ac = e7awgsw.AwgCtrl("${1}")
cc = e7awgsw.CaptureCtrl("${1}")
print(str(resolve_hw_type((ac.version(), cc.version(), ""))[0]).split(".")[1])
EOS
