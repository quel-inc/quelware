#!/bin/bash

set -eu

export PYTHONPATH=src:.

usage() {
  echo "Usage: $0 [ -d(S|Sr|F|8|8r|T|11r|N) ] [ -c(S|F|8) ] [ -l ] [ -h ]"
}

is_venv() {
  # TODO: find a better way
  pypath="$(which python3)"
  if [ "${pypath:0:5}" != "/usr/" ]; then
    echo "VENV"
  else
    echo "OS"
  fi
}

if [ "$(is_venv)" != "VENV" ]; then
    echo "ERROR: execute runtest.sh in a virtual environment"
    exit 1
fi

test_log_dir="artifacts/logs"
mkdir -p "${test_log_dir}"
rm -f "${test_log_dir}/*.txt"

quel1_parallel_linkup --conf helpers/quel_ci_env.yaml 2>&1 | tee "${test_log_dir}/linkup.txt"
# Sr: tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs --ignore=tests/with_devices/quel1/common/test_wave_generation_quel1.py
# 8: tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8
# 11r: tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/fujitsu11
# Ta: tests/with_devices/tempctrl/test_tempctrl_apis.py
# []: tests/without_devices
echo
echo
sleep 5
pytest --log-cli-level WARNING --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html \
  tests/without_devices \
  tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs --ignore=tests/with_devices/quel1/common/test_wave_generation_quel1.py \
  tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8 \
  tests/with_devices/tempctrl/test_tempctrl_apis.py \
  tests/with_devices/quel1se/fujitsu11 2>&1 | tee "${test_log_dir}/std.txt"
