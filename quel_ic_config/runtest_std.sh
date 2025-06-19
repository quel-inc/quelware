#!/bin/bash

set -eu

export MPLBACKEND=agg
export PYTHONPATH=.

usage() {
  echo "Usage: $0 [ -r ]"
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

use_spectrum_analyzer=1

while getopts "rh" flag; do
  case $flag in
    r) use_spectrum_analyzer=0 ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done


if [ "$(is_venv)" != "VENV" ]; then
    echo "ERROR: execute $0 in a virtual environment"
    exit 1
fi

test_log_dir="artifacts/logs"
mkdir -p "${test_log_dir}"
rm -f "${test_log_dir}/*.txt"

echo "rebooting 074 and 157"
helpers/powercycle_quel_ci_env.sh 074 157
echo
echo
sleep 5
quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a 2>&1 | tee -a "${test_log_dir}/linkup.txt" || true
quel1_linkstatus --ipaddr_wss 10.1.0.157 --boxtype quel1se-fujitsu11-a 2>&1 | tee -a "${test_log_dir}/linkup.txt" || true
echo
echo
sleep 5
quel1_syncstatus --conf helpers/quel_ci_env_v2.yaml || true
echo
echo
sleep 5
quel1_syncstatus --conf helpers/quel_ci_env_v2.yaml --ignore_unavailable || true
echo
echo
sleep 5
quel1_parallel_linkup --conf helpers/quel_ci_env_v2.yaml 2>&1 | tee "${test_log_dir}/linkup.txt"
echo
echo
sleep 5
quel1_sync --conf helpers/quel_ci_env_v2.yaml
echo
echo
sleep 5
python tests/cli_check/check_quel1_parallel_linkup.py
echo
echo
sleep 5
quel1_sync --conf helpers/quel_ci_env_v2.yaml  # Notes: the above test step breaks synchronization.
echo
echo
sleep 5
quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a 2>&1 | tee -a "${test_log_dir}/linkup.txt"
quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a 2>&1 | tee -a "${test_log_dir}/linkup.txt"
quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b 2>&1 | tee -a "${test_log_dir}/linkup.txt"
quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 2>&1 | tee -a "${test_log_dir}/linkup.txt"
quel1_linkstatus --ipaddr_wss 10.1.0.157 --boxtype quel1se-fujitsu11-a 2>&1 | tee -a "${test_log_dir}/linkup.txt"
quel1_linkstatus --ipaddr_wss 10.1.0.164 --boxtype quel1se-fujitsu11-b 2>&1 | tee -a "${test_log_dir}/linkup.txt"
echo
echo
sleep 5
quel1_syncstatus --conf helpers/quel_ci_env_v2.yaml
echo
echo
sleep 5
pytest --log-cli-level WARNING tests/with_devices/lock_abnormal 2>&1 | tee "${test_log_dir}/lock_abnormal.txt"
echo
echo
sleep 5
# Sr: tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs --ignore=tests/with_devices/quel1/common/test_wave_generation_quel1.py
# 8: tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8
# 11r: tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/fujitsu11
# Nr: tests/with_devices/nec --ignore=tests/with_devices/nec/test_wave_generation_quel1-nec.py
# Ta: tests/with_devices/tempctrl/test_tempctrl_apis.py
# []: tests/without_devices
if [ ${use_spectrum_analyzer} -eq 1 ]; then
  pytest --log-cli-level WARNING --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html \
    tests/without_devices \
    tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs --ignore=tests/with_devices/quel1/common/test_wave_generation_quel1.py \
    tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8 \
    tests/with_devices/nec --ignore=tests/with_devices/nec/test_wave_generation_quel1-nec.py \
    tests/with_devices/tempctrl/test_tempctrl_apis.py \
    tests/with_devices/quel1se/fujitsu11 2>&1 | tee "${test_log_dir}/std.txt"
else
  pytest --log-cli-level WARNING --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html \
    tests/without_devices \
    tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs --ignore=tests/with_devices/quel1/common/test_wave_generation_quel1.py \
    tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8 --ignore=tests/with_devices/quel1se/riken8/test_wave_generation_quel1se_riken8.py --ignore=tests/with_devices/quel1se/riken8/test_wave_generation_quel1se_riken8_port.py \
    tests/with_devices/nec --ignore=tests/with_devices/nec/test_wave_generation_quel1-nec.py \
    tests/with_devices/tempctrl/test_tempctrl_apis.py \
    tests/with_devices/quel1se/fujitsu11 2>&1 | tee "${test_log_dir}/std.txt"
fi
