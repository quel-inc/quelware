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

load_vivado() {
  if [ -z "$(which vivado)" ]; then
    source /tools/Xilinx/Vivado/2020.1/settings64.sh
  fi
}

test_with_device=
test_cli=
log_level=ERROR

while getopts "p:c:d:lh" flag; do
  case $flag in
    c) test_cli="${OPTARG}" ;;
    d) test_with_device="${OPTARG}" ;;
    l) log_level=INFO ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done


if [ "$(is_venv)" != "VENV" ]; then
    echo "ERROR: execute runtest.sh in a virtual environment"
    exit 1
fi


function test_cli_standard {
  echo "step 7 ================ (powercycleing staging-074, staging-050, and staging-060)"
  ./helpers/powercycle_quel_ci_env.sh 074 050 060
  sleep 15  # waiting for booting up
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a || true
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a || true
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b || true

  echo "step 8 ================ (link up first time after reboot)"
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1 --ignore_access_failure_of_adrf6780 0,1,2,3,4,5,6,7 --ignore_lock_failure_of_lmx2594 0,1,2,3,4,5,6,7,8,9
  quel1_linkup --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1 --ignore_access_failure_of_adrf6780 0,1,2,3,4,5,6,7 --ignore_lock_failure_of_lmx2594 0,1,2,3,4,5,6,7,8,9

  echo "step 8.1 ================ (check the linkstatus of staging-074, and -060, all the 4 mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1

  echo "step 8.2 ================ (link up again)"
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkup --ipaddr_wss 10.1.0.60 --boxtype quel1-b

  echo "step 8.3 ================ (check the linkstatus of staging-074, and -060, all the 4 mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1

  echo "step 9 ================ (link up all in parallel)"
  quel1_parallel_linkup --conf helpers/quel_ci_env_quel1only_v1.yaml

  echo "step 9.1 ================ (check the linkstatus of staging-xxx, all the mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b

  echo "step 10 ================ (dumping the current config of staging-xxx)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.74 --boxtype quel1-a > artifacts/dump_port_staging-074.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.50 --boxtype quel1-a > artifacts/dump_port_staging-050.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.60 --boxtype quel1-b > artifacts/dump_port_staging-060.txt
}

function test_cli_feedback {
  echo "Error: not supported yet"
}

function test_cli_8 {
  echo "step 7 ================ (powercycleing staging-094)"
  ./helpers/powercycle_quel_ci_env.sh 094
  sleep 15  # waiting for booting up
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 || true

  echo "step 8 ================ (link up first time after reboot)"
  quel1_linkup --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 --ignore_crc_error_of_mxfe 0,1 --ignore_access_failure_of_adrf6780 0,1 --ignore_lock_failure_of_lmx2594 0,1,2,3,4

  echo "step 8.1 ================ (check the linkstatus of staging-094, all the 2 mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 --ignore_crc_error_of_mxfe 0,1

  echo "step 8.2 ================ (link up again)"
  quel1_linkup --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 --ignore_crc_error_of_mxfe 0,1

  echo "step 8.3 ================ (check the linkstatus of staging-094, all the 2 mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 --ignore_crc_error_of_mxfe 0,1

  echo "step 9 ================ (link up all in parallel)"
  quel1_parallel_linkup --conf helpers/quel_ci_env_quel1se-riken8-only.yaml

  echo "step 9.1 ================ (check the linkstatus of staging-xxx, all the mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 --ignore_crc_error_of_mxfe 0,1

  echo "step 10 ================ (dumping the current config of staging-xxx)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 > artifacts/dump_port_staging-094.txt
}


if [ "${test_cli}" == "S" ]; then
  test_cli_standard
elif [ "${test_cli}" == "F" ]; then
  test_cli_feedback
elif [ "${test_cli}" == "8" ]; then
  test_cli_8
fi

if [ "${test_with_device}" == "S" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs
elif [ "${test_with_device}" == "Sr" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs --ignore=tests/with_devices/quel1/common/test_wave_generation_quel1.py
elif [ "${test_with_device}" == "F" ]; then
  echo "Error: not supoorted yet"
elif [ "${test_with_device}" == "8" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8
  pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8
elif [ "${test_with_device}" == "8r" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8
  pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8 --ignore=tests/with_devices/quel1se/riken8/test_wave_generation_quel1se_riken8.py --ignore=tests/with_devices/quel1se/riken8/test_wave_generation_quel1se_riken8_port.py
elif [ "${test_with_device}" == "11r" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.157 --boxtype x-quel1se-fujitsu11-a
  pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/fujitsu11
elif [ "${test_with_device}" == "N" ]; then
  pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/nec
elif [ "${test_with_device}" == "Ta" ]; then
  pytest --log-cli-level INFO --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/tempctrl/test_tempctrl_apis.py
elif [ "${test_with_device}" == "Tr" ]; then
  pytest --log-cli-level INFO --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/tempctrl/test_register_access.py
elif [ "${test_with_device}" == "L" ]; then
  pytest --log-cli-level INFO --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/lock_abnormal
else
  pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/without_devices
fi
