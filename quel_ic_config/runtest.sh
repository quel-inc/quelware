#!/bin/bash

set -eu

export PYTHONPATH=src:.

usage() {
  echo "Usage: $0 [ -d(C|S|Sr|F|8|8r|T|11r|N) ] [ -c(C|S|F|8|11) ] [ -l ] [ -h ]"
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

testenv_setup=
test_with_device=
test_cli=
log_level=ERROR

while getopts "p:c:d:lh" flag; do
  case $flag in
    p) testenv_setup="${OPTARG}" ;;
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

echo "INFO: installed e7awgsw is for $(./helpers/detect_e7awgsw_branch.sh)"
echo "INFO: firmware type of staging-074 is $(./helpers/detect_firmware_type.sh 10.1.0.74)"
echo "INFO: firmware type of staging-050 is $(./helpers/detect_firmware_type.sh 10.1.0.50)"
echo "INFO: firmware type of staging-060 is $(./helpers/detect_firmware_type.sh 10.1.0.60)"


function linkup_all_quel1 {
  quel1_parallel_linkup --conf helpers/quel_ci_env_quel1only_v1.yaml
}


function testenv_setup_classic {
  echo "================= setting up test environment for simplemulti_classic"
  if [ "$(./helpers/detect_e7awgsw_branch.sh)" != "SIMPLEMULTI_CLASSIC" ]; then
    pip install --force-reinstall ./dependency_pkgs/simplemulti_classic/*.whl
    echo "================= e7awgsw for simplemulti_classic is installed"
  fi
  load_vivado
  ./helpers/reboot_au50_with_arbitrary_firmware_parallel.sh simplemulti_20231228
  linkup_all_quel1
}

function testenv_setup_standard {
  echo "================= setting up test environment for simplemulti_standard"
  if [ "$(./helpers/detect_e7awgsw_branch.sh)" != "SIMPLEMULTI_STANDARD" ]; then
    pip install --force-reinstall ./dependency_pkgs/simplemulti_standard/*.whl
    echo "================= e7awgsw for simplemulti_standard is installed"
  fi
  load_vivado
  ./helpers/reboot_au50_with_arbitrary_firmware_parallel.sh simplemulti_20240125
  linkup_all_quel1
}

function testenv_setup_feedback {
  echo "================= setting up test environment for feedback"
  if [ "$(./helpers/detect_e7awgsw_branch.sh)" != "FEEDBACK" ]; then
    pip install --force-reinstall ./dependency_pkgs/feedback/*.whl
    echo "================= e7awgsw for feedback is installed"
  fi
  load_vivado
  ./helpers/reboot_au50_with_arbitrary_firmware_parallel.sh feedback_20240110
  linkup_all_quel1
}

# Notes: currently, SIMPLEMULTI_CLASSIC firmware is flahsed in the SPI flash memory.
function test_cli_CSF {
  echo "step 0 ================ (powercycling quel1 boxes )"
  ./helpers/powercycle_quel_ci_env.sh 074 050 060
  echo "step 2 ================ (get linkstatus of staging-050, should be 'no datalink' for both mxfe)"
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a || true
  echo "step 6 ================ (link up staging-050)"
  quel1_linkup --ipaddr_wss 10.1.0.50 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  echo "step 7 ================ (check the linkstatus of staging-050, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a || true
  echo "step 8 ================ (link up staging-074)"
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1 --ignore_access_failure_of_adrf6780 0,1,2,3,4,5,6,7 --ignore_lock_failure_of_lmx2594 0,1,2,3,4,5,6,7,8,9
  echo "step 8.1 ================ (link up all)"
  quel1_parallel_linkup --conf helpers/quel_ci_env_quel1only_v1.yaml
  echo "step 9 ================ (check the linkstatus of staging-074 and staging-060, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b
  echo "step 10 ================ (dumping the current config of staging-074)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.74 --boxtype quel1-a > artifacts/dump_port_staging-074.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.50 --boxtype quel1-a > artifacts/dump_port_staging-050.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.60 --boxtype quel1-b > artifacts/dump_port_staging-060.txt
  return 0
}

function test_cli_8_11 {
  echo "step 7 ================ (powercycleing staging-094)"
  ./helpers/powercycle_quel_ci_env.sh 094
  # no rebooter is connected to 157
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 || true
  quel1_linkstatus --ipaddr_wss 10.1.0.157 --boxtype x-quel1se-fujitsu11-a || true

  echo "step 8 ================ (link up first time after reboot)"
  quel1_linkup --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 --ignore_crc_error_of_mxfe 0,1 --ignore_access_failure_of_adrf6780 0,1 --ignore_lock_failure_of_lmx2594 0,1,2,3,4

  echo "step 8.1 ================ (check the linkstatus of staging-094, all the 2 mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8

  echo "step 8.2 ================ (link up again)"
  quel1_linkup --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 --ignore_crc_error_of_mxfe 0,1

  echo "step 8.3 ================ (check the linkstatus of staging-094, all the 2 mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8

  echo "step 9 ================ (link up all in parallel)"
  quel1_parallel_linkup --conf helpers/quel_ci_env_quel1se-only.yaml

  echo "step 9.1 ================ (check the linkstatus of staging-xxx, all the mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8
  quel1_linkstatus --ipaddr_wss 10.1.0.157 --boxtype x-quel1se-fujitsu11-a

  echo "step 10 ================ (dumping the current config of staging-xxx)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8 > artifacts/dump_port_staging-094.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.157 --boxtype x-quel1se-fujitsu11-a > artifacts/dump_port_staging-157.txt
}


if [ "${testenv_setup}" == "C" ]; then
  testenv_setup_classic
elif [ "${testenv_setup}" == "S" ]; then
  testenv_setup_standard
elif [ "${testenv_setup}" == "F" ]; then
  testenv_setup_feedback
fi


if [ "${test_cli}" == "C" ]; then
  test_cli_CSF
elif [ "${test_cli}" == "S" ]; then
  test_cli_CSF
elif [ "${test_cli}" == "F" ]; then
  test_cli_CSF
elif [ "${test_cli}" == "8" ]; then
  test_cli_8_11
elif [ "${test_cli}" == "11" ]; then
  test_cli_8_11
fi


if [ "${test_with_device}" == "C" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/simplemulti_classic
elif [ "${test_with_device}" == "S" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs
elif [ "${test_with_device}" == "Sr" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs --ignore=tests/with_devices/quel1/common/test_wave_generation_quel1.py
elif [ "${test_with_device}" == "F" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.50 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs
elif [ "${test_with_device}" == "8" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8
elif [ "${test_with_device}" == "8r" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/riken8 --ignore=tests/with_devices/quel1se/riken8/test_wave_generation_quel1se_riken8.py --ignore=tests/with_devices/quel1se/riken8/test_wave_generation_quel1se_riken8_port.py --ignore=tests/with_devices/quel1se/riken8/test_wave_generation_quel1se_riken8_rawrss.py
elif [ "${test_with_device}" == "11r" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.157 --boxtype x-quel1se-fujitsu11-a
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1se/common tests/with_devices/quel1se/fujitsu11
elif [ "${test_with_device}" == "N" ]; then
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/nec
elif [ "${test_with_device}" == "T" ]; then
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/tempctrl
else
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/without_devices
fi
