#!/bin/bash

set -eu

usage() {
  echo "Usage: $0 [ -p(C|S|F) ] [ -d(C|S|F|8|T) ] [ -c(C|S|F) ] [ -l ] [ -h ]"
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
echo "INFO: firmware type of staging-058 is $(./helpers/detect_firmware_type.sh 10.1.0.58)"
echo "INFO: firmware type of staging-060 is $(./helpers/detect_firmware_type.sh 10.1.0.60)"


function linkup_all {
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkup --ipaddr_wss 10.1.0.58 --boxtype quel1-a --use_204c
  quel1_linkup --ipaddr_wss 10.1.0.60 --boxtype quel1-b
  return 0
}


function testenv_setup_classic {
  echo "================= setting up test environment for simplemulti_classic"
  if [ "$(./helpers/detect_e7awgsw_branch.sh)" != "SIMPLEMULTI_CLASSIC" ]; then
    pip install --force-reinstall ./dependency_pkgs/simplemulti_classic/*.whl
    echo "================= e7awgsw for simplemulti_classic is installed"
  fi
  load_vivado
  ./helpers/reboot_au50_with_arbitrary_firmware_parallel.sh simplemulti_20231228
  linkup_all
}

# Notes: currently, SIMPLEMULTI_CLASSIC firmware is flahsed in the SPI flash memory.
function test_cli_classic {
  echo "step 0 ================ (powercycling staging-058)"
  pe_switch_powercycle --ipaddr 10.250.0.102 --idx 7  # for 10.1.0.58
  echo "step 1 ================ (powercycling staging-074)"
  pe_switch_powercycle --ipaddr 10.250.0.102 --idx 6  # for 10.1.0.74
  echo "step 2 ================ (get linkstatus of staging-058, should be 'no datalink' for both mxfe)"
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a || true
  echo "step 3 ================ (taking linkup statistics of staging-058:#0, it should link up five times)"
  quel1_test_linkup --ipaddr_wss 10.1.0.58 --boxtype quel1-a --count 5 --use_204c --mxfe 0
  echo "step 4 ================ (check the linkstatus of staging-058:#0, should be 'healthy datalink')"
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a --mxfe 0
  echo "step 5 ================ (check the linkstatus of staging-058:#1, should be still 'no datalink')"
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a --mxfe 1 || true
  echo "step 6 ================ (link up staging-058)"
  # Notes: link up one of #1 only causes the link down of #0 (!) because the refclks are reset.
  quel1_linkup --ipaddr_wss 10.1.0.58 --boxtype quel1-a --use_204c
  echo "step 7 ================ (check the linkstatus of staging-058, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a
  echo "step 8 ================ (link up staging-074)"
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1 --ignore_access_failure_of_adrf6780 0,1,2,3,4,5,6,7 --ignore_lock_failure_of_lmx2594 0,1,2,3,4,5,6,7,8,9
  echo "step 9 ================ (check the linkstatus of staging-074, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  echo "step 10 ================ (dumping the current config of staging-074)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.74 --boxtype quel1-a > artifacts/dump_port_staging-074.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.58 --boxtype quel1-a > artifacts/dump_port_staging-058.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.60 --boxtype quel1-b > artifacts/dump_port_staging-060.txt
  return 0
}


function testenv_setup_standard {
  echo "================= setting up test environment for simplemulti_standard"
  if [ "$(./helpers/detect_e7awgsw_branch.sh)" != "SIMPLEMULTI_STANDARD" ]; then
    pip install --force-reinstall ./dependency_pkgs/simplemulti_standard/*.whl
    echo "================= e7awgsw for simplemulti_standard is installed"
  fi
  load_vivado
  ./helpers/reboot_au50_with_arbitrary_firmware_parallel.sh simplemulti_20240125
  linkup_all
}

function test_cli_standard {
  echo "step 8 ================ (link up staging-074)"
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1 --ignore_access_failure_of_adrf6780 0,1,2,3,4,5,6,7 --ignore_lock_failure_of_lmx2594 0,1,2,3,4,5,6,7,8,9

  echo "step 9 ================ (check the linkstatus of staging-xxx, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b

  echo "step 10 ================ (dumping the current config of staging-xxx)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.74 --boxtype quel1-a > artifacts/dump_port_staging-074.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.58 --boxtype quel1-a > artifacts/dump_port_staging-058.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.60 --boxtype quel1-b > artifacts/dump_port_staging-060.txt
}


function testenv_setup_feedback {
  echo "================= setting up test environment for feedback"
  if [ "$(./helpers/detect_e7awgsw_branch.sh)" != "FEEDBACK" ]; then
    pip install --force-reinstall ./dependency_pkgs/feedback/*.whl
    echo "================= e7awgsw for feedback is installed"
  fi
  load_vivado
  ./helpers/reboot_au50_with_arbitrary_firmware_parallel.sh feedback_20240110
  linkup_all
}

function test_cli_feedback {
  echo "step 8 ================ (link up staging-074)"
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1 --ignore_access_failure_of_adrf6780 0,1,2,3,4,5,6,7 --ignore_lock_failure_of_lmx2594 0,1,2,3,4,5,6,7,8,9

  echo "step 9 ================ (check the linkstatus of staging-xxx, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b

  echo "step 10 ================ (dumping the current config of staging-xxx)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.74 --boxtype quel1-a > artifacts/dump_port_staging-074.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.58 --boxtype quel1-a > artifacts/dump_port_staging-058.txt
  quel1_dump_port_config --ipaddr_wss 10.1.0.60 --boxtype quel1-b > artifacts/dump_port_staging-060.txt
}


if [ "${testenv_setup}" == "C" ]; then
  testenv_setup_classic
elif [ "${testenv_setup}" == "S" ]; then
  testenv_setup_standard
elif [ "${testenv_setup}" == "F" ]; then
  testenv_setup_feedback
fi

if [ "${test_cli}" == "C" ]; then
  test_cli_classic
elif [ "${test_cli}" == "S" ]; then
  test_cli_standard
elif [ "${test_cli}" == "F" ]; then
  test_cli_feedback
fi

if [ "${test_with_device}" == "C" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/simplemulti_classic
elif [ "${test_with_device}" == "S" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs
elif [ "${test_with_device}" == "Sr" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs --ignore=tests/with_devices/quel1/common/test_wave_generation_quel1.py
elif [ "${test_with_device}" == "F" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1/common tests/with_devices/quel1/both_adcs
elif [ "${test_with_device}" == "8" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1se_riken8
elif [ "${test_with_device}" == "8r" ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.94 --boxtype quel1se-riken8
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/quel1se_riken8 --ignore=tests/with_devices/quel1se_riken8/test_wave_generation_quel1se_riken8.py --ignore=tests/with_devices/quel1se_riken8/test_wave_generation_quel1se_riken8_port.py
elif [ "${test_with_device}" == "N" ]; then
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/common tests/with_devices/nec
elif [ "${test_with_device}" == "T" ]; then
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/with_devices/tempctrl
else
  PYTHONPATH=src:. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html tests/without_devices
fi
