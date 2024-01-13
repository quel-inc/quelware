#!/bin/bash

set -eu

usage() {
  echo "Usage: $0 [ -p ] [ -P ] [ -d ] [ -D ] [ -c ] [ -C ] [ -l ] [ -h ]"
}

is_venv() {
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

setup_simplemulti=0
setup_feedback=0
test_with_device_simplemulti=0
test_with_device_feedback=0
test_cli_simplemulti=0
test_cli_feedback=0
log_level=ERROR

while getopts "pPcCdDlh" flag; do
  case $flag in
    p) setup_simplemulti=1 ;;
    P) setup_feedback=1 ;;
    c) test_cli_simplemulti=1 ;;
    C) test_cli_feedback=1 ;;
    d) test_with_device_simplemulti=1 ;;
    D) test_with_device_feedback=1 ;;
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


if [ ${setup_simplemulti} -eq 1 ]; then
  echo "================= setting up test environment for simplemulti"
  if [ "$(./helpers/detect_e7awgsw_branch.sh)" != "SIMPLEMULTI" ]; then
    pip install --force-reinstall ./dependency_pkgs/simplemulti/*.whl
    echo "================= e7awgsw for simplemulti is installed"
  fi
  for boxid in 74 58 60; do
    if [ "$(./helpers/detect_firmware_type.sh 10.1.0.${boxid})" != "SIMPLEMULTI_CLASSIC" ]; then
      load_vivado
      ./helpers/reboot_au50.sh 0${boxid}
      sleep 5
      case ${boxid} in
        "74") quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a ;;
        "58") quel1_linkup --ipaddr_wss 10.1.0.58 --boxtype quel1-a --use_204c ;;
        "60") quel1_linkup --ipaddr_wss 10.1.0.60 --boxtype quel1-b ;;
        *) echo "invalid boxid: $boxid"; exit 1 ;;
      esac
    fi
  done
elif [ ${setup_feedback} -eq 1 ]; then
  echo "================= setting up test environment for feedback"
  if [ "$(./helpers/detect_e7awgsw_branch.sh)" != "FEEDBACK" ]; then
    pip install --force-reinstall ./dependency_pkgs/feedback/*.whl
    echo "================= e7awgsw for feedback is installed"
  fi
  for boxid in 74 58 60; do
    if [ "$(./helpers/detect_firmware_type.sh 10.1.0.${boxid})" != "FEEDBACK_EARLY" ]; then
      load_vivado
      helpers/reboot_au50_with_feedback_firmware.sh 0$boxid
      sleep 5
      case ${boxid} in
        "74") quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a ;;
        "58") quel1_linkup --ipaddr_wss 10.1.0.58 --boxtype quel1-a --use_204c ;;
        "60") quel1_linkup --ipaddr_wss 10.1.0.60 --boxtype quel1-b ;;
        *) echo "invalid boxid: $boxid"; exit 1 ;;
      esac
    fi
  done
fi


if [ ${test_cli_simplemulti} -eq 1 ]; then
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
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  echo "step 9 ================ (check the linkstatus of staging-074, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  echo "step 10 ================ (dumping the current config of staging-074)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.74 --boxtype quel1-a > artifacts/dump_port_staging-074.txt
elif [ ${test_cli_feedback} -eq 1 ]; then
  echo "step 7 ================ (check the linkstatus of staging-058, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a
  echo "step 8 ================ (link up staging-074)"
  quel1_linkup --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  echo "step 9 ================ (check the linkstatus of staging-074, both mxfes should be 'healthy')"
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  echo "step 10 ================ (dumping the current config of staging-074)"
  quel1_dump_port_config --ipaddr_wss 10.1.0.74 --boxtype quel1-a > artifacts/dump_port_staging-074.txt
fi


if [ ${test_with_device_simplemulti} -eq 1 ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a --ignore_crc_error_of_mxfe 0,1
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html --ignore=deactivated --ignore=tests_with_devices/feedback --ignore=tests_with_devices/qube
elif [ ${test_with_device_feedback} -eq 1 ]; then
  quel1_linkstatus --ipaddr_wss 10.1.0.74 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.58 --boxtype quel1-a
  quel1_linkstatus --ipaddr_wss 10.1.0.60 --boxtype quel1-b --ignore_crc_error_of_mxfe 0,1
  PYTHONPATH=. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html --ignore=deactivated --ignore=tests_with_devices/simplemulti --ignore=tests_with_devices/qube
else
  PYTHONPATH=. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html --ignore=deactivated --ignore=tests_with_devices
fi
