#!/bin/bash

set -eu

usage() {
  echo "Usage: $0 [ -d ] [ -c ] [ -l ] [ -h ]"
}

au50_test_firmware_version=simplemulti_20231228
au50_test_host=172.30.2.203
au50_test_port=3121

exstickge_test_firmware_version=quel1se-riken8_20240206
exstickge_test_host=172.30.2.204
exstickge_test_port=5121

test_with_device=0
test_cli_command=0
log_level=WARNING

while getopts "cdlh" flag; do
  case "$flag" in
    c) test_cli_command=1 ;;
    d) test_with_device=1 ;;
    l) log_level=INFO ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [ ${test_cli_command} -eq 1 ]; then
  source /opt/Xilinx/Vivado/2020.1/settings64.sh
  echo "============ quel_program_au50 --help"
  quel_program_au50 --help
  echo "============ quel_program_au50 --dry"
  quel_program_au50 --ipaddr 10.1.0.74 --macaddr 00-0a-35-0e-74-18 --firmware "${au50_test_firmware_version}" --adapter 500202A50TIAA --host "${au50_test_host}" --port "${au50_test_port}" --dry

  source /opt/Xilinx/Vivado/2022.1/settings64.sh
  echo "============= quel_program_exstickge_1se --help"
  quel_program_exstickge_1se --help
  echo "============= quel_program_exstickge_1se --mcs"
  quel_program_exstickge_1se --ipaddr 10.5.0.83 --macaddr 00-0b-0a-ee-01-87 --firmware "${exstickge_test_firmware_version}" --adapter 210249B87F82 --host "${exstickge_test_host}" --port "${exstickge_test_port}"

  echo "============= quel_program_exstickge_clkdist --help"
  quel_program_exstickge_clkdist --help
fi

if [ ${test_with_device} -eq 1 ]; then
  PYTHONPATH=. pytest --log-cli-level="${log_level}" --cov=quel_staging_tool --cov-branch --cov-report=html --ignore=deactivated
else
  PYTHONPATH=. pytest --log-cli-level="${log_level}" --cov=quel_staging_tool --cov-branch --cov-report=html --ignore=deactivated --ignore tests_with_device
fi
