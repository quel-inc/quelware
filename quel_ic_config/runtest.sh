#!/bin/bash

set -eu

usage() {
  echo "Usage: $0 [ -d ] [ -c ] [ -l ] [ -h ]"
}

test_with_device=0
test_cli=0
log_level=ERROR

while getopts "cdlh" flag; do
  case $flag in
    c) test_cli=1 ;;
    d) test_with_device=1 ;;
    l) log_level=INFO ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [ ${test_cli} -eq 1 ]; then
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
fi

if [ ${test_with_device} -eq 1 ]; then
  PYTHONPATH=. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html --ignore=deactivated
else
  PYTHONPATH=. pytest --log-cli-level "${log_level}" --cov=quel_ic_config --cov=testlibs --cov-branch --cov-report=html --ignore=deactivated --ignore=tests_with_devices
fi
