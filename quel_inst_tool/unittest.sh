#!/bin/bash

set -eu

usage() {
  echo "Usage: $0 [ -d ] [ -w ] [ -l ] [ -a AVAILABLE_DEVICES ] [ -s AVAILABLE_WEBSERVERS] [ -h ]"
}


test_with_device=0
test_with_web=0
log_level=ERROR
available_devices=""
available_webservers=""

while getopts "dwla:s:h" flag; do
  case $flag in
    a) available_devices="${OPTARG}" ;;
    s) available_webservers="${OPTARG}" ;;
    d) test_with_device=1 ;;
    w) test_with_web=1 ;;
    l) log_level=DEBUG ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [ ${test_with_device} -eq 1 ]; then
  if [ ${test_with_web} -eq 1 ]; then
    QUEL_INST_AVAILABLE_DEVICES=${available_devices} QUEL_INST_AVAILABLE_WEBSERVERS=${available_webservers} PYTHONPATH=. pytest --log-cli-level="${log_level}" --cov=quel_inst_tool --cov-branch --cov-report=html --ignore=deactivated
  else
    QUEL_INST_AVAILABLE_DEVICES=${available_devices} PYTHONPATH=. pytest --log-cli-level="${log_level}" --cov=quel_inst_tool --cov-branch --cov-report=html --ignore=deactivated --ignore=tests_with_web
  fi
else
  if [ ${test_with_web} -eq 1 ]; then
    QUEL_INST_AVAILABLE_WEBSERVERS=${available_webservers} PYTHONPATH=. pytest --log-cli-level="${log_level}" --cov=quel_inst_tool --cov-branch --cov-report=html --ignore=deactivated --ignore=tests_with_devices
  else
    PYTHONPATH=. pytest --log-cli-level="${log_level}" --cov=quel_inst_tool --cov-branch --cov-report=html --ignore=deactivated --ignore=tests_with_devices --ignore=tests_with_web
  fi
fi
