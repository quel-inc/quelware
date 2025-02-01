#!/bin/bash

set -eu

if [ "$#" == "0" ]; then
  echo "Error: no boxes are given."
  exit 1
fi

while (( "$#" )); do
  case "$1" in
  "074")
    pe_switch_powercycle --rbtype PE6108AVA --ipaddr 10.250.0.102 --idx 6
    ;;
  "050")
    pe_switch_powercycle --rbtype PE6108AVA --ipaddr 10.250.0.102 --idx 7
    ;;
  "060")
    pe_switch_powercycle --rbtype PE6108AVA --ipaddr 10.250.0.102 --idx 5
    ;;
  "071")
    pe_switch_powercycle --rbtype PE6108AVA --ipaddr 10.250.0.102 --idx 3
    ;;
  "080")
    pe_switch_powercycle --rbtype PE4104AJ --ipaddr 10.250.0.107 --idx 1
    ;;
  "094")
    pe_switch_powercycle --rbtype PE4104AJ --ipaddr 10.250.0.107 --idx 4
    ;;
  "157")
    pe_switch_powercycle --rbtype PE4104AJ --ipaddr 10.250.0.107 --idx 2
    ;;
  "164")
    pe_switch_powercycle --rbtype PE4104AJ --ipaddr 10.250.0.107 --idx 3
    ;;
  *)
    echo "Error: invalid box $1, ignore it"
    ;;
  esac
  shift
done
echo "INFO: program completed successfully"
