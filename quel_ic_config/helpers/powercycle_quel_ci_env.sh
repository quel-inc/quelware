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
  "058")
    pe_switch_powercycle --rbtype PE6108AVA --ipaddr 10.250.0.102 --idx 7
    ;;
  "060")
    pe_switch_powercycle --rbtype PE6108AVA --ipaddr 10.250.0.102 --idx 5
    ;;
  "094")
    pe_switch_powercycle --rbtype PE4104AJ --ipaddr 10.250.0.107 --idx 1
    ;;
  *)
    echo "Error: invalid box $1, ignore it"
    ;;
  esac
  shift
done
echo "INFO: program completed successfully"
