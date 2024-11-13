#!/bin/bash

set -eu
HOST=172.30.2.203
PORT=3121

if [ "$#" == "0" ]; then
  echo "Error: no boxes are given."
  exit 1
fi

while (( "$#" )); do
  case "$1" in
  "074")
    quel_reboot_fpga --adapter 500202A50TIAA --host "$HOST" --port "$PORT"
    ;;
  "050")
    quel_reboot_fpga --adapter 500202A50TTAA --host "$HOST" --port "$PORT"
    ;;
  "060")
    quel_reboot_fpga --adapter 500202A5069AA --host "$HOST" --port "$PORT"
    ;;
  "071")
    quel_reboot_fpga --adapter 500202a50nhAA --host "$HOST" --port "$PORT"
    ;;
  "080")
    quel_reboot_fpga --adapter 500202A50B0AA --host "$HOST" --port "$PORT"
    ;;
  "094")
    quel_reboot_fpga --adapter 500202A50KCAA --host "$HOST" --port "$PORT"
    ;;
  *)
    echo "Error: invalid box $1, ignore it"
    ;;
  esac
  shift
done
echo "INFO: program completed successfully"
