#!/bin/bash

set -eu
HWSVR=172.30.2.204
HWSVR_PORT=3121

firmware_version=simplemulti_20231228


if [ "$#" == "0" ]; then
  echo "Error: no boxes are given."
  exit 1
fi

while (( "$#" )); do
  case "$1" in
  "083")
    quel_program_au50 --ipaddr 10.1.0.83 --macaddr 00-0a-35-16-64-c4 --firmware "${firmware_version}" --adapter 500202A50Q1AA --host "$HWSVR" --port "$HWSVR_PORT"
    ;;
  "084")
    quel_program_au50 --ipaddr 10.1.0.84 --macaddr 00-0a-35-16-65-68 --firmware "${firmware_version}" --adapter 500202A50HKAA --host "$HWSVR" --port "$HWSVR_PORT"
    ;;
  "085")
    quel_program_au50 --ipaddr 10.1.0.85 --macaddr 00-0a-35-16-66-34 --firmware "${firmware_version}" --adapter 500202A50R2AA --host "$HWSVR" --port "$HWSVR_PORT"
    ;;
  *)
    echo "Error: invalid box $1, ignore it"
    ;;
  esac
  shift
done
echo "INFO: program completed successfully"
