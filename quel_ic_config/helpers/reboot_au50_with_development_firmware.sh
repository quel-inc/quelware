#!/bin/bash

set -eu
HOST=172.30.2.203
PORT=3121

firmware_version=simplemulti_20240125

if [ "$#" == "0" ]; then
  echo "Error: no boxes are given."
  exit 1
fi

while (( "$#" )); do
  case "$1" in
  "074")
    quel_program_au50 --ipaddr 10.1.0.74 --macaddr 00-0a-35-0e-74-18 --firmware "${firmware_version}" --adapter 500202A50TIAA --host "$HOST" --port "$PORT" --bit
    ;;
  "050")
    quel_program_au50 --ipaddr 10.1.0.50 --macaddr 00-0a-35-0e-74-10 --firmware "${firmware_version}" --adapter 500202A50TTAA --host "$HOST" --port "$PORT" --bit
    ;;
  "060")
    quel_program_au50 --ipaddr 10.1.0.60 --macaddr 00-0a-35-0e-74-14 --firmware "${firmware_version}" --adapter 500202A5069AA --host "$HOST" --port "$PORT" --bit
    ;;
  *)
    echo "Error: invalid box $1, ignore it"
    ;;
  esac
  shift
done
echo "INFO: program completed successfully"
