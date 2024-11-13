#!/bin/bash

set -eu
HOST=localhost
PORT=0

if [ $# -lt 1 ]; then
  echo "Usage: $0 firmware version"
  exit 1
fi

firmware_version=${1}

quel_program_au50 --ipaddr 10.1.0.74 --macaddr 00-0a-35-0e-74-18 --firmware "${firmware_version}" --adapter 500202A50TIAA --host "$HOST" --port "$PORT" 2>&1 | tee updatelog_074.log &
sleep 5  # waiting for cs_server is starting
quel_program_au50 --ipaddr 10.1.0.50 --macaddr 00-0a-35-0e-74-10 --firmware "${firmware_version}" --adapter 500202A50TTAA --host "$HOST" --port "$PORT" 2>&1 | tee updatelog_050.log &
quel_program_au50 --ipaddr 10.1.0.60 --macaddr 00-0a-35-0e-74-14 --firmware "${firmware_version}" --adapter 500202A5069AA --host "$HOST" --port "$PORT" 2>&1 | tee updatelog_060.log &

wait
echo "INFO: program completed successfully"
