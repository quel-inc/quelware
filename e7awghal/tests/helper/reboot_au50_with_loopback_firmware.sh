#!/bin/bash

set -eu
HOST=172.30.2.203
PORT=3121

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
firmware_directory="${parent_path}/../fixture"

if [ "$#" -lt "2" ]; then
  echo "Usage: $0 firmware_version target_box_number ..."
  echo
  echo "possible firmware_versions: $(ls -1 "${firmware_directory}" | sed 's/au50_loopback_//' | xargs)"
  exit 1
fi

firmware_version="loopback_${1}"

while (( "$#" - 1 )); do
  case "$2" in
  "074")
    quel_program_au50 --ipaddr 10.1.0.74 --macaddr 00-0a-35-0e-74-18 --firmware_dir "${firmware_directory}" --firmware "${firmware_version}" --adapter 500202A50TIAA --host "$HOST" --port "$PORT" --bit
    ;;
  "058")
    quel_program_au50 --ipaddr 10.1.0.58 --macaddr 00-0a-35-0e-77-34 --firmware_dir "${firmware_directory}" --firmware "${firmware_version}" --adapter 500202A506VAA --host "$HOST" --port "$PORT" --bit
    ;;
  "060")
    quel_program_au50 --ipaddr 10.1.0.60 --macaddr 00-0a-35-0e-74-14 --firmware_dir "${firmware_directory}" --firmware "${firmware_version}" --adapter 500202A5069AA --host "$HOST" --port "$PORT" --bit
    ;;
  *)
    echo "Error: invalid box $2, ignore it"
    ;;
  esac
  shift
done
echo "INFO: programming completed successfully"
exit 0
