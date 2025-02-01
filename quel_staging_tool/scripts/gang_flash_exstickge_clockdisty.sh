#!/bin/bash

set -eu

function hs2_color_to_id {
  if [ "$1" == "red" ]; then
    echo 210249B87FB6
  elif [ "$1" == "ylw" ]; then
    echo 210249B87F98
  elif [ "$1" == "grn" ]; then
    echo 210249B87F82
  elif [ "$1" == "blk" ]; then
    echo 210249B2B992
  elif [ "$1" == "wht" ]; then
    echo 210249B87FAE
  elif [ "$1" == "gry" ]; then
    echo 210249B87F93
  elif [ "$1" == "org" ]; then
    echo 210249B87FA8
  elif [ "$1" == "prp" ]; then
    echo 210249B2B9DE
  else
    echo xxx
  fi
}

function program {
  echo "INFO: quel_program_exstickge_clkdisty --macaddr $1 --ipaddr $2 --port 0 --firmware clkdisty_20240731 --adapter $3"
  quel_program_exstickge_clkdisty --macaddr "$1" --ipaddr "$2" --port 0 --firmware clkdisty_20240731 --adapter "$3"

  set +e
  for i in {0..8}; do
    sleep 2
    if ping -c 1 -W 1 "$2" > /dev/null; then
      echo "INFO: get ping response from $2"
      break
    else
      echo "INFO: waiting for ping response from $2 (${i}/8)"
    fi
  done
  set -e

  if [ "$i" -eq 8 ]; then
    echo "ERROR: no response from $2"
    return 1
  fi

  v=$(coap-client -m get "coap://${2}/version/firmware")
  if [ "$v" != "v1.4.0" ]; then
    echo "ERROR: unexpected version of firmware $v"
    return 1
  else
    echo "INFO: programming firmware $v is finished successfully"
    return 0
  fi
}

function select_quel1se {
  case "$1" in
  "014")
    # RIKEN 202501 (updating existing one)
    program 00-1B-1A-EE-01-30 10.6.0.14 "$2"
    ;;
  "015")
    # NTT 202501 (improved hardware)
    program 00-1B-1A-EE-01-56 10.6.0.15 "$2"
    ;;
  "017")
    # RIKEN 202501 (improved hardware)
    program 00-1B-1A-EE-01-58 10.6.0.17 "$2"
    ;;
  "021")
    # FUJITSU 202502 (improved hardware)
    program 00-1B-1A-EE-01-7A 10.6.0.21 "$2"
    ;;
  "024")
    # NEC
    program 00-1B-1A-EE-01-8C	10.6.0.24 "$2"
    ;;
  "025")
    # RIKEN 202403 master
    program 00-1B-1A-EE-01-8D	10.6.0.25 "$2"
    ;;
  "026")
    # RIKEN 202403 subordinate-0
    program 00-1B-1A-EE-01-8E	10.6.0.26 "$2"
    ;;
  "027")
    # RIKEN 202403 subordinate-1
    program 00-1B-1A-EE-01-8F	10.6.0.27 "$2"
    ;;
  "028")
    # RIKEN 202403 subordinate-2
    program 00-1B-1A-EE-01-90	10.6.0.28 "$2"
    ;;
  "030")
    program 00-1B-1A-EE-01-85	10.6.0.30 "$2"
    ;;
  "031")
    program 00-1B-1A-EE-01-8A	10.6.0.31 "$2"
    ;;
  "032")
    program 00-1B-1A-EE-01-E5	10.6.0.32 "$2"
    ;;
  "033")
    program 00-1B-1A-EE-01-E6	10.6.0.33 "$2"
    ;;
  "034")
    program 00-1B-1A-EE-01-E0	10.6.0.34 "$2"
    ;;
  "035")
    program 00-1B-1A-EE-01-E1	10.6.0.35 "$2"
    ;;
  "036")
    program 00-1B-1A-EE-01-E2	10.6.0.36 "$2"
    ;;
  "037")
    program 00-1B-1A-EE-01-DF	10.6.0.37 "$2"
    ;;
  *)
    echo "Error: no such clock distributor: $1"
    exit 1
  esac
}

if [ $# -lt 2 ]; then
  echo "Usage: $0 box_number_in_3_digits red|ylw|grn|blk|wht|gry"
  exit 1
fi

adapter=$(hs2_color_to_id "$2")
if [ "${adapter}" == "xxx" ]; then
  echo "Error: invalid color of HS2: $2"
  exit 1
fi
select_quel1se "$1" "$adapter"
