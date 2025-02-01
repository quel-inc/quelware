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

R8F=quel1se-riken8_20240731
R8V="v1.2.1"
F11FA=quel1se-fujitsu11-a_20240731
F11FB=quel1se-fujitsu11-b_20240731
F11V="v1.2.1"

function program {
  case "$3" in
  "R8")
    fwid=${R8F}
    vstr=${R8V}
    bxtpstr="quel1se-riken8"
    ;;
  "F11A")
    fwid=${F11FA}
    vstr=${F11V}
    bxtpstr="quel1se-fujitsu11-a"
    ;;
  "F11B")
    fwid=${F11FB}
    vstr=${F11V}
    bxtpstr="quel1se-fujitsu11-b"
    ;;
  *)
    echo "invalid type of firmware: $3"
    exit 1
    ;;
  esac

  echo "INFO: quel_program_exstickge_1se --macaddr $1 --ipaddr $2 --port 0 --firmware ${fwid} --adapter $4"
  quel_program_exstickge_1se --macaddr "$1" --ipaddr "$2" --port 0 --firmware "${fwid}" --adapter "$4"

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

  v=$(coap-client -m get "coap://${2}/conf/boxtype")
  if [ "$v" != "${bxtpstr}" ]; then
    echo "ERROR: unexpected boxtype of firmware $v"
    return 1
  fi

  u=$(coap-client -m get "coap://${2}/version/firmware")
  if [ "$u" != "${vstr}" ]; then
    echo "ERROR: unexpected version of firmware $u"
    return 1
  else
    echo "INFO: programming firmware $v:$u is finished successfully"
    return 0
  fi
}

function select_quel1se {
  case "$1" in
  "083")
    program 00-1b-1a-ee-01-87	10.5.0.83 R8 "$2"
    ;;
  "084")
    program 00-1b-1a-ee-01-86	10.5.0.84 R8 "$2"
    ;;
  "085")
    program 00-1b-1a-ee-01-88	10.5.0.85 R8 "$2"
    ;;
  "087")
    program 00-1B-1A-EE-01-97	10.5.0.87 R8 "$2"
    ;;
  "088")
    program 00-1B-1A-EE-01-98	10.5.0.88 R8 "$2"
    ;;
  "089")
    program 00-1B-1A-EE-01-99	10.5.0.89 R8 "$2"
    ;;
  "090")
    program 00-1B-1A-EE-01-9A	10.5.0.90 R8 "$2"
    ;;
  "091")
    program 00-1B-1A-EE-01-9B	10.5.0.91 R8 "$2"
    ;;
  "092")
    program 00-1B-1A-EE-01-9C	10.5.0.92 R8 "$2"
    ;;
  "093")
    program 00-1B-1A-EE-01-9D	10.5.0.93 R8 "$2"
    ;;
  "094")
    program 00-1B-1A-EE-01-9E	10.5.0.94 R8 "$2"
    ;;
  "095")
    program 00-1B-1A-EE-01-9F	10.5.0.95 R8 "$2"
    ;;
  "096")
    program 00-1B-1A-EE-01-A0 10.5.0.96 R8 "$2"
    ;;
  "097")
    program 00-1B-1A-EE-01-A1	10.5.0.97 R8 "$2"
    ;;
  "098")
    program 00-1B-1A-EE-01-A2	10.5.0.98 R8 "$2"
    ;;
  "099")
    program 00-1B-1A-EE-01-A3 10.5.0.99 R8 "$2"
    ;;
  "100")
    program 00-1B-1A-EE-01-A4 10.5.0.100 R8 "$2"
    ;;
  "101")
    program 00-1B-1A-EE-01-A5 10.5.0.101 R8 "$2"
    ;;
  "102")
    program 00-1B-1A-EE-01-A6	10.5.0.102 R8 "$2"
    ;;
  "103")
    program 00-1B-1A-EE-01-A7	10.5.0.103 R8 "$2"
    ;;
  "104")
    program 00-1B-1A-EE-01-A8	10.5.0.104 R8 "$2"
    ;;
  "105")
    program 00-1B-1A-EE-01-A9	10.5.0.105 R8 "$2"
    ;;
  "106")
    program 00-1B-1A-EE-01-AA	10.5.0.106 R8 "$2"
    ;;
  "107")
    program 00-1B-1A-EE-01-AB	10.5.0.107 R8 "$2"
    ;;
  "108")
    program 00-1B-1A-EE-01-AC	10.5.0.108 R8 "$2"
    ;;
  "109")
    program 00-1B-1A-EE-01-AD	10.5.0.109 R8 "$2"
    ;;
  "110")
    program 00-1B-1A-EE-01-AE	10.5.0.110 R8 "$2"
    ;;
  "111")
    program 00-1B-1A-EE-01-AF	10.5.0.111 R8 "$2"
    ;;
  "112")
    program 00-1B-1A-EE-01-B0	10.5.0.112 R8 "$2"
    ;;
  "113")
    program 00-1B-1A-EE-01-B1	10.5.0.113 R8 "$2"
    ;;
  "114")
    program 00-1B-1A-EE-01-B2	10.5.0.114 R8 "$2"
    ;;
  "115")
    program 00-1B-1A-EE-01-B3	10.5.0.115 R8 "$2"
    ;;
  "116")
    program 00-1B-1A-EE-01-B4	10.5.0.116 R8 "$2"
    ;;
  "117")
    program 00-1B-1A-EE-01-B5	10.5.0.117 R8 "$2"
    ;;
  "118")
    program 00-1B-1A-EE-01-B6	10.5.0.118 R8 "$2"
    ;;
  "119")
    program 00-1B-1A-EE-01-B7	10.5.0.119 R8 "$2"
    ;;
  "120")
    program 00-1B-1A-EE-01-B8	10.5.0.120 R8 "$2"
    ;;
  "121")
    program 00-1B-1A-EE-01-B9	10.5.0.121 R8 "$2"
    ;;
  "122")
    program 00-1B-1A-EE-01-BA	10.5.0.122 R8 "$2"
    ;;
  "123")
    program 00-1B-1A-EE-01-BB	10.5.0.123 R8 "$2"
    ;;
  "124")
    program 00-1B-1A-EE-01-BC	10.5.0.124 R8 "$2"
    ;;
  "125")
    program 00-1B-1A-EE-01-BD	10.5.0.125 R8 "$2"
    ;;
  "126")
    program 00-1B-1A-EE-01-BE	10.5.0.126 R8 "$2"
    ;;
  "127")
    program 00-1B-1A-EE-01-BF	10.5.0.127 R8 "$2"
    ;;
  "128")
    program 00-1B-1A-EE-01-C0	10.5.0.128 R8 "$2"
    ;;
  "129")
    program 00-1B-1A-EE-01-C1	10.5.0.129 R8 "$2"
    ;;
  "130")
    program 00-1B-1A-EE-01-C2	10.5.0.130 R8 "$2"
    ;;
  "131")
    program 00-1B-1A-EE-01-C3	10.5.0.131 R8 "$2"
    ;;
  "132")
    program 00-1B-1A-EE-01-C4	10.5.0.132 R8 "$2"
    ;;
  "133")
    program 00-1B-1A-EE-01-C5	10.5.0.133 R8 "$2"
    ;;
  "134")
    program 00-1B-1A-EE-01-C6	10.5.0.134 R8 "$2"
    ;;
  "135")
    program 00-1B-1A-EE-01-C7	10.5.0.135 R8 "$2"
    ;;
  "136")
    program 00-1B-1A-EE-01-C8	10.5.0.136 R8 "$2"
    ;;
  "137")
    program 00-1B-1A-EE-01-C9	10.5.0.137 R8 "$2"
    ;;
  "138")
    program 00-1B-1A-EE-01-CA	10.5.0.138 R8 "$2"
    ;;
  "139")
    program 00-1B-1A-EE-01-CB	10.5.0.139 R8 "$2"
    ;;
  "140")
    program 00-1B-1A-EE-01-CC	10.5.0.140 R8 "$2"
    ;;
  "141")
    program 00-1B-1A-EE-01-CD	10.5.0.141 R8 "$2"
    ;;
  "142")
    program 00-1B-1A-EE-01-CE	10.5.0.142 R8 "$2"
    ;;
  "143")
    program 00-1B-1A-EE-01-CF	10.5.0.143 R8 "$2"
    ;;
  "144")
    program 00-1B-1A-EE-01-D0	10.5.0.144 R8 "$2"
    ;;
  "145")
    program 00-1B-1A-EE-01-D1	10.5.0.145 R8 "$2"
    ;;
  "146")
    program 00-1B-1A-EE-01-D2	10.5.0.146 R8 "$2"
    ;;
  "147")
    program 00-1B-1A-EE-01-D3	10.5.0.147 R8 "$2"
    ;;
  "148")
    program 00-1B-1A-EE-01-D4	10.5.0.148 R8 "$2"
    ;;
  "149")
    program 00-1B-1A-EE-01-D5	10.5.0.149 R8 "$2"
    ;;
  "150")
    program 00-1B-1A-EE-01-D6	10.5.0.150 R8 "$2"
    ;;
  "151")
    program 00-1B-1A-EE-01-D7	10.5.0.151 R8 "$2"
    ;;
  "152")
    program 00-1B-1A-EE-01-D8	10.5.0.152 R8 "$2"
    ;;
  "153")
    program 00-1B-1A-EE-01-D9	10.5.0.153 R8 "$2"
    ;;
  "154")
    program 00-1B-1A-EE-01-DA	10.5.0.154 R8 "$2"
    ;;
  "155")
    program 00-1B-1A-EE-01-DB	10.5.0.155 R8 "$2"
    ;;
  "156")
    program 00-1B-1A-EE-01-DC	10.5.0.156 R8 "$2"
    ;;
  "157")
    program 00-1B-1A-EE-01-E3 10.5.0.157 F11A "$2"
    ;;
  "158")
    program 00-1B-1A-EE-01-EF 10.5.0.158 F11A "$2"
    ;;
  "159")
    program 00-1B-1A-EE-01-E9 10.5.0.159 F11A "$2"
    ;;
  "160")
    program 00-1B-1A-EE-01-EA 10.5.0.160 F11A "$2"
    ;;
  "161")
    program 00-1B-1A-EE-01-EB 10.5.0.161 F11A "$2"
    ;;
  "162")
    program 00-1B-1A-EE-01-EC 10.5.0.162 F11A "$2"
    ;;
  "163")
    program 00-1B-1A-EE-01-ED 10.5.0.163 F11B "$2"
    ;;
  "164")
    program 00-1B-1A-EE-01-EE 10.5.0.164 F11B "$2"
    ;;
  "165")
    program 00-1B-1A-EE-01-F1 10.5.0.165 F11B "$2"
    ;;
  "166")
    program 00-1B-1A-EE-01-F2 10.5.0.166 R8 "$2"
    ;;
  "167")
    program 00-1B-1A-EE-01-F3 10.5.0.167 R8 "$2"
    ;;
  "168")
    program 00-1B-1A-EE-01-F4 10.5.0.168 R8 "$2"
    ;;
  "169")
    program 00-1B-1A-EE-01-F5 10.5.0.169 R8 "$2"
    ;;
  "170")
    program 00-1B-1A-EE-01-F6 10.5.0.170 R8 "$2"
    ;;
  "171")
    program 00-1B-1A-EE-01-F7 10.5.0.171 R8 "$2"
    ;;
  "172")
    program 00-1B-1A-EE-01-F8 10.5.0.172 R8 "$2"
    ;;
  "173")
    program 00-1B-1A-EE-01-F9 10.5.0.173 R8 "$2"
    ;;
  "174")
    program 00-1B-1A-EE-01-FA 10.5.0.174 R8 "$2"
    ;;
  "175")
    program 00-1B-1A-EE-01-FB 10.5.0.175 R8 "$2"
    ;;
  "176")
    program 00-1B-1A-EE-01-92 10.5.0.176 R8 "$2"
    ;;
  "177")
    program 00-1B-1A-EE-01-93 10.5.0.177 R8 "$2"
    ;;
  "178")
    program 00-1B-1A-EE-01-94 10.5.0.178 R8 "$2"
    ;;
  "000")
    program 00-1B-1A-EE-01-4A 192.168.192.4 R8 "$2"
    ;;
  *)
    echo "Error: no such box: $1"
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
