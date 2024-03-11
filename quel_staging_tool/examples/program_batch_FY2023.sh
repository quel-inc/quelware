#!/bin/bash

set -eu

adapter=210249B87F82

# 00-1B-1A-EE-01-97	10.5.0.87
#                :          :
#                :          :
# 00-1B-1A-EE-01-C8	10.5.0.136

macaddr="00-1B-1A-EE-01-${1}"
last_octet=$(( 0x${1} - 64 ))
ipaddr="10.5.0.${last_octet}"

echo INFO: programming  "${ipaddr}"  "${macaddr}"
quel_program_exstickge_1se --ipaddr "${ipaddr}" --macaddr "${macaddr}" --port 5121 --firmware_dir ./firmware --firmware dev --adapter "${adapter}"  --verbose

sleep 10
echo "INFO: fpga version:  $(coap-client -m get coap://${ipaddr}/version/fpga)"
actual_macaddr="$(arp -an $ipaddr | awk '{ print $4 }' | cut --delimiter=: -f 6 -)"
echo "INFO: the last octet of the detected macaddr: ${actual_macaddr}"
if [ "$((0x${actual_macaddr}))" == "$((0x${1}))" ]; then
    echo "INFO: *** success ***"
else
    echo "ERROR: *** failed ***  ${actual_macaddr} != ${macaddr}"
fi
