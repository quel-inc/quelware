#!/bin/bash

set -eu

uribase=https://github.com/quel-inc/quelware/releases/download
version="$(grep __version__ quel_ic_config/__init__.py | awk '{print $3}' | sed 's/\"//g')"
archivename=quelware_prebuilt.tgz

echo "INFO: downloading ${uribase}/${version}/${archivename} ..."
wget -q "${uribase}/${version}/${archivename}" -O ${archivename} || (echo "ERROR: no prebuilt archive is available for ${version}" && exit 1)
echo "INFO: download completed"
