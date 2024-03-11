#!/bin/bash

set -eu

uribase=https://github.com/quel-inc/quelware/releases/download
version="$(grep __version__ quel_ic_config/__init__.py | awk '{print $3}')"
archivename=quelware_prebuilt.tgz

wget -q "${uribase}/${version}/${archivename}" -O ${archivename} || (echo "ERROR: no prebuilt archive is available for ${version}" && exit 1)
echo "INFO: download completed"
