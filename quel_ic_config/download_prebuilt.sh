#!/bin/bash

set -eu

uribase=https://github.com/quel-inc/quelware/releases/download
version="$(grep ^version pyproject.toml | awk '{print $3}' | sed 's/\"//g')"
archivename=quelware_prebuilt.tgz

tag_with_v="v${version}"
tag_without_v="${version}"

echo "INFO: downloading ${uribase}/${tag_with_v}/${archivename} ..."
if ! wget -q "${uribase}/${tag_with_v}/${archivename}" -O "${archivename}"; then
    echo "INFO: ${uribase}/${tag_with_v}/${archivename} not found."
    echo "INFO: downloading ${uribase}/${tag_without_v}/${archivename} ..."
    if ! wget -q "${uribase}/${tag_without_v}/${archivename}" -O "${archivename}"; then
        echo "INFO: ${uribase}/${tag_without_v}/${archivename} not found."
        echo "ERROR: no prebuilt archive is available for version ${version}"
        exit 1
    fi
fi

echo "INFO: download completed"
