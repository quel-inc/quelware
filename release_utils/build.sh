#!/bin/bash

set -eu

script_dir="$(dirname "$(readlink -f "$0")")"
repository_root=$script_dir/..

wheels_dirname="wheels"

build_targets=(
quel_ic_config
quel_inst_tool
quel_pyxsdb
quel_staging_tool
quel_cmod_scripting
quel_clock_master
)


requirements_simplemulti_classic=(
quel_ic_config
quel_clock_master
) # + e7awgsw from fixtures dir

requirements_simplemulti_standard=(
quel_ic_config
quel_clock_master
) # + e7awgsw from fixtures dir

requirements_feedback=(
quel_ic_config
quel_clock_master
) # + e7awgsw from fixtures dir

requirements_dev=(
quel_ic_config
quel_inst_tool
quel_pyxsdb
quel_staging_tool
quel_cmod_scripting
)

# building all the packages
for target in "${build_targets[@]}"; do
  echo "### ${target}"
  pushd "${repository_root}/${target}" >> /dev/null || exit 1
  rm -rf dist
  python -m build
  popd >> /dev/null || exit 1
done

rm -rf "${script_dir:?}/${wheels_dirname}"
mkdir "${script_dir}/${wheels_dirname}"

cp -r "${script_dir}/prebuilt_fixtures/"* "${script_dir}/${wheels_dirname}/"

for target in "${build_targets[@]}"; do
  cp "$(ls "${repository_root}/${target}/dist/"*.whl)" "${script_dir}/${wheels_dirname}/"
done

# creating requirements*.txt
pushd "${script_dir}" >> /dev/null || exit 1

rm -f requirements*.txt

for pkg_name in "${requirements_simplemulti_classic[@]}"; do
  ls -1 "${wheels_dirname}/${pkg_name}"*.whl >> requirements_simplemulti_classic.txt
done
ls -1 "${wheels_dirname}/simplemulti_classic"/* >> requirements_simplemulti_classic.txt

for pkg_name in "${requirements_simplemulti_standard[@]}"; do
  ls -1 "${wheels_dirname}/${pkg_name}"*.whl >> requirements_simplemulti_standard.txt
done
ls -1 "${wheels_dirname}/simplemulti_standard"/* >> requirements_simplemulti_standard.txt

for pkg_name in "${requirements_feedback[@]}"; do
  ls -1 "${wheels_dirname}/${pkg_name}"*.whl >> requirements_feedback.txt
done
ls -1 "${wheels_dirname}/feedback"/* >> requirements_feedback.txt

for pkg_name in "${requirements_dev[@]}"; do
  ls -1 "${wheels_dirname}/${pkg_name}"*.whl >> requirements_dev.txt
done

# creating a tarball
tar zcfv quelware_prebuilt.tgz ${wheels_dirname} requirements_simplemulti_classic.txt requirements_simplemulti_standard.txt requirements_feedback.txt requirements_dev.txt

# finalize
popd >> /dev/null || exit 1
exit 0
