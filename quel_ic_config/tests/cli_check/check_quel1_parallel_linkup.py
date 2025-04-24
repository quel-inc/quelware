import logging
import os
import re
import subprocess
from typing import Any, Union

logger = logging.getLogger(os.path.basename(__file__))


CASES_QPLU = {
    "nominal": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_quel1only.yaml --check_duration 10",
        "rc": 0,
        "msg": r"(.*)SUCCESS: all the boxes described in in '(.*)' are ready",
    },
    "nominal_v2": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_quel1only_v2.yaml --check_duration 10",
        "rc": 0,
        "msg": r"(.*)SUCCESS: all the boxes described in in '(.*)' are ready",
    },
    "nominal_v2_opt": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_quel1only_v2_with_options.yaml --check_duration 10",
        "rc": 0,
        "msg": r"(.*)SUCCESS: all the boxes described in in '(.*)' are ready",
    },
    "nonexistent": {
        "args": "--conf tests/cli_check/conf_files/nonexistent.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: failed to load '(.*)' due to exception: '(.*)'",
    },
    "empty": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_empty.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: empty config file '(.*)'",
    },
    "broken1": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_errors1.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: broken configuration file '(.*)', quitting",
    },
    "broken2": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_errors2.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: broken configuration file '(.*)', quitting",
    },
    "broken3": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_errors3.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: broken configuration file '(.*)', quitting",
    },
    "broken4": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_errors4.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: broken configuration file '(.*)', quitting",
    },
    "broken5": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_errors5.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: broken configuration file '(.*)', quitting",
    },
    "nobox": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_no_boxes.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: no 'boxes' key is found in the config file '(.*)'",
    },
    "noversion": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_no_version.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: version is not specified in the config file '(.*)'",
    },
    "wrong_version": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_wrong_version.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: wrong version '999' \(!= 1, 2\)",
    },
    "noex_boxes": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_nonexistent_boxes.yaml --check_duration 10",
        "rc": 1,
        "msg": "(.*)FAILED: some boxes described in '(.*)' are unavailable",
    },
    "noex_boxes_igunav": {
        "args": (
            "--conf tests/cli_check/conf_files/quel_ci_env_with_nonexistent_boxes.yaml --check_duration 10 "
            "--ignore_unavailable"
        ),
        "rc": 0,
        "msg": "(.*)SUCCESS: but some boxes described in '(.*)' are unavailable",
    },
    "noex_boxes_only": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_nonexistent_boxes_only.yaml --check_duration 10",
        "rc": 1,
        "msg": "(.*)FAILED: some boxes described in '(.*)' are unavailable",
    },
    "noex_boxes_only_igunav": {
        "args": (
            "--conf tests/cli_check/conf_files/quel_ci_env_with_nonexistent_boxes_only.yaml --check_duration 10 "
            "--ignore_unavailable"
        ),
        "rc": 0,
        "msg": "(.*)SUCCESS: but some boxes described in '(.*)' are unavailable",
    },
    "wrong_boxtype": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_wrong_boxtype.yaml --check_duration 10",
        "rc": 1,
        "msg": "(.*)FAILED: some boxes described in '(.*)' are unavailable",
    },
    "wrong_options1": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_wrong_options1.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: broken configuration file '(.*)', quitting",
    },
    "wrong_options2": {
        "args": "--conf tests/cli_check/conf_files/quel_ci_env_with_wrong_options2.yaml",
        "rc": 1,
        "msg": r"(.*)CANCELED: broken configuration file '(.*)', quitting",
    },
}


def run_tests(cmd: str, cases: dict[str, dict[str, Any]]) -> tuple[int, int]:
    n_success: int = 0
    n_failure: int = 0
    for label, c in cases.items():
        cmdlbl = f"{cmd}/{label}"
        args = [cmd]
        args.extend(c["args"].split())
        logger.debug(f"{cmdlbl:40s}: executing '{' '.join(args)}'")
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rc = p.returncode
        is_success = True
        if rc != c["rc"]:
            logger.error(f"{cmdlbl:40s}: FAILED, unexpected return code: {rc} (!= {c['rc']})")
            is_success = False

        msg0: Union[str, None] = None
        if c["msg"] is not None:
            msg0 = p.stderr.decode().split("\n")[-2]
            if not re.match(c["msg"], msg0):
                logger.error(f"{cmdlbl:40s}: FAILED, unexpected message: '{msg0}'")
                is_success = False

        if is_success:
            if msg0:
                logger.debug(f"{cmdlbl:40s}:           ==> '{msg0}'")
            logger.info(f"{cmdlbl:40s}: SUCCESS")
            n_success += 1
        else:
            n_failure += 1

    return n_success, n_failure


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="[{levelname:.4}]: {message}", style="{")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_qplu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.skip_qplu:
        n_s0, n_f0 = 0, 0
    else:
        n_s0, n_f0 = run_tests("quel1_parallel_linkup", CASES_QPLU)

    logger.info(f"** quel1_parallel_linkup: {n_s0} success, {n_f0} failure")

    if n_f0 == 0:
        logger.info("PASS ALL")
        sys.exit(0)
    else:
        logger.error("FAILED")
        sys.exit(1)
