import argparse
import logging
import pathlib
import threading
import time
from collections.abc import Collection
from concurrent.futures import Future, ThreadPoolExecutor
from ipaddress import IPv4Address
from typing import Any, Dict, Final, List, Set, Tuple, Union, cast

import yaml

from quel_ic_config import Quel1Box, Quel1BoxType, QuelClockMasterV1

logger = logging.getLogger()

NUM_CONNECT_RETRY: Final[int] = 2
NUM_LINKUP_RETRY: Final[int] = 3
DEFAULT_MAX_ALLOWABLE_COUNTERDELTA: Final[int] = 256  # = 2048ns


def validate_collection_of_index(v: Collection[int]):
    is_valid: bool = True
    if isinstance(v, Collection):
        for i in v:
            if not isinstance(i, int) or i < 0:
                is_valid = False
                break
    else:
        is_valid = False
    return is_valid


def validate_options_of_box(box_opt: dict[str, Any]) -> bool:
    is_valid: bool = True
    for k, v in box_opt.items():
        if k in {"ignore_crc_error_of_mxfe", "ignore_access_failure_of_adrf6780", "ignore_lock_failure_of_lmx2594"}:
            if not validate_collection_of_index(v):
                logger.error(f"invalid list of indices: '{v}'")
                is_valid = False
        else:
            logger.error(f"invalid option '{k}'")
            is_valid = False
    return is_valid


def validate_boxes_conf(box_confs) -> bool:
    valid_conf: bool = True
    box_names: Set[str] = set()  # Notes: for checking duplicated box_names
    ipaddrs: Set[IPv4Address] = set()  # Notes: for checking duplicated addresses

    # Notes: validating box_confs
    if not isinstance(box_confs, list):
        logger.error("unexpected format of box descriptions")
        return False

    for idx, spec in enumerate(box_confs):
        if not isinstance(spec, dict):
            logger.error(f"unexpected format of descriptions of the {idx}-th box")
            valid_conf = False
            continue

        if "name" not in spec:
            logger.error(f"no 'name' found at the {idx}-th box")
            valid_conf = False
            continue

        box_name = spec["name"]
        if box_name in box_names:
            logger.error(f"duplicated name '{box_name}' at the {idx}-th box")
            valid_conf = False
            continue
        box_names.add(box_name)

        if "ipaddr" not in spec:
            logger.error(f"no 'ipaddr' found of a box '{box_name}'")
            valid_conf = False
            continue

        try:
            box_ipaddr = IPv4Address(spec["ipaddr"])
        except Exception:
            logger.error(f"invalid value '{spec['ipaddr']}' for 'ipaddr' of a box '{box_name}'")
            valid_conf = False
            continue

        if box_ipaddr in ipaddrs:
            logger.error(f"IP address {box_ipaddr:s} of a box '{box_name}' is duplicated")
            valid_conf = False
            continue

        ipaddrs.add(box_ipaddr)

        if "boxtype" not in spec:
            logger.error(f"no 'boxtype' found of a box '{box_name}'")
            valid_conf = False
            continue

        try:
            _ = Quel1BoxType.fromstr(spec["boxtype"])
        except KeyError:
            logger.error(f"invalid boxtype '{spec['boxtype']}' of a box '{box_name}'")
            valid_conf = False
            continue

        if "options" in spec and not validate_options_of_box(spec["options"]):
            logger.error(f"invalid option '{spec['options']}' of a box '{box_name}'")
            valid_conf = False
            continue

    return valid_conf


def validate_clockmaster_conf(cm_conf) -> bool:
    if not (isinstance(cm_conf, list) and len(cm_conf) == 1 and isinstance(cm_conf[0], dict)):
        logger.error("unexpected format of clockmaster descriptions")
        return False

    if "ipaddr" not in cm_conf[0]:
        logger.error("no 'ipaddr' found of clock master")
        return False

    return True


# Notes: no leakage of exceptions is allowed. this function is submitted to pool.
#        all the possible exceptions are handled within function for better diagnosis.
def load_box(box_conf: dict[str, Any]) -> tuple[str, Union[Quel1Box, str]]:
    box_name = box_conf["name"]
    threading.current_thread().name = box_name  # Notes: thread name is printed in log messages.

    box_ipaddr = IPv4Address(box_conf["ipaddr"])
    box_type = Quel1BoxType.fromstr(box_conf["boxtype"])
    box_opt: dict[str, Any] = box_conf.get("options", {})

    exception: str = ""
    for _ in range(NUM_CONNECT_RETRY):
        try:
            retval: Union[Quel1Box, str] = Quel1Box.create(
                ipaddr_wss=str(box_ipaddr),
                ipaddr_sss=str(box_ipaddr + 0x010000),
                ipaddr_css=str(box_ipaddr + 0x040000),
                boxtype=box_type,
                ignore_crc_error_of_mxfe=box_opt.get("ignore_crc_error_of_mxfe", set()),
                ignore_access_failure_of_adrf6780=box_opt.get("ignore_access_failure_of_adrf6780", set()),
                ignore_lock_failure_of_lmx2594=box_opt.get("ignore_lock_failure_of_lmx2594", set()),
            )
            logger.info(f"connected to {box_name} successfully")
            break
        except Exception as e:
            exception = str(e.args[0])
            logger.warning(f"failed to connect to {box_name} due to exception: '{e}'")
    else:
        retval = exception
        logger.error(f"give up to connect to {box_name} due to repeated failures")

    return box_name, retval


def load_conf_v1(conf, filename: pathlib.Path) -> tuple[bool, list[dict[str, str]]]:
    if len(set(conf.keys()) - {"version", "boxes"}) != 0:
        logger.error(f"CANCELED: unexpected keys are found in the config file '{filename}'")

    if "boxes" not in conf:
        logger.error(f"CANCELED: no 'boxes' key is found in the config file '{filename}'")
        return False, []

    if not validate_boxes_conf(conf["boxes"]):
        logger.error(f"CANCELED: broken configuration file '{filename}', quitting")
        return False, []

    return True, conf["boxes"]


def load_conf_v2(conf, filename: pathlib.Path) -> tuple[bool, list[dict[str, str]], dict[str, str]]:
    if len(set(conf.keys()) - {"version", "clockmaster", "boxes"}) != 0:
        logger.error(f"CANCELED: unexpected keys are found in the config file '{filename}'")

    if "clockmaster" not in conf:
        logger.error(f"CANCELED: no 'clockmaster' key is found in the config file '{filename}'")
        return False, [], {}

    if "boxes" not in conf:
        logger.error(f"CANCELED: no 'boxes' key is found in the config file '{filename}'")
        return False, [], {}

    if not (validate_boxes_conf(conf["boxes"]) and validate_clockmaster_conf(conf["clockmaster"])):
        logger.error(f"CANCELED: broken configuration file '{filename}', quitting")
        return False, [], {}

    return True, conf["boxes"], conf["clockmaster"][0]


def load_conf(filename: pathlib.Path) -> Tuple[bool, int, list[dict[str, str]], dict[str, str]]:
    try:
        with open(filename) as f:
            conf = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"CANCELED: failed to load '{filename}' due to exception: '{e}'")
        return False, 0, [], {}

    if conf is None:
        logger.error(f"CANCELED: empty config file '{filename}'")
        return False, 0, [], {}

    if "version" not in conf:
        logger.error(f"CANCELED: version is not specified in the config file '{filename}'")
        return False, 0, [], {}

    version: int = conf["version"]
    if version == 1:
        validity, boxes = load_conf_v1(conf, filename)
        return validity, version, boxes, {}
    elif version == 2:
        validity, boxes, cm = load_conf_v2(conf, filename)
        return validity, version, boxes, cm
    else:
        logger.error(f"CANCELED: wrong version '{version}' (!= 1, 2)")
        return False, version, [], {}


def check_link_validity(name: str, box: Quel1Box, args: argparse.Namespace) -> bool:
    status = box.reconnect(
        background_noise_threshold=args.background_noise_threshold,
        ignore_crc_error_of_mxfe=box.css.get_all_mxfes(),
        ignore_invalid_linkstatus=True,
    )
    if not all(status.values()):
        logger.warning("no valid link status, it is subject to be linked up")
        return False
    return True


def get_crc_error_count(box: Quel1Box) -> Dict[int, List[int]]:
    cntr: Dict[int, List[int]] = {}
    for mxfe_idx in box.css.get_all_mxfes():
        cntr[mxfe_idx] = box.css.get_crc_error_counts(mxfe_idx)
    return cntr


def diff_crc_error_count(
    cnts0: Dict[int, List[int]], cnts1: Dict[int, List[int]], ignore_crc_error_of_mxfe: Collection[int]
) -> bool:
    for mxfe_idx in cnts0:
        for lane_idx, cnt in enumerate(cnts0[mxfe_idx]):
            if cnt != cnts1[mxfe_idx][lane_idx]:
                if mxfe_idx in ignore_crc_error_of_mxfe:
                    logger.warning(
                        f"a new crc error is detected at the {lane_idx}-th lane of the mxfe-#{mxfe_idx}: "
                        f"{cnt} -> {cnts1[mxfe_idx][lane_idx]}, but ignore it."
                    )
                    return True
                else:
                    logger.warning(
                        f"a new crc error is detected at the {lane_idx}-th lane of the mxfe-#{mxfe_idx}: "
                        f"{cnt} -> {cnts1[mxfe_idx][lane_idx]}"
                    )
                    return False

    return True


def check_crc_error(name: str, box: Quel1Box, duration: int) -> bool:
    # Notes: all the mxfe should be linked up in advance.
    step = 0.5
    t = t0 = time.perf_counter()
    e0 = get_crc_error_count(box)
    while t < t0 + duration:
        time.sleep(step)
        t = time.perf_counter()
        e1 = get_crc_error_count(box)
        if not diff_crc_error_count(e0, e1, box.options["ignore_crc_error_of_mxfe"]):
            return False
        step = min(10.0, step * 2)

    # Notes: clear CRC error flag if no CRC error is detected.
    for mxfe_idx in box.css.get_all_mxfes():
        box.css.clear_crc_error(mxfe_idx)
    return True


# Notes: no leakage of exceptions is allowed. this function is submitted to pool.
#        all the possible exceptions are handled within function for better diagnosis.
def linkup_a_box(name: str, box: Quel1Box, args: argparse.Namespace) -> Union[bool, None]:
    threading.current_thread().name = name  # Notes: thread name is printed in log messages.

    if args.force:
        total_status = False
    else:
        try:
            total_status = check_link_validity(name, box, args)
        except Exception as e:
            logger.error(f"abort linking up due to {e}")
            return None

    for retry_idx in range(NUM_LINKUP_RETRY):
        if not total_status:
            # Notes: no log messages are shown during the link-up process which can last 30 seconds or so.
            logger.info("link-up is in progress")
            try:
                status = box.relinkup(
                    hard_reset_wss=args.hard_reset_wss,
                    background_noise_threshold=args.background_noise_threshold,
                )
                total_status = all(status.values())
                if not total_status:
                    continue
            except Exception as e:
                logger.error(e)
                # Notes: allow to retry anyway.
                continue

        # Notes: here total_status must be True
        logger.info(f"observing the link stability for {args.check_duration} seconds")
        try:
            if check_crc_error(name, box, args.check_duration):
                logger.info("no CRC error is detected")
                break
            else:
                total_status = False
        except Exception as e:
            logger.error(e)
            # Notes: total_status is not changed because no new information is available.
            continue
    else:
        return False

    return True


# Notes: no leakage of exceptions is allowed. this function is submitted to pool.
#        all the possible exceptions are handled within function for better diagnosis.
def reconnect_a_box(name: str, box: Quel1Box, args: argparse.Namespace) -> Union[bool, None]:
    threading.current_thread().name = name  # Notes: thread name is printed in log messages.

    try:
        total_status = check_link_validity(name, box, args)  # Notes: don't care of CRC error flags.
        if not total_status:
            logger.info("link is NOT ready")
    except Exception as e:
        logger.error(f"abort reconnecting due to {e}")
        return None

    return total_status


def _suppressing_noisy_loggers() -> None:
    logging.getLogger("quel_ic_config.ad9082").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.lmx2594").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1_config_subsystem_common").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1_config_subsystem_tempctrl").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1_config_loader").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.exstickge_coap_client").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1se_config_subsystem").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1se_riken8_config_subsystem").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1se_fujitsu11_config_subsystem").setLevel(logging.WARNING)
    logging.getLogger("e7awghal.versionchecker").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.exstickge_sock_client").disabled = True
    logging.getLogger("parallel_linkup").disabled = True
    logging.getLogger("quel1_parallel_linkup").disabled = True
    logging.getLogger("coap").disabled = True
    logging.getLogger("flufl.lock").disabled = True


def parallel_linkup_main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: ({threadName}) {message}", style="{"
    )
    parser = argparse.ArgumentParser(description="parallelized link-up of QuEL-1 series control box")
    parser.add_argument("--conf", type=pathlib.Path, required=True, help="path to a configuration file")
    parser.add_argument("--ignore_unavailable", action="store_true", help="just ignoring unavailable boxes")
    parser.add_argument("--force", action="store_true", help="force to re-linkup all the boxes")
    parser.add_argument(
        "--check_duration", type=int, default=120, help="duration in seconds for checking the stability of the link"
    )
    parser.add_argument(
        "--background_noise_threshold",
        type=float,
        default=None,
        help="maximum allowable background noise amplitude of the ADCs at relinkup and at reconnect",
    )
    parser.add_argument(
        "--hard_reset_wss",
        action="store_true",
        default=False,
        help="hard reset AWG units and CAP units to clear hardware error flags, should avoid using it when unnecessary",
    )
    parser.add_argument("--verbose", action="store_true", help="show verbose log")
    args = parser.parse_args()

    if not args.verbose:
        _suppressing_noisy_loggers()

    # Notes: phase-1 loading configurations of boxes
    validity, version, box_confs, cm_conf = load_conf(pathlib.Path(args.conf))
    if not validity:
        return 1

    # Notes: phase-2 creating box objects
    pool = ThreadPoolExecutor(max_workers=len(box_confs))

    futures_box: list[Future[tuple[str, Union[Quel1Box, str]]]] = [
        pool.submit(load_box, box_conf) for box_conf in box_confs
    ]
    boxes: dict[str, Union[Quel1Box, str]] = {}
    for future_box in futures_box:
        box_name, box_obj = future_box.result()
        boxes[box_name] = box_obj

    # Notes: phase-3 linking up the boxes
    futures_status: Dict[str, Union[Future[Union[bool, None]], None]] = {}
    for name, obj in boxes.items():
        if isinstance(obj, Quel1Box):
            futures_status[name] = pool.submit(linkup_a_box, name, obj, args)
        else:
            # Notes: to show the final results in the same order as the given YAML file.
            futures_status[name] = None

    # Notes: taking a barrier sync to show the results after finishing all the threads
    num_failed: int = 0
    num_unavailable: int = 0
    results: Dict[str, str] = {}
    for name, future in futures_status.items():
        if future is not None:
            if future.result():
                results[name] = "ready"
            else:
                results[name] = "failed"
                num_failed += 1
        else:
            results[name] = cast(str, boxes[name])  # Notes: the content of boxes[name] is definitely str.
            num_unavailable += 1

    # Notes: phase-4 showing the results
    print("----------")
    for name, result in results.items():
        if result == "ready":
            print(f"{name} is ready to use")
        elif result == "failed":
            print(f"{name} fails to link up")
        else:
            print(f"{name} is unavailable due to '{result}'")
    print("----------")

    del boxes

    is_success: bool = False
    if num_failed == 0:
        if num_unavailable == 0:
            is_success = True
            logger.info(f"SUCCESS: all the boxes described in in '{args.conf}' are ready")
        elif args.ignore_unavailable:
            is_success = True
            logger.info(f"SUCCESS: but some boxes described in '{args.conf}' are unavailable")
        else:
            logger.error(f"FAILED: some boxes described in '{args.conf}' are unavailable")
    else:
        logger.error(f"FAILED: some boxes described in '{args.conf}' are not ready")

    return 0 if is_success else 1


def _calc_counter_delta(status: dict[str, tuple[bool, bool, int]]) -> tuple[bool, int, int, bool]:
    crl = []
    for name, (av, ls, tc) in status.items():
        if av and ls:
            cr = tc % 2000
            crl.append(cr)

    if len(crl) == 0:
        return False, 0, 0, False

    shifted: bool = False
    mincr = min(crl)
    maxcr = max(crl)
    if maxcr - mincr >= 1000:
        crl = [(cr - 1000) % 2000 + 1000 for cr in crl]
        mincr = min(crl)
        maxcr = max(crl)
        if maxcr - mincr < 1000:
            shifted = True
    return True, int(maxcr - mincr), mincr, shifted


def _print_counter(status: dict[str, tuple[bool, bool, int]], min_cntr: int, shifted: bool):
    print(f"base_sysref_offset: {min_cntr}")
    print()

    for name, (av, ls, tc) in status.items():
        if av and ls:
            if shifted:
                cr = (tc - 1000) % 2000 + 1000
            else:
                cr = tc % 2000
            print(f"{name}: {cr - min_cntr}")

    for name, (av, ls, tc) in status.items():
        if not av:
            print(f"{name}: not available, ignored")
        elif not ls:
            print(f"{name}: not ready to use, ignored")


def _print_sync_judge(delta: int, max_delta: int) -> bool:
    if delta >= 1000:
        logger.error(f"FAILED SYNCHRONIZATION, delta of the time counters is over 999 counts (> {max_delta:d} counts)")
    else:
        if delta <= max_delta:
            logger.info(
                "SUCCESSFUL SYNCHRONIZATION, "
                f"delta of the time counters is {delta:d} counts (<= {max_delta:d} counts)"
            )
            return True
        else:
            logger.error(
                f"FAILED SYNCHRONIZATION, delta of the time counters is {delta:d} counts (> {max_delta:d} counts)"
            )

    return False


def _sync_main(check_only: bool) -> int:
    logging.basicConfig(
        level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: ({threadName}) {message}", style="{"
    )
    parser = argparse.ArgumentParser(description="parallelized link-up of QuEL-1 series control box")
    parser.add_argument("--conf", type=pathlib.Path, required=True, help="path to a configuration file")
    parser.add_argument(
        "--background_noise_threshold",
        type=float,
        default=None,
        help="maximum allowable background noise amplitude of the ADCs at relinkup and at reconnect",
    )
    parser.add_argument("--ignore_unavailable", action="store_true", help="just ignore unavailable boxes exist")
    parser.add_argument(
        "--max_delta",
        type=int,
        default=DEFAULT_MAX_ALLOWABLE_COUNTERDELTA,
        help="worst allowable synchronization error",
    )
    parser.add_argument("--force", action="store_true", help="force to re-linkup all the boxes")
    parser.add_argument("--verbose", action="store_true", help="show verbose log")
    args = parser.parse_args()

    if not args.verbose:
        _suppressing_noisy_loggers()

    # Notes: phase-1 loading configurations of boxes
    validity, version, box_confs, cm_conf = load_conf(pathlib.Path(args.conf))
    if not validity:
        return 1

    # Notes: phase-2 creating box objects
    pool = ThreadPoolExecutor(max_workers=len(box_confs))

    futures_box: list[Future[tuple[str, Union[Quel1Box, str]]]] = [
        pool.submit(load_box, box_conf) for box_conf in box_confs
    ]
    boxes: dict[str, Union[Quel1Box, str]] = {}
    for future_box in futures_box:
        box_name, box_obj = future_box.result()
        boxes[box_name] = box_obj

    clock_master: Union[QuelClockMasterV1, None] = None
    if not check_only:
        if version >= 2:
            clock_master = QuelClockMasterV1(
                ipaddr=cm_conf["ipaddr"], boxes=[b for b in boxes.values() if isinstance(b, Quel1Box)]
            )
            if not clock_master.check_availability():
                logger.error(
                    f"CANCELED SYNCHRONIZATION due to the unavailability of the clockmaster {cm_conf['ipaddr']}"
                )
                return 1
        else:
            logger.error(
                f"CANCELED SYNCHRONIZATION because no clockmaster is specified in the config file '{args.conf}'"
            )
            return 1

    # Notes: phase-3 reconnecting to the boxes
    futures_status: Dict[str, Union[Future[Union[bool, None]], None]] = {}
    for name, obj in boxes.items():
        if isinstance(obj, Quel1Box):
            futures_status[name] = pool.submit(reconnect_a_box, name, obj, args)
        else:
            # Notes: to show the final results in the same order as the given YAML file.
            futures_status[name] = None

    # Notes: taking a barrier sync to show the results after finishing all the threads
    status: Dict[str, Tuple[bool, bool, int]] = {}
    ignored_box: list[str] = []
    for name, future in futures_status.items():
        if future is not None:
            box = cast(Quel1Box, boxes[name])
            if future.result():
                status[name] = True, True, int(box.get_latest_sysref_timecounter())
            else:
                status[name] = True, False, int(box.get_latest_sysref_timecounter())
                ignored_box.append(name)
        else:
            status[name] = False, False, 0
            ignored_box.append(name)

    # Notes: phase-4 showing the results
    if not args.ignore_unavailable and len(ignored_box) > 0:
        logger.error(f"CANCELED SYNCHRONIZATION because some boxes are not ready to use: {', '.join(ignored_box)}")
        return 1

    valid, delta, min_cntr, shifted = _calc_counter_delta(status)
    if not valid:
        logger.error("CANCELED SYNCHRONIZATION due to no available boxes")
        return 1

    # Notes: phase-5 kick
    if not check_only:
        if delta <= args.max_delta and not args.force:
            logger.info("all the boxes are synchronized well, skipping resynchronization")
        else:
            assert clock_master is not None  # TODO: replace assert with an appropriate mypy feature in future
            logger.info("conducting resynchronization")
            clock_master.sync_boxes()

            for name, (av, ls, tc) in status.items():
                if av:
                    box = cast(Quel1Box, boxes[name])
                    status[name] = (av, ls, int(box.get_latest_sysref_timecounter()))
            valid, delta, min_cntr, shifted = _calc_counter_delta(status)

    # Notes: phase-6 reporting
    print("----------")
    _print_counter(status, min_cntr, shifted)
    print("----------")
    sync_judge = _print_sync_judge(delta, args.max_delta)

    del boxes
    return 0 if sync_judge else 1


def syncstatus_main() -> int:
    return _sync_main(check_only=True)


def sync_main() -> int:
    return _sync_main(check_only=False)


if __name__ == "__main__":
    import sys

    sys.exit(parallel_linkup_main())
