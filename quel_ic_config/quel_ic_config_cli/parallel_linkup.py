import argparse
import logging
import pathlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from ipaddress import IPv4Address
from typing import Dict, Final, List, Set, Tuple, Union

import yaml

from quel_ic_config import Quel1Box, Quel1BoxType

logger = logging.getLogger()

NUM_CONNECT_RETRY: Final[int] = 2
NUM_LINKUP_RETRY: Final[int] = 3


def load_conf(filename: pathlib.Path) -> Tuple[bool, Dict[str, Union[None, Quel1Box]]]:
    objs: Dict[str, Union[Quel1Box, None]] = {}

    try:
        with open(filename) as f:
            conf = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"failed to load '{filename}' due to exception: '{e}'")
        return False, objs

    if conf is None:
        logger.error(f"empty file '{filename}'")
        return False, objs

    if "version" not in conf:
        logger.error(f"version is not specified in '{filename}'")
        return False, objs

    version: int = conf["version"]
    if version != 1:
        logger.error(f"wrong version '{version}' (!= 1)")
        return False, objs

    if "boxes" not in conf:
        logger.error(f"no 'boxes' key is found in the config file '{filename}'")
        return False, objs

    specs = conf["boxes"]
    valid_conf: bool = True
    box_names: Set[str] = set()  # Notes: for checking duplicated box_names
    ipaddrs: Set[IPv4Address] = set()  # Notes: for checking duplicated addresses
    for idx, spec in enumerate(specs):
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
            box_type = Quel1BoxType.fromstr(spec["boxtype"])
        except KeyError:
            logger.error(f"invalid boxtype '{spec['boxtype']}' of a box '{box_name}'")
            valid_conf = False
            continue

    if not valid_conf:
        logger.error(f"broken configuration file '{filename}', quitting")
        return False, objs

    for idx, spec in enumerate(specs):
        box_name = spec["name"]
        box_ipaddr = IPv4Address(spec["ipaddr"])
        box_type = Quel1BoxType.fromstr(spec["boxtype"])
        for _ in range(NUM_CONNECT_RETRY):
            try:
                objs[box_name] = Quel1Box.create(
                    ipaddr_wss=str(box_ipaddr),
                    ipaddr_sss=str(box_ipaddr + 0x010000),
                    ipaddr_css=str(box_ipaddr + 0x040000),
                    boxtype=box_type,
                )
                break
            except Exception as e:
                logger.warning(f"failed to connect to {box_name} due to exception: '{e}'")
        else:
            objs[box_name] = None
            logger.error(f"give up to connect to {box_name} due to repeated failures")

    return True, objs


def check_established(name: str, box: Quel1Box) -> bool:
    status = box.reconnect(ignore_crc_error_of_mxfe=box.css.get_all_mxfes())
    if not all(status.values()):
        logger.warning("no valid link status, it is subject to be linked up")
        return False
    return True


def get_crc_error_count(box: Quel1Box) -> Dict[int, List[int]]:
    cntr: Dict[int, List[int]] = {}
    for mxfe_idx in box.css.get_all_mxfes():
        cntr[mxfe_idx] = box.css.ad9082[mxfe_idx].get_crc_error_counts()
    return cntr


def diff_crc_error_count(cnts0: Dict[int, List[int]], cnts1: Dict[int, List[int]]) -> bool:
    for mxfe_idx in cnts0:
        for lane_idx, cnt in enumerate(cnts0[mxfe_idx]):
            if cnt != cnts1[mxfe_idx][lane_idx]:
                logger.warning(
                    f"a new crc error is detected at the {lane_idx}-th lane of the mxfe-#{mxfe_idx}: "
                    f"{cnt} -> {cnts1[mxfe_idx][lane_idx]}"
                )
                return False
    return True


def check_link_quality(name: str, box: Quel1Box, duration: int) -> bool:
    # Notes: all the mxfe should be established link

    step = 0.5
    t = t0 = time.perf_counter()
    e0 = get_crc_error_count(box)
    while t < t0 + duration:
        time.sleep(step)
        t = time.perf_counter()
        e1 = get_crc_error_count(box)
        if not diff_crc_error_count(e0, e1):
            return False
        step = min(10.0, step * 2)

    return True


def linkup_a_box(name: str, box: Quel1Box, args) -> bool:
    threading.current_thread().name = name

    use_bgcal: bool = True
    if args.nouse_bgcal:
        if args.use_bgcal:
            raise ValueError("it is not allowed to specify both --use_bgcal and --nouse_bgcal at the same time")
        else:
            use_bgcal = False

    not_bad_link: bool = True
    if args.force:
        # Notes: force to execute link-up at the first attempt even if the link state is fine
        not_bad_link = False

    for retry_idx in range(NUM_LINKUP_RETRY):
        if not check_established(name, box) or not not_bad_link:
            logger.info("link-up is in progress")
            not_bad_link = True
            try:
                status = box.relinkup(use_204b=False, use_bg_cal=use_bgcal)
                if not all(status.values()):
                    continue
            except Exception as e:
                logger.error(e)
                # Notes: allow to retry anyway.
                continue

        logger.info(f"observing the link stability for {args.check_duration} seconds")
        not_bad_link = check_link_quality(name, box, args.check_duration)
        if not_bad_link:
            logger.info("no CRC error is detected")
            break
    else:
        return False

    return True


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: ({threadName}) {message}", style="{"
    )
    logging.getLogger("quel_ic_config.ad9082_v106").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.lmx2594").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1_config_subsystem_common").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1_config_subsystem_tempctrl").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.exstickge_coap_client").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.quel1se_riken8_config_subsystem").setLevel(logging.WARNING)
    logging.getLogger("quel_ic_config.e7workaround").setLevel(logging.WARNING)
    # Notes: workaround for preventing e7awgsw from generating unnecessary logs
    logging.getLogger("nullLibLog").disabled = True
    logging.getLogger("parallel_linkup").disabled = True
    logging.getLogger("quel1_parallel_linkup").disabled = True

    parser = argparse.ArgumentParser(description="parallelized link-up of QuEL-1 series control box")
    parser.add_argument("--conf", type=pathlib.Path, required=True, help="path to a configuration file")
    parser.add_argument("--use_bgcal", action="store_true", help="activate background calibration (default)")
    parser.add_argument(
        "--nouse_bgcal", action="store_true", help="deactivate background calibration (not recommended)"
    )
    parser.add_argument("--force", action="store_true", help="force to re-linkup all the boxes")
    parser.add_argument(
        "--check_duration", type=int, default=120, help="duration in seconds for checking the stability of the link"
    )
    args = parser.parse_args()

    validity, boxes = load_conf(pathlib.Path(args.conf))
    if not validity:
        return 1

    pool = ThreadPoolExecutor(max_workers=len(boxes))
    futures: Dict[str, Union[Future[bool], None]] = {}
    for name, box in boxes.items():
        if box is not None:
            futures[name] = pool.submit(linkup_a_box, name, box, args)
        else:
            # Notes: to show the final results in the same order as the given YAML file.
            futures[name] = None

    # Notes: taking a barrier sync to show the results after finishing all the threads
    num_failed: int = 0
    results: Dict[str, str] = {}
    for name, future in futures.items():
        if future is not None:
            if future.result():
                results[name] = "ready"
            else:
                results[name] = "failed"
                num_failed += 1
        else:
            results[name] = "unavailable"
            num_failed += 1

    print("----------")
    for name, result in results.items():
        if result == "ready":
            print(f"{name} is ready to use")
        elif result == "failed":
            print(f"{name} fails to link up")
        else:
            print(f"{name} looks unavailable")

    return 0 if num_failed == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
