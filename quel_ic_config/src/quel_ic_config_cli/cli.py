import argparse
import logging
import socket
import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple

from quel_ic_config import LinkupFpgaMxfe, LinkupStatus, Quel1Box, Quel1BoxIntrinsic, Quel1seTempctrlState
from quel_ic_config_utils.common_arguments import (
    add_common_arguments,
    add_common_workaround_arguments,
    complete_ipaddrs,
)

logger = logging.getLogger()


def quel1_linkup() -> None:
    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="making a QuEL-1 device ready to use. "
        "be aware that all the device configurations will be reset completely."
    )
    add_common_arguments(parser, use_mxfe=True, allow_implicit_mxfe=True)
    add_common_workaround_arguments(
        parser,
        use_ignore_crc_error_of_mxfe=True,
        use_ignore_access_failure_of_adrf6780=True,
        use_ignore_lock_failure_of_lmx2594=True,
    )
    parser.add_argument(
        "--background_noise_threshold",
        type=float,
        default=None,
        help="maximum allowable background noise amplitude of the ADCs",
    )
    parser.add_argument(
        "--use_204c",
        action="store_true",
        help="enable JESD204C link calibration instead of the conventional 204B one (default)",
    )
    parser.add_argument(
        "--use_204b",
        action="store_true",
        help="dare to use the conventional JESD204B link calibration instead of the 204C one",
    )
    parser.add_argument(
        "--use_bgcal",
        action="store_true",
        help="enable background calibration of JESD204C link (default)",
    )
    parser.add_argument(
        "--nouse_bgcal",
        action="store_true",
        help="disable background calibration of JESD204C link",
    )
    parser.add_argument(
        "--restart_tempctrl",
        action="store_true",
        help="restart temperature control to update setpoints",
    )
    parser.add_argument(
        "--skip_init",
        action="store_true",
        default=False,
        help="skip initialization of ICs other than AD9082",
    )
    parser.add_argument(
        "--hard_reset_wss",
        action="store_true",
        default=False,
        help="hard reset AWG units and CAP units to clear hardware error flags, should avoid using it when unnecessary",
    )
    parser.add_argument(
        "--save_dirpath",
        type=Path,
        default=None,
        help="path to directory to save captured wave data during link-up",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show verbose log",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        retcode = quel1_linkup_body(args)
        sys.exit(retcode)
    except AssertionError as e:
        if args.verbose:
            logger.exception(
                "internal error, please contact customer support with how to reproduce this issue", exc_info=e
            )
        else:
            logger.error("internal error, please contact customer support with how to reproduce this issue")
    except Exception as e:
        if args.verbose:
            logger.exception(e, exc_info=e)
        else:
            logger.error(e)
        sys.exit(-1)


def quel1_linkup_body(args: argparse.Namespace) -> int:
    mxfe_list: Tuple[int, ...] = args.mxfe

    try:
        box = Quel1Box.create(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
            ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
            ignore_access_failure_of_adrf6780=args.ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=args.ignore_lock_failure_of_lmx2594,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    use_204b: bool = False
    if args.use_204b:
        if args.use_204c:
            raise ValueError("it is not allowed to specify both --use_204b and --use_204c at the same time")
        else:
            use_204b = True

    use_bgcal: bool = True
    if args.nouse_bgcal:
        if args.use_bgcal:
            raise ValueError("it is not allowed to specify both --use_bgcal and --nouse_bgcal at the same time")
        else:
            use_bgcal = False

    cli_retcode: int = 0
    linkup_ok: Dict[int, bool] = box.relinkup(
        mxfes_to_linkup=mxfe_list,
        config_root=args.config_root,
        config_options=args.config_options,
        hard_reset=args.boxtype.is_quel1se(),
        use_204b=use_204b,
        use_bg_cal=use_bgcal,
        restart_tempctrl=args.restart_tempctrl,
        skip_init=args.skip_init,
        hard_reset_wss=args.hard_reset_wss,
        background_noise_threshold=args.background_noise_threshold,
    )

    for mxfe_idx in mxfe_list:
        if linkup_ok[mxfe_idx]:
            print(f"ad9082-#{mxfe_idx} linked up successfully")
        else:
            logger.error(f"ad9082-#{mxfe_idx} failed to link up")
            cli_retcode = -1

    return cli_retcode


def quel1_linkstatus() -> None:
    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(description="show the link status of AD9082s")
    add_common_arguments(
        parser, use_mxfe=True, allow_implicit_mxfe=True, use_config_root=False, use_config_options=False
    )
    add_common_workaround_arguments(parser, use_ignore_crc_error_of_mxfe=True)
    parser.add_argument(
        "--background_noise_threshold",
        type=float,
        default=None,
        help="maximum allowable background noise amplitude of the ADCs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show verbose log",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        retcode = quel1_linkstatus_body(args)
        sys.exit(retcode)
    except AssertionError as e:
        if args.verbose:
            logger.exception(
                "internal error, please contact customer support with how to reproduce this issue", exc_info=e
            )
        else:
            logger.error("internal error, please contact customer support with how to reproduce this issue")
    except Exception as e:
        if args.verbose:
            logger.exception(e, exc_info=e)
        else:
            logger.error(e)
        sys.exit(-1)


def quel1_linkstatus_body(args: argparse.Namespace) -> int:
    mxfe_list: Tuple[int, ...] = args.mxfe

    try:
        box = Quel1BoxIntrinsic.create(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    if not isinstance(box, Quel1BoxIntrinsic):
        raise ValueError(f"boxtype {args.boxtype} is not supported currently")

    linkup_ok: Dict[int, bool] = box.reconnect(
        background_noise_threshold=args.background_noise_threshold,
        ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
        ignore_invalid_linkstatus=True,
    )

    for mxfe_idx in mxfe_list:
        try:
            ls, _, diag = box.css.check_link_status(
                mxfe_idx, ignore_crc_error=(mxfe_idx in args.ignore_crc_error_of_mxfe)
            )
            if ls and not linkup_ok[mxfe_idx]:
                diag = f"{box.css.ipaddr_css}:AD9082-#{mxfe_idx} has badly configured tx-link {diag[diag.rfind('('):]}"
            else:
                linkup_ok[mxfe_idx] = ls
            print(diag)
        except Exception as e:
            print(f"AD9082-#{mxfe_idx}: failed to sync with the hardware due to {e}")

    return 0 if all(linkup_ok.values()) else -1


def quel1_test_linkup() -> None:
    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="identifying the failed step at the establishment of JESD204B/C link between FPGA and AD9082"
    )
    add_common_arguments(parser, use_mxfe=True, use_hard_reset=True)
    add_common_workaround_arguments(
        parser,
        use_ignore_crc_error_of_mxfe=True,
        use_ignore_access_failure_of_adrf6780=True,
        use_ignore_lock_failure_of_lmx2594=True,
    )
    parser.add_argument(
        "--background_noise_threshold",
        type=float,
        default=None,
        help="maximum allowable background noise amplitude of the ADCs",
    )
    parser.add_argument(
        "--use_204c",
        action="store_true",
        help="enable JESD204C link calibration instead of the conventional 204B one (default)",
    )
    parser.add_argument(
        "--use_204b",
        action="store_true",
        help="dare to use the conventional JESD204B link calibration instead of the 204C one",
    )
    parser.add_argument(
        "--use_bgcal",
        action="store_true",
        help="enable background calibration of JESD204C link (default)",
    )
    parser.add_argument(
        "--nouse_bgcal",
        action="store_true",
        help="disable background calibration of JESD204C link",
    )
    parser.add_argument(
        "--skip_init",
        action="store_true",
        default=False,
        help="skip initialization of ICs other than AD9082",
    )
    parser.add_argument(
        "--hard_reset_wss",
        action="store_true",
        default=False,
        help="hard reset AWG units and CAP units to clear hardware error flags, should avoid using it when unnecessary",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="count of test iterations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show verbose log",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        retcode = quel1_test_linkup_body(args)
        sys.exit(retcode)
    except AssertionError as e:
        if args.verbose:
            logger.exception(
                "internal error, please contact customer support with how to reproduce this issue", exc_info=e
            )
        else:
            logger.error("internal error, please contact customer support with how to reproduce this issue")
    except Exception as e:
        if args.verbose:
            logger.exception(e, exc_info=e)
        else:
            logger.error(e)
        sys.exit(-1)


def quel1_test_linkup_body(args: argparse.Namespace) -> int:
    mxfe_list: Tuple[int, ...] = args.mxfe

    try:
        box = Quel1BoxIntrinsic.create(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
            ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
            ignore_access_failure_of_adrf6780=args.ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=args.ignore_lock_failure_of_lmx2594,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    if not isinstance(box, Quel1BoxIntrinsic):
        raise ValueError(f"boxtype {args.boxtype} is not supported currently")

    use_204b: bool = False
    if args.use_204b:
        if args.use_204c:
            raise ValueError("it is not allowed to specify both --use_204b and --use_204c at the same time")
        else:
            use_204b = True

    use_bgcal: bool = True
    if args.nouse_bgcal:
        if args.use_bgcal:
            raise ValueError("it is not allowed to specify both --use_bgcal and --nouse_bgcal at the same time")
        else:
            use_bgcal = False

    n_success: Dict[int, int] = {}
    n_failure: Dict[int, int] = {}
    for mxfe_idx in mxfe_list:
        n_success[mxfe_idx] = 0
        n_failure[mxfe_idx] = 0

    for _ in range(args.count):
        linkup_ok: Dict[int, bool] = box.relinkup(
            mxfes_to_linkup=mxfe_list,
            config_root=args.config_root,
            config_options=args.config_options,
            hard_reset=args.hard_reset,
            use_204b=use_204b,
            use_bg_cal=use_bgcal,
            skip_init=args.skip_init,
            hard_reset_wss=args.hard_reset_wss,
            background_noise_threshold=args.background_noise_threshold,
        )
        for mxfe_idx in linkup_ok:
            if linkup_ok[mxfe_idx]:
                n_success[mxfe_idx] += 1
            else:
                n_failure[mxfe_idx] += 1

    if args.background_noise_threshold is None:
        background_noise_threshold: float = LinkupFpgaMxfe._DEFAULT_BACKGROUND_NOISE_THRESHOLD_RELINKUP
    else:
        background_noise_threshold = args.background_noise_threshold

    for mxfe_idx in mxfe_list:
        print(f"mxfe-#{mxfe_idx}:")
        print(f"    SUCCESS: {n_success[mxfe_idx]}")
        print(f"    FAILURE: {n_failure[mxfe_idx]}")

        s = [x.categorize(background_noise_threshold) for x in box.linkupper.linkup_statistics[mxfe_idx]]
        for label, v in LinkupStatus.__members__.items():
            print(f"        {label}: {s.count(v)}")

    return 0


def quel1_dump_port_config() -> None:
    logging.basicConfig(level=logging.ERROR, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(description="Dump the current configuration settings of the QuEL1 device.")
    add_common_arguments(
        parser,
        use_config_root=False,
        use_config_options=False,
    )
    parser.add_argument("--json", action="store_true", default=False, help="dump port configuration in JSON format")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show verbose log",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        _quel1_dump_port_config_body(args)
        sys.exit(0)
    except AssertionError as e:
        if args.verbose:
            logger.exception(
                "internal error, please contact customer support with how to reproduce this issue", exc_info=e
            )
        else:
            logger.error("internal error, please contact customer support with how to reproduce this issue")
    except Exception as e:
        if args.verbose:
            logger.exception(e, exc_info=e)
        else:
            logger.error(e)
    sys.exit(-1)


def _quel1_dump_port_config_body(args: argparse.Namespace) -> None:
    try:
        box = Quel1Box.create(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
        )
    except socket.timeout:
        raise Exception(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")

    if not isinstance(box, Quel1Box):
        raise ValueError(f"boxtype {args.boxtype} is not supported currently")

    if not all(box.reconnect(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values()):
        raise Exception("failed to syncing with the hardware, check the power and link status before retrying.")

    if args.json:
        print(box.dump_box_to_jsonstr())
    else:
        pprint(box.dump_box(), sort_dicts=False)


def quel1_firmware_version() -> None:
    logging.basicConfig(level=logging.ERROR, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(description="show the firmware version")
    add_common_arguments(parser, use_config_root=False, use_config_options=False)
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show verbose log",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        retcode = quel1_firmware_version_body(args)
        sys.exit(retcode)
    except AssertionError as e:
        if args.verbose:
            logger.exception(
                "internal error, please contact customer support with how to reproduce this issue", exc_info=e
            )
        else:
            logger.error("internal error, please contact customer support with how to reproduce this issue")
    except Exception as e:
        if args.verbose:
            logger.exception(e, exc_info=e)
        else:
            logger.error(e)
        sys.exit(-1)


def quel1_firmware_version_body(args: argparse.Namespace) -> int:
    try:
        box = Quel1Box.create(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    print(f"e7awghw firmware version: {box.wss.fw_version}")
    print(f"type of the firmware: {box.wss.fw_type}")
    return 0


def quel1se_tempctrl_state() -> None:
    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(description="show the state of temperature control subsystem of QuEL-1 SE")
    add_common_arguments(parser, use_config_root=False, use_config_options=False)
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show verbose log",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        retcode = quel1se_tempctrl_state_body(args)
        time.sleep(0.1)  # Notes: waiting for releasing the lock and closing coap context
        sys.exit(retcode)
    except AssertionError as e:
        if args.verbose:
            logger.exception(
                "internal error, please contact customer support with how to reproduce this issue", exc_info=e
            )
        else:
            logger.error("internal error, please contact customer support with how to reproduce this issue")
    except Exception as e:
        if args.verbose:
            logger.exception(e, exc_info=e)
        else:
            logger.error(e)
        sys.exit(-1)


def quel1se_tempctrl_state_body(args: argparse.Namespace) -> int:
    try:
        box = Quel1Box.create(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    state: Quel1seTempctrlState = box.css.get_tempctrl_state()
    if state == Quel1seTempctrlState.INIT:
        expl: str = "temperature control is not started yet"
    elif state == Quel1seTempctrlState.PRERUN:
        count: int = box.css.get_tempctrl_state_count()
        expl = f"warming up will finish in {count*10} seconds before starting the active control"
    elif state == Quel1seTempctrlState.RUN:
        expl = "active control is working"
    else:
        expl = "developer mode"

    print(f"tempctrl state is '{state.tostr().upper()}', {expl}.")
    return 0


def quel1se_tempctrl_reset() -> None:
    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="restart the temperature control after the warming-up of the specified duration"
    )
    add_common_arguments(parser, use_config_root=False, use_config_options=False)
    parser.add_argument(
        "--duration",
        type=int,
        default=1200,
        help="duration of warming-up in seconds, default value is 1200s",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show verbose log",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        retcode = quel1se_tempctrl_reset_body(args)
        time.sleep(0.1)  # Notes: waiting for releasing the lock and closing coap context
        sys.exit(retcode)
    except AssertionError as e:
        if args.verbose:
            logger.exception(
                "internal error, please contact customer support with how to reproduce this issue", exc_info=e
            )
        else:
            logger.error("internal error, please contact customer support with how to reproduce this issue")
    except Exception as e:
        if args.verbose:
            logger.exception(e, exc_info=e)
        else:
            logger.error(e)
        sys.exit(-1)


def quel1se_tempctrl_reset_body(args: argparse.Namespace) -> int:
    try:
        box = Quel1Box.create(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    # Notes: css.start_tempctrl() is defined for QuEL-1 for the convenience.
    #        so, not enough for rejecting unsupported boxtypes.
    if args.duration < 0:
        logger.error("warm-up duration must be positive")
        return -1

    count = args.duration // 10
    if count == 0:
        count = 1
        logger.warning("warmup duration is set to 10 seconds, the shortest duration")
    elif count > 0xFFFFFFFF:
        count = 0xFFFFFFFF
        logger.warning(f"warmup duration is set to too long, clipped to {count*10} seconds, the longest duration")

    box.css.start_tempctrl(new_count=count)
    print(f"the active temperature control starts after {count*10}-sencod warming-up")
    return 0
