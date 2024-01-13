import argparse
import json
import logging
import socket
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple

from quel_ic_config import Quel1BoxType
from quel_ic_config_utils import LinkupFpgaMxfe, LinkupStatus, SimpleBox, SimpleBoxIntrinsic, create_box_objects, linkup
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
        default=False,
        help="enable JESD204C link instead of the conventional 204B one",
    )
    parser.add_argument(
        "--skip_init",
        action="store_true",
        default=False,
        help="skip initialization of ICs other than AD9082",
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
        _, _, _, linkupper, box = create_box_objects(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
            config_root=args.config_root,
            config_options=args.config_options,
            refer_by_port=False,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    if not isinstance(box, SimpleBoxIntrinsic):
        raise ValueError(f"boxtype {args.boxtype} is not supported currently")

    cli_retcode: int = 0
    linkup_ok: Dict[int, bool] = linkup(
        linkupper=linkupper,
        mxfe_list=mxfe_list,
        use_204b=not args.use_204c,
        skip_init=args.skip_init,
        background_noise_threshold=args.background_noise_threshold,
        ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
        ignore_access_failure_of_adrf6780=args.ignore_access_failure_of_adrf6780,
        ignore_lock_failure_of_lmx2594=args.ignore_lock_failure_of_lmx2594,
        save_dirpath=args.save_dirpath,
    )

    for mxfe_idx in mxfe_list:
        if linkup_ok[mxfe_idx]:
            print(f"ad9082-#{mxfe_idx} linked up successfully")
        else:
            logger.error(f"ad9082-#{mxfe_idx} failed to link up")
            cli_retcode = -1

    return cli_retcode


def quel1_linkstatus() -> None:
    logging.basicConfig(level=logging.ERROR, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(description="show the link status of AD9082s")
    add_common_arguments(
        parser, use_mxfe=True, allow_implicit_mxfe=True, use_config_root=False, use_config_options=False
    )
    add_common_workaround_arguments(parser, use_ignore_crc_error_of_mxfe=True)
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
        _, _, _, _, box = create_box_objects(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
            config_root=args.config_root,
            config_options=args.config_options,
            refer_by_port=False,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    if not isinstance(box, SimpleBoxIntrinsic):
        raise ValueError(f"boxtype {args.boxtype} is not supported currently")

    _ = box.init(ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe)

    cli_retcode: int = 0
    for mxfe_idx in mxfe_list:
        try:
            link_status, error_flag = box.css.get_link_status(mxfe_idx)
            if link_status == 0xE0:
                if error_flag == 0x01:
                    judge: str = "healthy datalink"
                else:
                    judge = "unhealthy datalink (CRC errors are detected)"
                    if mxfe_idx not in args.ignore_crc_error_of_mxfe:
                        cli_retcode = -1
            else:
                judge = "no datalink available"
                cli_retcode = -1
            print(f"AD9082-#{mxfe_idx}: {judge}  (linkstatus = 0x{link_status:02x}, error_flag = 0x{error_flag:02x})")
        except Exception as e:
            print(f"AD9082-#{mxfe_idx}: failed to sync with the hardware due to {e}")

    return cli_retcode


def quel1_test_linkup() -> None:
    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="identifying the failed step at the establishment of JESD204B/C link between FPGA and AD9082"
    )
    add_common_arguments(parser, use_mxfe=True)
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
        default=False,
        help="enable JESD204C link instead of the conventional 204B one",
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
        _, _, _, linkupper, box = create_box_objects(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
            config_root=args.config_root,
            config_options=args.config_options,
            refer_by_port=False,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    if not isinstance(box, SimpleBoxIntrinsic):
        raise ValueError(f"boxtype {args.boxtype} is not supported currently")

    for _ in range(args.count):
        linkup(
            linkupper=linkupper,
            mxfe_list=mxfe_list,
            use_204b=not args.use_204c,
            background_noise_threshold=args.background_noise_threshold,
            ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
            ignore_access_failure_of_adrf6780=args.ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=args.ignore_lock_failure_of_lmx2594,
        )

    if args.background_noise_threshold is None:
        background_noise_threshold: float = LinkupFpgaMxfe._DEFAULT_BACKGROUND_NOISE_THRESHOLD
    else:
        background_noise_threshold = args.background_noise_threshold

    for mxfe_idx in mxfe_list:
        s = [x.categorize(background_noise_threshold) for x in linkupper.linkup_statistics[mxfe_idx]]
        print(f"mxfe-#{mxfe_idx}:")
        for label, v in LinkupStatus.__members__.items():
            print(f"    {label}: {s.count(v)}")

    return 0


def quel1_dump_port_config() -> None:
    logging.basicConfig(level=logging.ERROR, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="identifying the failed step at the establishment of JESD204B/C link between FPGA and AD9082"
    )
    add_common_arguments(
        parser,
        use_config_root=False,
        use_config_options=False,
    )
    parser.add_argument("--json", action="store_true", default=False, help="dump port configuration in json format")
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
        retcode = quel1_dump_port_config_body(args)
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


def quel1_dump_port_config_body(args: argparse.Namespace) -> int:
    try:
        _, _, _, linkupper, box = create_box_objects(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
            config_root=args.config_root,
            config_options=args.config_options,
            refer_by_port=True,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    if not isinstance(box, SimpleBox):
        raise ValueError(f"boxtype {args.boxtype} is not supported currently")

    if not all(box.init(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values()):
        logger.error("failed to syncing with the hardware, check the power and link status before retrying.")
        return -1

    cli_retcode = 0
    try:
        if args.json:
            print(json.dumps(box.dump_config()))
        else:
            pprint(box.dump_config(), sort_dicts=False)
    except Exception as e:
        logger.error(f"failed due to {e}")
        cli_retcode = -1
    return cli_retcode


def quel1_start_all_ports():
    logging.basicConfig(level=logging.ERROR, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="identifying the failed step at the establishment of JESD204B/C link between FPGA and AD9082"
    )
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
        _, _, _, linkupper, box = create_box_objects(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
            config_root=args.config_root,
            config_options=args.config_options,
            refer_by_port=True,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        sys.exit(-1)

    if not isinstance(box, SimpleBox):
        raise AssertionError

    if args.boxtype == Quel1BoxType.QuEL1_TypeA:
        box.config_port(1, lo_freq=8.5e9, cnco_freq=1.5e9, vatt=0xA00, sideband="U")  # 10GHz
        box.config_port(2, lo_freq=11.5e9, cnco_freq=1.55e9, vatt=0xA00, sideband="L")  # 9.95GHz
        box.config_port(3, lo_freq=8.5e9, cnco_freq=1.4e9, vatt=0xA00, sideband="U")  # 9.9GHz
        box.config_port(4, lo_freq=11.5e9, cnco_freq=1.65e9, vatt=0xA00, sideband="L")  # 9.85GHz
        box.config_port(8, lo_freq=8.5e9, cnco_freq=1.3e9, vatt=0xA00, sideband="U")  # 9.8GHz
        box.config_port(9, lo_freq=11.5e9, cnco_freq=1.75e9, vatt=0xA00, sideband="L")  # 9.75GHz
        box.config_port(10, lo_freq=8.5e9, cnco_freq=1.2e9, vatt=0xA00, sideband="U")  # 9.7GHz
        box.config_port(11, lo_freq=11.5e9, cnco_freq=1.85e9, vatt=0xA00, sideband="L")  # 9.65GHz
        for port in (1, 2, 3, 4, 8, 9, 10, 11):
            box.start_channel(port)
    elif args.boxtype == Quel1BoxType.QuEL1_TypeB:
        box.config_port(1, lo_freq=11.5e9, cnco_freq=1.5e9, vatt=0xA00, sideband="L")  # 10GHz
        box.config_port(2, lo_freq=11.5e9, cnco_freq=1.55e9, vatt=0xA00, sideband="L")  # 9.95GHz
        box.config_port(3, lo_freq=11.5e9, cnco_freq=1.6e9, vatt=0xA00, sideband="L")  # 9.9GHz
        box.config_port(4, lo_freq=11.5e9, cnco_freq=1.65e9, vatt=0xA00, sideband="L")  # 9.85GHz
        box.config_port(8, lo_freq=11.5e9, cnco_freq=1.7e9, vatt=0xA00, sideband="L")  # 9.8GHz
        box.config_port(9, lo_freq=11.5e9, cnco_freq=1.75e9, vatt=0xA00, sideband="L")  # 9.75GHz
        box.config_port(10, lo_freq=11.5e9, cnco_freq=1.8e9, vatt=0xA00, sideband="L")  # 9.7GHz
        box.config_port(11, lo_freq=11.5e9, cnco_freq=1.85e9, vatt=0xA00, sideband="L")  # 9.65GHz
        for port in (1, 2, 3, 4, 8, 9, 10, 11):
            box.start_channel(port)
    else:
        logger.error(f"boxtype {args.boxtype} is not supported")
        sys.exit(-1)

    input("hit enter to stop the RF signals")

    box.easy_stop_all()
    sys.exit(0)


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
        _, wss, _, _, _ = create_box_objects(
            ipaddr_wss=str(args.ipaddr_wss),
            ipaddr_sss=str(args.ipaddr_sss),
            ipaddr_css=str(args.ipaddr_css),
            boxtype=args.boxtype,
            config_root=args.config_root,
            config_options=args.config_options,
        )
    except socket.timeout:
        logger.error(f"cannot access to the given IP addresses {args.ipaddr_wss} / {args.ipaddr_css}")
        return -1

    print(f"e7awg firmware version: {wss.hw_version}")
    print(f"type of the firmware: {wss.hw_type}")
    print(f"life cycle of the firmware: {wss.hw_lifestage}")
    return 0
