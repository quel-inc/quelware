import logging

from quel_ic_config import Quel1Box

logger = logging.getLogger()


def relinkup():
    global box, args

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

    link_ok = box.relinkup(
        use_204b=use_204b,
        use_bg_cal=use_bgcal,
    )
    return link_ok


if __name__ == "__main__":
    import argparse

    from quel_ic_config_utils.common_arguments import (
        add_common_arguments,
        add_common_workaround_arguments,
        complete_ipaddrs,
    )

    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="check the basic functionalities of QuEL-1",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--use_204c",
        action="store_true",
        help="enable JESD204C link calibration instead of the conventional 204B one (default)",
    )
    parser.add_argument(
        "--use_204b",
        action="store_true",
        help="dare to use the conventional JESD204B link calibration instead of 204C one",
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
    add_common_workaround_arguments(
        parser, use_ignore_crc_error_of_mxfe=True, use_ignore_access_failure_of_adrf6780=True
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

    box = Quel1Box.create(
        ipaddr_wss=str(args.ipaddr_wss),
        ipaddr_sss=str(args.ipaddr_sss),
        ipaddr_css=str(args.ipaddr_css),
        boxtype=args.boxtype,
        config_root=args.config_root,
        config_options=args.config_options,
        ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
        ignore_access_failure_of_adrf6780=args.ignore_access_failure_of_adrf6780,
    )
    status = box.reconnect()
    for mxfe_idx, s in status.items():
        if not s:
            logger.error(f"be aware that mxfe-#{mxfe_idx} is not linked-up properly")
