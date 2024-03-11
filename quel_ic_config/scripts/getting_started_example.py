import logging

from quel_ic_config import Quel1Box

logger = logging.getLogger()


def relinkup():
    global box, args

    is_quel1se = args.boxtype in {"quel1se-riken8", "x-quel1se-riken8"}
    link_ok = box.relinkup(
        use_204b=(not args.use_204c) if not is_quel1se else False,
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
        default=False,
        help="enable JESD204C link calibration instead of the conventional 204B one",
    )
    add_common_workaround_arguments(
        parser, use_ignore_crc_error_of_mxfe=True, use_ignore_access_failure_of_adrf6780=True
    )
    args = parser.parse_args()
    complete_ipaddrs(args)

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
