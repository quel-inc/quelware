import logging

from quel_ic_config_utils import create_box_objects

logger = logging.getLogger()


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
    add_common_arguments(parser, use_mxfe=True, allow_implicit_mxfe=True)
    add_common_workaround_arguments(
        parser, use_ignore_crc_error_of_mxfe=True, use_ignore_extraordinary_converter_select_of_mxfe=True
    )
    parser.add_argument("--dev", action="store_true", help="use (group, line, channel) instead of port")
    parser.add_argument("--linkup", action="store_true", help="conducting link-up just after the initialization")
    args = parser.parse_args()
    complete_ipaddrs(args)

    css, wss, rmap, linkupper, box = create_box_objects(
        ipaddr_wss=str(args.ipaddr_wss),
        ipaddr_sss=str(args.ipaddr_sss),
        ipaddr_css=str(args.ipaddr_css),
        boxtype=args.boxtype,
        config_root=args.config_root,
        config_options=args.config_options,
        refer_by_port=not args.dev,
    )

    if box is not None:
        status = box.init(
            ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select_of_mxfe=args.ignore_extraordinary_converter_select_of_mxfe,
        )

        for mxfe_idx, s in status.items():
            if not s:
                logger.error(f"be aware that mxfe-#{mxfe_idx} is not linked-up properly")
    else:
        logger.error(f"boxtype {args.boxtype} is not ready for test with SimpleBox object.")
