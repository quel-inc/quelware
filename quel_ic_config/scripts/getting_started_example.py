import logging

from quel_ic_config_utils import create_box_objects

logger = logging.getLogger()


if __name__ == "__main__":
    import argparse

    from quel_ic_config_utils.common_arguments import add_common_arguments, complete_ipaddrs

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="check the basic functionalities of QuEL-1",
    )
    add_common_arguments(parser, use_mxfe=True, allow_implicit_mxfe=True)
    args = parser.parse_args()
    complete_ipaddrs(args)

    css, wss, linkupper, box = create_box_objects(
        ipaddr_wss=str(args.ipaddr_wss),
        ipaddr_sss=str(args.ipaddr_sss),
        ipaddr_css=str(args.ipaddr_css),
        boxtype=args.boxtype,
        config_root=args.config_root,
        config_options=args.config_options,
        refer_by_port=True,
    )

    if box is not None:
        box.init()
