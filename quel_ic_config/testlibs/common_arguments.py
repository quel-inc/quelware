from argparse import ArgumentParser
from ipaddress import ip_address
from pathlib import Path
from typing import List

from quel_ic_config import QUEL1_BOXTYPE_ALIAS, Quel1BoxType, Quel1ConfigOption


def parse_boxtype(boxtypename: str) -> Quel1BoxType:
    if boxtypename not in QUEL1_BOXTYPE_ALIAS:
        raise ValueError
    return Quel1BoxType.fromstr(boxtypename)


def parse_config_options(optstr: str) -> List[Quel1ConfigOption]:
    return [Quel1ConfigOption(s) for s in optstr.split(",") if s != ""]


def add_common_arguments(
    parser: ArgumentParser,
    use_ipaddr_wss: bool = True,
    use_ipaddr_sss: bool = True,
    use_ipaddr_css: bool = True,
    use_boxtype: bool = True,
    use_config_root: bool = True,
    use_config_options: bool = True,
) -> None:
    """adding common arguments to testlibs of quel_ic_config for manual tests. allowing to accept unused arguments for
    your convenience
    :param parser: ArgumentParser object to register arguments
    :param use_ipaddr_wss:
    :param use_ipaddr_sss:
    :param use_ipaddr_css:
    :param use_boxtype:
    :param use_config_root:
    :param use_config_options:
    :return:
    """

    non_existent_ipaddress = ip_address("241.3.5.6")

    if use_ipaddr_wss:
        parser.add_argument(
            "--ipaddr_wss",
            type=ip_address,
            required=True,
            help="IP address of the wave generation/capture subsystem of the target box",
        )
    else:
        parser.add_argument(
            "--ipaddr_wss", type=ip_address, required=False, default=non_existent_ipaddress, help="IGNORED"
        )

    if use_ipaddr_sss:
        parser.add_argument(
            "--ipaddr_sss",
            type=ip_address,
            required=True,
            help="IP address of the wave sequencer subsystem of the target box",
        )
    else:
        parser.add_argument(
            "--ipaddr_sss", type=ip_address, required=False, default=non_existent_ipaddress, help="IGNORED"
        )

    if use_ipaddr_css:
        parser.add_argument(
            "--ipaddr_css",
            type=ip_address,
            required=True,
            help="IP address of the configuration subsystem of the target box",
        )
    else:
        parser.add_argument(
            "--ipaddr_css", type=ip_address, required=False, default=non_existent_ipaddress, help="IGNORED"
        )

    if use_boxtype:
        parser.add_argument(
            "--boxtype",
            type=parse_boxtype,
            required=True,
            help=f"a type of the target box: either of "
            f"{', '.join([t for t in QUEL1_BOXTYPE_ALIAS if not t.startswith('x_')])}",
        )
    else:
        raise NotImplementedError

    if use_config_root:
        parser.add_argument(
            "--config_root",
            type=Path,
            default=Path("settings"),
            help="path to configuration file root",
        )
    else:
        raise NotImplementedError

    if use_config_options:
        parser.add_argument(
            "--config_options",
            type=parse_config_options,
            default=[],
            help=f"comma separated list of config options: ("
            f"{' '.join([o for o in Quel1ConfigOption if not o.startswith('x_')])})",
        )
    else:
        raise NotImplementedError
