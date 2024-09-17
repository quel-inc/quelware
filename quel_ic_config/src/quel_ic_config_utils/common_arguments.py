from argparse import ArgumentParser, Namespace
from ipaddress import ip_address
from pathlib import Path
from typing import Collection, List, Set, Tuple, Union

from quel_ic_config.quel_config_common import QUEL1_BOXTYPE_ALIAS, Quel1BoxType, Quel1ConfigOption


def parse_boxtype(boxtypename: str) -> Quel1BoxType:
    if boxtypename not in QUEL1_BOXTYPE_ALIAS:
        raise ValueError(f"invalid boxtype: '{boxtypename}'")
    return Quel1BoxType.fromstr(boxtypename)


def parse_config_options(optstr: str) -> List[Quel1ConfigOption]:
    return [Quel1ConfigOption(s) for s in optstr.split(",") if s != ""]


def _parse_mxfe_combination(mxfe_combination: str) -> Tuple[int, ...]:
    if mxfe_combination == "0":
        mxfe_list: Tuple[int, ...] = (0,)
    elif mxfe_combination == "1":
        mxfe_list = (1,)
    elif mxfe_combination in {"both", "0,1"}:
        mxfe_list = (0, 1)
    elif mxfe_combination == "1,0":
        mxfe_list = (1, 0)
    elif mxfe_combination == "none":
        mxfe_list = ()
    else:
        raise ValueError(f"invalid combination of MxFEs: {mxfe_combination}")
    return mxfe_list


def _parse_comma_separated_adrf6780_list(cslist: str) -> Set[int]:
    parsed = [int(x) for x in cslist.split(",")]
    invalid = [x for x in parsed if not 0 <= x <= 7]
    if len(invalid) != 0:
        raise ValueError(f"invalid indices of ADRF6780s: {','.join([str(x) for x in invalid])}")
    return set(parsed)


def _parse_comma_separated_lmx2594_list(cslist: str) -> Set[int]:
    parsed = [int(x) for x in cslist.split(",")]
    invalid = [x for x in parsed if not 0 <= x <= 9]
    if len(invalid) != 0:
        raise ValueError(f"invalid indices of LMX2594s: {','.join([str(x) for x in invalid])}")
    return set(parsed)


def complete_ipaddrs(args: Namespace):
    if int(args.ipaddr_sss) == 0:
        args.ipaddr_sss = args.ipaddr_wss + (1 << 16)
    if int(args.ipaddr_css) == 0:
        args.ipaddr_css = args.ipaddr_wss + (4 << 16)


def add_common_arguments(
    parser: ArgumentParser,
    strict_ipaddrs: bool = False,
    use_ipaddr_wss: bool = True,
    use_ipaddr_sss: bool = True,
    use_ipaddr_css: bool = True,
    use_boxtype: bool = True,
    use_config_root: bool = True,
    use_config_options: bool = True,
    use_mxfe: bool = False,
    default_boxtype: Union[str, None] = None,
    default_config_root: Union[Path, None] = None,
    default_config_options: Union[Collection[str], None] = None,
    allow_implicit_mxfe: bool = False,
    use_hard_reset: bool = False,
) -> None:
    """adding common arguments to testlibs of quel_ic_config for manual tests. allowing to accept unused arguments for
    your convenience

    :param parser:
    :param strict_ipaddrs:
    :param use_ipaddr_wss:
    :param use_ipaddr_sss:
    :param use_ipaddr_css:
    :param use_boxtype:
    :param use_config_root:
    :param use_config_options:
    :param use_mxfe:
    :param default_boxtype:
    :param default_config_root:
    :param default_config_options:
    :param allow_implicit_mxfe:
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
        if strict_ipaddrs:
            parser.add_argument(
                "--ipaddr_sss",
                type=ip_address,
                required=True,
                help="IP address of the wave sequencer subsystem of the target box",
            )
        else:
            parser.add_argument(
                "--ipaddr_sss",
                type=ip_address,
                default=ip_address(0),
                help="IP address of the wave sequencer subsystem of the target box",
            )
    else:
        parser.add_argument(
            "--ipaddr_sss", type=ip_address, required=False, default=non_existent_ipaddress, help="IGNORED"
        )

    if use_ipaddr_css:
        if strict_ipaddrs:
            parser.add_argument(
                "--ipaddr_css",
                type=ip_address,
                required=True,
                help="IP address of the configuration subsystem of the target box",
            )
        else:
            parser.add_argument(
                "--ipaddr_css",
                type=ip_address,
                default=ip_address(0),
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
        if default_boxtype is None:
            raise ValueError("default boxtype is required when boxtype parameter is disabled")
        parser.add_argument(
            "--boxtype", type=parse_boxtype, default=default_boxtype, help="default value should be used"
        )

    if use_config_root:
        parser.add_argument(
            "--config_root",
            type=Path,
            default=None,
            help="path to configuration file root",
        )
    else:
        parser.add_argument(
            "--config_root",
            action="store_const",
            const=default_config_root,
            help="constant value",
        )

    if use_config_options:
        parser.add_argument(
            "--config_options",
            type=parse_config_options,
            default=[],
            help=f"comma separated list of config options: ("
            f"{' '.join([o for o in Quel1ConfigOption if not o.startswith('x_')])})",
        )
    else:
        if default_config_options is None:
            default_config_options = []
        parser.add_argument(
            "--config_options",
            action="store_const",
            const=default_config_options,
            help="constant value",
        )

    if use_mxfe:
        if allow_implicit_mxfe:
            parser.add_argument(
                "--mxfe",
                metavar="MxFEs",
                type=_parse_mxfe_combination,
                default=(0, 1),
                help="combination of MxFEs under test, possible values are '0', '1', 'both', '0,1', '1,0', and 'none'",
            )
        else:
            parser.add_argument(
                "--mxfe",
                metavar="MxFEs",
                type=_parse_mxfe_combination,
                required=True,
                help="combination of MxFEs under test, possible values are '0', '1', 'both', '0,1', '1,0', and 'none'",
            )

    if use_hard_reset:
        parser.add_argument(
            "--hard_reset",
            action="store_true",
            default=False,
            help="enable hard reset of AD9082 at the initialization",
        )


def add_common_workaround_arguments(
    parser: ArgumentParser,
    use_ignore_crc_error_of_mxfe: bool = False,
    use_ignore_access_failure_of_adrf6780: bool = False,
    use_ignore_lock_failure_of_lmx2594: bool = False,
    use_ignore_extraordinary_converter_select_of_mxfe: bool = False,
):
    if use_ignore_crc_error_of_mxfe:
        parser.add_argument(
            "--ignore_crc_error_of_mxfe",
            metavar="MxFEs",
            type=_parse_mxfe_combination,
            default=(),
            help="a comma-separated list of MxFEs to ignore crc_error_flag during initialization, "
            "possible values are '0', '1', 'both', '0,1', '1,0', and 'none'",
        )

    if use_ignore_access_failure_of_adrf6780:
        parser.add_argument(
            "--ignore_access_failure_of_adrf6780",
            metavar="ADRF6780s",
            type=_parse_comma_separated_adrf6780_list,
            default={},
            help="a comma-separated list of ADRF6780s to ignore communication failures",
        )

    if use_ignore_lock_failure_of_lmx2594:
        parser.add_argument(
            "--ignore_lock_failure_of_lmx2594",
            metavar="LMX2594s",
            type=_parse_comma_separated_lmx2594_list,
            default={},
            help="a comma-separated list of LMX2594s to ignore lock failures",
        )

    if use_ignore_extraordinary_converter_select_of_mxfe:
        parser.add_argument(
            "--ignore_extraordinary_converter_select_of_mxfe",
            metavar="MxFEs",
            type=_parse_mxfe_combination,
            default=(),
            help="a comma-separated list of MxFEs to ignore the abnormally configured converter select. "
            "possible values are '0', '1', 'both', '0,1', '1,0', and 'none'",
        )
