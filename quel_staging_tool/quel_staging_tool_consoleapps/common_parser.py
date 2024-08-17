import argparse
import logging
import os
from ipaddress import ip_address
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _dir_path(path: str) -> Path:
    if os.path.isdir(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid path to a directory")


def _firmware_key(v: str) -> Tuple[int, int]:
    kind_order = ("simplemulti", "feedback")
    try:
        v_kind, v_date = v.split("_")
        if v_kind in kind_order:
            return (kind_order.index(v_kind), -int(v_date))
    except Exception:
        pass

    # Notes: this should not happen.
    logger.error(f"a firmware which has unexpected name '{v}' exists in the package")
    return (len(kind_order), 0)


def common_parser(
    progname: str,
    description: str,
    target_name: str,
    bitfile_names: List[str],
    *,
    use_ipaddr: bool = True,
    use_macaddr: bool = True,
    use_firmware: bool = True,
    use_adapter: bool = True,
    use_port: bool = True,
    use_bit: bool = False,
    use_dry: bool = False,
    use_save: bool = False,
    use_firmware_dir: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=progname, description=description)

    if use_ipaddr:
        parser.add_argument(
            "--ipaddr",
            type=ip_address,
            required=True,
            help=f"IP address of {target_name}",
        )

    if use_macaddr:
        parser.add_argument(
            "--macaddr",
            type=str,
            required=True,
            help=f"MAC address of {target_name}",
        )

    if target_name == "Alveo U50":
        bitfile_names.sort(key=_firmware_key)
    if use_firmware:
        parser.add_argument(
            "--firmware",
            type=str,
            required=True,
            help=(
                f"name of firmware to program, the current quel_staging_tool package has "
                f"the following firmwares: {','.join(bitfile_names)}"
            ),
        )

    if use_adapter:
        parser.add_argument(
            "--adapter",
            type=str,
            required=True,
            help="id of JTAG adapter",
        )

    if use_port:
        parser.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="host where the target hw_server is running",
        )

        parser.add_argument(
            "--port",
            type=int,
            required=True,
            help="port to hw_server, you need to specify 3121 to use the default hw_server",
        )

    if use_bit:
        parser.add_argument(
            "--bit",
            action="store_true",
            default=False,
            help="programming FPGA instead of non-volatile configuration memory",
        )

    if use_save:
        parser.add_argument(
            "--save",
            action="store_true",
            default=False,
            help="keep the generated bit and mcs files at the current working directory",
        )

    if use_firmware_dir:
        parser.add_argument(
            "--firmware_dir",
            type=_dir_path,
            default=None,
            help="a path to user's firmware repository",
        )

    if use_dry:
        parser.add_argument(
            "--dry",
            action="store_true",
            default=False,
            help="enable dry-run mode, just connecting to programming adapter.",
        )

    parser.add_argument("--verbose", action="store_true", default=False, help="show verbose log")
    return parser
