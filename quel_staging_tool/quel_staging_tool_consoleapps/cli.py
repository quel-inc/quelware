import argparse
import logging
import os
import shutil
import sys
from ipaddress import ip_address
from pathlib import Path

from quel_staging_tool import Au50Programmer, ExstickgeProgrammer, QuelXilinxFpgaProgrammer

logger = logging.getLogger()


def _dir_path(path: str) -> Path:
    if os.path.isdir(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid path to a directory")


def _common_parser(
    obj: QuelXilinxFpgaProgrammer,
    progname: str,
    description: str,
    target_name: str,
    use_ipaddr: bool = True,
    use_macaddr: bool = True,
    use_firmware: bool = True,
    use_adapter: bool = True,
    use_port: bool = True,
    use_bit: bool = False,
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

    if use_firmware:
        parser.add_argument(
            "--firmware",
            type=str,
            required=True,
            help=f"name of firmware: {' '.join(obj.get_bits())}",
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

    return parser


def program_exstickge():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = ExstickgeProgrammer()
    parser = _common_parser(
        obj,
        "quel_program_exstickge",
        "writing the specified firmware with MAC and IP addresses into the flash memory of the specified ExStickGE",
        "ExStickGE",
    )
    args = parser.parse_args()

    bitfiles = obj.get_bits()
    if args.firmware not in bitfiles:
        logger.error(f"invalid firmware: {args.firmware}")
        sys.exit(1)

    try:
        memfile_path = obj.make_mem(
            macaddr=args.macaddr, ipaddr=str(args.ipaddr), netmask="255.0.0.0", default_gateway="10.0.0.1"
        )
    except Exception as e:
        logger.error("IP address or MAC address looks invalid")
        logger.error(e)
        sys.exit(1)

    try:
        e_path = obj.make_embedded_bit(bitpath=bitfiles[args.firmware], mempath=memfile_path)
        m_path = obj.make_mcs(e_path)
        obj.program(m_path, args.host, args.port, args.adapter)

    except Exception as e:
        logger.error(e)
        sys.exit(1)


def program_au50():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = Au50Programmer()
    # TODO: provides a way to modify cache directory.
    parser = _common_parser(
        obj,
        "quel_program_au50",
        "writing the specified firmware with MAC and IP addresses into the flash memory of the specified Alveo U50",
        "Alveo U50",
        use_bit=True,
        use_save=True,
        use_firmware_dir=True,
    )
    args = parser.parse_args()

    bitfiles = obj.get_bits(args.firmware_dir)
    if args.firmware not in bitfiles:
        logger.error(f"invalid firmware: {args.firmware}")
        sys.exit(1)

    bitcache_path = (
        Path(os.getenv("HOME", "."))
        / ".quel_staging_cache"
        / os.path.basename(os.path.dirname(bitfiles[args.firmware]))
    )
    os.makedirs(bitcache_path, exist_ok=True)
    cached_e_path = bitcache_path / f"{str(args.ipaddr)}.bit"

    if os.path.exists(cached_e_path):
        e_path = cached_e_path
    else:
        try:
            memfile_path = obj.make_mem(
                macaddr=args.macaddr, ipaddr=str(args.ipaddr), netmask="255.0.0.0", default_gateway="10.0.0.1"
            )
        except Exception as e:
            logger.error("IP address or MAC address looks invalid")
            logger.error(e)
            sys.exit(1)

        try:
            e_path = obj.make_embedded_bit(bitpath=bitfiles[args.firmware], mempath=memfile_path)
            shutil.copy(e_path, cached_e_path)
        except Exception as e:
            logger.error(e)
            sys.exit(1)

    try:
        if args.bit:
            if args.save:
                shutil.copy(e_path, ".")
            obj.program_bit(e_path, args.host, args.port, args.adapter)
        else:
            m_path = obj.make_mcs(e_path)
            if args.save:
                shutil.copy(m_path, ".")
            obj.program(m_path, args.host, args.port, args.adapter)
    except Exception as e:
        logger.error(e)
        sys.exit(1)


def reboot_xil_fpga():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = Au50Programmer()
    parser = _common_parser(
        obj,
        "quel_reboot_fpga",
        "reboot the specified Xilinx FPGA via JTAG adapter",
        "",
        use_ipaddr=False,
        use_macaddr=False,
        use_firmware=False,
    )
    args = parser.parse_args()

    try:
        obj.reboot(args.host, args.port, args.adapter)
    except Exception as e:
        logger.error(e)
        sys.exit(1)
