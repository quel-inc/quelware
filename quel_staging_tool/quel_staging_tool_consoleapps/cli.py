import argparse
import logging
import os
import shutil
import sys
from ipaddress import ip_address
from pathlib import Path
from typing import Tuple

from quel_staging_tool import (
    Au50Programmer,
    Au200Programmer,
    ExstickgeProgrammer,
    QuelXilinxFpgaProgrammer,
    QuelXilinxFpgaProgrammerZephyr,
)
from quel_staging_tool.programmer_for_e7udpip import AuxxxProgrammer

logger = logging.getLogger()


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


def _common_parser(
    obj: QuelXilinxFpgaProgrammer,
    progname: str,
    description: str,
    target_name: str,
    bitfile_name: str,
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

    bits = list(obj.get_bits(bitfile_name=bitfile_name))
    if target_name == "Alveo U50":
        bits.sort(key=_firmware_key)
    if use_firmware:
        parser.add_argument(
            "--firmware",
            type=str,
            required=True,
            help=(
                f"name of firmware to program, the current quel_staging_tool package has "
                f"the following firmwares: {','.join(bits)}"
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


def program_exstickge_1():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = ExstickgeProgrammer()
    parser = _common_parser(
        obj,
        "quel_program_exstickge_1",
        "writing the specified e7udpip-based firmware with MAC and IP addresses into the flash memory"
        "of the specified ExStickGE",
        "ExStickGE",
        "top.bit",
        use_firmware_dir=True,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    bitfiles = obj.get_bits(bitdir_path=args.firmware_dir)
    if args.firmware not in bitfiles:
        logger.error(f"invalid firmware: {args.firmware}")
        sys.exit(1)

    try:
        memfile_path = obj.make_mem(
            macaddr=args.macaddr, ipaddr=str(args.ipaddr), netmask="255.0.0.0", default_gateway="10.0.0.1"
        )
    except Exception as e:
        logger.error("given IP address or MAC address looks invalid")
        logger.error(e)
        sys.exit(1)

    try:
        e_path = obj.make_embedded_bit(bitpath=bitfiles[args.firmware], mempath=memfile_path)
        m_path = obj.make_mcs(e_path)
        obj.program(m_path, args.host, args.port, args.adapter)
        obj.reboot(args.host, args.port, args.adapter)

    except Exception as e:
        logger.error(e)
        sys.exit(1)


def program_exstickge_clockdisty():
    program_exstickge_zephyr_common("quel_clk_distributor.bit")


def program_exstickge_1se():
    program_exstickge_zephyr_common("quel1_config.bit")


def program_exstickge_zephyr_common(bitfile_name: str):
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = QuelXilinxFpgaProgrammerZephyr()
    parser = _common_parser(
        obj,
        "quel_program_exstickge_1se",
        "writing the specified zephyr-based firmware with MAC and IP addresses into the flash memory of"
        "the specified ExStickGE",
        "ExStickGE",
        bitfile_name,
        use_firmware_dir=True,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    bitfiles = obj.get_bits(bitdir_path=args.firmware_dir, bitfile_name=bitfile_name)
    if args.firmware not in bitfiles:
        logger.error(f"invalid firmware: {args.firmware}")
        sys.exit(1)

    bitfile = bitfiles[args.firmware]
    elffile = bitfile.parent / "zephyr.elf"
    mmifile = bitfile.parent / "itcm.mmi"

    try:
        eelfpath = obj.make_embedded_elf(elfpath=elffile, ipaddr=args.ipaddr)
        macaddrpath = obj.make_macaddr_bin(args.macaddr)
    except Exception as e:
        logger.error("given IP address or MAC address looks invalid")
        logger.error(e)
        sys.exit(1)

    try:
        ebitpath = obj.make_embedded_bit(bitpath=bitfile, mmipath=mmifile, elfpath=eelfpath)
        mcspath = obj.make_mcs_with_macaddr(bitpath=ebitpath, macaddrpath=macaddrpath)
        obj.program(mcspath, args.host, args.port, args.adapter, "Digilent/JTAG-HS2/")
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
        "top.bit",
        use_bit=True,
        use_save=True,
        use_firmware_dir=True,
        use_dry=True,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        retval = program_alveoxxx_body(obj, args, "Xilinx/Alveo-DMBv1 FT4232H/")
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    sys.exit(retval)


def program_au200():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = Au200Programmer()
    # TODO: provides a way to modify cache directory.
    parser = _common_parser(
        obj,
        "quel_program_au200",
        "writing the specified firmware with MAC and IP addresses into the flash memory of the specified Alveo U200",
        "Alveo U200",
        "top.bit",
        use_bit=True,
        use_save=True,
        use_firmware_dir=True,
        use_dry=True,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        retval = program_alveoxxx_body(obj, args, "Xilinx/A-U200-A64G FT4232H/")
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    sys.exit(retval)


def program_alveoxxx_body(obj: AuxxxProgrammer, args: argparse.Namespace, adapter_typeid: str) -> int:
    bitfiles = obj.get_bits(bitdir_path=args.firmware_dir)
    if args.firmware not in bitfiles:
        logger.error(f"invalid firmware: {args.firmware}")
        return 1

    bitcache_path = (
        Path(os.getenv("HOME", "."))
        / ".quel_staging_cache"
        / os.path.basename(os.path.dirname(bitfiles[args.firmware]))
    )
    os.makedirs(bitcache_path, exist_ok=True)
    cached_e_path = bitcache_path / f"{str(args.ipaddr)}.bit"

    if os.path.exists(cached_e_path):
        e_path = cached_e_path
        logger.info(f"using a cached file '{e_path}'")
    else:
        try:
            memfile_path = obj.make_mem(
                macaddr=args.macaddr, ipaddr=str(args.ipaddr), netmask="255.0.0.0", default_gateway="10.0.0.1"
            )
        except Exception:
            logger.error("given IP address or MAC address looks invalid")
            raise

        if args.dry:
            e_path = Path("nonexistent.bit")
        else:
            e_path = obj.make_embedded_bit(bitpath=bitfiles[args.firmware], mempath=memfile_path)
            shutil.copy(e_path, cached_e_path)

    if args.dry:
        obj.dry_run(args.host, args.port, args.adapter, adapter_typeid)
    elif args.bit:
        if args.save:
            shutil.copy(e_path, ".")
        obj.program_bit(e_path, args.host, args.port, args.adapter, adapter_typeid)
    else:
        m_path = obj.make_mcs(e_path)
        if args.save:
            shutil.copy(m_path, ".")
        obj.program(m_path, args.host, args.port, args.adapter, adapter_typeid)
    return 0


def reboot_xil_fpga():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = Au50Programmer()
    parser = _common_parser(
        obj,
        "quel_reboot_fpga",
        "reboot the specified Xilinx FPGA via JTAG adapter",
        "",
        "",
        use_ipaddr=False,
        use_macaddr=False,
        use_firmware=False,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        obj.reboot(args.host, args.port, args.adapter)
    except Exception as e:
        logger.error(e)
        sys.exit(1)


def reboot_xil_au50():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = Au50Programmer()
    parser = _common_parser(
        obj,
        "quel_reboot_fpga",
        "reboot the specified Xilinx FPGA via JTAG adapter",
        "",
        "",
        use_ipaddr=False,
        use_macaddr=False,
        use_firmware=False,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        obj.reboot(args.host, args.port, args.adapter, "Xilinx/Alveo-DMBv1 FT4232H/")
    except Exception as e:
        logger.error(e)
        sys.exit(1)


def reboot_xil_au200():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = Au200Programmer()
    parser = _common_parser(
        obj,
        "quel_reboot_fpga",
        "reboot the specified Xilinx FPGA via JTAG adapter",
        "",
        "",
        use_ipaddr=False,
        use_macaddr=False,
        use_firmware=False,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        obj.reboot(args.host, args.port, args.adapter, "Xilinx/A-U200-A64G FT4232H/")
    except Exception as e:
        logger.error(e)
        sys.exit(1)
