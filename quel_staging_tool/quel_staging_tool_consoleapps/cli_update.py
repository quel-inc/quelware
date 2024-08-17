import asyncio
import hashlib
import logging
import sys
from ipaddress import IPv4Address
from pathlib import Path
from typing import Final

from aiocoap import GET, PUT, Context, Message

from quel_staging_tool import QuelXilinxFpgaProgrammerZephyr
from quel_staging_tool_consoleapps.common_parser import common_parser

logger = logging.getLogger()


EMPTY_HEXDIGEST: Final[str] = "7d44bb0955baeee64b2134d10e114fc0"


class _ProgrammingTransportTuning:
    ACK_TIMEOUT = 2.0
    ACK_RANDOM_FACTOR = 1.5
    MAX_LATENCY = 180.0
    DEFAULT_BLOCK_SIZE_EXP = 10
    MAX_RETRANSMIT = 4


async def update(ipaddr: IPv4Address, payload: bytes):
    protocol = await Context.create_client_context()
    request = Message(
        code=PUT,
        uri=f"coap://{str(ipaddr)}/sys/firmware/update",
        payload=payload,
        transport_tuning=_ProgrammingTransportTuning,
    )

    try:
        response = await protocol.request(request).response
    except Exception as e:
        logger.error(f"failed to program the update region: {e}")
        return None

    else:
        return response


async def verify(ipaddr: IPv4Address, size: int = 0):
    protocol = await Context.create_client_context()
    if size <= 0:
        request = Message(code=GET, uri=f"coap://{str(ipaddr)}/sys/firmware/md5")
    else:
        request = Message(code=GET, uri=f"coap://{str(ipaddr)}/sys/firmware/md5?s={size}")

    try:
        response = await protocol.request(request).response
    except Exception as e:
        logger.error(f"failed to acquire md5 hash of the update region: {e}")
        return None
    else:
        return response


async def erase(ipaddr: IPv4Address):
    protocol = await Context.create_client_context()
    # Notes: I couldn't find a good way to set block_size to 1024 with null payload. this is just a workaround.
    request = Message(
        code=PUT,
        uri=f"coap://{str(ipaddr)}/sys/firmware/update",
        payload=bytes([0xFF]) * 2048,
        transport_tuning=_ProgrammingTransportTuning,
    )

    try:
        response = await protocol.request(request).response
    except Exception as e:
        logger.error(f"failed to erase the update region: {e}")
        return None
    else:
        return response


def update_exstickge_clockdisty():
    update_exstickge_zephyr_common("quel_update_exstickge_clkdisty", "quel_clk_distributor_for_update.bit")


def update_exstickge_1se():
    update_exstickge_zephyr_common("quel_update_exstickge_1se", "quel1_config_for_update.bit")


def update_exstickge_zephyr_common(cmd_name: str, bitfile_name: str):
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    obj = QuelXilinxFpgaProgrammerZephyr()
    parser = common_parser(
        cmd_name,
        "writing the specified zephyr-based firmware with MAC and IP addresses into the **update region** of "
        "flash memory of the specified ExStickGE",
        "ExStickGE",
        bitfile_names=list(obj.get_bits(bitfile_name=bitfile_name)),
        use_firmware_dir=True,
        use_macaddr=False,
        use_port=False,
        use_adapter=False,
        use_dry=False,
        use_bit=False,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    bitfiles = obj.get_bits(bitdir_path=args.firmware_dir, bitfile_name=bitfile_name)
    if args.firmware not in bitfiles:
        logger.error(f"invalid firmware: {args.firmware}")
        sys.exit(1)

    bitfile: Path = bitfiles[args.firmware]
    elffile: Path = bitfile.parent / "zephyr.elf"
    mmifile: Path = bitfile.parent / "itcm.mmi"

    try:
        eelfpath: Path = obj.make_embedded_elf(elfpath=elffile, ipaddr=args.ipaddr)
    except Exception as e:
        logger.error("given IP address looks invalid")
        logger.error(e)
        sys.exit(1)

    if update_body(obj, args.ipaddr, bitfile, mmifile, eelfpath):
        logger.info("successful update, you need to reboot the target to run the updated firmware")
        sys.exit(0)
    else:
        logger.error(
            "failed update, retry this command to fix the firmware or fall-back firmware will start at the next boot"
        )
        sys.exit(1)


def update_body(
    obj: QuelXilinxFpgaProgrammerZephyr, ipaddr: IPv4Address, bitfile: Path, mmifile: Path, eelfpath: Path
) -> bool:
    try:
        ebitpath: Path = obj.make_embedded_bit(bitpath=bitfile, mmipath=mmifile, elfpath=eelfpath)
        ebinpath: Path = obj.make_bin(bitpath=ebitpath)
        with open(ebinpath, "rb") as f:
            payload: bytes = f.read()
        hexdigest: str = hashlib.md5(payload).hexdigest()
        res1 = asyncio.run(update(ipaddr, payload))
        if res1 is None:
            return False
        res2 = asyncio.run(verify(ipaddr, size=len(payload)))
        if res2 is None:
            return False
        if res2.payload.decode() == hexdigest:
            logger.info("the updated content is verified")
        else:
            logger.error(
                "failed to verify the update content, "
                f"md5 digest should be {hexdigest} but actually is {res2.payload.decode()}"
            )
            return False
    except Exception as e:
        logger.error(e)
        return False

    return True


def erase_exstickge_clockdisty():
    erase_exstickge_zephyr_common("quel_erase_exstickge_clockdisty")


def erase_exstickge_1se():
    erase_exstickge_zephyr_common("quel_erase_exstickge_1se")


def erase_exstickge_zephyr_common(cmd_name: str):
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = common_parser(
        cmd_name,
        "writing the specified zephyr-based firmware with MAC and IP addresses into the update region of "
        "flash memory of the specified ExStickGE",
        "ExStickGE",
        bitfile_names=[],
        use_firmware=False,
        use_firmware_dir=False,
        use_macaddr=False,
        use_port=False,
        use_adapter=False,
        use_dry=False,
        use_bit=False,
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if erase_body(args.ipaddr):
        logger.info("successful removal of the updated firmware, fall-back firmware will run at the next reboot")
        sys.exit(0)
    else:
        logger.error("failed clean-up of the update firmware although the update firmware is probably inactivated")
        sys.exit(1)


def erase_body(ipaddr: IPv4Address) -> bool:
    try:
        res1 = asyncio.run(erase(ipaddr))
        if res1 is None:
            return False
        res2 = asyncio.run(verify(ipaddr))
        if res2 is None:
            return False
        if res2.payload.decode() == EMPTY_HEXDIGEST:
            logger.info("the clean-up of the update region is verified")
        else:
            logger.error(
                "failed to verify the clean-up of the update region, "
                f"md5 digest should be {EMPTY_HEXDIGEST} but actually is {res2.payload.decode()}"
            )
            return False
    except Exception as e:
        logger.error(e)
        return False

    return True
