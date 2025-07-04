import asyncio
import hashlib
import json
import logging
import os
import sys
from ipaddress import IPv4Address
from pathlib import Path
from typing import Final

from aiocoap import GET, PUT, Context, Message
from packaging.version import Version

from quel_staging_tool import QuelXilinxFpgaProgrammerZephyr
from quel_staging_tool_consoleapps.common_parser import common_parser

logger = logging.getLogger()


EMPTY_HEXDIGEST: Final[str] = "7d44bb0955baeee64b2134d10e114fc0"


class _ProgrammingTransportTuning:
    MAX_LATENCY = 180.0
    DEFAULT_BLOCK_SIZE_EXP = 11

    ACK_TIMEOUT = 30.0
    ACK_RANDOM_FACTOR = 1.5
    MAX_RETRANSMIT = 4
    DEFAULT_LEISURE = 5
    EMPTY_ACK_DELAY = 0.1


async def update_and_verify(cmd_name: str, ipaddr: IPv4Address, payload: bytes, verify_len: int = 0):
    protocol = await Context.create_client_context()
    request0 = Message(
        code=GET,
        uri=f"coap://{str(ipaddr)}/version/firmware",
    )
    request1 = Message(
        code=GET,
        uri=f"coap://{str(ipaddr)}/lock/acquire?t=180",
    )
    request2 = Message(
        code=PUT,
        uri=f"coap://{str(ipaddr)}/sys/firmware/update",
        payload=payload,
        transport_tuning=_ProgrammingTransportTuning,
    )
    if verify_len != 0:
        request3 = Message(code=GET, uri=f"coap://{str(ipaddr)}/sys/firmware/md5?s={verify_len}")
    else:
        request3 = Message(code=GET, uri=f"coap://{str(ipaddr)}/sys/firmware/md5")
    request4 = Message(
        code=GET,
        uri=f"coap://{str(ipaddr)}/lock/release",
    )

    lock_acquired: bool = False
    try:
        response0 = await protocol.request(request0).response
        if not response0.code.is_successful():
            raise RuntimeError("failed to get the current firmware version")
        logger.info(f"current firmware version: {response0.payload.decode()}")
        need_to_lock: bool = cmd_name in {"quel_update_exstickge_1se", "quel_erase_exstickge_1se"} and Version(
            response0.payload.decode()
        ) >= Version("v1.3.0")
        if need_to_lock:
            response1 = await protocol.request(request1).response
            if not response1.code.is_successful():
                logger.error("failed to acquire lock, programming is aborted.")
                return None
            else:
                lock_acquired = True
                logger.info("lock is acquired")
        response2 = await protocol.request(request2).response
        if not response2.code.is_successful():
            logger.error(f"programming failure with code {response2.code}")
            return None
        else:
            logger.info("programming is done")
        # Notes: 'verify' is executed in the locked environment in order to prevent the other client from halting.
        #        be aware that the server cannot respond to any other requests during processing 'verify.'
        response3 = await protocol.request(request3).response
        if not response3.code.is_successful():
            logger.error(f"hash calculation failure with code {response3.code}")
            return None
        else:
            logger.info("hash calculation is done")

    except Exception as e:
        logger.error(f"failed to program the update region: {e}")
        return None
    finally:
        if lock_acquired:
            response4 = await protocol.request(request4).response
            if response4.code.is_successful():
                logger.info("lock is released")

    return response3


async def erase(cmd_name: str, ipaddr: IPv4Address):
    payload = bytes([0xFF]) * 2048
    return await update_and_verify(cmd_name, ipaddr, payload)


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
    pchfile = bitfile.parent / "bin_patch.json"

    try:
        if os.path.exists(pchfile):
            with open(pchfile) as f:
                pchdict = json.load(f)
        else:
            pchdict = {}
        eelfpath: Path = obj.make_embedded_elf(elfpath=elffile, ipaddr=args.ipaddr, patch_dict=pchdict)
    except Exception as e:
        logger.error("given IP address looks invalid")
        logger.error(e)
        sys.exit(1)

    if update_body(cmd_name, obj, args.ipaddr, bitfile, mmifile, eelfpath):
        logger.info("successful update, you need to reboot the target to run the updated firmware")
        sys.exit(0)
    else:
        logger.error(
            "failed update, retry this command to fix the firmware or fall-back firmware will start at the next boot"
        )
        sys.exit(1)


def update_body(
    cmd_name: str,
    obj: QuelXilinxFpgaProgrammerZephyr,
    ipaddr: IPv4Address,
    bitfile: Path,
    mmifile: Path,
    eelfpath: Path,
) -> bool:
    try:
        ebitpath: Path = obj.make_embedded_bit(bitpath=bitfile, mmipath=mmifile, elfpath=eelfpath)
        ebinpath: Path = obj.make_bin(bitpath=ebitpath)
        with open(ebinpath, "rb") as f:
            payload: bytes = f.read()
        hexdigest: str = hashlib.md5(payload).hexdigest()
        res1 = asyncio.run(update_and_verify(cmd_name, ipaddr, payload, len(payload)))
        if res1 is None:
            # Notes: error message is already generated.
            return False
        if res1.payload.decode() == hexdigest:
            logger.info("the update is verified")
        else:
            logger.error(
                "failed to verify the update content, "
                f"md5 digest should be {hexdigest} but actually is {res1.payload.decode()}"
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

    if erase_body(cmd_name, args.ipaddr):
        logger.info("successful removal of the updated firmware, fall-back firmware will run at the next reboot")
        sys.exit(0)
    else:
        logger.error(
            "failed clean-up of the update firmware. be aware that the update firmware may be disrupted partially"
        )
        sys.exit(1)


def erase_body(cmd_name: str, ipaddr: IPv4Address) -> bool:
    try:
        res1 = asyncio.run(erase(cmd_name, ipaddr))
        if res1 is None:
            # Notes: error message is already generated.
            return False
        if res1.payload.decode() == EMPTY_HEXDIGEST:
            logger.info("the update region is confirmed to be blank")
        else:
            logger.error(
                "failed to verify the clean-up of the update region, "
                f"md5 digest should be {EMPTY_HEXDIGEST} but actually is {res1.payload.decode()}"
            )
            return False
    except Exception as e:
        logger.error(e)
        return False

    return True
