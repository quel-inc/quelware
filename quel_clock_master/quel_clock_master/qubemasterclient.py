import logging
import struct
from typing import Final, List, Sequence, Tuple, Union

from quel_clock_master.simpleudpclient import SimpleUdpClient

logger = logging.getLogger(__name__)


class QuBEMasterClient(SimpleUdpClient):
    DEFAULT_PORT: Final[int] = 16384
    DEFAULT_RESET_PORT: Final[int] = 16385

    def __init__(
        self,
        master_ipaddr: str,
        master_port: Union[int, None] = None,
        master_reset_port: Union[int, None] = None,
        receiver_limit_by_bind: bool = False,
        timeout=SimpleUdpClient.DEFAULT_TIMEOUT,
    ):
        super().__init__(master_ipaddr, receiver_limit_by_bind, timeout)
        self._master_port: int = self.DEFAULT_PORT if master_port is None else master_port
        self._master_reset_port: int = self.DEFAULT_RESET_PORT if master_reset_port is None else master_reset_port

    # TODO: improving packet assembling. the current implementation causes many buffer-copy.
    def kick_clock_synch(self, target_addrs: Sequence[str]) -> bool:
        targets: List[Tuple[int, int]] = [(self._conv2addr(a), 0x4001) for a in target_addrs]

        data = struct.pack("BBHHH", 0x32, 0, 0, 0, 0)
        for addr, port in targets:
            data += struct.pack(">I", addr)
            data += struct.pack(">I", port)

        logger.debug(f"sending {':'.join(['{0:02x}'.format(x) for x in data])}")
        reply, raddr = self._send_recv_generic(self._master_port, data)
        if raddr is None:
            logger.warning("communication failure in kick_clock_synch()")
        else:
            logger.debug(f"receiving {':'.join(['{0:02x}'.format(x) for x in reply])} from {raddr[0]:s}:{raddr[1]:d}")
            if reply[0] in (0xFE, 0xFF):
                logger.warning(
                    "a state machine of the clock master may be hanged up, please **RESET** it before re-trying 'kick'"
                )
            elif reply[0] != 0x33:
                logger.warning(f"unexpected reply packet starting with {reply[0]:02x} is received")

        return (raddr is not None) and (reply[0] == 0x33)

    def clear_clock(self, value: int = 0) -> bool:
        data = struct.pack("BBHHH", 0x34, 0, 0, 0, 0)
        data += struct.pack("<Q", value)

        logger.debug(f"sending {':'.join(['{0:02x}'.format(x) for x in data])}")
        reply, raddr = self._send_recv_generic(self._master_port, data)
        if raddr is None:
            logger.warning("communication failure in clear_clock()")
        else:
            logger.debug(f"receiving {':'.join(['{0:02x}'.format(x) for x in reply])} from {raddr[0]:s}:{raddr[1]:d}")
            if reply[0] != 0x33:
                logger.warning(f"unexpected reply packet starting with {reply[0]:02x} is received")

        return (raddr is not None) and (reply[0] == 0x33)

    def read_clock(self, value: int = 0) -> Tuple[bool, int]:
        data = struct.pack("BBHHH", 0x30, 0, 0, 0, 0)
        data += struct.pack("<Q", value)

        logger.debug(f"sending {':'.join(['{0:02x}'.format(x) for x in data])}")
        reply, raddr = self._send_recv_generic(self._master_port, data)
        if raddr is None:
            logger.warning("communication failure in read_clock()")
            clock = -1
        else:
            logger.debug(f"receiving {':'.join(['{0:02x}'.format(x) for x in reply])} from {raddr[0]:s}:{raddr[1]:d}")
            clock = struct.unpack("<Q", reply[8:])[0]

        return (raddr is not None) and (reply[0] == 0x33), clock

    def reset(self) -> bool:
        data = "dummy message".encode("utf-8")
        logger.debug(f"sending {':'.join(['{0:02x}'.format(x) for x in data])}")
        reply, raddr = self._send_recv_generic(self._master_reset_port, data)
        if raddr is None:
            logger.warning("communication failure in reset()")
        else:
            logger.debug(
                f"receiving from reset {':'.join(['{0:02x}'.format(x) for x in reply])} from {raddr[0]:s}:{raddr[1]:d}"
            )
        return (raddr is not None) and (reply == b"dummy message")

    def _conv2addr(self, addr_str: str) -> int:
        a = 0
        for s in addr_str.split("."):
            a = (a << 8) | int(s)
        return a


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.DEBUG, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ipaddr_master", required=True)
    parser.add_argument("--port", type=int, default=QuBEMasterClient.DEFAULT_PORT)
    parser.add_argument("--reset_port", type=int, default=QuBEMasterClient.DEFAULT_RESET_PORT)
    parser.add_argument("--command", choices=("clear", "read", "kick", "reset"), required=True)
    parser.add_argument("--value", type=int, default=0)
    parser.add_argument("ipaddr_targets", type=str, nargs="*")
    args = parser.parse_args()

    proxy = QuBEMasterClient(args.ipaddr_master, args.port, args.reset_port)
    retcode: bool = False

    if args.command == "clear":
        retcode = proxy.clear_clock(value=args.value)
        if retcode:
            logger.info("cleared successfully")
        else:
            logger.error("failure in cleaning")
    elif args.command == "read":
        retcode, clock = proxy.read_clock(value=args.value)
        if retcode:
            logger.info(f"the clock reading is '{clock:d}'")
        else:
            logger.error("failure in reading the clock")
    elif args.command == "kick":
        if len(args.ipaddr_targets) > 0:
            retcode = proxy.kick_clock_synch(args.ipaddr_targets)
            if retcode:
                logger.info("kicked successfully")
            else:
                logger.error("failure in kicking the targets")
        else:
            logger.error("'kick' command requires IP addresses of targets")
    elif args.command == "reset":
        retcode = proxy.reset()
        if retcode:
            logger.info("reset successfully")
        else:
            logger.error("failure in reset")
    else:
        # this cannot happen.
        raise AssertionError

    if not retcode:
        sys.exit(1)
