import logging
import struct
from typing import Final, Tuple, Union

from quel_clock_master.simpleudpclient import SimpleUdpClient

logger = logging.getLogger()


class SequencerClient(SimpleUdpClient):
    DEFAULT_SEQR_PORT: Final[int] = 16384
    DEFAULT_SYNCH_PORT: Final[int] = 16385

    def __init__(
        self,
        target_ipaddr: str,
        seqr_port: Union[int, None] = None,
        synch_port: Union[int, None] = None,
        receiver_limit_by_bind: bool = False,
        timeout=SimpleUdpClient.DEFAULT_TIMEOUT,
    ):
        super().__init__(target_ipaddr, receiver_limit_by_bind, timeout)
        self._seqr_port: int = self.DEFAULT_SEQR_PORT if seqr_port is None else seqr_port
        self._synch_port: int = self.DEFAULT_SYNCH_PORT if synch_port is None else synch_port

    @property
    def ipaddress(self) -> str:
        """returns IP adderess to allow QubeMasterClient to kick this box
        :return: IP address of the sequencer subsystem of the box
        """
        return self._server_ipaddr

    def kick_softreset(self) -> bool:
        data = struct.pack("BBBB", 0xE0, 0x00, 0x00, 0x00)
        _, raddr = self._send_recv_generic(self._seqr_port, data)
        if raddr is None:
            logger.warning(f"communication failure with {self._server_ipaddr} in kick_softreset()")
        else:
            logger.info(f"{self._server_ipaddr} is reset successfully")
        return raddr is not None

    def add_sequencer(self, clock: int, awg_bitmap: int = 0xFFFF) -> bool:
        data = struct.pack("BB", 0x22, 0)
        data += struct.pack("HH", 0, 0)
        data += struct.pack(">H", 16)  # 1-command = 16bytes
        data += struct.pack("<Q", clock)  # start time
        data += struct.pack("<H", awg_bitmap)  # target AWG
        data += struct.pack("BBBBB", 0, 0, 0, 0, 0)  # padding
        data += struct.pack("B", 0)  # entry id
        _, raddr = self._send_recv_generic(self._seqr_port, data)
        if raddr is None:
            logger.warning(f"communication failure with {self._server_ipaddr} in kick_sequencer()")
        else:
            logger.info(f"scheduled command is added to {self._server_ipaddr} successfully")
        return raddr is not None

    def read_clock(self) -> Tuple[bool, int, int]:
        data = struct.pack("BBBB", 0x00, 0x00, 0x00, 0x04)

        logger.debug(f"sending {':'.join(['{0:02x}'.format(x) for x in data])}")
        reply_, raddr = self._send_recv_generic(self._synch_port, data)
        reply = memoryview(reply_)

        clock: int = -1
        sysref_latch: int = -1
        flag: bool = False

        if raddr is None:
            logger.warning("communication failure in clear_clock()")
        else:
            logger.debug(f"receiving {':'.join(['{0:02x}'.format(x) for x in reply])} from {raddr[0]:s}:{raddr[1]:d}")
            if reply[0] == 0x00 and reply[1] == 0x00 and reply[2] == 0x00 and reply[3] == 0x05:
                try:
                    if len(reply) == 12:
                        clock = struct.unpack(">Q", reply[4:12])[0]
                        flag = True
                    elif len(reply) == 20:
                        clock, sysref_latch = struct.unpack(">QQ", reply[4:20])
                        flag = True
                    else:
                        logger.warning(f"a packet with wrong length ({len(reply)}) is received, ignore it")
                except struct.error as e:
                    logger.error(e)
            else:
                logger.warning(f"unexpected reply packet starting with {reply[0]:02x} is received")

        return flag, clock, sysref_latch


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.DEBUG, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = argparse.ArgumentParser()
    parser.add_argument("ipaddr_targets", type=str, nargs="+", help="IP addresses of the target boxes")
    parser.add_argument("--seqr_port", type=int, default=SequencerClient.DEFAULT_SEQR_PORT)
    parser.add_argument("--synch_port", type=int, default=SequencerClient.DEFAULT_SYNCH_PORT)
    parser.add_argument(
        "--command", type=str, choices=("sched", "reset", "read"), required=True, help="command to execute"
    )
    parser.add_argument("--delta", type=float, default=5.0, help="delta time from now to start AWGs")
    args = parser.parse_args()

    current: int = 0
    if args.command == "sched":
        target0 = SequencerClient(args.ipaddr_targets[0], args.seqr_port, args.synch_port)
        retcode0, current, _ = target0.read_clock()
        if not retcode0:
            logger.error(f"failed to read clock from {args.ipaddr_targets[0]}, aborted.")
            sys.exit(1)

    flag: bool = True
    for ipaddr_target in args.ipaddr_targets:
        target = SequencerClient(ipaddr_target, args.seqr_port, args.synch_port)
        if args.command == "reset":
            retcode = target.kick_softreset()
            if retcode:
                logger.warning(f"{ipaddr_target}: success")
            else:
                flag = False
                logger.warning(f"{ipaddr_target}: failed")
        elif args.command == "sched":
            ttx = current + int(args.delta * 125000000)  # 125M = 1sec
            retcode = target.add_sequencer(ttx)
            if retcode:
                logger.warning(f"{ipaddr_target}: success")
            else:
                flag = False
                logger.warning(f"{ipaddr_target}: failed")
        elif args.command == "read":
            retcode, clock, last_sysref = target.read_clock()
            if retcode:
                logger.info(f"{ipaddr_target}: {clock} {last_sysref}")
            else:
                flag = False
                logger.info(f"{ipaddr_target}: failed")
        else:
            # never happens
            raise AssertionError

    sys.exit(0 if flag else 1)
