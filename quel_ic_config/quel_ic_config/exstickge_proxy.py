import logging
import socket
import struct
import threading
import time
from enum import IntEnum
from typing import Tuple, Union

logger = logging.getLogger(__name__)


class LsiKindId(IntEnum):
    AD9082 = 1
    ADRF6780 = 2
    LMX2594 = 4
    AD5328 = 6
    GPIO = 7


class _LsiSpiId(IntEnum):
    AD9082_IF = LsiKindId.AD9082
    ADRF6780_IF_0 = LsiKindId.ADRF6780
    ADRF6780_IF_1 = LsiKindId.ADRF6780 + 1
    LMX2594_IF_0 = LsiKindId.LMX2594
    LMX2594_IF_1 = LsiKindId.LMX2594 + 1
    AD5328_IF = LsiKindId.AD5328
    GPIO_IF = LsiKindId.GPIO


class ExstickgeProxyQuel1:
    _PACKET_FORMAT = "!BBLH"  # MODE, I/F, ADDR, VALUE

    _MODE_READ_CMD = 0x80
    _MODE_READ_RPL = 0x81
    _MODE_WRITE_CMD = 0x82
    _MODE_WRITE_RPL = 0x83

    _NUM_AD9082 = 2
    _AD9082_AMASK = 0x7FFF
    _NUM_ADRF6780 = 8
    _ADRF6780_AMASK = 0x003F
    _NUM_LMX2594 = 10
    _LMX2594_AMASK = 0x007F
    _NUM_AD5328 = 1
    _AD5328_AMASK = 0x000F
    _NUM_GPIO = 1
    _GPIO_AMASK = 0x0000

    _MAX_RECV_TRIAL = 1000000  # for safety, must be harmless under the bombardment of wrong packtes.

    def __init__(
        self,
        target_address,
        target_port=16384,
        timeout: float = 2.0,
        receiver_limit_by_binding: bool = False,
        sock: Union[socket.socket, None] = None,
    ):
        self._target = (target_address, target_port)
        self._socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if sock is None else sock
        self._timeout = timeout
        self._socket.settimeout(self._timeout)
        if receiver_limit_by_binding:
            self._socket.bind((self._get_my_ip_addr(), 0))
        else:
            self._socket.bind(("", 0))
        self._lock: threading.Lock = threading.Lock()
        self._dump_enable = False

    def dump_enable(self):
        self._dump_enable = True

    def dump_disable(self):
        self._dump_enable = False

    def _get_my_ip_addr(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((self._target[0], 0))
        my_ip_addr = sock.getsockname()[0]
        sock.close()
        return my_ip_addr

    def _xlate(self, kind: LsiKindId, idx: int, addr: int) -> Tuple[int, int]:
        """Translating a triplet of lsi_type, lsi_idx, and register address into two-byte packet field.
        :param kind: a type of LSI.
        :param idx: an index of the LSI in the specified type.
        :param addr: an address of a register of the LSI to be accessed.
        :return: an encoded address word of a command packet.
        """
        if kind == LsiKindId.ADRF6780:
            if not (0 <= idx < self._NUM_ADRF6780):
                raise ValueError("invalid index of ADRF6780")
            elif idx < 4:
                return _LsiSpiId.ADRF6780_IF_0, (idx << 6) | (addr & self._AD9082_AMASK)
            else:
                return _LsiSpiId.ADRF6780_IF_1, ((idx - 4) << 6) | (addr & self._AD9082_AMASK)
        elif kind == LsiKindId.LMX2594:
            if not (0 <= idx < self._NUM_LMX2594):
                raise ValueError("invalid index of LMX2594")
            elif idx < 5:
                return _LsiSpiId.LMX2594_IF_0, (idx << 7) | (addr & self._LMX2594_AMASK)
            else:
                return _LsiSpiId.LMX2594_IF_1, ((idx - 5) << 7) | (addr & self._LMX2594_AMASK)
        elif kind == LsiKindId.AD5328:
            if not (0 <= idx < self._NUM_AD5328):
                raise ValueError("invalid index of AD5328")
            else:
                # most significant 4 bits of a command is considered as being an address.
                return _LsiSpiId.AD5328_IF, addr & self._AD5328_AMASK
        elif kind == LsiKindId.GPIO:
            if not (0 <= idx < self._NUM_GPIO):
                raise ValueError("invalid index of GPIO")
            else:
                # the second item of return value is always 0, actually.
                return _LsiSpiId.GPIO_IF, addr & self._GPIO_AMASK
        elif kind == LsiKindId.AD9082:
            # Note: the class is not in charge of accessing to MFE, usually.
            if not (0 <= idx < self._NUM_AD9082):
                raise ValueError("invalid index of AD9082")
            else:
                return _LsiSpiId.AD9082_IF, (idx << 15) | (addr & self._AD9082_AMASK)
        else:
            raise ValueError(f"invalid Lsi identifier '{kind}'")

    def _make_pkt(self, mode, if_idx, addr, value) -> bytes:
        return struct.pack(self._PACKET_FORMAT, mode, if_idx, addr, value)

    def _make_readpkt(self, kind: LsiKindId, idx: int, addr: int) -> bytes:
        if_idx, csel_addr = self._xlate(kind, idx, addr)
        return self._make_pkt(self._MODE_READ_CMD, if_idx, csel_addr, 0)

    def _make_writepkt(self, kind: LsiKindId, idx: int, addr: int, value: int) -> bytes:
        if_idx, csel_addr = self._xlate(kind, idx, addr)
        return self._make_pkt(self._MODE_WRITE_CMD, if_idx, csel_addr, value)

    def _send_and_recv(self, cmd: bytes) -> Union[bytes, None]:
        try:
            if self._dump_enable:
                cmdrepr = " ".join([f"{x:02x}" for x in cmd])
                logger.info(f"-> {cmdrepr} to {self._target}")

            time0 = time.time()
            self._socket.sendto(cmd, self._target)
            for _ in range(self._MAX_RECV_TRIAL):
                rpl, addr = self._socket.recvfrom(16)
                if self._dump_enable:
                    rplrepr = " ".join([f"{x:02x}" for x in rpl])
                    logger.info(f"-> {rplrepr} from {addr}")

                if len(rpl) == 8 and addr == self._target and rpl[0] == cmd[0] + 1 and rpl[1:6] == cmd[1:6]:
                    return rpl
                else:
                    rplrepr = " ".join([f"{x:02x}" for x in rpl])
                    if len(rplrepr) > 26:
                        rplrepr = rplrepr[:26] + "..."
                    logger.warning(f"unexpected packet '{rplrepr:s}' is read from {addr}, ignore it.")

                if time.time() - time0 >= self._timeout:
                    logger.error(f"failed to receive a packet from {self._target} due to time out.")
                    break
            else:
                logger.error("too much unexpected packets, give up to get the reply")
        except socket.timeout:
            logger.error(f"failed to send/receive a packet to/from {self._target} due to time out.")

        return None

    def read_reg(self, kind, idx, addr) -> Union[int, None]:
        with self._lock:
            cmd = self._make_readpkt(kind, idx, addr)
            rpl = self._send_and_recv(cmd)
            if rpl:
                return struct.unpack("!H", rpl[6:8])[0]
        return None

    def write_reg(self, kind, idx, addr, value) -> bool:
        with self._lock:
            cmd = self._make_writepkt(kind, idx, addr, value)
            rpl = self._send_and_recv(cmd)
            if rpl:
                return struct.unpack("!H", rpl[6:8])[0] == value
        return False
