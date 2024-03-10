import logging
import socket
import struct
import threading
import time
from typing import Mapping, Tuple, Union

from quel_ic_config.exstickge_proxy import LsiKindId, _ExstickgeProxyBase

logger = logging.getLogger(__name__)


class _ExstickgeSockClientBase(_ExstickgeProxyBase):
    _DEFAULT_RESPONSE_TIMEOUT: float = 0.5
    _DEFAULT_PORT: int = 16384

    _PACKET_FORMAT = "!BBLH"  # MODE, I/F, ADDR, VALUE

    _MODE_READ_CMD = 0x80
    _MODE_READ_RPL = 0x81
    _MODE_WRITE_CMD = 0x82
    _MODE_WRITE_RPL = 0x83

    _MAX_RECV_TRIAL = 1000000  # for safety, must be harmless under the bombardment of wrong packtes.

    _SPIIF_MAPPINGS: Mapping[LsiKindId, Mapping[int, Tuple[int, int]]] = {}

    def __init__(
        self,
        target_address: str,
        target_port: int = _DEFAULT_PORT,
        timeout: float = _DEFAULT_RESPONSE_TIMEOUT,
        receiver_limit_by_binding: bool = False,
        sock: Union[socket.socket, None] = None,
    ):
        super().__init__(target_address, target_port, timeout)
        self._socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if sock is None else sock
        self._socket.settimeout(self._timeout)
        if receiver_limit_by_binding:
            self._socket.bind((self._get_my_ip_addr(), 0))
        else:
            self._socket.bind(("", 0))
        self._lock: threading.Lock = threading.Lock()

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

        if not LsiKindId.has_value(kind):
            raise ValueError(f"invalid Lsi identifier '{kind}'")

        if kind not in self._SPIIF_MAPPINGS:
            raise ValueError(f"no instance of '{kind}' is no available on this board")

        spiif_idx, cs_idx = self._SPIIF_MAPPINGS[kind][idx]

        if kind == LsiKindId.ADRF6780:
            return spiif_idx, (cs_idx << 6) | (addr & self._ADDR_MASKS[kind])
        elif kind == LsiKindId.LMX2594:
            return spiif_idx, (cs_idx << 7) | (addr & self._ADDR_MASKS[kind])
        elif kind == LsiKindId.AD5328:
            return spiif_idx, (cs_idx << 4) | (addr & self._ADDR_MASKS[kind])
        elif kind == LsiKindId.GPIO:
            return spiif_idx, cs_idx | (addr & self._ADDR_MASKS[kind])
        elif kind == LsiKindId.AD9082:
            return spiif_idx, (idx << 15) | (addr & self._ADDR_MASKS[kind])
        else:
            raise ValueError(f"unexpected LsiKindId for socket client: {kind.name}")

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

    def terminate(self):
        self._socket.close()
