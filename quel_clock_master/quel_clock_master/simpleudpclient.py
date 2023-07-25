import logging
import socket
import threading
import time
from typing import Final, Tuple, Union, cast

logger = logging.getLogger(__name__)


# TODO: use ipaddress.ip_address class for validation
class SimpleUdpClient:
    DEFAULT_TIMEOUT: Final[float] = 2.0  # s
    RECV_BUFSIZE: Final[int] = 16384  # Bytes
    SEND_MAX_PKTSIZE: Final[int] = 1440  # Bytes

    def __init__(self, ipaddr: str, receiver_limit_by_bind: bool = False, timeout=DEFAULT_TIMEOUT):
        self._server_ipaddr: str = ipaddr
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout * 0.5)
        self._timeout = timeout
        if receiver_limit_by_bind:
            self._sock.bind((self._get_my_ip_addr(), 0))
        else:
            self._sock.bind(("", 0))
        self._lock: threading.Lock = threading.Lock()

    def _get_my_ip_addr(self) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((self._server_ipaddr, 0))
        my_ipaddr = sock.getsockname()[0]
        sock.close()
        return my_ipaddr

    def _send_recv_generic(self, port: int, data: bytes) -> Tuple[bytes, Union[Tuple[str, int], None]]:
        if len(data) > self.SEND_MAX_PKTSIZE:
            raise ValueError(f"too large packget size ({len(data)} > {self.SEND_MAX_PKTSIZE})")

        with self._lock:
            self._sock.sendto(data, (self._server_ipaddr, port))
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < self._timeout:
                try:
                    reply, addr = self._sock.recvfrom(self.RECV_BUFSIZE)
                    if addr == (self._server_ipaddr, port):
                        return reply, cast(Tuple[str, int], addr)
                except socket.timeout:
                    pass
            else:
                logger.warning(f"failed to receive a reply from {(self._server_ipaddr, port)} due to timeout")
                return b"", None
