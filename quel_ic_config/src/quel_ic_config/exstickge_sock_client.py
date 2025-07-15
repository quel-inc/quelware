import atexit
import logging
import os
import socket
import struct
import threading
import time
from abc import ABCMeta, abstractmethod
from itertools import cycle
from pathlib import Path
from typing import Final, Mapping, Tuple, Union
from weakref import WeakSet

import flufl.lock as fl

from quel_ic_config.box_lock import BoxLockError
from quel_ic_config.exstickge_proxy import LsiKindId, _ExstickgeProxyBase
from quel_ic_config.quel_config_common import _DEFAULT_LOCK_DIRECTORY

logger = logging.getLogger(__name__)


class AbstractLockKeeper(threading.Thread, metaclass=ABCMeta):
    _DEFAULT_LOOP_WAIT: Final[float] = 0.25  # [s]
    _DEFAULT_RELEASE_TIMEOUT: Final[float] = 2.5
    _DEFAULT_LOCK_EXPIRATION: float

    _clients: WeakSet["AbstractLockKeeper"] = WeakSet()
    _clients_lock: threading.RLock = threading.RLock()

    @classmethod
    def release_lock_all(cls):
        with cls._clients_lock:
            # Notes: just for speeding up
            for c in cls._clients:
                c._to_release = True
            for c in cls._clients:
                c.deactivate()

    def __init__(self, *, target: tuple[str, int], loop_wait: float = _DEFAULT_LOOP_WAIT):
        super().__init__()
        self._target: tuple[str, int] = target
        self._loop_wait = loop_wait
        self._lock_timestamp: float = 0.0
        self._to_release: bool = False
        # Note: must be daemon to allow invocation of release_lock_all at exit
        #       even if some lock_keepers are still alive.
        self.daemon = True

    def _register_self(self):
        with self._clients_lock:
            if not self.has_lock:
                raise RuntimeError(f"try to register proxy object for {self._target[0]} which doesn't have lock")

            for c in self._clients:
                if c._target[0] == self._target[0] and c.has_lock:
                    # Notes: this doesn't happen usually, just in case.
                    raise RuntimeError(f"duplicated proxy object for {self._target[0]}")
            else:
                self._clients.add(self)

    def _unregister_self(self):
        with self._clients_lock:
            if not self.has_lock:
                try:
                    self._clients.remove(self)
                except KeyError:
                    pass
            else:
                raise RuntimeError(f"try to unregister live lock for {self._target[0]}")

    @property
    @abstractmethod
    def has_lock(self) -> bool: ...

    @abstractmethod
    def _take_lock(self) -> bool: ...

    @abstractmethod
    def _keep_lock(self) -> bool: ...

    @abstractmethod
    def _release_lock(self) -> None: ...

    def activate(self) -> bool:
        if self._take_lock():
            self._lock_timestamp = time.perf_counter()
            self._register_self()
            self.start()
            return True
        else:
            return False

    def run(self):
        while not self._to_release:
            if self.has_lock:
                cur = time.perf_counter()
                if (cur - self._lock_timestamp) > self._DEFAULT_LOCK_EXPIRATION * 0.5:
                    if self._keep_lock():
                        self._lock_timestamp = cur
                    else:
                        break
            time.sleep(self._loop_wait)
        logger.info(f"quitting run() of {self.__class__.__name__} of {self._target[0]}")
        self._release_lock()
        self._unregister_self()

    def deactivate(self, timeout: float = _DEFAULT_RELEASE_TIMEOUT) -> bool:
        # Notes: must be called from the other thread than one running self.
        self._to_release = True
        joined = False
        try:
            self.join(timeout)
            joined = True
        except TimeoutError:
            logger.warning(f"failed to join the lock keeper of {self._target[0]}")

        if self.has_lock:
            # Notes: this should not happen if joined successfully, and join should work.
            logger.error(f"failed to unlock {self._target[0]} due to timeout")

        return joined


class DummyLockKeeper(AbstractLockKeeper):
    _DEFAULT_LOCK_EXPIRATION: float = 0  # Notes: dummy value

    def __init__(self, *, target: tuple[str, int], loop_wait: float = AbstractLockKeeper._DEFAULT_LOOP_WAIT):
        super().__init__(target=target, loop_wait=loop_wait)
        self._locked: bool = False  # Notes: state of dummy lock

    @property
    def has_lock(self) -> bool:
        return self._locked

    def _take_lock(self) -> bool:
        self._locked = True
        return True

    def _keep_lock(self) -> bool:
        return True

    def _release_lock(self) -> None:
        self._locked = False


class FileLockKeeper(AbstractLockKeeper):
    _DEFAULT_LOCK_EXPIRATION = 15.0  # [s]
    _FLOCK_TIMEOUT: int = 5  # [s]
    _KILL_CHECK_PERIOD: int = 10

    def __init__(
        self,
        *,
        target: tuple[str, int],
        loop_wait: float = AbstractLockKeeper._DEFAULT_LOOP_WAIT,
        lock_directory: Path = _DEFAULT_LOCK_DIRECTORY,
    ):
        super().__init__(target=target, loop_wait=loop_wait)
        if not lock_directory.is_dir():
            raise RuntimeError(f"lock directory '{lock_directory}' is unavailable")
        self._lockfile: str = str(lock_directory / self._target[0])
        self._lockobj: fl.Lock = fl.Lock(lockfile=self._lockfile, lifetime=int(self._DEFAULT_LOCK_EXPIRATION))

        self._killfile: str = f"{self._lockobj.claimfile}" + ".kill"
        self._kill_check_counter = cycle([True] + (self._KILL_CHECK_PERIOD - 1) * [False])

    @property
    def has_lock(self) -> bool:
        try:
            return self._lockobj.is_locked
        except FileNotFoundError as e:
            logger.warning(e)
            return False

    def _take_lock(self) -> bool:
        try:
            self._lockobj.lock(timeout=self._FLOCK_TIMEOUT)
            logger.info(f"successfully acquired the lock of {self._target[0]}")
            return True
        except fl.TimeOutError:
            logger.warning(f"failed to acquire lock of {self._target[0]}")
            return False

    def _keep_lock(self) -> bool:
        if next(self._kill_check_counter):
            if self._check_if_killfile_exists():
                logger.info(f"Cancelled to refresh lock of {self._target[0]} due to the killfile.")
                return False

        try:
            self._lockobj.refresh()
        except Exception as e:
            logger.warning(e)
            return False
        return True

    def _check_if_killfile_exists(self) -> bool:
        return os.path.exists(self._killfile)

    def _release_lock(self) -> None:
        try:
            self._lockobj.unlock()
            logger.info(f"lock of {self._target[0]} is released successfully")
        except fl.NotLockedError:
            pass


atexit.register(AbstractLockKeeper.release_lock_all)


class _ExstickgeSockClientBase(_ExstickgeProxyBase):
    DEFAULT_RESPONSE_TIMEOUT: float = 0.5
    DEFAULT_PORT: int = 16384

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
        target_port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_RESPONSE_TIMEOUT,
        receiver_limit_by_binding: bool = False,
    ):
        super().__init__(target_address, target_port, timeout)
        self._receiver_limit_by_binding: bool = receiver_limit_by_binding
        self._sock: Union[socket.socket, None] = None
        self._lock: threading.Lock = threading.Lock()
        # TODO: consider the best timing of lock keeper creation.
        self._lock_keeper: Union[AbstractLockKeeper, None] = self._create_lockkeeper()

    def initialize(self):
        with self._lock:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(self._timeout)
            if self._receiver_limit_by_binding:
                self._sock.bind((self._get_my_ip_addr(), 0))
            else:
                self._sock.bind(("", 0))

    @abstractmethod
    def _create_lockkeeper(self) -> AbstractLockKeeper: ...

    @property
    def _socket(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError(f"proxy of {self._target[0]} is not initialized")
        return self._sock

    @property
    def has_lock(self) -> bool:
        return (self._lock_keeper is not None) and self._lock_keeper.has_lock

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
        # Notes: lock should be acquired by caller
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
            # Notes: for CSS firmware corresponding to the sock client, any simultaneous accesses can fail.
            if not self.has_lock:
                raise BoxLockError(f"lock of {self._target[0]} is not available")
            cmd = self._make_readpkt(kind, idx, addr)
            rpl = self._send_and_recv(cmd)
            if rpl:
                return struct.unpack("!H", rpl[6:8])[0]
        return None

    def write_reg(self, kind, idx, addr, value) -> bool:
        with self._lock:
            if not self.has_lock:
                raise BoxLockError(f"lock of {self._target[0]} is not available")
            cmd = self._make_writepkt(kind, idx, addr, value)
            rpl = self._send_and_recv(cmd)
            if rpl:
                return struct.unpack("!H", rpl[6:8])[0] == value
        return False

    def terminate(self):
        # Notes: terminate() is call by __del__() of Quel1ConfigSubsystemRoot and __del__() of self.__del__() defined
        #        at _ExstickgeProxyBase. terminate() should be defined to be idempotent.
        if self._sock is not None:
            self._sock.close()
            self._sock = None

        if hasattr(self, "_lock_keeper") and self._lock_keeper is not None and self._lock_keeper.is_alive():
            if self._lock_keeper.deactivate():
                self._lock_keeper = None
