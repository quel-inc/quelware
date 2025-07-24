import asyncio
import atexit
import logging
import os
import queue
import threading
import time
from abc import ABCMeta, abstractmethod
from enum import Enum
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Set, Tuple, Union
from weakref import WeakSet

import aiocoap
import aiocoap.error
import flufl.lock as fl
import ulid
from packaging.version import Version

from quel_ic_config.box_lock import BoxLockError
from quel_ic_config.exstickge_proxy import LsiKindId, _ExstickgeProxyBase
from quel_ic_config.quel_config_common import _DEFAULT_LOCK_DIRECTORY

_DEFAULT_COAP_SERVER_DETECTION_TIMEOUT: Final[float] = 30.0

logger = logging.getLogger(__name__)


class SyncAsyncThunk:
    def __init__(self, data: Any):
        self._data = data
        self._cv = threading.Condition()
        self._result_is_available: bool = False
        self._result: Any = None
        self._exception: Union[Exception, None] = None

    def set_result(self, result: Any) -> None:
        if self._result_is_available:
            raise RuntimeError("result was already put")
        with self._cv:
            self._result = result
            self._result_is_available = True
            self._cv.notify()

    def set_exception(self, exception: Exception) -> None:
        if self._result_is_available:
            raise RuntimeError("result was already put")
        with self._cv:
            self._exception = exception
            self._result_is_available = True
            self._cv.notify()

    def result(self, timeout=None) -> Any:
        with self._cv:
            self._cv.wait_for(lambda: self._result_is_available, timeout=timeout)

        if self._exception is not None:
            raise self._exception
        else:
            return self._result


class _BoxFailedUnlockError(Exception):
    pass


class AbstractSyncAsyncCoapClient(threading.Thread, metaclass=ABCMeta):
    DEFAULT_COAP_PORT = 5683
    DEFAULT_COAP_RESPONSE_TIMEOUT: Final[float] = 10.0

    _DEFAULT_LOOPING_TIMEOUT: Final[float] = 0.25

    _clients: WeakSet["AbstractSyncAsyncCoapClient"] = WeakSet()
    _create_lock: threading.Lock = threading.Lock()

    class _ProximityTransportTuning:
        # Notes: this tuning parameter is tailored for the default timeout (= 3.0 second).
        #        i.e., 0.3 * (1+2+4+8) * 1.25 = 5.625 < 10.0
        ACK_TIMEOUT = 0.3
        ACK_RANDOM_FACTOR = 1.25
        MAX_RETRANSMIT = 3

    @classmethod
    def release_lock_all(cls):
        with cls._create_lock:
            for c in cls._clients:
                if c._locked:
                    c._cleanup()
            # Notes: reconsider the following check is effective or not.
            for c in cls._clients:
                for _ in range(10):
                    if not c.has_lock:
                        break
                    time.sleep(cls._DEFAULT_LOOPING_TIMEOUT)
                else:
                    logger.error(f"failed to unlock {c._target[0]} due to timeout")

    @classmethod
    def release_lock_all_at_exit(cls):
        with cls._create_lock:
            for c in cls._clients:
                if c._locked:
                    c._cleanup_at_exit()

    def __init__(self, target: Tuple[str, int], looping_timeout: float = _DEFAULT_LOOPING_TIMEOUT):
        super().__init__()
        self._target = target
        self._request_queue: queue.SimpleQueue[Union[SyncAsyncThunk, None]] = queue.SimpleQueue()
        self._looping_timeout: float = looping_timeout
        self._lock_timestamp: float = 0.0
        self._request_to_init_context: bool = False
        self._locked: bool = False
        self._old_recovery_key: Union[ulid.ULID, None] = None

    def _register_self(self):
        with self._create_lock:
            for c in self._clients:
                if c._target[0] == self._target[0] and c.has_lock:
                    # Notes: should not happen
                    raise RuntimeError(f"duplicated proxy object for {self._target[0]}")
            else:
                self._clients.add(self)

    def _unregister_self(self):
        with self._create_lock:
            if not self.has_lock:
                try:
                    self._clients.remove(self)
                except KeyError:
                    # Notes: this means already removed automatically
                    pass
            else:
                raise _BoxFailedUnlockError(f"try to delete live lock for {self._target[0]}")

    @property
    def has_lock(self) -> bool:
        return self._locked

    def run(self):
        loop = asyncio.new_event_loop()
        task = loop.create_task(
            self._async_main(),
        )

        while not task.done():
            try:
                loop.run_until_complete(task)
            except KeyboardInterrupt:
                raise
            except RuntimeError as e:
                if len(e.args) > 0 and e.args[0] == "cannot schedule new futures after shutdown":
                    # asyncio/base_events.py(647) raises those after the object is deleted.
                    break
                elif len(e.args) > 0 and e.args[0] == "cannot schedule new futures after interpreter shutdown":
                    # concurrent/features/thread.py(169) raises those after the object is deleted.
                    break
                else:
                    raise
            except SystemExit:
                # TODO: understand it carefully.
                #         asyncio/tasks.py(242) raises those after setting them as results,
                #         but we particularly want them back in the loop
                continue

        loop.close()

    def request(self, **args) -> SyncAsyncThunk:
        req = SyncAsyncThunk(args)
        self._request_queue.put(req)
        return req

    def request_and_wait(self, *, timeout: Union[float, None] = None, **args):
        for _ in range(3):
            req = SyncAsyncThunk(args)
            self._request_queue.put(req)
            try:
                return req.result(timeout=timeout)
            except aiocoap.error.NetworkError as e:
                logger.info(f"network error is detected: {e}")
                if not (len(e.args) > 0 and e.args[0].startswith("[Errno 111]")):
                    raise
        else:
            raise aiocoap.error.NetworkError("connection is refused by server repeatedly")

    def terminate(self) -> None:
        if hasattr(self, "_request_queue"):
            self._request_queue.put(None)
            for _ in range(5):
                if not self._locked:
                    break
                time.sleep(self._looping_timeout / 2)
            self._unregister_self()  # Notes: raise RuntimeException for ClientWithDeviceLock holding a lock at exit.

    @abstractmethod
    async def _take_lock(self, context, with_token: bool) -> None: ...

    @abstractmethod
    async def _keep_lock(self, context) -> bool: ...

    @abstractmethod
    async def _release_lock(self, context, key: Union[ulid.ULID, None] = None) -> bool: ...

    @abstractmethod
    def _check_lock_at_host(self, data: Any) -> bool: ...

    async def _async_main(self):
        context = await aiocoap.Context.create_client_context()

        is_first_loop = True
        while True:
            try:
                if self._old_recovery_key is not None:
                    _ = await self._release_lock(context, self._old_recovery_key)
                    self._old_recovery_key = None
                    logger.info(f"the previous lock of {self._target[0]} is released successfully")

                if self._request_to_init_context:
                    _ = await context.shutdown()
                    logger.info(f"recreating communication context of {self._target[0]} for recovery")
                    context = await aiocoap.Context.create_client_context()
                    self._request_to_init_context = False

                if is_first_loop or self.has_lock:
                    if not (await self._keep_lock(context)):
                        logger.info(f"fails to keep lock of {self._target[0]}")
                        await self._release_lock(context)

                req = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._request_queue.get(timeout=self._looping_timeout)
                )  # Notes: queue.Empty exception is raised if no requests come.

                if req is None:
                    break
                else:
                    if self._check_lock_at_host(req._data):
                        _ = asyncio.create_task(self._single_access(req, context))
                    else:
                        _ = asyncio.create_task(self._failed_access(req))

            except queue.Empty:
                # Notes: timeout is set to avoid blocking
                pass
            finally:
                is_first_loop = False

        # Notes: executed only when Box object is explicitly deleted.
        logger.info(f"quitting main loop of {self.__class__.__name__} of {self._target[0]}")
        _ = await self._release_lock(context)
        _ = await context.shutdown()

    async def _single_access(self, req: SyncAsyncThunk, context: aiocoap.Context):
        if "transport_tuning" not in req._data:
            msg = aiocoap.Message(**req._data, transport_tuning=self._ProximityTransportTuning)
        else:
            msg = aiocoap.Message(**req._data)
        try:
            res = await context.request(msg).response
            req.set_result(res)
        except aiocoap.error.NetworkError as e:
            if len(e.args) > 0 and e.args[0].startswith("[Errno 111]"):
                self._request_to_init_context = True
            if len(e.args) > 0 and e.args[0] == "Retransmissions exceeded":
                # Notes: QuEL-1 boxes are always connected to the same hub that the Host PC is connected to.
                #        so, both timeout and retransmission exceeded mean the unavailability of the box.
                req.set_result(None)
            else:
                req.set_exception(e)
        except Exception as e:
            req.set_exception(e)

    async def _failed_access(self, req: SyncAsyncThunk):
        req.set_exception(BoxLockError(f"device lock of {self._target[0]} is not available now"))

    @abstractmethod
    def _cleanup(self): ...

    @abstractmethod
    def _cleanup_at_exit(self): ...


# Notes: for bootstrap of CSS and debug. (no lock is required for the bootstrap, actually)
# Notes: be aware that this lock should not be delegated to the access control of e7awghw APIs.
class SyncAsyncCoapClientWithDummyLock(AbstractSyncAsyncCoapClient):
    async def _take_lock(self, context, with_token: bool) -> None:
        self._locked = True  # Notes: not locked, actually.
        # Notes: do not invoke self._register_self() here, it is meaningless and harmful.

    async def _keep_lock(self, context) -> bool:
        if not self._locked:
            await self._take_lock(context, False)
        return True

    async def _release_lock(self, context, key: Union[ulid.ULID, None] = None) -> bool:
        self._release_lock_body()
        return True

    def _release_lock_body(self) -> None:
        self._locked = False

    def _check_lock_at_host(self, data: Any) -> bool:
        return True

    def _cleanup(self):
        self._release_lock_body()

    def _cleanup_at_exit(self):
        pass

    def terminate(self) -> None:
        super().terminate()
        self._release_lock_body()


class SyncAsyncCoapClientWithFileLock(AbstractSyncAsyncCoapClient):
    # Notes: no lock mechanism is unavailable in v1.2.1 and former firmware.
    #        the substitute lock mechanism is implemented.

    _DEFAULT_LOCK_EXPIRATION: Final[float] = 15.0  # [s]
    _FLOCK_TIMEOUT: int = 5  # [s]
    _KILL_CHECK_PERIOD: int = 10  # cycles

    def __init__(
        self,
        target: Tuple[str, int],
        lock_directory: Path = _DEFAULT_LOCK_DIRECTORY,
        looping_timeout: float = AbstractSyncAsyncCoapClient._DEFAULT_LOOPING_TIMEOUT,
    ):
        super().__init__(target=target, looping_timeout=looping_timeout)
        if not lock_directory.is_dir():
            raise RuntimeError(f"lock directory '{lock_directory}' is unavailable")
        self._lockfile: str = str(lock_directory / self._target[0])
        self._lockobj: fl.Lock = fl.Lock(lockfile=self._lockfile, lifetime=int(self._DEFAULT_LOCK_EXPIRATION))
        self._killfile: str = f"{self._lockobj.claimfile}" + ".kill"
        self._kill_check_counter = cycle([True] + (self._KILL_CHECK_PERIOD - 1) * [False])

    async def _take_lock(self, context, with_token: bool) -> None:
        self._lock_timestamp = time.perf_counter()  # no next chance to take a lock.
        try:
            self._lockobj.lock(timeout=self._FLOCK_TIMEOUT)
            self._locked = True
            self._register_self()
            logger.info(f"successfully acquired the lock of {self._target[0]}")
        except fl.TimeOutError:
            logger.error(f"failed to acquire lock of {self._target[0]}")

    async def _keep_lock(self, context) -> bool:
        if self._lock_timestamp == 0.0:
            await self._take_lock(context, False)
        elif self.has_lock:
            if next(self._kill_check_counter):
                if self._check_if_killfile_exists():
                    logger.info(f"Cancelled to refresh lock of {self._target[0]} due to the killfile.")
                    return False

            try:
                self._lockobj.refresh()
                self._lock_timestamp = time.perf_counter()
            except fl.NotLockedError:
                logger.error(f"failed to refresh lock of {self._target[0]}")
                return False
        return self.has_lock

    def _check_if_killfile_exists(self) -> bool:
        return os.path.exists(self._killfile)

    async def _release_lock(self, context, key: Union[ulid.ULID, None] = None) -> bool:
        self._release_lock_body()
        return True

    def _release_lock_body(self) -> None:
        self._lockobj.unlock(unconditionally=True)
        self._locked = False

    def _check_lock_at_host(self, data: Any) -> bool:
        return self.has_lock

    def _cleanup(self):
        self._release_lock_body()

    def _cleanup_at_exit(self):
        self._release_lock_body()


class SyncAsyncCoapClientWithDeviceLock(AbstractSyncAsyncCoapClient):
    _LOCK_EXPIRATION = 60.0  # [s]
    _LOCK_REACQUIRE_PERIOD = 45.0  # [s]
    _RECOVERY_KEYLEN = 10  # Notes: this comes from option length constraint of CoAP without enabling
    #        CONFIG_COAP_EXTENDED_OPTIONS_LEN.
    _RECOVERY_KEY_DIRECTORY: Path = Path(os.getenv("HOME", "/tmp")) / ".quelware" / "recovery_keys"

    def __init__(
        self,
        target: Tuple[str, int],
        looping_timeout: float = AbstractSyncAsyncCoapClient._DEFAULT_LOOPING_TIMEOUT,
    ):
        super().__init__(target=target, looping_timeout=looping_timeout)
        self._target: Tuple[str, int] = target
        self._old_recovery_key = self._load_old_recovery_key()
        self._recovery_key: ulid.ULID = ulid.ULID()

    def _load_old_recovery_key(self) -> Union[ulid.ULID, None]:
        filepath = self._RECOVERY_KEY_DIRECTORY / f"{self._target[0]}.txt"
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:

                    k = ulid.ULID.from_str(f.read())
                os.remove(filepath)  # Notes: saved key can be used only once.
                delta = time.time() - k.timestamp
                if delta < self._LOCK_EXPIRATION * 1.2:
                    return k
                else:
                    logger.info(f"the previous key is created {round(delta)} seconds ago, already expired")
            except Exception as e:
                logger.info(f"failed to load the previous key due to {e}")
        else:
            logger.info("no previous key is found")
        return None

    def _shorten_key(self, k: Union[ulid.ULID, None] = None) -> str:
        if k is None:
            return str(self._recovery_key)[-self._RECOVERY_KEYLEN :]
        else:
            return str(k)[-self._RECOVERY_KEYLEN :]

    async def _take_lock(self, context, with_key: bool) -> None:
        self._lock_timestamp = time.perf_counter()
        if with_key:
            msg = aiocoap.Message(
                code=aiocoap.GET, uri=f"coap://{self._target[0]}/lock/acquire?k={self._shorten_key()}"
            )
            res = await context.request(msg).response
            if res.code.is_successful():
                self._locked = True
                self._register_self()
                logger.info(f"successfully acquired the lock of {self._target[0]}")
            elif res.code == aiocoap.FORBIDDEN:
                raise BoxLockError(f"device lock of {self._target[0]} is not available now")
            else:
                raise RuntimeError(f"failed to acquire the lock of {self._target[0]}")
        else:
            msg = aiocoap.Message(code=aiocoap.GET, uri=f"coap://{self._target[0]}/lock/acquire")
            res = await context.request(msg).response
            if not res.code.is_successful():
                self._locked = False
                raise RuntimeError(f"failed to extend the lock of {self._target[0]}")

    async def _keep_lock(self, context) -> bool:
        if self._lock_timestamp == 0.0:
            try:
                await self._take_lock(context, with_key=True)
            except BaseException as e:
                logger.warning(e)
        elif self._locked:
            if self._request_to_init_context:
                # Notes: reacquiring the lock with my token
                await self._take_lock(context, with_key=True)
            else:
                cur = time.perf_counter()
                if cur - self._lock_timestamp >= self._LOCK_REACQUIRE_PERIOD:
                    # Notes: just extending a lock
                    await self._take_lock(context, with_key=False)
        return self._locked

    async def _release_lock(self, context, key: Union[ulid.ULID, None] = None) -> bool:
        if key is None:
            msg = aiocoap.Message(code=aiocoap.GET, uri=f"coap://{self._target[0]}/lock/release")
        else:
            msg = aiocoap.Message(
                code=aiocoap.GET, uri=f"coap://{self._target[0]}/lock/release?k={self._shorten_key(key)}"
            )

        try:
            res = await context.request(msg).response
            if res.code.is_successful():
                self._locked = False
                logger.info(f"lock of {self._target[0]} is released successfully")
                return True
            else:
                # Notes: try to unlock irrespective of has_lock, however, don't show failure log if not has_lock.
                if self.has_lock:
                    logger.warning(f"failed to release the lock of {self._target[0]} with code: {res.code}")
                return False
        except Exception as e:
            logger.warning(f"failed to release the lock of {self._target[0]} with exception: {e}")
            return False

    def _check_lock_at_host(self, data: Any) -> bool:
        # Notes: host check is skipped. firmware is in charge of judging permission.
        return True

    def _cleanup(self):
        if hasattr(self, "_request_queue"):
            res1 = self.request_and_wait(
                code=aiocoap.GET,
                uri=f"coap://{self._target[0]}/lock/release",
                timeout=self.DEFAULT_COAP_RESPONSE_TIMEOUT,
            )
            if res1.code.is_successful():
                self._locked = False
                logger.info(f"lock of {self._target[0]} is released successfully")

    def _cleanup_at_exit(self):
        os.makedirs(self._RECOVERY_KEY_DIRECTORY, exist_ok=True)
        filepath = self._RECOVERY_KEY_DIRECTORY / f"{self._target[0]}.txt"
        with open(filepath, "w") as f:
            # Notes: write the recovery key after updating its timestamp (i know this is an unusual usage of ULID)
            z = ulid.ULID()
            f.write(f"{str(z)[:10]}{str(self._recovery_key)[10:]}")
            f.flush()
            f.close()
        logger.info(f"recovery key of {self._target[0]} is written out to {filepath}")

    def terminate(self) -> None:
        try:
            super().terminate()
        except _BoxFailedUnlockError:
            pass


atexit.register(AbstractSyncAsyncCoapClient.release_lock_all_at_exit)


class Quel1seBoard(str, Enum):
    MIXER0 = "mx0"
    MIXER1 = "mx1"
    POWER = "pwr"
    PATHSEL0 = "ps0"
    PATHSEL1 = "ps1"


class _ExstickgeCoapClientBase(_ExstickgeProxyBase):
    DEFAULT_PORT: int = AbstractSyncAsyncCoapClient.DEFAULT_COAP_PORT
    DEFAULT_RESPONSE_TIMEOUT: float = AbstractSyncAsyncCoapClient.DEFAULT_COAP_RESPONSE_TIMEOUT

    _URI_MAPPINGS: Mapping[Tuple[LsiKindId, int], str]
    _READ_REG_PATHS: Mapping[LsiKindId, Callable[[int], str]]
    _WRITE_REG_PATHS_AND_PAYLOADS: Mapping[LsiKindId, Callable[[int, int], Tuple[str, str]]]

    _AVAILABLE_BOARDS: Tuple[Quel1seBoard, ...]

    _DATA_MASKS: Mapping[LsiKindId, int] = {
        LsiKindId.AD9082: 0xFF,
        LsiKindId.ADRF6780: 0xFFFF,
        LsiKindId.LMX2594: 0xFFFF,
        LsiKindId.AD5328: 0xFFFF,
        LsiKindId.MIXERBOARD_GPIO: 0xFFFF,
        LsiKindId.PATHSELECTORBOARD_GPIO: 0x003F,
        LsiKindId.AD7490: 0xFFFF,
        LsiKindId.POWERBOARD_PWM: 0xFFFF,
    }

    _VALID_BOXTYPE: Set[str] = set()
    _VERSION_SPEC: Tuple[Version, Version, Set[Version]] = Version("0.0.0"), Version("0.0.0"), set()

    @classmethod
    def matches(cls, btstr: str, vstr: str):
        minv, maxv, excvs = cls._VERSION_SPEC
        v = Version(vstr)
        return (btstr in cls._VALID_BOXTYPE) and (minv <= v <= maxv) and (v not in excvs)

    def __init__(
        self,
        target_address: str,
        target_port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_RESPONSE_TIMEOUT,
    ):
        super().__init__(target_address, target_port, timeout)
        self._core: AbstractSyncAsyncCoapClient = self._creating_core()
        self._core.start()

    @property
    def has_lock(self) -> bool:
        return self._core.has_lock

    @property
    def available_boards_with_cpld(self) -> Tuple[Quel1seBoard, ...]:
        return self._AVAILABLE_BOARDS

    def _creating_core(self) -> AbstractSyncAsyncCoapClient:
        return SyncAsyncCoapClientWithDummyLock(target=self._target)

    def _coap_return_check(self, res: Union[aiocoap.Message, None], uri: str, raise_exception: bool = True) -> bool:
        if res is None:
            if raise_exception:
                raise RuntimeError(f"failed access to the end-point '{uri}' due to timeout")
            else:
                logger.error(f"failed access to the end-point '{uri}' due to timeout")
                return False
        if not res.code.is_successful():
            # Notes: DeviceLockException is not subject to be controlled by raise_exception argument.
            if res.code == aiocoap.FORBIDDEN:
                raise BoxLockError(f"device lock of {self._target[0]} is not available now")
            if raise_exception:
                raise RuntimeError(f"failed access to the end-point '{uri}' with an error code {res.code}")
            else:
                logger.error(f"failed access to the end-point '{uri}' with an error code {res.code}")
                return False
        return True

    def read_fpga_version(self) -> Tuple[str, str]:
        uri1 = f"coap://{self._target[0]}/version/fpga"
        uri2 = f"coap://{self._target[0]}/hash/fpga"
        res1 = self._core.request_and_wait(code=aiocoap.GET, uri=uri1, timeout=self._timeout)
        res2 = self._core.request_and_wait(code=aiocoap.GET, uri=uri2, timeout=self._timeout)
        self._coap_return_check(res1, uri1)
        self._coap_return_check(res2, uri2)
        return res1.payload.decode(), res2.payload.decode()

    def read_firmware_version(self) -> str:
        uri1 = f"coap://{self._target[0]}/version/firmware"
        res1 = self._core.request_and_wait(code=aiocoap.GET, uri=uri1, timeout=self._timeout)
        self._coap_return_check(res1, uri1)
        return res1.payload.decode()

    def read_boxtype(self) -> str:
        # Notes: some early-stage firmware doesn't implement this API.
        #        those will be eliminated soon.
        uri1 = f"coap://{self._target[0]}/conf/boxtype"
        res1 = self._core.request_and_wait(code=aiocoap.GET, uri=uri1, timeout=self._timeout)
        self._coap_return_check(res1, uri1)
        return res1.payload.decode()

    def read_current_config(self) -> str:
        # Notes: some early-stage firmware doesn't implement this API.
        #        those will be eliminated soon.
        uri1 = f"coap://{self._target[0]}/conf/current"
        res1 = self._core.request_and_wait(code=aiocoap.GET, uri=uri1, timeout=self._timeout)
        self._coap_return_check(res1, uri1)
        return res1.payload.decode()

    def write_current_config(self, cfg: str):
        # Notes: some early-stage firmware doesn't implement this API.
        #        those will be eliminated soon.
        if len(cfg) > 256:
            raise ValueError("too long config data")
        uri1 = f"coap://{self._target[0]}/conf/current"
        res1 = self._core.request_and_wait(code=aiocoap.PUT, payload=cfg.encode(), uri=uri1, timeout=self._timeout)
        self._coap_return_check(res1, uri1)

    def read_board_active(self, board: Quel1seBoard) -> bool:
        if board in self._AVAILABLE_BOARDS:
            uri1 = f"coap://{self._target[0]}/{board.value}/xbar/reset"
            res1 = self._core.request_and_wait(code=aiocoap.GET, uri=uri1, timeout=self._timeout)
            if res1 is not None:
                if res1.code.is_successful():
                    v = res1.payload.decode()
                    if v == "0":
                        return False  # the board is in reset state
                    elif v == "1":
                        return True
                    else:
                        raise RuntimeError(f"unexpected return value: '{res1}' from the end-point '{uri1}'")
                else:
                    raise RuntimeError(f"failed to read active status of the board {board}")
            else:
                raise RuntimeError(f"failed access to the end-point '{uri1}'")
        else:
            raise ValueError(f"the specified board {board} is not available")

    def write_board_active(self, board: Quel1seBoard, active: bool) -> None:
        if board in self._AVAILABLE_BOARDS:
            uri1 = f"coap://{self._target[0]}/{board.value}/xbar/reset"
            v = "1" if active else "0"
            res1 = self._core.request_and_wait(code=aiocoap.PUT, payload=v.encode(), uri=uri1, timeout=self._timeout)
            self._coap_return_check(res1, uri1)
        else:
            raise ValueError(f"the specified board {board} is not available")

    def read_reset(self, kind: LsiKindId, idx: int) -> Union[int, None]:
        if (kind, idx) not in self._URI_MAPPINGS:
            raise ValueError(f"invalid IC ({kind.name}, {idx})")

        if kind == LsiKindId.AD9082:
            uri = f"coap://{self._target[0]}/{self._URI_MAPPINGS[kind, idx]}/reset"
            res = self._core.request_and_wait(code=aiocoap.GET, uri=uri, timeout=self._timeout)
            self._coap_return_check(res, uri)
            return int(res.payload.decode())
        else:
            return None

    def write_reset(self, kind: LsiKindId, idx: int, value: int) -> bool:
        if (kind, idx) not in self._URI_MAPPINGS:
            raise ValueError(f"invalid IC ({kind.name}, {idx})")
        if value not in {0, 1}:
            raise ValueError(f"invalid value for reset: {value}")

        if kind == LsiKindId.AD9082:
            uri = f"coap://{self._target[0]}/{self._URI_MAPPINGS[kind, idx]}/reset"
            res = self._core.request_and_wait(
                code=aiocoap.PUT, uri=uri, payload=str(value).encode(), timeout=self._timeout
            )
            self._coap_return_check(res, uri)
            return True  # processed
        else:
            return False  # delegated

    def read_reg(self, kind: LsiKindId, idx: int, addr: int) -> Union[int, None]:
        if (kind, idx) not in self._URI_MAPPINGS:
            raise ValueError(f"invalid IC ({kind.name if isinstance(kind, LsiKindId) else kind}, {idx})")
        if (addr & self._ADDR_MASKS[kind]) != addr:
            raise ValueError(f"address {addr} of {kind.name} is out of range ")

        if kind in self._READ_REG_PATHS:
            uri = f"coap://{self._target[0]}/{self._URI_MAPPINGS[kind, idx]}" + self._READ_REG_PATHS[kind](addr)
        else:
            raise ValueError(f"no read register API is available for {kind}")

        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri, timeout=self._timeout)
        if self._coap_return_check(res, uri, False):
            return int(res.payload.decode(), 16) & self._DATA_MASKS[kind]
        else:
            return None

    def write_reg(self, kind: LsiKindId, idx: int, addr: int, value: int) -> bool:
        if (kind, idx) not in self._URI_MAPPINGS:
            raise ValueError(f"invalid IC ({kind.name if isinstance(kind, LsiKindId) else kind}, {idx})")
        if (addr & self._ADDR_MASKS[kind]) != addr:
            raise ValueError(f"address {addr} of {kind.name} is out of range ")
        if (value & self._DATA_MASKS[kind]) != value:
            raise ValueError(f"value {value} of {kind.name} is out of range ")

        if kind in self._WRITE_REG_PATHS_AND_PAYLOADS:
            api_path, payload = self._WRITE_REG_PATHS_AND_PAYLOADS[kind](addr, value)
            uri = f"coap://{self._target[0]}/{self._URI_MAPPINGS[kind, idx]}" + api_path
        else:
            raise ValueError(f"no write register API is available for {kind}")

        res = self._core.request_and_wait(code=aiocoap.PUT, uri=uri, payload=payload.encode(), timeout=self._timeout)
        return self._coap_return_check(res, uri, False)

    def terminate(self):
        if hasattr(self, "_core"):
            self._core.terminate()
            self._core.join()
            if self._core._locked:
                logger.warning(
                    f"you should delete a box object ({self._target[0]}) explicitly to release its lock properly"
                )


def get_exstickge_server_info(
    ipaddr_css: str,
    port: int = AbstractSyncAsyncCoapClient.DEFAULT_COAP_PORT,
    timeout_ping: float = _DEFAULT_COAP_SERVER_DETECTION_TIMEOUT,
    timeout_coap: float = AbstractSyncAsyncCoapClient.DEFAULT_COAP_RESPONSE_TIMEOUT,
) -> Tuple[bool, str, str]:
    # Notes: assuming ipaddr_css is already validated
    # Notes: timeout_ping of 10.0 second comes from the booting duration of the server
    #        it was previously implemented with ping3.
    # Notes: port is not currently used
    uri1 = f"coap://{ipaddr_css}/version/firmware"
    uri2 = f"coap://{ipaddr_css}/conf/boxtype"
    res1: Union[aiocoap.Message, None] = None
    res2: Union[aiocoap.Message, None] = None

    # Notes: reading box information with temporary coap client
    core = SyncAsyncCoapClientWithDummyLock(target=(ipaddr_css, port))
    core.start()

    t0 = time.perf_counter()
    while (time.perf_counter() < t0 + timeout_ping) and (res1 is None):
        try:
            res1 = core.request_and_wait(code=aiocoap.GET, uri=uri1, timeout=timeout_coap)
            if res1 is None:
                logger.info(f"no CoAP response from {ipaddr_css}")
                time.sleep(3)
        except aiocoap.error.NetworkError:
            core.terminate()
            time.sleep(3)
            core = SyncAsyncCoapClientWithDummyLock(target=(ipaddr_css, port))
            core.start()
            res1 = None

    if res1 is not None:
        res2 = core.request_and_wait(code=aiocoap.GET, uri=uri2, timeout=timeout_coap)
    core.terminate()
    del core

    # Notes: validating the box information
    if res1 is None:
        logger.info(f"timed out the detection of CoAP server on {ipaddr_css}")
        return False, "", ""  # UDP server, probably
    elif not res1.code.is_successful():
        raise RuntimeError(f"failed to access to end-point '{uri1}' with error code {res1.code}")
    else:
        v = res1.payload.decode()
        logger.info(f"CoAP server version of {ipaddr_css} is '{v}'")
        if res2 and res2.code.is_successful():
            u = res2.payload.decode()
            logger.info(f"reported boxtype of {ipaddr_css} is '{u}'")
            return True, v, u
        else:
            logger.info(f"boxtype information of {ipaddr_css} is not available due to error")
            return True, v, ""
