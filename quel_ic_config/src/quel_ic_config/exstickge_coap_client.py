import asyncio
import logging
import queue
import threading
import time
from enum import Enum
from typing import Any, Callable, Final, Mapping, Set, Tuple, Union

import aiocoap
import aiocoap.error
from packaging.version import Version

from quel_ic_config.exstickge_proxy import LsiKindId, _ExstickgeProxyBase

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


class SyncAsyncCoapClient(threading.Thread):
    DEFAULT_COAP_PORT = 5683
    DEFAULT_COAP_RESPONSE_TIMEOUT: Final[float] = 3.0

    _DEFAULT_LOOPING_TIMEOUT: Final[float] = 0.25

    class _ProximityTransportTuning:
        # Notes: this tuning parameter is tailored for the default timeout (= 3.0 second).
        #        i.e., 0.15 * (1+2+4+8) * 1.25 = 2.8125 < 3.0
        ACK_TIMEOUT = 0.15
        ACK_RANDOM_FACTOR = 1.25
        MAX_RETRANSMIT = 3

    def __init__(self, looping_timeout: float = _DEFAULT_LOOPING_TIMEOUT):
        super().__init__()
        self.request_queue: queue.SimpleQueue[Union[SyncAsyncThunk, None]] = queue.SimpleQueue()
        self._looping_timeout = looping_timeout
        self._request_to_init_context = False

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
        self.request_queue.put(req)
        return req

    def request_and_wait(self, *, timeout: Union[float, None] = None, **args):
        for _ in range(3):
            req = SyncAsyncThunk(args)
            self.request_queue.put(req)
            try:
                return req.result(timeout=timeout)
            except aiocoap.error.NetworkError as e:
                logger.info(f"network error is detected: {e}")
                if not (len(e.args) > 0 and e.args[0].startswith("[Errno 111]")):
                    raise
        else:
            raise aiocoap.error.NetworkError("connection is refused by server repeatedly")

    def terminate(self) -> None:
        self.request_queue.put(None)

    async def _async_main(self):
        context = await aiocoap.Context.create_client_context()

        while True:
            try:
                req = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.request_queue.get(timeout=self._looping_timeout)
                )
                if req is None:
                    logger.info("None is detected in request_queue of SyncAsyncCoapClient")
                    break
                logger.debug(req._data)
                if self._request_to_init_context:
                    _ = await context.shutdown()
                    logger.info("recreating communication context for recovery")
                    context = await aiocoap.Context.create_client_context()
                    self._request_to_init_context = False
                _ = asyncio.create_task(self._single_access(req, context))
            except queue.Empty:
                # Notes: timeout is set to avoid blocking
                pass

        # Notes: executed only when Box object is explicitly deleted.
        logger.info(f"quitting main loop of {self.__class__.__name__}")
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


class Quel1seBoard(str, Enum):
    MIXER0 = "mx0"
    MIXER1 = "mx1"
    POWER = "pwr"
    PATHSEL0 = "ps0"
    PATHSEL1 = "ps1"


class _ExstickgeCoapClientBase(_ExstickgeProxyBase):
    DEFAULT_PORT: int = SyncAsyncCoapClient.DEFAULT_COAP_PORT
    DEFAULT_RESPONSE_TIMEOUT: float = SyncAsyncCoapClient.DEFAULT_COAP_RESPONSE_TIMEOUT

    _URI_MAPPINGS: Mapping[Tuple[LsiKindId, int], str]
    _READ_REG_PATHS: Mapping[LsiKindId, Callable[[int], str]]
    _WRITE_REG_PATHS_AND_PAYLOADS: Mapping[LsiKindId, Callable[[int, int], Tuple[str, str]]]

    _AVAILABLE_BOARDS: Set[Quel1seBoard]

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
        self._core = SyncAsyncCoapClient()
        self._core.start()

    def _coap_return_check(self, res: Union[aiocoap.Message, None], uri: str, raise_exception: bool = True) -> bool:
        if res is None:
            if raise_exception:
                raise RuntimeError(f"failed access to the end-point '{uri}' due to timeout")
            else:
                logger.error(f"failed access to the end-point '{uri}' due to timeout")
                return False
        if not res.code.is_successful():
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
        self._core.terminate()
        self._core.join()


def get_exstickge_server_info(
    ipaddr_css: str,
    port: int = SyncAsyncCoapClient.DEFAULT_COAP_PORT,
    timeout_ping: float = _DEFAULT_COAP_SERVER_DETECTION_TIMEOUT,
    timeout_coap: float = SyncAsyncCoapClient.DEFAULT_COAP_RESPONSE_TIMEOUT,
) -> Tuple[bool, str, str]:
    # Notes: assuming ipaddr_css is already validated
    # Notes: timeout_ping of 10.0 second comes from the booting duration of the server
    #        it was previously implemented with ping3.
    # Notes: port is not currently used
    uri1 = f"coap://{ipaddr_css}/version/firmware"
    uri2 = f"coap://{ipaddr_css}/conf/boxtype"
    res1: Union[aiocoap.Message, None] = None
    res2: Union[aiocoap.Message, None] = None

    core = SyncAsyncCoapClient()
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
            core = SyncAsyncCoapClient()
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
