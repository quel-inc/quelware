import ipaddress
import logging
import time
from abc import ABCMeta, abstractmethod
from collections.abc import Collection
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from enum import Enum
from functools import cached_property
from threading import RLock
from typing import Callable, Final, Generic, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from e7awghal import (
    AbstractCapCtrl,
    AbstractCapParam,
    AbstractCapUnit,
    AbstractQuel1Au50Hal,
    AwgCtrl,
    AwgParam,
    AwgUnit,
    CapCtrlSimpleMulti,
    CapIqDataReader,
    ClockcounterCtrl,
    E7awgCaptureDataError,
    E7FwType,
    SimplemultiAwgTriggers,
    SimplemultiSequencer,
    create_quel1au50hal,
)
from e7awghal.common_defs import _DEFAULT_POLLING_PERIOD

logger = logging.getLogger(__name__)


class CaptureReturnCode(Enum):
    CAPTURE_TIMEOUT = 1
    CAPTURE_ERROR = 2
    BROKEN_DATA = 3
    SUCCESS = 4


RT = TypeVar("RT")  # notes: return type
CVRT = TypeVar("CVRT")  # notes: converted return type
CCT = TypeVar("CCT", bound=AbstractCapCtrl)  # notes: capture controller type


class AbstractCancellableTask(
    Generic[RT],
    metaclass=ABCMeta,
):
    __slots__ = {
        "_pool",
        "_args",
        "_request_cancel",
        "_cancel_failed",
        "_cancelled",
        "_future",
        "_lock",
    }

    def __init__(self, pool: ThreadPoolExecutor, *args):
        self._pool = pool
        self._args = args
        self._request_cancel: bool = False
        self._cancel_failed: bool = False
        self._cancelled: bool = False
        self._future: Union[Future[RT], None] = None
        self._lock = RLock()

    @abstractmethod
    def _body(self, *args) -> RT: ...

    def _future_cancel_hook(self) -> None:
        pass

    def _cancellable_wait_loop(
        self,
        check_callback: Callable[[], bool],
        timeout_callback: Callable[[], None],
        cancel_callback: Callable[[], None],
        timeout: Optional[float],
        polling_period: float,
    ) -> None:
        t0 = time.perf_counter()
        while timeout is None or (time.perf_counter() < t0 + timeout):
            time.sleep(polling_period)
            if check_callback():
                break
            if self._request_cancel:
                cancel_callback()
        else:
            timeout_callback()

    def start(self) -> None:
        with self._lock:
            if self._future is None:
                self._future = self._pool.submit(self._body, *self._args)
            else:
                raise RuntimeError("already started")

    def cancel(self, timeout: Optional[float] = None, polling_period: Optional[float] = None) -> bool:
        if polling_period is None:
            if hasattr(self, "_polling_period"):
                polling_period_: float = getattr(self, "_polling_period") or _DEFAULT_POLLING_PERIOD
            else:
                polling_period_ = _DEFAULT_POLLING_PERIOD
        else:
            polling_period_ = polling_period

        if self._future is None:
            raise RuntimeError("not started yet")

        with self._lock:
            self._cancelled = self._future.cancel()
            if not self._cancelled:
                self._request_cancel = True
                t0 = time.perf_counter()
                while timeout is None or time.perf_counter() < t0 + timeout:
                    time.sleep(polling_period_)
                    if isinstance(self._future.exception(), CancelledError):
                        self._cancelled = True
                    if not self._future.running():
                        break
                else:
                    logger.warning(f"failed to stop a task {self.__class__.__name__}({self._args})")
                    self._cancel_failed = True
            else:
                self._future_cancel_hook()

            return self._cancelled

    def result(self, timeout: Optional[float] = None) -> RT:
        if self._future is None:
            raise RuntimeError("not started yet")

        with self._lock:
            if self._cancel_failed:
                if not self.cancel(timeout):
                    raise RuntimeError("cancel request is still pending")
            return self._future.result(timeout)

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        if self._future is None:
            raise RuntimeError("not started yet")
        return self._future.exception(timeout)

    def is_started(self) -> bool:
        return self._future is not None

    def running(self) -> bool:
        if self._future:
            return self._future.running()
        else:
            return False

    def done(self) -> bool:
        if self._future:
            # TODO: clarify done() should True or not when canncelled.
            return self._future.done() and not self._cancelled
        else:
            return False

    def cancelled(self) -> bool:
        if self._future:
            return self._future.cancelled() or self._cancelled
        else:
            return False


class AbstractCancellableTaskWrapper(Generic[RT, CVRT], metaclass=ABCMeta):
    def __init__(self, task: AbstractCancellableTask[RT]):
        self._task = task

    def start(self) -> None:
        self._task.start()

    def cancel(self, timeout: Optional[float] = None, polling_period: Optional[float] = None) -> bool:
        return self._task.cancel(timeout, polling_period)

    @abstractmethod
    def _conveter(self, orig: RT) -> CVRT: ...

    def result(self, timeout: Optional[float] = None) -> CVRT:
        return self._conveter(self._task.result(timeout))

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        return self._task.exception(timeout)

    def is_started(self) -> bool:
        return self._task.is_started()

    def running(self) -> bool:
        return self._task.running()

    def done(self) -> bool:
        return self._task.done()

    def cancelled(self) -> bool:
        return self._task.cancelled()


class AbstractStartAwgunitsTask(AbstractCancellableTask[None]):
    _DONE_TIMEOUT_MARGIN: Final[float] = 0.25
    _CANCEL_DONE_TIMEOUT: Final[float] = 1.0
    _WORD_RATE: Final[int] = 125_000_000  # [Hz]

    def __init__(
        self,
        pool: ThreadPoolExecutor,
        awgctrl: AwgCtrl,
        awgunits: dict[int, AwgUnit],
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ):
        if len(awgunits) == 0:
            raise ValueError("no awg units are specified")
        super().__init__(pool)
        self._awgctrl: AwgCtrl = awgctrl
        self._awgunits: dict[int, AwgUnit] = awgunits
        self._timeout: Union[float, None] = None if disable_timeout else (timeout or self._default_done_timeout)
        self._polling_period: float = polling_period or _DEFAULT_POLLING_PERIOD
        self._has_started_emission: bool = False

    def has_started_emission(self) -> bool:
        return self._has_started_emission

    def wait_for_starting_emission(self) -> None:
        # Notes: usually you don't have to use this method. possible use cases is taking spectrum snapshot with
        #        a spectrum analyzer after confirming RF signal is actually generating.
        #        so, blocking API is enough.
        # Notes: this API should not raise exception since any failures in the main task should be noticed when
        #        either of task.result() or task.exception() is invoked.
        # Notes: this loop should exit in 0.5 (= _PRELOAD_TIMEOUT + _START_TIMEOUT) second or so for TaskNow.
        while not (self._has_started_emission or self.cancelled()):
            if self.done() and not self._has_started_emission:
                # Notes: it should not happen, basically.
                logger.error("emission has not been started due to failures in the task thread")
                break
            time.sleep(self._polling_period)

    @cached_property
    def _hwidxs(self) -> set[int]:
        return {u.unit_index for u in self._awgunits.values()}

    @cached_property
    def _idxs_string(self) -> str:
        return "awgunits" + ", ".join([f"-#{u}" for u in self._awgunits])

    @cached_property
    def _default_done_timeout(self) -> float:
        return (
            max([u.get_wave_duration() for u in self._awgunits.values()]) / self._WORD_RATE + self._DONE_TIMEOUT_MARGIN
        )

    def _clear_done_all(self, raise_exception: bool = False):
        for u in self._awgunits.values():
            u.clear_done()
        if self._awgctrl.are_done_any(self._hwidxs):
            if raise_exception:
                raise RuntimeError(f"failed to clear done flag, some of {self._idxs_string} are still asserted")
            else:
                logger.warning(f"failed to clear done flag, some of {self._idxs_string} are still asserted")

    def _terminate_all(self):
        for u in self._awgunits.values():
            u.terminate(timeout=self._CANCEL_DONE_TIMEOUT, polling_period=self._polling_period).result()

    @abstractmethod
    def _activate_awgunits(self) -> None: ...

    @abstractmethod
    def _wait_for_preparation(self) -> None: ...

    @abstractmethod
    def _wait_for_start(self) -> None: ...

    def _check_start_callback(self) -> bool:
        with self._awgctrl.lock:
            if self._awgctrl.have_started_all(self._hwidxs):
                self._awgctrl.check_error(self._hwidxs)
                return True
            else:
                return False

    def _check_done_callback(self) -> bool:
        with self._awgctrl.lock:
            if self._awgctrl.are_done_all(self._hwidxs):
                self._clear_done_all()
                self._awgctrl.check_error(self._hwidxs)
                return True
            else:
                return False

    def _timeout_callback(self, timeout_msg: str):
        # Notes: don't use hal.awgctrl.terminate(), it wouldn't work well...
        self._terminate_all()
        self._awgctrl.check_error(self._hwidxs)
        raise TimeoutError(timeout_msg)

    def _cancel_callback(self, cancel_msg: str):
        # Notes: don't use hal.awgctrl.terminate(), it wouldn't work well...
        self._terminate_all()
        self._awgctrl.check_error(self._hwidxs)
        raise CancelledError(cancel_msg)

    def _body(self) -> None:
        with self._awgctrl.lock:
            if self._awgctrl.are_busy_any(self._hwidxs):
                raise RuntimeError(f"stop preloading because some of {self._idxs_string} are still busy")
            self._clear_done_all(raise_exception=True)
            self._activate_awgunits()

        self._wait_for_preparation()

        self._wait_for_start()

        self._cancellable_wait_loop(
            self._check_done_callback,
            lambda: self._timeout_callback(f"failed to finish operation of some of {self._idxs_string}"),
            lambda: self._cancel_callback("cancelled during waiting for the completion of the operation"),
            self._timeout,
            self._polling_period,
        )
        return None


class StartAwgunitsNowTask(AbstractStartAwgunitsTask):
    _PRELOAD_TIMEOUT: Final[float] = 0.25
    _START_TIMEOUT: Final[float] = 0.25

    def _activate_awgunits(self) -> None:
        self._awgctrl.prepare(self._hwidxs)

    def _wait_for_preparation(self) -> None:
        self._cancellable_wait_loop(
            self._check_prepare_callback,
            lambda: self._timeout_callback(f"failed to finish preloading at some of {self._idxs_string}"),
            lambda: self._cancel_callback("cancelled during waiting for the completion of preloading"),
            self._PRELOAD_TIMEOUT,
            self._polling_period,
        )

    def _check_prepare_callback(self) -> bool:
        with self._awgctrl.lock:
            if self._awgctrl.are_ready_all(self._hwidxs):
                self._awgctrl.check_error(self._hwidxs)
                return True
            else:
                return False

    def _wait_for_start(self) -> None:
        with self._awgctrl.lock:
            if not self._awgctrl.are_ready_all(self._hwidxs):
                raise RuntimeError(f"stop starting because some of {self._idxs_string} are not ready yet")
            self._awgctrl.start_now(self._hwidxs)

        self._cancellable_wait_loop(
            self._check_start_callback,
            lambda: self._timeout_callback(f"failed to start some of {self._idxs_string}"),
            lambda: self._cancel_callback("cancelled during waiting for starting"),
            self._START_TIMEOUT,
            self._polling_period,
        )

    def _check_start_callback(self) -> bool:
        with self._awgctrl.lock:
            if self._awgctrl.have_started_all(self._hwidxs):
                self._awgctrl.check_error(self._hwidxs)
                self._has_started_emission = True
                return True
            else:
                return False


class StartAwgunitsTimedTask(AbstractStartAwgunitsTask):
    _TRIGGER_SETTABLE_MARGIN: Final[float] = 0.05  # [s]
    _START_TIMEOUT_MARGIN: Final[float] = 0.25  # [s]

    def __init__(
        self,
        pool: ThreadPoolExecutor,
        awgctrl: AwgCtrl,
        awgunits: dict[int, AwgUnit],
        sqrctrl: SimplemultiSequencer,
        clkctrl: ClockcounterCtrl,
        timecount: int,
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ):
        super().__init__(
            pool, awgctrl, awgunits, timeout=timeout, polling_period=polling_period, disable_timeout=disable_timeout
        )
        self._sqrctrl: SimplemultiSequencer = sqrctrl
        self._clkctrl: ClockcounterCtrl = clkctrl
        self._timecount: int = timecount
        self._delta: float = 0.0

    def _activate_awgunits(self) -> None:
        cur, _ = self._clkctrl.read_counter()
        if self._timecount < cur + self._TRIGGER_SETTABLE_MARGIN * self._clkctrl.CLOCK_FREQUENCY:
            raise RuntimeError(f"specified timecount (= {self._timecount}) is too late to schedule")
        self._delta = (self._timecount - cur) / self._clkctrl.CLOCK_FREQUENCY

        trig = SimplemultiAwgTriggers()
        trig.append(self._timecount, self._hwidxs)
        self._sqrctrl.add_awg_start(trig)

    def _wait_for_preparation(self) -> None:
        # do nothing
        pass

    def _wait_for_start(self) -> None:
        self._cancellable_wait_loop(
            self._check_start_callback,
            lambda: self._timeout_before_start_callback(f"failed to start some of {self._idxs_string}"),
            lambda: self._cancel_before_start_callback("cancelled during waiting for starting"),
            self._delta + self._START_TIMEOUT_MARGIN,
            self._polling_period,
        )

    def _check_start_callback(self) -> bool:
        with self._awgctrl.lock:
            if self._awgctrl.have_started_all(self._hwidxs):
                self._awgctrl.check_error(self._hwidxs)
                self._has_started_emission = True
                return True
            else:
                return False

    def _timeout_before_start_callback(self, timeout_msg: str):
        self._sqrctrl.cancel_triggers()
        # Notes: don't use hal.awgctrl.terminate(), it wouldn't work well...
        self._terminate_all()  # Notes: to avoid race condition
        self._awgctrl.check_error(self._hwidxs)
        raise TimeoutError(timeout_msg)

    def _cancel_before_start_callback(self, cancel_msg: str):
        self._sqrctrl.cancel_triggers()
        # Notes: don't use hal.awgctrl.terminate(), it wouldn't work well...
        self._terminate_all()
        self._awgctrl.check_error(self._hwidxs)
        raise CancelledError(cancel_msg)


class AbstractStartCapunitsTask(Generic[CCT], AbstractCancellableTask[dict[tuple[int, int], CapIqDataReader]]):
    # Notes: it is no problem to set long _START_TIMEOUT because upper layer API gives more realistic value as argument.
    _DONE_TIMEOUT_MARGIN: Final[float] = 0.25
    _CANCEL_DONE_TIMEOUT: Final[float] = 1.0
    _WORD_RATE: Final[int] = 125_000_000  # [Hz]

    def __init__(
        self,
        pool: ThreadPoolExecutor,
        capctrl: CCT,
        capunits: dict[tuple[int, int], AbstractCapUnit],
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ):
        if len(capunits) == 0:
            raise ValueError("no capture units are specified")
        super().__init__(pool)
        self._capctrl: CCT = capctrl
        self._capunits: dict[tuple[int, int], AbstractCapUnit] = capunits
        self._timeout: Optional[float] = None if disable_timeout else (timeout or self._default_done_timeout)
        self._polling_period: float = polling_period or _DEFAULT_POLLING_PERIOD

    @cached_property
    def _hwidxs(self) -> set[int]:
        return {u.unit_index for u in self._capunits.values()}

    @cached_property
    def _idxs_string(self) -> str:
        return "capunits" + ", ".join([f"-#{m}:{u}" for m, u in self._capunits])

    @cached_property
    def _default_done_timeout(self) -> float:
        return (
            max([u.get_capture_duration() for u in self._capunits.values()]) / self._WORD_RATE
            + self._DONE_TIMEOUT_MARGIN
        )

    def _clear_done_all(self, raise_exception: bool = False):
        for u in self._capunits.values():
            u.clear_done()
        if self._capctrl.are_done_any(self._hwidxs):
            if raise_exception:
                raise RuntimeError(f"failed to clear done flag, some of {self._idxs_string} are still asserted")
            else:
                logger.warning(f"failed to clear done flag, some of {self._idxs_string} are still asserted")

    def _terminate_all(self):
        for i, u in self._capunits.items():
            u.terminate(timeout=self._CANCEL_DONE_TIMEOUT, polling_period=self._polling_period).result()
            u.hard_reset()  # Notes: to flush buffer in FPGA

    @abstractmethod
    def _wait_for_starting(self): ...

    def _check_done_callback(self) -> bool:
        with self._capctrl.lock:
            if self._capctrl.are_done_all(self._hwidxs):
                self._clear_done_all()
                self._capctrl.check_error(self._hwidxs)
                return True
            else:
                return False

    def _timeout_done_callback(self) -> None:
        self._terminate_all()
        self._capctrl.check_error(self._hwidxs)
        raise TimeoutError(f"failed to wait for the completion of {self._idxs_string}")

    @abstractmethod
    def _cancel_callback(self) -> None: ...

    @abstractmethod
    def _activate_capunits(self): ...

    def _body(self) -> dict[tuple[int, int], CapIqDataReader]:
        with self._capctrl.lock:
            for idx, u in self._capunits.items():
                if not u.is_loaded():
                    raise RuntimeError(f"capunit-#{idx} is not configured yet")

            if self._capctrl.are_busy_any(self._hwidxs):
                raise RuntimeError(f"stop starting capture because some of {self._idxs_string} are still busy")
            self._clear_done_all(raise_exception=True)
            self._activate_capunits()

        self._wait_for_starting()

        self._cancellable_wait_loop(
            self._check_done_callback,
            self._timeout_done_callback,
            self._cancel_callback,
            self._timeout,
            self._polling_period,
        )

        readers: dict[tuple[int, int], CapIqDataReader] = {}
        broken_data: bool = False
        for idx, u in self._capunits.items():
            assert u.is_loaded()
            r = u.get_reader()
            n = u.get_num_captured_sample()
            m = r.total_size_in_sample
            if n == m:
                readers[idx] = r
            elif n < m:
                broken_data = True
                logger.error(
                    f"size of the data captured by capunit-#{idx} is shorter than expected "
                    f"({n} samples < {m} samples)"
                )
            else:
                broken_data = True
                logger.error(
                    f"size of the data captured by capunit-#{idx} is longer than expected "
                    f"({n} samples > {m} samples)"
                )

        if broken_data:
            raise E7awgCaptureDataError(f"the captured data is broken at some of {self._idxs_string}")

        return readers


class StartCapunitsNowTask(AbstractStartCapunitsTask[AbstractCapCtrl]):
    def __init__(
        self,
        pool: ThreadPoolExecutor,
        capctrl: AbstractCapCtrl,
        capunits: dict[tuple[int, int], AbstractCapUnit],
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ):
        super().__init__(
            pool, capctrl, capunits, timeout=timeout, polling_period=polling_period, disable_timeout=disable_timeout
        )

    def _wait_for_starting(self) -> None:
        pass

    def _cancel_callback(self) -> None:
        self._terminate_all()
        self._capctrl.check_error(self._hwidxs)
        raise CancelledError(f"{self._idxs_string} are stopped during the capture")

    def _activate_capunits(self):
        self._capctrl.start_now(self._hwidxs)


class StartCapunitsByTriggerTask(AbstractStartCapunitsTask[CapCtrlSimpleMulti]):
    _DEFAULT_START_TIMEOUT: Final[float] = 10.0  # [s]

    def __init__(
        self,
        pool: ThreadPoolExecutor,
        capctrl: CapCtrlSimpleMulti,
        capunits: dict[tuple[int, int], AbstractCapUnit],
        *,
        timeout_before_trigger: Optional[float] = None,
        timeout_after_trigger: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ):
        super().__init__(
            pool,
            capctrl,
            capunits,
            timeout=timeout_after_trigger,
            polling_period=polling_period,
            disable_timeout=disable_timeout,
        )
        self._capmod_idxs: set[int] = {cu[0] for cu in capunits}
        self._trigger_activated: bool = False
        self._timeout_before_trigger: Union[float, None] = (
            None if disable_timeout is None else (timeout_before_trigger or self._DEFAULT_START_TIMEOUT)
        )

    def _activate_capunits(self):
        self._capctrl.add_triggerable_units(self._hwidxs)
        self._trigger_activated = True

    def _deactivate_trigger(self):
        if self._trigger_activated:
            self._capctrl.remove_triggerable_units(self._hwidxs)
            for capmod_idx in self._capmod_idxs:
                self._capctrl.set_triggering_awgunit_idx(capmod_idx, None)  # unset
            self._trigger_activated = False

    def _check_start_callback(self) -> bool:
        with self._capctrl.lock:
            has_started: bool = self._capctrl.have_started_all(self._hwidxs)
            has_done: bool = self._capctrl.are_done_all(self._hwidxs)
            if has_started or has_done:
                self._deactivate_trigger()
            return has_started or has_done

    def _timeout_start_callback(self) -> None:
        self._deactivate_trigger()
        self._terminate_all()
        self._capctrl.check_error(self._hwidxs)
        raise TimeoutError(f"failed to wait for the completion of {self._idxs_string}")

    def _wait_for_starting(self) -> None:
        capmod_not_ready: list[int] = []
        for capmod_idx in self._capmod_idxs:
            if self._capctrl.get_triggering_awgunit_idx(capmod_idx) is None:
                capmod_not_ready.append(capmod_idx)
        if len(capmod_not_ready) > 0:
            capmod_str = ", ".join([f"-#{cm}" for cm in capmod_not_ready])
            self._deactivate_trigger()
            self._terminate_all()
            self._capctrl.check_error(self._hwidxs)
            raise RuntimeError(f"no triggering awg unit is set to capture modules{capmod_str}")

        self._cancellable_wait_loop(
            self._check_start_callback,
            self._timeout_start_callback,
            self._cancel_callback,
            self._timeout_before_trigger,
            self._polling_period,
        )

    def _cancel_callback(self) -> None:
        self._deactivate_trigger()
        self._terminate_all()
        self._deactivate_trigger()
        self._capctrl.check_error(self._hwidxs)
        raise CancelledError(f"{self._idxs_string} are stopped during the capture")

    def _future_cancel_hook(self) -> None:
        # Notes: triggering_awgunit is set in advance. need to clear it even if the task is cancelled before start.
        for capmod_idx in self._capmod_idxs:
            self._capctrl.set_triggering_awgunit_idx(capmod_idx, None)


class Quel1WaveSubsystem:
    def __init__(
        self, ipaddr_wss: str, ipaddr_sss: Optional[str] = None, auth_callback: Optional[Callable[[], bool]] = None
    ):
        self._ipaddr_wss: str = ipaddr_wss
        self._ipaddr_sss: str = ipaddr_sss or str(ipaddress.IPv4Address(self._ipaddr_wss) + 0x00010000)
        self._auth_callback: Optional[Callable[[], bool]] = auth_callback
        self._hal: AbstractQuel1Au50Hal = create_quel1au50hal(
            ipaddr_wss=self._ipaddr_wss, ipaddr_sss=self._ipaddr_sss, auth_callback=self._auth_callback
        )

    def initialize(self):
        self.hal.initialize()

    @property
    def ipaddr_wss(self) -> str:
        return self._ipaddr_wss

    @cached_property
    def ipaddr_sss(self) -> str:
        return self._ipaddr_sss

    @property
    def hal(self) -> AbstractQuel1Au50Hal:
        return self._hal

    @cached_property
    def fw_type(self) -> E7FwType:
        return self.hal.fw_type()

    @cached_property
    def fw_version(self) -> str:
        return self.hal.fwversion

    @property
    def num_awgunit(self) -> int:
        return self.hal.awgctrl.num_unit

    @property
    def num_capmod(self) -> int:
        return self.hal.capctrl.num_module

    @property
    def num_capunit(self) -> int:
        return self.hal.capctrl.num_unit

    def num_capunit_of_capmod(self, capmod_idx) -> int:
        return self.hal.capctrl.num_unit_of_module(capmod_idx)

    def _get_awgunit(self, awgunit_idx: int) -> AwgUnit:
        if not (0 <= awgunit_idx < self.num_awgunit):
            raise ValueError(f"invalid index of awgunit: {awgunit_idx}")
        else:
            return self.hal.awgunit(awgunit_idx)

    def _capunit_idx_to_hwidx(self, capunit_idx: tuple[int, int]) -> int:
        m, u = capunit_idx
        return self.hal.capctrl.units_of_module(m)[u]

    def _get_capunit(self, capunit_idx: tuple[int, int]) -> AbstractCapUnit:
        m, u = capunit_idx
        if not ((0 <= m < self.hal.capctrl.num_module) and (0 <= u < self.hal.capctrl.num_unit_of_module(m))):
            raise ValueError(f"invalid index of capunit: {capunit_idx}")
        return self.hal.capunit(self._capunit_idx_to_hwidx(capunit_idx))

    def initialize_sequencer(self):
        self.hal.sqrctrl.initialize()

    def initialize_awgunits(self, awgunit_idxs: Collection[int]):
        for i in awgunit_idxs:
            self._get_awgunit(i).initialize()

    def initialize_all_awgunits(self):
        self.initialize_awgunits(self.hal.awgctrl.units)
        self.hal.awgctrl.initialize()

    def get_current_timecounter(self) -> int:
        return self.hal.clkcntr.read_counter()[0]

    def get_latest_sysref_timecounter(self) -> int:
        cntr = self.hal.clkcntr.read_counter()[1]
        if cntr:
            return cntr
        else:
            raise RuntimeError("installed wss firmware doesn't support SYSREF timecounter")

    def get_averaged_sysref_offset(self, num_iteration: int = 100) -> float:
        cntr = self.hal.clkcntr.read_counter()[1]
        if cntr is None:
            raise RuntimeError("installed wss firmware doesn't support SYSREF timecounter")

        cntr %= 2000
        for _ in range(num_iteration - 1):
            cntr += cast(int, self.hal.clkcntr.read_counter()[1]) % 2000
        return cntr / num_iteration

    def timecounter_to_second(self, tctr: int) -> float:
        return tctr / self.hal.clkcntr.CLOCK_FREQUENCY

    def get_names_of_wavedata(self, awgunit_idx) -> set[str]:
        u = self._get_awgunit(awgunit_idx)
        return u.get_names_of_wavedata()

    def register_wavedata(
        self, awgunit_idx: int, name: str, iq: npt.NDArray[np.complex64], allow_update: bool = True, **kwdargs
    ) -> None:
        u = self._get_awgunit(awgunit_idx)
        if allow_update and u.has_wavedata(name):
            u.delete_wavedata(name)
        u.register_wavedata_from_complex64vector(name, iq, **kwdargs)

    def has_wavedata(self, awgunit_idx: int, label: str) -> bool:
        u = self._get_awgunit(awgunit_idx)
        return u.has_wavedata(label)

    def delete_wavedata(self, awgunit_idx: int, label: str) -> None:
        u = self._get_awgunit(awgunit_idx)
        if u.has_wavedata(label):
            u.delete_wavedata(label)

    def config_awgunit(self, awgunit_idx: int, param: AwgParam):
        u = self._get_awgunit(awgunit_idx)
        u.load_parameter(param)

    def start_awgunits_now(
        self,
        awgunit_idxs: Collection[int],
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
        return_after_start_emission: bool = True,
    ) -> StartAwgunitsNowTask:
        awgunits = {idx: self.hal.awgunit(idx) for idx in awgunit_idxs}
        task = StartAwgunitsNowTask(
            self.hal.awgctrl._pool,
            self.hal.awgctrl,
            awgunits,
            timeout=timeout,
            polling_period=polling_period,
            disable_timeout=disable_timeout,
        )
        task.start()
        if return_after_start_emission:
            task.wait_for_starting_emission()
        return task

    def start_awgunits_timed(
        self,
        awgunit_idxs: Collection[int],
        timecount: int,
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
        return_after_start_emission: bool = False,
    ) -> StartAwgunitsTimedTask:
        awgunits = {idx: self.hal.awgunit(idx) for idx in awgunit_idxs}
        task = StartAwgunitsTimedTask(
            self.hal.awgctrl._pool,
            self.hal.awgctrl,
            awgunits,
            self.hal.sqrctrl,
            self.hal.clkcntr,
            timecount=timecount,
            timeout=timeout,
            polling_period=polling_period,
            disable_timeout=disable_timeout,
        )
        task.start()
        if return_after_start_emission:
            task.wait_for_starting_emission()
        return task

    def initialize_capunits(self, capunit_idxs: Collection[tuple[int, int]]):
        for i in capunit_idxs:
            self._get_capunit(i).initialize()

    def initialize_all_capunits(self):
        cc = self.hal.capctrl
        if isinstance(cc, CapCtrlSimpleMulti):
            for i in cc.modules:
                cc.set_triggering_awgunit_idx(i, None)
        for i in self.hal.capctrl.units:
            self.hal.capunit(i).initialize()

    def config_capunit(self, capunit_idx: tuple[int, int], param: AbstractCapParam):
        u = self._get_capunit(capunit_idx)
        u.load_parameter(param)

    def start_capunits_now(
        self,
        capunits_idxs: Collection[tuple[int, int]],
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ) -> StartCapunitsNowTask:
        capunits = {idx: self.hal.capunit(self._capunit_idx_to_hwidx(idx)) for idx in capunits_idxs}
        task = StartCapunitsNowTask(
            self.hal.capctrl._pool,
            self.hal.capctrl,
            capunits,
            timeout=timeout,
            polling_period=polling_period,
            disable_timeout=disable_timeout,
        )
        task.start()
        return task

    def set_triggering_awg_to_line(self, capmod_idx: int, awg_idx: int):
        cc = self.hal.capctrl
        if isinstance(cc, CapCtrlSimpleMulti):
            cc.set_triggering_awgunit_idx(capmod_idx, awg_idx)
        else:
            raise TypeError(f"the specified capture controller {cc.__class__.__name__} is not supported")

    def unset_triggering_awg_to_line(self, capmod_idx: int):
        cc = self.hal.capctrl
        if isinstance(cc, CapCtrlSimpleMulti):
            cc.set_triggering_awgunit_idx(capmod_idx, None)
        else:
            raise TypeError(f"the specified capture controller {cc.__class__.__name__} is not supported")

    def get_triggering_awg_to_line(self, capmod_idx: int) -> Union[int, None]:
        cc = self.hal.capctrl
        if isinstance(cc, CapCtrlSimpleMulti):
            return cc.get_triggering_awgunit_idx(capmod_idx)
        else:
            raise TypeError(f"the specified capture controller {cc.__class__.__name__} is not supported")

    def start_capunits_by_trigger(
        self,
        capunits_idxs: Collection[tuple[int, int]],
        *,
        timeout_before_trigger: Optional[float] = None,
        timeout_after_trigger: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ) -> StartCapunitsByTriggerTask:
        capunits = {idx: self.hal.capunit(self._capunit_idx_to_hwidx(idx)) for idx in capunits_idxs}
        cc = self.hal.capctrl
        if isinstance(cc, CapCtrlSimpleMulti):
            task = StartCapunitsByTriggerTask(
                self.hal.capctrl._pool,
                cc,
                capunits,
                timeout_before_trigger=timeout_before_trigger,
                timeout_after_trigger=timeout_after_trigger,
                polling_period=polling_period,
                disable_timeout=disable_timeout,
            )
            task.start()
            return task
        else:
            raise TypeError(f"the specified capture controller {cc.__class__.__name__} is not supported")
