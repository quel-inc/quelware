import logging
import time
from concurrent.futures import CancelledError, Future
from typing import Any

from e7awghal.capctrl import AbstractCapCtrl
from e7awghal.capdata import CapIqDataReader
from e7awghal.capunit import (
    AbstractCapParam,
    CapSection,
    CapUnitSimplified,
    _CapParamCfirRegFile,
    _CapParamClassificationRegFile,
    _CapParamMainRegFile,
    _CapParamRfirsRegFile,
    _CapParamSectionRegFile,
    _CapParamWindowRegFile,
)
from e7awghal.common_defs import (
    _DEFAULT_POLLING_PERIOD,
    _DEFAULT_TIMEOUT,
    _DEFAULT_TIMEOUT_FOR_CAPTURE_RESERVE,
    E7awgCaptureDataError,
)
from e7awghal.e7awg_memoryobj import E7awgAbstractMemoryManager, E7awgMemoryObj
from e7awghal.hbmctrl import HbmCtrl

logger = logging.getLogger(__name__)


class CapUnitSimplifiedHL(CapUnitSimplified):
    def __init__(
        self,
        unit_idx: int,
        capctrl: AbstractCapCtrl,
        hbmctrl: HbmCtrl,
        mm: E7awgAbstractMemoryManager,
        settings: dict[str, Any],
    ):
        super().__init__(unit_idx, capctrl, hbmctrl, mm, settings)

    def initialize(self):
        self._cancel()
        super().initialize()

    def hard_reset(self) -> None:
        self._cancel()
        super().hard_reset()

    # TODO: move this method to utility function for tests
    def start_now(
        self, timeout: float = _DEFAULT_TIMEOUT, polling_period: float = _DEFAULT_POLLING_PERIOD
    ) -> Future[CapIqDataReader]:
        if self._waiting_for_capture:
            raise RuntimeError(f"capture task of cap_unit-#{self._unit_idx:02d} is still running")

        with self._unit_lock:
            if self._current_reader is None:
                raise RuntimeError(f"cap_unit-#{self._unit_idx:02d} is not configured yet")
            if self.is_busy():
                raise RuntimeError(f"cap_unit-#{self._unit_idx:02d} is busy")
            with self._master_lock:
                if self.is_done():
                    self.clear_done()
                self.start()

            self._capture_cancel_request = False
            self._waiting_for_capture = True
            return self._pool.submit(self._wait_done_loop, timeout, polling_period)

    # TODO: move this method to utility function for tests
    def wait_for_triggered_capture(
        self, timeout: float = _DEFAULT_TIMEOUT_FOR_CAPTURE_RESERVE, polling_period: float = _DEFAULT_POLLING_PERIOD
    ) -> Future[CapIqDataReader]:
        if self._waiting_for_capture:
            raise RuntimeError(f"capture task of cap_unit-#{self._unit_idx:02d} is still running")

        with self._unit_lock:
            if self._current_reader is None:
                raise RuntimeError(f"cap_unit-#{self._unit_idx:02d} is not configured yet")
            with self._master_lock:
                if self.is_busy():
                    raise RuntimeError(f"cap_unit-#{self._unit_idx:02d} is busy")
                if self.is_done():
                    self.clear_done()

            self._capture_cancel_request = False
            self._waiting_for_capture = True
            return self._pool.submit(self._wait_done_loop, 0.0, polling_period)  # Notes: no timeout

    # TODO: move this method to utility function for tests
    def _cancel(self) -> None:
        # Notes: _unit_lock is already locked if _waiting_for_capture is True.
        #        so, the current capture task should be cancelled without taking the _unit_lock.
        #        in other words, cancellation forces to unlock the _unit_lock.
        with self._cancel_lock:
            if self._waiting_for_capture:
                self._capture_cancel_request = True
                t1 = time.perf_counter() + _DEFAULT_TIMEOUT
                while self._waiting_for_capture and time.perf_counter() < t1:
                    time.sleep(_DEFAULT_POLLING_PERIOD)
                self._capture_cancel_request = False

    # TODO: move this method to utility function for tests
    def cancel(self) -> None:
        # Notes: no lock should be taken, see the comments in _cancel().
        with self._cancel_lock:
            if self._waiting_for_capture:
                self._cancel()
            else:
                raise RuntimeError(f"no active capture task of cap_unit-#{self._unit_idx:02d}")

    # TODO: move this method to utility function for tests
    def _wait_done_loop(self, timeout: float, polling_period: float) -> CapIqDataReader:
        cancelled: bool = False

        with self._unit_lock:
            assert self._waiting_for_capture, "_wait_done_loop() seems to be executed in an unexpected context"
            assert self._current_reader is not None, "_wait_done_loop() seems to be executed in an unexpected context"

            t1 = time.perf_counter() + timeout if timeout > 0 else 0
            while t1 == 0 or time.perf_counter() < t1:
                time.sleep(polling_period)
                with self._master_lock:
                    if self.is_done():
                        self.clear_done()
                        self.check_error()
                        break
                if self._capture_cancel_request:
                    logger.info(f"cancellation of cap_unit-#{self._unit_idx:02d} is conducted")
                    self.terminate()
                    cancelled = True
                    break
            else:
                self.unload_parameter()
                self._waiting_for_capture = False
                self.check_error()
                raise TimeoutError(
                    f"failed to wait for the completion of cap_unit-#{self._unit_idx:02d} due to timeout"
                )

            if cancelled:
                self.unload_parameter()
                self._waiting_for_capture = False
                # Notes: self.check_error() is conducted before 'break' above.
                raise CancelledError(f"capture task of cap_unit-#{self._unit_idx:02d} is cancelled")

            n = self.get_num_captured_sample()
            rdr = self._current_reader  # Notes: keeping possibly valid result reader.
            self.unload_parameter()
            self._waiting_for_capture = False  # Notes: capture is completed now.

            # Notes: check the validity of the capture data in terms of the length of captured data.
            m = rdr.total_size_in_sample
            if n < m:
                raise E7awgCaptureDataError(
                    f"size of the data captured by cap_unit-#{self._unit_idx:02d} is shorter than expected "
                    f"({n} samples < {m} samples)"
                )
            elif n > m:
                raise E7awgCaptureDataError(
                    f"size of the data captured by cap_unit-#{self._unit_idx:02d} is longer than expected "
                    f"({n} samples > {m} samples)"
                )

            return rdr


# Notes: this is just copied from capunit.py
class CapUnitHL(CapUnitSimplifiedHL):
    # Notes: to be relocated in a right place with some modifications
    def _set_cap_param(self, param: AbstractCapParam[CapSection], mobj: E7awgMemoryObj):
        # Notes: lock should be acquired by caller.
        main_regs: _CapParamMainRegFile = _CapParamMainRegFile.fromcapparam(param)
        main_regs.capture_address = (self._mm._address_offset + mobj.address_top) >> 5
        self._write_param_regs(0x0000_0000, main_regs.build())
        sec_regs: _CapParamSectionRegFile = _CapParamSectionRegFile.fromcapparam(param)
        b = sec_regs.build()
        self._write_param_regs(0x0000_1000, b[0 : main_regs.num_sum_section])
        self._write_param_regs(0x0000_5000, b[0x1000 : 0x1000 + main_regs.num_sum_section])
        if main_regs.dsp_switch.cfir_en:
            c = _CapParamCfirRegFile.fromcapparam(param).build()
            self._write_param_regs(0x0000_9000, c)
        if main_regs.dsp_switch.rfirs_en:
            r = _CapParamRfirsRegFile.fromcapparam(param).build()
            self._write_param_regs(0x0000_A000, r)
        if main_regs.dsp_switch.wdw_en:
            w = _CapParamWindowRegFile.fromcapparam(param).build()
            self._write_param_regs(0x0000_B000, w)
        if main_regs.dsp_switch.clsfy_en:
            f = _CapParamClassificationRegFile.fromcapparam(param).build()
            self._write_param_regs(0x0000_F000, f)

        num_word = 0
        for i in range(main_regs.num_sum_section):
            num_word += sec_regs.nums_capture_word[i] + sec_regs.nums_blank_word[i]
        self._capture_duration = main_regs.capture_delay + num_word * main_regs.num_integ_section

    def _get_cap_param(self) -> dict[str, Any]:
        # Notes: for debug purposes only.
        with self._unit_lock:
            r = super()._get_cap_param()
            r["main"] = _CapParamMainRegFile.parse(self._read_param_regs(0x0000_0000, 8))
            r["cfir"] = _CapParamCfirRegFile.parse(self._read_param_regs(0x0000_9000, 32))
            r["rfirs"] = _CapParamRfirsRegFile.parse(self._read_param_regs(0x0000_A000, 16))
            r["wdw"] = _CapParamWindowRegFile.parse(self._read_param_regs(0x0000_B000, 4096))
            r["clsfy"] = _CapParamClassificationRegFile.parse(self._read_param_regs(0x0000_F000, 6))
            return r

    def _validate_parameter(self, param: AbstractCapParam[CapSection]) -> None:
        # Notes: validate param
        _CapParamMainRegFile.fromcapparam(param)
