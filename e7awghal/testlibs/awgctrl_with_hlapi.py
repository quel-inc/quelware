from concurrent.futures import Future
import time

from e7awghal.awgctrl import AwgCtrl
from e7awghal.awgunit import AwgUnit, AwgUnitCtrlReg
from e7awghal.common_defs import _DEFAULT_TIMEOUT, _DEFAULT_POLLING_PERIOD


class AwgUnitHL(AwgUnit):

    def start_now(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
        polling_period: float = _DEFAULT_POLLING_PERIOD,
    ) -> Future[None]:
        def _preload_and_start() -> None:
            # Notes: generating a positive edge of prepare bit
            v0 = AwgUnitCtrlReg()
            v1 = AwgUnitCtrlReg(prepare=True)
            v2 = AwgUnitCtrlReg(start=True)

            with self._unit_lock:
                with self._master_lock:
                    if self.is_busy():
                        raise RuntimeError(f"stop preloading because awg_uint-#{self._unit_idx:02d} is still busy")
                    if self.is_done():
                        self.clear_done()
                    self.check_error()
                    self._set_ctrl(v0)
                    self._set_ctrl(v1)

                t0 = time.perf_counter()
                while time.perf_counter() < t0 + timeout:
                    time.sleep(polling_period)
                    if self.is_ready():
                        break
                else:
                    raise TimeoutError(f"failed to make awg_unit-#{self._unit_idx:02d} ready due to timeout")

                with self._master_lock:
                    self.check_error()
                    if not self.is_ready():
                        raise RuntimeError(f"awg_unit-#{self._unit_idx:02d} is not ready yet")
                    # Notes: cleer done flag before starting
                    self._set_ctrl(v0)
                    self._set_ctrl(v2)

        return self._pool.submit(_preload_and_start)

    def _wait_to_start(self, timeout: float, polling_period: float, timeout_msg: str) -> Future[None]:
        def _wait_to_start_unit_loop() -> None:
            # Notes: the timing of check_error() is considered carefully with repsect to the efficiency (less register
            #        access is better) and the priority (more important for the users than TimeoutError).
            with self._unit_lock:
                t0 = time.perf_counter()
                while time.perf_counter() < t0 + timeout:
                    time.sleep(polling_period)
                    with self._master_lock:
                        if self.has_started():
                            self.check_error()
                            break
                else:
                    self.check_error()
                    raise TimeoutError(timeout_msg)

        return self._pool.submit(_wait_to_start_unit_loop)

    def wait_to_start(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
        polling_period: float = _DEFAULT_POLLING_PERIOD,
    ) -> Future[None]:
        return self._wait_to_start(
            timeout,
            polling_period,
            f"failed to wait for the completion of awg_unit-#{self._unit_idx:02d} due to timeout",
        )

    def _wait_done(self, timeout: float, polling_period: float, timeout_msg: str) -> Future[None]:
        def _wait_done_unit_loop() -> None:
            # Notes: the timing of check_error() is considered carefully with respect to the efficiency (less register
            #        access is better) and the priority (more important for the users than TimeoutError).
            with self._unit_lock:
                t1 = time.perf_counter() + timeout
                while time.perf_counter() < t1:
                    time.sleep(polling_period)
                    with self._master_lock:
                        if self.is_done():
                            self.clear_done()
                            self.check_error()
                            break
                else:
                    self.check_error()
                    raise TimeoutError(timeout_msg)

        return self._pool.submit(_wait_done_unit_loop)

    def wait_done(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
        polling_period: float = _DEFAULT_POLLING_PERIOD,
    ) -> Future[None]:
        return self._wait_done(
            timeout,
            polling_period,
            f"failed to wait for the completion of awg_unit-#{self._unit_idx:02d} due to timeout",
        )


class AwgCtrlHL(AwgCtrl):
    def _wait_to_start(self, units: set[int], timeout: float, polling_period: float, timeout_msg: str) -> Future[None]:
        def _wait_to_start_unit_loop() -> None:
            # Notes: the timing of check_error() is considered carefully with respect to the efficiency (less register
            #        access is better) and the priority (more important for the users than TimeoutError).
            t0 = time.perf_counter()
            while time.perf_counter() < t0 + timeout:
                time.sleep(polling_period)
                with self._lock:
                    if self.have_started_all(units):
                        self.check_error(units)
                        break
            else:
                self.check_error(units)
                raise TimeoutError(timeout_msg)

        return self._pool.submit(_wait_to_start_unit_loop)

    def wait_to_start(
        self,
        units: set[int],
        timeout: float = _DEFAULT_TIMEOUT,
        polling_period: float = _DEFAULT_POLLING_PERIOD,
    ) -> Future[None]:
        self._validate_units(units)
        return self._wait_to_start(
            units,
            timeout,
            polling_period,
            "failed to wait for the completion of some awg_units due to timeout",
        )

    def _wait_done(self, units: set[int], timeout: float, polling_period: float, timeout_msg: str) -> Future[None]:
        def _wait_done_unit_loop() -> None:
            # Notes: the timing of check_error() is considered carefully with respect to the efficiency (less register
            #        access is better) and the priority (more important for the users than TimeoutError).
            t0 = time.perf_counter()
            while time.perf_counter() < t0 + timeout:
                time.sleep(polling_period)
                with self._lock:
                    if self.are_done_all(units):
                        self.clear_done(units)
                        self.check_error(units)
                        break
            else:
                self.check_error(units)
                raise TimeoutError(timeout_msg)

        return self._pool.submit(_wait_done_unit_loop)

    def wait_done(
        self,
        units: set[int],
        timeout: float = _DEFAULT_TIMEOUT,
        polling_period: float = _DEFAULT_POLLING_PERIOD,
    ) -> Future[None]:
        self._validate_units(units)
        return self._wait_done(
            units,
            timeout,
            polling_period,
            "failed to wait for the completion of some awg_units due to timeout",
        )
