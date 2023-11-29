import logging
import time
from typing import Final, Set, Tuple, Union

from quel_inst_tool.ms2xxxx import Ms2xxxx, Ms2xxxxTraceMode
from quel_inst_tool.spectrum_analyzer import InstDev

logger = logging.getLogger(__name__)


class Ms2720t(Ms2xxxx):
    __slots__ = ("_max_peaksearch_trials", "_holdmode_nsweeps", "_continuous_sweep")
    _RESOURCE_TYPE: Final[str] = "INSTR"
    _FREQ_MAX: Final[float] = 3.2e10
    _FREQ_MIN: Final[float] = 9e3
    _VIDEO_BANDWIDTH_RATIO_MIN: Final[float] = 1e-6
    _VIDEO_BANDWIDTH_RATIO_MAX: Final[float] = 1
    _VIDEO_BANDWIDTH_MIN: Final[float] = 1
    _VIDEO_BANDWIDTH_MAX: Final[float] = 1e7
    _RESOLUTION_BANDWIDTH_MIN: Final[float] = 1
    _RESOLUTION_BANDWIDTH_MAX: Final[float] = 1e7
    _INPUT_ATT_MIN: Final[float] = 0
    _INPUT_ATT_MAX: Final[float] = 65
    _SWEEP_POINTS: Final[int] = 551
    _SUPPORTED_PROD_ID: Final[Set[str]] = {"MS2720T"}
    _DEFUALT_HOLDMODE_NSWEEPS: Final[int] = 10
    _SUPPORTED_TRACE_MODE: Final[Set[Ms2xxxxTraceMode]] = {
        Ms2xxxxTraceMode.NORM,
        Ms2xxxxTraceMode.MAXHOLD,
        Ms2xxxxTraceMode.MINHOLD,
        Ms2xxxxTraceMode.AVER,
    }

    def __init__(self, dev: InstDev):
        super().__init__(dev)
        self._holdmode_nsweeps = self._DEFUALT_HOLDMODE_NSWEEPS

    @classmethod
    def get_visa_name(cls, ipaddr: str, port: Union[int, None] = None) -> str:
        return "TCPIP::" + ipaddr + "::" + cls._RESOURCE_TYPE

    @property
    def holdmode_nsweeps(self) -> int:
        return self._holdmode_nsweeps

    @holdmode_nsweeps.setter
    def holdmode_nsweeps(self, nsweeps: int) -> None:
        self._holdmode_nsweeps = nsweeps

    @property
    def sweep_points(self) -> int:
        return self._SWEEP_POINTS

    @sweep_points.setter
    def sweep_points(self, num_points: int):
        raise ValueError(f"Cannot change sweep points for {self.prod_id:s}. It is fixed to be {self.sweep_points:d}")

    @property
    def trace_mode(self, index: int = 1) -> Ms2xxxxTraceMode:
        self._check_trace_index(index)
        return Ms2xxxxTraceMode(self._dev.query(f":TRAC{index:d}:OPER?").strip())

    @trace_mode.setter
    def trace_mode(self, mode: Ms2xxxxTraceMode, index: int = 1) -> None:
        if self._is_supported_trace_mode(mode):
            self._check_trace_index(index)
            self._dev.write(f":TRAC{index:d}:OPER {mode.value:s}")
        else:
            raise ValueError(f"trace mode '{mode}' is not supported for {self.prod_id}")

    def _trace_capture(self, timeout) -> None:
        # Notes: must be called from locked environment.
        if self.continuous_sweep is False:
            if self.trace_mode == Ms2xxxxTraceMode.AVER:
                self._dev.write(":INIT:IMM AVER")
            else:
                self._dev.write(":INIT:IMM")
        else:
            self.init_cont = False  # If continuous_sweep is True, sweeping is stopped for the capture.

        t0 = time.time()
        logger.info("waiting for capture...")
        flag = False
        sweep_cnt: int = 0
        nsweeps: int = (
            1
            if self.continuous_sweep is True
            or self.trace_mode == Ms2xxxxTraceMode.AVER
            or self.trace_mode == Ms2xxxxTraceMode.NORM
            else self.holdmode_nsweeps
        )
        while time.time() - t0 < timeout:
            stb = int(self._dev.query(":STAT:OPER?"))
            if stb & 0x100 != 0:
                sweep_cnt = sweep_cnt + 1
                if sweep_cnt == nsweeps:
                    flag = True
                    break
                else:
                    self._dev.write(":INIT:IMM")
            time.sleep(min(0.2, timeout / 25))
        if not flag:
            raise RuntimeError("measurement command timeout")

        logger.info("capture completed")

    def _trace_and_peak_read(
        self, enable_trace: bool, enable_peak: bool, idx: int, timeout: float
    ) -> Tuple[bytes, str]:
        # Notes: must be called from locked environment
        self._check_trace_index(idx)
        self._trace_capture(timeout)

        # Notes: write_and_read_raw() method fails for MS2720T
        self._dev.write(":FORM INT,32")
        self._dev.write(f"TRAC? {idx}")
        trace_raw: bytes = self._dev.read_raw() if enable_trace else b""

        peaks_raw: str = self._peak_search()
        if self.continuous_sweep is True:
            self.init_cont = True  # going back to continuous sweep if its mode is on.
        return trace_raw, peaks_raw
