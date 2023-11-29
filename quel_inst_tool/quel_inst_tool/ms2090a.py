import logging
import time
from typing import Final, Set, Tuple, Union

from quel_inst_tool.ms2xxxx import Ms2xxxx, Ms2xxxxTraceMode
from quel_inst_tool.spectrum_analyzer import InstDev

logger = logging.getLogger(__name__)


class Ms2090a(Ms2xxxx):
    __slots__ = ("_max_peaksearch_trials", "_holdmode_nsweeps", "_continuous_sweep")
    _RESOURCE_TYPE: Final[str] = "SOCKET"
    _PORT: Final[int] = 9001
    _FREQ_MAX: Final[float] = 3.2e10
    _FREQ_MIN: Final[float] = 9e3
    _VIDEO_BANDWIDTH_RATIO_MIN: Final[float] = 1e-5
    _VIDEO_BANDWIDTH_RATIO_MAX: Final[float] = 1
    _VIDEO_BANDWIDTH_MIN: Final[float] = 1
    _VIDEO_BANDWIDTH_MAX: Final[float] = 5e7
    _RESOLUTION_BANDWIDTH_MIN: Final[float] = 1
    _RESOLUTION_BANDWIDTH_MAX: Final[float] = 5e7
    _INPUT_ATT_MIN: Final[float] = 0
    _INPUT_ATT_MAX: Final[float] = 65
    _SUPPORTED_PROD_ID: Final[Set[str]] = {"MS2090A"}
    _SUPPORTED_TRACE_MODE: Final[Set[Ms2xxxxTraceMode]] = {
        Ms2xxxxTraceMode.NORM,
        Ms2xxxxTraceMode.MAX,
        Ms2xxxxTraceMode.MIN,
        Ms2xxxxTraceMode.AVER,
        Ms2xxxxTraceMode.RMAXHOLD,
        Ms2xxxxTraceMode.RMINHOLD,
        Ms2xxxxTraceMode.RAVER,
        Ms2xxxxTraceMode.MATH,
    }

    def __init__(self, dev: InstDev):
        super().__init__(dev)

    @classmethod
    def get_visa_name(cls, ipaddr: str, port: Union[int, None] = None) -> str:
        if port is None:
            port = cls._PORT
        return "TCPIP::" + ipaddr + "::" + str(port) + "::" + cls._RESOURCE_TYPE

    # For Ms2090a, MAXHOLD, RMAXHOLD, NIMHOLD and RMINHOLD modes use average count
    @property
    def holdmode_nsweeps(self) -> int:
        return self.average_count

    @holdmode_nsweeps.setter
    def holdmode_nsweeps(self, nsweeps: int) -> None:
        self.average_count = nsweeps

    @property
    def sweep_points(self):
        if self._sweep_points is None:
            self._sweep_points = int(self._dev.query(":DISP:POIN?"))
        return self._sweep_points

    @sweep_points.setter
    def sweep_points(self, num_points: int):
        with self._lock:
            self._dev.write(f":DISP:POIN {num_points}")
            self._cache_flush()

    @property
    def trace_mode(self, index: int = 1) -> Ms2xxxxTraceMode:
        self._check_trace_index(index)
        return Ms2xxxxTraceMode(self._dev.query(f":TRAC{index:d}:TYPE?").strip())

    @trace_mode.setter
    def trace_mode(self, mode: Ms2xxxxTraceMode, index: int = 1) -> None:
        if self._is_supported_trace_mode(mode):
            self._check_trace_index(index)
            self._dev.write(f":TRAC{index:d}:TYPE {mode.value:s}")
        else:
            raise ValueError(f"trace mode '{mode}' is not supported for {self.prod_id}")

    def _trace_capture(self, timeout) -> None:
        # Notes: must be called from locked environment.
        if self.continuous_sweep is False:
            if self.trace_mode == Ms2xxxxTraceMode.NORM:
                self._dev.write(":INIT:IMM")
            else:
                self._dev.write(":INIT:IMM:ALL")
        else:
            self.init_cont = False  # If continuous_sweep is True, sweeping is stopped just for the capture.

        t0 = time.time()
        logger.info("waiting for capture...")
        flag = False
        while time.time() - t0 < timeout:
            stb = int(self._dev.query(":STAT:OPER?"))
            if stb & 0x100 != 0:
                flag = True
                break
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
        self.display_enable = True
        self._dev.write(":FORM INT,32")
        logger.info("Getting Trace Data")
        self._dev.write(f":TRACe:DATA? {idx}")

        self._dev.suppress_end_enabled = False
        self._dev.termchar_enabled = False
        trace_raw: bytes = self._dev.read_raw() if enable_trace else b""
        self._dev.suppress_end_enabled = True
        self._dev.termchar_enabled = True

        peaks_raw: str = self._peak_search()
        if self.continuous_sweep is True:
            self.init_cont = True  # going back to continuous sweep if its mode is on.

        return trace_raw, peaks_raw
