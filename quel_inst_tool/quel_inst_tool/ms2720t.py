import logging
import time
from enum import Enum
from typing import Final, List, Set, Tuple

import numpy as np
import numpy.typing as npt

from quel_inst_tool.spectrum_analyzer import InstDev, SpectrumAnalyzer, SpectrumAnalyzerParams

logger = logging.getLogger(__name__)


class Ms2720tTraceMode(str, Enum):
    NORM = "NORM"
    MAXHOLD = "MAXH"
    MINHOLD = "MINH"
    AVER = "AVER"


class Ms2720tAverageType(str, Enum):
    NONE = "NONE"
    SCAL = "SCAL"
    MAX = "MAX"
    MIN = "MIN"


class Ms2720t(SpectrumAnalyzer):
    __slots__ = ("_max_peaksearch_trials", "_holdmode_nsweeps", "_continuous_sweep")
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
    _DEFUALT_MAX_PEAKSEARCH_TRIALS: Final[int] = 10
    _TRACE_DATA_ENDIAN: Final[str] = "little"
    _DEFUALT_HOLDMODE_NSWEEPS: Final[int] = 10
    _PYVISA_TIMEOUT: Final[float] = 10  # default : 2s

    def __init__(self, dev: InstDev):
        super().__init__(dev)
        self._max_peaksearch_trials = self._DEFUALT_MAX_PEAKSEARCH_TRIALS
        self._holdmode_nsweeps = self._DEFUALT_HOLDMODE_NSWEEPS
        self._continuous_sweep = False

    @property
    def continuous_sweep(self) -> bool:
        return self._continuous_sweep

    @continuous_sweep.setter
    def continuous_sweep(self, exp_mode: bool) -> None:
        if exp_mode is True:
            self.init_cont = True
        else:
            self.init_cont = False
        self._continuous_sweep = exp_mode

    @property
    def max_peaksearch_trials(self) -> int:
        return self._max_peaksearch_trials

    @max_peaksearch_trials.setter
    def max_peaksearch_trials(self, max_pk: int) -> None:
        self._max_peaksearch_trials = max_pk

    @property
    def holdmode_nsweeps(self) -> int:
        return self._holdmode_nsweeps

    @holdmode_nsweeps.setter
    def holdmode_nsweeps(self, nsweeps: int) -> None:
        self._holdmode_nsweeps = nsweeps

    @property
    def peak_threshold(self) -> float:
        return float(self._dev.query(":CALC:MARK:PEAK:THR?"))

    @peak_threshold.setter
    def peak_threshold(self, pk_thr: float):
        self._dev.write(f":CALC:MARK:PEAK:THR {pk_thr:f}")

    @property
    def sweep_points(self) -> int:
        return self._SWEEP_POINTS

    @sweep_points.setter
    def sweep_points(self, num_points: int):
        raise ValueError(f"Cannot change sweep points for {self.prod_id:s}. It is fixed to be {self.sweep_points:d}")

    @property
    def display_enable(self):
        return bool(int(self._dev.query(":TRAC:DISP:STAT?")))

    @display_enable.setter
    def display_enable(self, enable):
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":TRAC:DISP:STAT {on_or_off:s}")

    def average_clear(self) -> None:
        average_type: str = self._dev.query(":AVER:TYPE?")
        self._dev.write(f":AVER:TYPE {Ms2720tAverageType.NONE}")
        if Ms2720tAverageType.SCAL in average_type:
            self._dev.write(f":AVER:TYPE {Ms2720tAverageType.SCAL}")

    @property
    def average_enable(self) -> bool:
        average_type: str = self._dev.query(":AVER:TYPE?")
        if Ms2720tAverageType.SCAL in average_type:
            return True
        else:
            return False

    @average_enable.setter
    def average_enable(self, enable) -> None:
        on_or_off = Ms2720tAverageType.SCAL if enable else Ms2720tAverageType.NONE
        self._dev.write(f":AVER:TYPE {on_or_off:s}")

    @property
    def trace_mode(self, index: int = 1) -> Ms2720tTraceMode:
        self._check_trace_index(index)
        return Ms2720tTraceMode(self._dev.query(f":TRAC{index:d}:OPER?").strip())

    @trace_mode.setter
    def trace_mode(self, mode: Ms2720tTraceMode, index: int = 1) -> None:
        self._check_trace_index(index)
        self._dev.write(f":TRAC{index:d}:OPER {mode.value:s}")

    def _trace_capture(self, timeout) -> None:
        # Notes: must be called from locked environment.
        if self.init_cont is False:
            if self.trace_mode == Ms2720tTraceMode.AVER:
                self._dev.write(":INIT:IMM AVER")
            else:
                self._dev.write(":INIT:IMM")
        else:
            self.trace_mode == Ms2720tTraceMode.NORM

        t0 = time.time()
        logger.info("waiting for capture...")
        flag = False
        sweep_cnt: int = 0
        nsweeps: int = (
            1
            if self.trace_mode == Ms2720tTraceMode.AVER or self.trace_mode == Ms2720tTraceMode.NORM
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
        peaks: List[Tuple[float, float]] = []
        self._dev.write("CALC:MARKer1:MAX")
        freq_max: float = float(self._dev.query("CALC:MARKer1:X?"))
        amp_max: float = float(self._dev.query("CALC:MARKer1:Y?"))
        peaks.append((freq_max, amp_max))
        freq_prev: float = freq_max
        nsearches: int = 0
        while nsearches < self.max_peaksearch_trials:
            self._dev.write("CALC:MARKer1:MAX:NEXT")
            freq_peak: float = float(self._dev.query("CALC:MARKer1:X?"))
            amp_peak: float = float(self._dev.query("CALC:MARKer1:Y?"))
            if freq_peak == freq_prev:
                break
            else:
                peaks.append((freq_peak, amp_peak))
                freq_prev = freq_peak
            nsearches = nsearches + 1
        peaks.sort(key=lambda x: x[0])
        peaks_raw: str = ""  # need to add the peak search
        for p in peaks:
            peaks_raw = peaks_raw + str(p[0]) + "," + str(p[1]) + ","

        return trace_raw, peaks_raw

    @staticmethod
    def _convert_peaks(peaks_raw: str, minimum_power: float) -> npt.NDArray[np.float64]:
        if len(peaks_raw) == 0:
            peaks_split = []
        else:
            peaks_split = [float(p) for p in peaks_raw.split(",") if len(p) > 0]
        peaks = np.fromiter(peaks_split, float)
        peaks = peaks.reshape(peaks.shape[0] // 2, 2)
        return peaks[peaks[:, 1] > minimum_power]

    def check_prms(self, wprms: SpectrumAnalyzerParams) -> SpectrumAnalyzerParams:
        raise NotImplementedError("This function is not yet defined")
