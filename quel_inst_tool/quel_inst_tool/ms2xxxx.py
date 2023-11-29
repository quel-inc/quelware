import logging
from abc import abstractmethod
from enum import Enum
from typing import Final, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from quel_inst_tool.spectrum_analyzer import InstDev, SpectrumAnalyzer, SpectrumAnalyzerParams

logger = logging.getLogger(__name__)


class Ms2xxxxTraceMode(str, Enum):
    NORM = "NORM"
    MAX = "MAX"
    MIN = "MIN"
    MAXHOLD = "MAXH"
    MINHOLD = "MINH"
    AVER = "AVER"
    RMAXHOLD = "RMAX"
    RMINHOLD = "RMIN"
    RAVER = "RAV"
    MATH = "MATH"


class Ms2xxxx(SpectrumAnalyzer):
    __slots__ = ("_device_name", "_device_port", "_max_peaksearch_trials", "_holdmode_nsweeps", "_continuous_sweep")
    _DEFUALT_MAX_PEAKSEARCH_TRIALS: Final[int] = 10
    _TRACE_DATA_ENDIAN: Final[str] = "little"
    _PYVISA_TIMEOUT: Final[float] = 10  # default : 2s

    def __init__(self, dev: InstDev):
        super().__init__(dev)
        self._max_peaksearch_trials = self._DEFUALT_MAX_PEAKSEARCH_TRIALS
        self._continuous_sweep = False

    @classmethod
    @abstractmethod
    def get_visa_name(cls, ipaddr: str, port: Union[int, None] = None) -> str:
        pass

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
    @abstractmethod
    def holdmode_nsweeps(self) -> int:
        pass

    @holdmode_nsweeps.setter
    @abstractmethod
    def holdmode_nsweeps(self, nsweeps: int) -> None:
        pass

    @property
    def peak_threshold(self) -> float:
        return float(self._dev.query(":CALC:MARK:PEAK:THR?"))

    @peak_threshold.setter
    def peak_threshold(self, pk_thr: float):
        self._dev.write(f":CALC:MARK:PEAK:THR {pk_thr:f}")

    @property
    def display_enable(self):
        return bool(int(self._dev.query(":TRAC:DISP:STAT?")))

    @display_enable.setter
    def display_enable(self, enable):
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":TRAC:DISP:STAT {on_or_off:s}")

    def average_clear(self) -> None:
        trace_mode: Ms2xxxxTraceMode = self.trace_mode
        self.trace_mode = Ms2xxxxTraceMode.NORM
        if trace_mode == Ms2xxxxTraceMode.AVER:
            self.trace_mode = trace_mode

    @property
    def average_enable(self) -> bool:
        if self.trace_mode == Ms2xxxxTraceMode.AVER:
            return True
        else:
            return False

    @average_enable.setter
    def average_enable(self, enable) -> None:
        if enable:
            self.trace_mode = Ms2xxxxTraceMode.AVER
        else:
            if self.trace_mode == Ms2xxxxTraceMode.AVER:
                self.trace_mode = Ms2xxxxTraceMode.NORM

    @property
    @abstractmethod
    def trace_mode(self, index: int = 1) -> Ms2xxxxTraceMode:
        pass

    @trace_mode.setter
    @abstractmethod
    def trace_mode(self, mode: Ms2xxxxTraceMode, index: int = 1) -> None:
        pass

    def _is_supported_trace_mode(self, trace_mode: Ms2xxxxTraceMode) -> bool:
        if hasattr(self, "_SUPPORTED_TRACE_MODE"):
            if trace_mode in self._SUPPORTED_TRACE_MODE:
                return True
            else:
                return False
        else:
            raise AssertionError

    def _peak_search(self) -> str:
        peaks: List[Tuple[float, float]] = []
        self._dev.write("CALC:MARKer1:MAX")
        logger.info("Getting Marker Data")
        self._dev.write("CALC:MARKer1:X?")
        freq_max: float = float(self._dev.read_raw())
        self._dev.write("CALC:MARKer1:Y?")
        amp_max: float = float(self._dev.read_raw())
        peaks.append((freq_max, amp_max))
        freq_prev: float = freq_max
        nsearches: int = 0
        while nsearches < self.max_peaksearch_trials:
            self._dev.write("CALC:MARKer1:MAX:NEXT")
            self._dev.write("CALC:MARKer1:X?")
            freq_peak: float = float(self._dev.read_raw())
            self._dev.write("CALC:MARKer1:Y?")
            amp_peak: float = float(self._dev.read_raw())
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
        return peaks_raw

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
