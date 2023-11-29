import logging
import time
from abc import ABCMeta
from enum import Enum
from typing import Final, Tuple, Union

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict

from quel_inst_tool.spectrum_analyzer import InstDev, SpectrumAnalyzer, SpectrumAnalyzerParams

logger = logging.getLogger(__name__)


class E440xbTraceMode(str, Enum):
    WRITE = "WRIT"
    MAXHOLD = "MAXH"
    MINHOLD = "MINH"
    VIEW = "VIEW"
    BLANK = "BLAN"


class E440xbParams(SpectrumAnalyzerParams):
    model_config = ConfigDict(extra="forbid")

    trace_mode: Union[E440xbTraceMode, None] = None
    resolution_bandwidth: Union[float, None] = None
    resolution_bandwidth_auto: Union[bool, None] = None
    sweep_points: Union[int, None] = None
    display_enable: Union[bool, None] = None
    average_enable: Union[bool, None] = None
    average_count: Union[int, None] = None
    video_bandwidth: Union[float, None] = None
    video_bandwidth_auto: Union[bool, None] = None
    video_bandwidth_ratio: Union[float, None] = None
    video_bandwidth_ratio_auto: Union[bool, None] = None
    input_attenuation: Union[float, None] = None


class E440xbReadableParams(E440xbParams):
    prod_id: Union[str, None] = None
    max_freq_error: Union[float, None] = None

    @classmethod
    def from_e440xb(cls, obj: "E440xb") -> "E440xbReadableParams":
        model = E440xbReadableParams()
        fields = cls.model_fields
        if not isinstance(fields, dict):
            raise RuntimeError("unexpected field data")
        for k in fields.keys():
            setattr(model, k, getattr(obj, k))

        # TODO: validate the model before returning it.
        return model


class E440xbWritableParams(E440xbParams):
    def update_device_parameter(self, obj: "E440xb") -> bool:
        fields = E440xbParams.model_fields
        if not isinstance(fields, dict):
            raise RuntimeError("unexpected field data")

        freq_center: Union[float, None] = self.freq_center
        freq_span: Union[float, None] = self.freq_span
        if freq_center is not None or freq_span is not None:
            if freq_center is None or freq_span is None or not obj.freq_range_check(freq_center, freq_span):
                return False
            obj.freq_range_set(freq_center, freq_span)

        for k in fields.keys():
            if k not in {"freq_center", "freq_span"}:
                v0 = getattr(self, k)
                if v0 is None:
                    continue
                setattr(obj, k, v0)
        return True


class E440xb(SpectrumAnalyzer, metaclass=ABCMeta):
    _VIDEO_BANDWIDTH_RATIO_MIN: Final[float] = 1e-5
    _VIDEO_BANDWIDTH_RATIO_MAX: Final[float] = 3e6
    _VIDEO_BANDWIDTH_MIN: Final[float] = 1
    _VIDEO_BANDWIDTH_MAX: Final[float] = 3e6
    _RESOLUTION_BANDWIDTH_MIN: Final[float] = 1
    _RESOLUTION_BANDWIDTH_MAX: Final[float] = 5e6
    _INPUT_ATT_MIN: Final[float] = 0
    _INPUT_ATT_MAX: Final[float] = 75
    _PYVISA_TIMEOUT: Final[float] = 10  # default : 1s
    _TRACE_DATA_ENDIAN: Final[str] = "big"

    __slots__ = ()

    def __init__(self, dev: InstDev):
        super().__init__(dev)
        _ = self.freq_max  # for confirming the existence of self._FREQ_MAX
        _ = self.freq_min  # for confirming the existence of self._FREQ_MIN

    def reset(self) -> None:
        with self._lock:
            logger.info(f"resetting {self._dev.prod_id}")
            self._dev.reset()
        super().reset()

    @property
    def sweep_points(self):
        if self._sweep_points is None:
            self._sweep_points = int(self._dev.query(":SWE:POIN?"))
        return self._sweep_points

    @sweep_points.setter
    def sweep_points(self, num_points: int):
        with self._lock:
            self._dev.write(f":SWE:POIN {num_points}")
            self._cache_flush()

    @property
    def display_enable(self):
        return bool(int(self._dev.query(":DISP:ENAB?")))

    @display_enable.setter
    def display_enable(self, enable):
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":DISP:ENAB {on_or_off:s}")

    def average_clear(self) -> None:
        self._dev.write(":AVER:CLE")

    @property
    def average_enable(self) -> bool:
        return bool(int(self._dev.query(":AVER?")))

    @average_enable.setter
    def average_enable(self, enable) -> None:
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":AVER {on_or_off:s}")

    @property
    def trace_mode(self, index: int = 1) -> E440xbTraceMode:
        self._check_trace_index(index)
        return E440xbTraceMode(self._dev.query(f":TRAC{index:d}:MODE?").strip())

    @trace_mode.setter
    def trace_mode(self, mode: E440xbTraceMode, index: int = 1) -> None:
        self._check_trace_index(index)
        self._dev.write(f":TRAC{index:d}:MODE {mode.value:s}")

    def _trace_capture(self, timeout) -> None:
        # Notes: must be called from locked environment.
        self._dev.write("*CLS")
        self._dev.write("*ESE 1")
        self._dev.write(":INIT:IMM")
        self._dev.write("*OPC")

        t0 = time.time()
        logger.info("waiting for capture...")
        flag = False
        while time.time() - t0 < timeout:
            stb = int(self._dev.query("*STB?"))
            if stb & 0x20 != 0:
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
        trace_raw: bytes = (
            self._dev.write_and_read_raw(f":FORM INT,32\n:FORM:BORD NORM\nTRAC? TRACE{idx}") if enable_trace else b""
        )
        peaks_raw: str = self._dev.query(":TRAC:MATH:PEAK?") if enable_peak else ""
        return trace_raw, peaks_raw

    @staticmethod
    def _convert_peaks(peaks_raw: str, minimum_power: float) -> npt.NDArray[np.float64]:
        peaks_raw = peaks_raw.strip()
        if len(peaks_raw) == 0:
            peaks_split = []
        else:
            peaks_split = peaks_raw.split(",")
        peaks = np.fromiter(peaks_split, float)
        peaks = peaks.reshape(peaks.shape[0] // 2, 2)
        return peaks[peaks[:, 1] > minimum_power]

    def check_prms(self, wprms: SpectrumAnalyzerParams) -> "E440xbReadableParams":
        if not isinstance(wprms, E440xbWritableParams):
            raise TypeError("given parameter object has wrong type")

        rprms: E440xbReadableParams = E440xbReadableParams.from_e440xb(self)
        field = E440xbParams.model_fields
        if not isinstance(field, dict):
            raise RuntimeError("unexpected field data")
        for k in field.keys():
            v0 = getattr(wprms, k)
            if v0 is None:
                continue
            v1 = getattr(rprms, k)
            if v0 != v1:
                logger.info(f"{self._dev.prod_id}.{k} is supposed to be {v0}, but is actually set to {v1}")
        return rprms
