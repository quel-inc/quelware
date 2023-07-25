import logging
import struct
import threading
import time
from typing import Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from quel_inst_tool.spectrum_analyzer import DEFAULT_TIMEOUT_FOR_SWEEP, InstDev, SpectrumAnalyzer

logger = logging.getLogger(__name__)


class E4405b(SpectrumAnalyzer):
    supported_prod_id: Set[str] = {"E4405B"}
    __slots__ = (
        "_dev",
        "_sweep_points",
        "_freq_start",
        "_freq_span",
        "_freq_points",
        "_freq_center",
        "_resolution_bandwidth",
        "_lock",
    )

    def __init__(self, dev: InstDev):
        if dev.prod_id not in self.supported_prod_id:
            raise ValueError(f"unsupported device (prod_id = {dev.prod_id} is given")
        self._dev = dev
        # NOTE: all the members should be defined in __init__()
        self._sweep_points = None  # cache
        self._freq_start = None  # cache
        self._freq_center = None  # cache
        self._freq_span = None  # cache
        self._freq_points = None  # synthesized
        self._resolution_bandwidth: Union[float, None] = None  # cache
        self._configure_for_automation()
        self._lock = threading.Lock()

    def _configure_for_automation(self):
        self._init_cont_set(False)

    def _freq_flush(self):
        self._sweep_points = None
        self._freq_start = None
        self._freq_center = None
        self._freq_span = None
        self._freq_points = None

    def cache_flush(self) -> None:
        self._freq_flush()
        self._resolution_bandwidth = None

    @staticmethod
    def _check_trace_index(index: int):
        if not (isinstance(index, int) and 1 <= index <= 3):
            raise ValueError(f"invalid trace index: '{index}'")

    def reset(self):
        logger.info(f"resetting {self._dev.prod_id}")
        self._dev.reset()

    @property
    def freq_points(self):
        if self._freq_points is None:
            n = self.sweep_points
            self._freq_points = self.freq_start + np.arange(0, n, dtype=np.double) / (n - 1) * self.freq_span
        return self._freq_points

    @property
    def freq_center(self):
        if self._freq_center is None:
            self._freq_center = float(self._dev.query(":FREQ:CENTER?"))
        return self._freq_center

    @freq_center.setter
    def freq_center(self, freq_in_hz: float):
        with self._lock:
            self._dev.write(f":FREQ:CENT {freq_in_hz:g}Hz")
            self._freq_flush()

    @property
    def freq_span(self):
        if self._freq_span is None:
            self._freq_span = float(self._dev.query(":FREQ:SPAN?"))
        return self._freq_span

    @freq_span.setter
    def freq_span(self, freq_in_hz: float):
        with self._lock:
            self._dev.write(f":FREQ:SPAN {freq_in_hz:g}Hz")
            self._freq_flush()

    @property
    def freq_start(self):
        if self._freq_start is None:
            self._freq_start = float(self._dev.query(":FREQ:STAR?"))
        return self._freq_start

    @property
    def sweep_points(self):
        if self._sweep_points is None:
            self._sweep_points = int(self._dev.query(":SWE:POIN?"))
        return self._sweep_points

    @sweep_points.setter
    def sweep_points(self, num_points: int):
        with self._lock:
            self._dev.write(f":SWE:POIN {num_points}")
            self._freq_flush()

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
    def average_count(self) -> int:
        return int(self._dev.query(":AVER:COUN?"))

    # TODO: fix it, this doesn't work.
    @average_count.setter
    def average_count(self, value: int) -> None:
        if 1 <= value <= 8192:
            self._dev.write(":AVER:COUN {value:d}")
        else:
            raise ValueError("invalid average count: {value:d}")

    @property
    def average_enable(self) -> bool:
        return bool(int(self._dev.query(":AVER?")))

    @average_enable.setter
    def average_enable(self, enable) -> None:
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":AVER {on_or_off:s}")

    @property
    def resolution_bandwidth(self) -> float:
        if self._resolution_bandwidth is None:
            self._resolution_bandwidth = float(self._dev.query(":BWID?"))
        return self._resolution_bandwidth

    @resolution_bandwidth.setter
    def resolution_bandwidth(self, value: float) -> None:
        with self._lock:
            if 1 <= value <= 5e6:
                self._dev.write(f":BWID {value:f}")
                self._resolution_bandwidth = None  # flush cache
            else:
                raise ValueError(f"invalid resolution bandwidth: {value:f}Hz")

    @property
    def resolution_bandwidth_auto(self) -> bool:
        return bool(int(self._dev.query(":BWID:AUTO?")))

    @resolution_bandwidth_auto.setter
    def resolution_bandwidth_auto(self, enable: bool) -> None:
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":BWID:AUTO {on_or_off:s}")

    @property
    def video_bandwidth(self) -> float:
        return float(self._dev.query(":BWID:VID?"))

    @video_bandwidth.setter
    def video_bandwidth(self, value: float) -> None:
        if 1 <= value <= 3e6:
            self._dev.write(f":BWID:VID {value:d}")
        else:
            raise ValueError(f"invalid video bandwidth: {value:d}Hz")

    @property
    def video_bandwidth_auto(self) -> float:
        return bool(int(self._dev.query(":BWID:VID:AUTO?")))

    @video_bandwidth_auto.setter
    def video_bandwidth_auto(self, enable: bool) -> None:
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":BWID:VID:AUTO {on_or_off:s}")

    @property
    def video_bandwidth_ratio(self) -> float:
        return float(self._dev.query(":BWID:VID:RAT?"))

    @video_bandwidth_ratio.setter
    def video_bandwidth_ratio(self, value: float) -> None:
        if 0.00001 <= value <= 3e6:
            self._dev.write(f":BWID:VID:RAT {value:d}")
        else:
            raise ValueError(f"invalid video bandwidth ratio: {value:d}Hz")

    @property
    def video_bandwidth_ratio_auto(self) -> bool:
        return bool(int(self._dev.query(":BWID:VID:RAT:AUTO?")))

    @video_bandwidth_ratio_auto.setter
    def video_bandwidth_ratio_auto(self, enable: bool) -> None:
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":BWID:VID:RAT:AUTO {on_or_off:s}")

    @property
    def input_attenuation(self) -> float:
        return float(self._dev.query(":POW:ATT?"))

    @input_attenuation.setter
    def input_attenuation(self, value: float) -> None:
        if 0 <= value <= 75:
            self._dev.write(f":POW:ATT {value:f}")
        else:
            raise ValueError("invalid input attenuation: '{value:d}'dB")

    @property
    def trace_mode(self, index: int = 1) -> str:
        self._check_trace_index(index)
        return self._dev.query(f":TRAC{index:d}:MODE?").strip()

    @trace_mode.setter
    def trace_mode(self, mode: str, index: int = 1) -> None:
        self._check_trace_index(index)
        mode = mode.upper()
        if mode in {"WRIT", "MAXH", "MINH", "VIEW", "BLAN"}:
            self._dev.write(f":TRAC{index:d}:MODE {mode:s}")
        else:
            raise ValueError(f"invalid trace mode: '{mode}'")

    def _init_cont_get(self) -> bool:
        return bool(int(self._dev.query(":INIT:CONT?")))

    def _init_cont_set(self, enable) -> None:
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":INIT:CONT {on_or_off:s}")

    def _trace_capture(self, timeout) -> None:
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
            time.sleep(min(0.1, timeout / 25))
        if not flag:
            raise RuntimeError("measurement command timeout")
        logger.info("capture completed")

    def _trace_and_peak_read(
        self, enable_trace: bool, enable_peak: bool, idx: int, timeout: float
    ) -> Tuple[bytes, str]:
        self._freq_flush()
        self._check_trace_index(idx)
        self._trace_capture(timeout)

        if enable_trace:
            self._dev.write(f":FORM INT,32\n:FORM:BORD NORM\nTRAC? TRACE{idx}")
            trace_raw: bytes = self._dev.read_raw()
        else:
            trace_raw = b""

        if enable_peak:
            peaks_raw: str = self._dev.query(":TRAC:MATH:PEAK?")
        else:
            peaks_raw = ""

        return trace_raw, peaks_raw

    def _convert_trace(self, trace_raw: bytes) -> npt.NDArray[np.float_]:
        h = int(chr(trace_raw[1]))
        d = int(trace_raw[2 : 2 + h])  # noqa: E203
        s = trace_raw[2 + h : 2 + h + d]  # noqa: E203
        data = np.array(struct.unpack(f">{d // 4:d}l", s)) / 1000.0
        return np.vstack((self.freq_points, data)).transpose()

    @staticmethod
    def _convert_peaks(peaks_raw: str, minimum_power: float) -> npt.NDArray[np.float_]:
        peaks_raw = peaks_raw.strip()
        if len(peaks_raw) == 0:
            peaks_split = []
        else:
            peaks_split = peaks_raw.split(",")
        peaks = np.fromiter(peaks_split, float)
        peaks = peaks.reshape(peaks.shape[0] // 2, 2)
        return peaks[peaks[:, 1] > minimum_power]

    def trace_get(self, idx: int = 1, timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP) -> npt.NDArray[np.float_]:
        with self._lock:
            trace_raw, _ = self._trace_and_peak_read(True, False, idx=idx, timeout=timeout)
            return self._convert_trace(trace_raw)

    def peak_get(
        self, minimum_power: float = -100.0, idx: int = 1, timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP
    ) -> npt.NDArray[np.float_]:
        with self._lock:
            _, peaks_raw = self._trace_and_peak_read(False, True, idx=idx, timeout=timeout)
            return self._convert_peaks(peaks_raw, minimum_power)

    def trace_and_peak_get(
        self, minimum_power: float = -100.0, idx: int = 1, timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        with self._lock:
            trace_raw, peaks_raw = self._trace_and_peak_read(True, True, idx=idx, timeout=timeout)
            return self._convert_trace(trace_raw), self._convert_peaks(peaks_raw, minimum_power)

    def max_freq_error_get(self) -> float:
        # once do not think about resolution bandwidth
        # a half of a frequency step doens't work well.
        return self.freq_span / (self.sweep_points - 1) * 1.05

    @property
    def max_freq_error(self):
        return self.max_freq_error_get()

    @property
    def prod_id(self):
        return self._dev.prod_id


if __name__ == "__main__":
    from quel_inst_tool.spectrum_analyzer import InstDevManager

    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.INFO)

    # simple tests
    im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so", blacklist=["GPIB0::6::INSTR"])
    e4405b = E4405b(im.lookup(prod_id="E4405B"))

    e4405b.reset()

    e4405b.freq_center = 8e9
    e4405b.freq_span = 2e9
    e4405b.resolution_bandwidth = 1e5

    logger.info(f"center_start = {e4405b.freq_start}Hz")
    logger.info(f"center_freq = {e4405b.freq_center}Hz, freq_span = {e4405b.freq_span}Hz")
    # logger.info(f"freq_points = {e4405b.freq_points}")
    logger.info(f"sweep_points = {e4405b.sweep_points}")
    logger.info(f"disp_enable = {e4405b.display_enable}")
    logger.info(f"average_enable, average_count = {e4405b.average_enable}, {e4405b.average_count}")
    logger.info(
        "resolution_bandwidth_auto, resolution_bandwidth ="
        f"{e4405b.resolution_bandwidth_auto}, {e4405b.resolution_bandwidth}Hz"
    )
    logger.info(f"video_bandwidth_auto, video_bandwidth = {e4405b.video_bandwidth_auto}, {e4405b.video_bandwidth}Hz")
    logger.info(
        "video_bandwidth_ratio_auto, video_bandwidth_ratio = "
        f"{e4405b.video_bandwidth_ratio_auto}, {e4405b.video_bandwidth_ratio}"
    )
    logger.info(f"trace_mode = {e4405b.trace_mode}")
    logger.info(f"input_attenuation = {e4405b.input_attenuation}dB")

    cap0 = e4405b.trace_get()
    peak0 = e4405b.peak_get(minimum_power=-40.0)

    e4405b.freq_center = 10e9
    e4405b.freq_span = 2e9
    logger.info(f"center_freq = {e4405b.freq_center}, freq_span = {e4405b.freq_span}")
    cap1, peak1 = e4405b.trace_and_peak_get(minimum_power=-40.0)

    import pprint

    pprint.pprint(peak0)
    pprint.pprint(peak1)
