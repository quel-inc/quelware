import logging
import struct
import threading
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, Final, List, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pyvisa
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class InstDev:
    def __init__(self, resource: pyvisa.resources.MessageBasedResource):
        self._resource = resource
        self._lock = threading.RLock()
        prod_dev = self._identify()
        self._prod_id: str = prod_dev[0]
        self._dev_id: str = prod_dev[1]

    def _identify(self) -> Tuple[str, str]:
        reply = self.query("*IDN?")
        r = [e.strip() for e in reply.split(",")]

        # TODO: more precise parsing should be implemented...
        prod_id: str = ""
        dev_id: str = ""
        if len(r) >= 2:
            # Ignore "/xxxx" which contains the option type
            rtmp = [e.strip() for e in r[1].split("/")]
            prod_id = rtmp[0]
        if len(r) >= 3:
            dev_id = r[2]
        return prod_id, dev_id

    def reset(self) -> None:
        # Notes: this method requires reentrant lock to prevent the other threads from accessing the device before
        #        completing the reset.
        if "gpib" in str(self._resource.interface_type):
            with self._lock:
                self.write("*RST")
                time.sleep(0.5)
        else:
            logger.warning("no *RST command is available")

    def is_matched(self, prod_id: Union[None, str], dev_id: Union[None, str]) -> bool:
        matched = True
        if prod_id is not None:
            matched &= prod_id == self._prod_id
        if dev_id is not None:
            matched &= dev_id == self._dev_id
        return matched

    @property
    def timeout(self) -> float:
        return self._resource.timeout / 1000

    @timeout.setter
    def timeout(self, timeout: float) -> None:
        self._resource.timeout = timeout * 1000

    @property
    def prod_id(self) -> str:
        return self._prod_id

    @property
    def dev_id(self) -> str:
        return self._dev_id

    def write(self, cmd: str) -> None:
        with self._lock:
            length = self._resource.write(cmd)
            if length != len(cmd) + 2:
                raise RuntimeError(f"failed to send a command '{cmd}'")

    def read_raw(self) -> bytes:
        with self._lock:
            return self._resource.read_raw()

    def write_and_read_raw(self, cmd: str) -> bytes:
        with self._lock:
            length = self._resource.write(cmd)
            if length != len(cmd) + 2:
                raise RuntimeError(f"failed to send a command '{cmd}'")
            return self._resource.read_raw()

    def query(self, cmd: str) -> str:
        with self._lock:
            return self._resource.query(cmd)


class InstDevManager:
    def __init__(self, ivi: Union[str, None] = None, blacklist: Union[None, List[str]] = None):
        self._blacklist: List[str] = blacklist if blacklist is not None else []
        if ivi is not None:
            self._resource_manager = pyvisa.ResourceManager(ivi)
        else:
            self._resource_manager = pyvisa.ResourceManager()
        self._lock = threading.Lock()
        self._insts: Dict[str, InstDev] = {}
        self.scan()

    def show(self) -> None:
        for addr, inst in self._insts.items():
            print(f"{addr:s}, {inst.prod_id:s}, {inst.dev_id:s}")

    def scan(self) -> None:
        with self._lock:
            rl: Tuple[str, ...] = self._resource_manager.list_resources()
            new_insts = {}
            for r in rl:
                try:
                    if r not in self._blacklist:
                        rsrc = self._resource_manager.open_resource(r)
                        if isinstance(rsrc, pyvisa.resources.MessageBasedResource):
                            new_insts[r] = InstDev(cast(pyvisa.resources.MessageBasedResource, rsrc))
                            logger.info(f"a resource '{r}' is recognized.")
                        else:
                            logger.info(f"a resource '{r}' is ignored because it is not a supported resource")
                    else:
                        logger.info(f"a resource '{r}' is ignored because it is blacklisted.")
                except pyvisa.errors.VisaIOError:
                    logger.info(f"a resource '{r}' is ignored because it is not responding.")
                    pass
            self._insts = new_insts

    def lookup(
        self, resource_id: Union[None, str] = None, prod_id: Union[None, str] = None, dev_id: Union[None, str] = None
    ):
        with self._lock:
            if resource_id is not None:
                inst = self._insts[resource_id]
                return inst if inst.is_matched(prod_id, dev_id) else None

            if prod_id is not None or dev_id is not None:
                insts = [inst for rsrc, inst in self._insts.items() if inst.is_matched(prod_id, dev_id)]
                if len(insts) == 0:
                    return None
                elif len(insts) == 1:
                    return insts[0]
                else:
                    raise ValueError(
                        "multiple devices matches with the specified keys prod_id='{prod_id}', dev_id='{dev_id}'."
                    )

            raise ValueError("invalid keys")


class SpectrumAnalyzerParams(BaseModel, metaclass=ABCMeta):
    # TODO: Must decide which parameters should be in this class later.
    freq_center: Union[float, None] = None
    freq_span: Union[float, None] = None


class SpectrumAnalyzer(metaclass=ABCMeta):
    DEFAULT_TIMEOUT_FOR_SWEEP: Final[float] = 60.0  # [s]
    DEFAULT_MINIMUM_PEAK_POWER: Final[float] = -100.0  # [dBm]

    __slots__ = (
        "_dev",
        "_lock",
        "_sweep_points",
        "_freq_center",
        "_freq_span",
        "_resolution_bandwidth",
        "_freq_start",
        "_freq_points",
    )

    def __init__(self, dev: "InstDev"):
        self._dev = dev
        self._lock = threading.Lock()
        # NOTE: all the members should be defined in __init__()
        self._sweep_points: Union[int, None] = None  # cached, configurable
        self._freq_center: Union[float, None] = None  # cached, configurable
        self._freq_span: Union[float, None] = None  # cached, configurable
        self._resolution_bandwidth: Union[float, None] = None  # cached, configurable
        self._freq_start: Union[float, None] = None  # cached
        self._freq_points: Union[npt.NDArray[np.float64], None] = None  # synthesized

        self._check_prod_id()
        self.reset()
        if hasattr(self, "_PYVISA_TIMEOUT"):
            dev.timeout = self._PYVISA_TIMEOUT
        else:
            raise AssertionError

    def __del__(self):
        try:
            self.init_cont = True
        except pyvisa.errors.InvalidSession:
            logger.info(
                "failed to restart continuous trace acquisition. "
                "to avoid this, delete this object explicitly with del statement."
            )

    def _cache_flush(self):
        self._sweep_points = None
        self._freq_center = None
        self._freq_span = None
        self._resolution_bandwidth = None
        self._freq_start = None
        self._freq_points = None

    def _check_prod_id(self) -> None:
        if hasattr(self, "_SUPPORTED_PROD_ID"):
            if self._dev.prod_id in self._SUPPORTED_PROD_ID:
                return
            else:
                raise ValueError(
                    f"unsupported device (prod_id = {self._dev.prod_id} for class {self.__class__.__name__} is given"
                )
        else:
            raise AssertionError

    @property
    def prod_id(self):
        return self._dev.prod_id

    @property
    def freq_max(self) -> float:
        if hasattr(self, "_FREQ_MAX"):
            return self._FREQ_MAX
        else:
            raise AssertionError

    @property
    def freq_min(self) -> float:
        if hasattr(self, "_FREQ_MIN"):
            return self._FREQ_MIN
        else:
            raise AssertionError

    @property
    def resolution_bandwidth_max(self) -> float:
        if hasattr(self, "_RESOLUTION_BANDWIDTH_MAX"):
            return self._RESOLUTION_BANDWIDTH_MAX
        else:
            raise AssertionError

    @property
    def resolution_bandwidth_min(self) -> float:
        if hasattr(self, "_RESOLUTION_BANDWIDTH_MIN"):
            return self._RESOLUTION_BANDWIDTH_MIN
        else:
            raise AssertionError

    @property
    def video_bandwidth_max(self) -> float:
        if hasattr(self, "_VIDEO_BANDWIDTH_MAX"):
            return self._VIDEO_BANDWIDTH_MAX
        else:
            raise AssertionError

    @property
    def video_bandwidth_min(self) -> float:
        if hasattr(self, "_VIDEO_BANDWIDTH_MIN"):
            return self._VIDEO_BANDWIDTH_MIN
        else:
            raise AssertionError

    @property
    def video_bandwidth_ratio_max(self) -> float:
        if hasattr(self, "_VIDEO_BANDWIDTH_RATIO_MAX"):
            return self._VIDEO_BANDWIDTH_RATIO_MAX
        else:
            raise AssertionError

    @property
    def video_bandwidth_ratio_min(self) -> float:
        if hasattr(self, "_VIDEO_BANDWIDTH_RATIO_MIN"):
            return self._VIDEO_BANDWIDTH_RATIO_MIN
        else:
            raise AssertionError

    @property
    def input_att_max(self) -> float:
        if hasattr(self, "_INPUT_ATT_MAX"):
            return self._INPUT_ATT_MAX
        else:
            raise AssertionError

    @property
    def input_att_min(self) -> float:
        if hasattr(self, "_INPUT_ATT_MIN"):
            return self._INPUT_ATT_MIN
        else:
            raise AssertionError

    def reset(self) -> None:
        """
        :return: None
        """
        self.init_cont = False

    @property
    def freq_points(self):
        if self._freq_points is None:
            n = self.sweep_points
            self._freq_points = self.freq_start + np.arange(0, n, dtype=np.double) / (n - 1) * self.freq_span
        return self._freq_points

    @property
    @abstractmethod
    def display_enable(self) -> bool:
        pass

    @display_enable.setter
    @abstractmethod
    def display_enable(self, val: bool) -> None:
        pass

    @staticmethod
    def _check_trace_index(index: int):
        if not (isinstance(index, int) and 1 <= index <= 3):
            raise ValueError(f"invalid trace index: '{index}'")

    @property
    def freq_center(self):
        if self._freq_center is None:
            self._freq_center = float(self._dev.query(":FREQ:CENT?"))
        return self._freq_center

    @property
    def freq_span(self):
        if self._freq_span is None:
            self._freq_span = float(self._dev.query(":FREQ:SPAN?"))
        return self._freq_span

    @property
    def freq_start(self):
        if self._freq_start is None:
            self._freq_start = float(self._dev.query(":FREQ:STAR?"))
        return self._freq_start

    def freq_range_set(self, freq_center_in_hz: float, freq_span_in_hz: float):
        if self.freq_range_check(freq_center_in_hz, freq_span_in_hz):
            with self._lock:
                self._dev.write(f":FREQ:CENT {freq_center_in_hz:g}Hz")
                self._dev.write(f":FREQ:SPAN {freq_span_in_hz:g}Hz")
                self._cache_flush()
        else:
            raise ValueError("frequency range is out of range")

    def freq_range_check(self, freq_center_in_hz: float, freq_span_in_hz: float) -> bool:
        f0 = freq_center_in_hz - freq_span_in_hz / 2
        f1 = freq_center_in_hz + freq_span_in_hz / 2
        return (self.freq_min <= f0 <= self.freq_max) and (self.freq_min < f1 < self.freq_max)

    @property
    @abstractmethod
    def sweep_points(self) -> int:
        pass

    @sweep_points.setter
    @abstractmethod
    def sweep_points(self, num_points: int) -> None:
        pass

    @abstractmethod
    def average_clear(self) -> None:
        pass

    @property
    def average_count(self) -> int:
        return int(self._dev.query(":AVER:COUN?"))

    # TODO: fix it, this doesn't work for E440xB
    @average_count.setter
    def average_count(self, value: int) -> None:
        if 1 <= value <= 8192:
            self._dev.write(f":AVER:COUN {value:d}")
        else:
            raise ValueError(f"invalid average count: {value:d}")

    @property
    @abstractmethod
    def average_enable(self) -> bool:
        pass

    @average_enable.setter
    @abstractmethod
    def average_enable(self, enable) -> None:
        pass

    @property
    def resolution_bandwidth(self) -> float:
        if self._resolution_bandwidth is None:
            self._resolution_bandwidth = float(self._dev.query(":BWID?"))
        return self._resolution_bandwidth

    @resolution_bandwidth.setter
    def resolution_bandwidth(self, value: float) -> None:
        with self._lock:
            if self.resolution_bandwidth_min <= value <= self.resolution_bandwidth_max:
                self._dev.write(f":BWID {value:f}")
                self._cache_flush()
            else:
                raise ValueError(f"invalid resolution bandwidth: {value:f}Hz")

    @property
    def resolution_bandwidth_auto(self) -> bool:
        return bool(int(self._dev.query(":BWID:AUTO?")))

    @resolution_bandwidth_auto.setter
    def resolution_bandwidth_auto(self, enable: bool) -> None:
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":BWID:AUTO {on_or_off:s}")
        self._cache_flush()

    @property
    def video_bandwidth(self) -> float:
        return float(self._dev.query(":BWID:VID?"))

    @video_bandwidth.setter
    def video_bandwidth(self, value: float) -> None:
        if self.video_bandwidth_min <= value <= self.video_bandwidth_max:
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
        if self.video_bandwidth_ratio_min <= value <= self.video_bandwidth_ratio_max:
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
        if self.input_att_min <= value <= self.input_att_max:
            self._dev.write(f":POW:ATT {value:f}")
        else:
            raise ValueError("invalid input attenuation: '{value:d}'dB")

    @property
    def init_cont(self) -> bool:
        return bool(int(self._dev.query(":INIT:CONT?")))

    @init_cont.setter
    def init_cont(self, enable: bool) -> None:
        on_or_off = "ON" if enable else "OFF"
        self._dev.write(f":INIT:CONT {on_or_off:s}")

    @abstractmethod
    def _trace_and_peak_read(
        self, enable_trace: bool, enable_peak: bool, idx: int, timeout: float
    ) -> Tuple[bytes, str]:
        pass

    def _convert_trace(self, trace_raw: bytes) -> npt.NDArray[np.float64]:
        if hasattr(self, "_TRACE_DATA_ENDIAN"):
            h = int(chr(trace_raw[1]))
            d = int(trace_raw[2 : 2 + h])  # noqa: E203
            s = trace_raw[2 + h : 2 + h + d]  # noqa: E203
            endian: str = "<" if self._TRACE_DATA_ENDIAN == "little" else ">"
            data = np.array(struct.unpack(f"{endian:s}{d // 4:d}l", s)) / 1000.0
            return np.vstack((self.freq_points, data)).transpose()
        else:
            raise AssertionError

    @staticmethod
    @abstractmethod
    def _convert_peaks(peaks_raw: str, minimum_power: float) -> npt.NDArray[np.float64]:
        pass

    def trace_get(self, idx: int = 1, timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP) -> npt.NDArray[np.float64]:
        with self._lock:
            trace_raw, _ = self._trace_and_peak_read(True, False, idx=idx, timeout=timeout)
            return self._convert_trace(trace_raw)

    def peak_get(
        self,
        minimum_power: float = DEFAULT_MINIMUM_PEAK_POWER,
        idx: int = 1,
        timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP,
    ) -> npt.NDArray[np.float64]:
        with self._lock:
            _, peaks_raw = self._trace_and_peak_read(False, True, idx=idx, timeout=timeout)
            return self._convert_peaks(peaks_raw, minimum_power)

    def trace_and_peak_get(
        self,
        minimum_power: float = DEFAULT_MINIMUM_PEAK_POWER,
        idx: int = 1,
        timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    @abstractmethod
    def check_prms(self, wprms: SpectrumAnalyzerParams) -> SpectrumAnalyzerParams:
        pass
