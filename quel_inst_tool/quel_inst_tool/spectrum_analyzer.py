import logging
import threading
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pyvisa

DEFAULT_TIMEOUT_FOR_SWEEP = 60.0

logger = logging.getLogger(__name__)


class InstDev:
    def __init__(self, resource: pyvisa.resources.MessageBasedResource):
        self._resource = resource
        self._prod_id: Union[None, str] = None
        self._dev_id: Union[None, str] = None
        self._lock = threading.Lock()
        self.identify()

    def identify(self) -> None:
        # TODO: more precise parsing should be implemented...
        self._resource.write("*IDN?")
        r = [e.strip() for e in self._resource.read().split(",")]
        if len(r) >= 2:
            self._prod_id = r[1]
        if len(r) >= 3:
            self._dev_id = r[2]

    def reset(self) -> None:
        with self._lock:
            self._resource.write("*RST")
            time.sleep(0.5)

    def is_matched(self, prod_id: Union[None, str], dev_id: Union[None, str]) -> bool:
        matched = True
        if prod_id is not None:
            matched &= prod_id == self._prod_id
        if dev_id is not None:
            matched &= dev_id == self._dev_id
        return matched

    @property
    def prod_id(self) -> Union[None, str]:
        return self._prod_id

    @property
    def dev_id(self):
        return self._dev_id

    def write(self, cmd: str) -> None:
        with self._lock:
            length = self._resource.write(cmd)
            if length != len(cmd) + 2:
                raise RuntimeError(f"failed to send a command '{cmd}'")

    def read_raw(self) -> bytes:
        with self._lock:
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


class SpectrumAnalyzer(metaclass=ABCMeta):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def cache_flush(self) -> None:
        pass

    @property
    @abstractmethod
    def display_enable(self) -> bool:
        pass

    @display_enable.setter
    @abstractmethod
    def display_enable(self, val: bool) -> None:
        pass

    @property
    @abstractmethod
    def freq_center(self) -> float:
        pass

    @freq_center.setter
    @abstractmethod
    def freq_center(self, val: float) -> None:
        pass

    @property
    @abstractmethod
    def freq_span(self) -> float:
        pass

    @freq_span.setter
    @abstractmethod
    def freq_span(self, val: float) -> None:
        pass

    @property
    @abstractmethod
    def sweep_points(self) -> int:
        pass

    @sweep_points.setter
    @abstractmethod
    def sweep_points(self, value: int) -> None:
        pass

    @abstractmethod
    def average_clear(self) -> None:
        pass

    @property
    @abstractmethod
    def average_count(self) -> int:
        pass

    @average_count.setter
    @abstractmethod
    def average_count(self, value: int) -> None:
        pass

    @property
    @abstractmethod
    def average_enable(self) -> bool:
        pass

    @average_enable.setter
    @abstractmethod
    def average_enable(self, enable) -> None:
        pass

    @property
    @abstractmethod
    def resolution_bandwidth(self) -> float:
        pass

    @resolution_bandwidth.setter
    @abstractmethod
    def resolution_bandwidth(self, value: float) -> None:
        pass

    @property
    @abstractmethod
    def input_attenuation(self) -> float:
        pass

    @input_attenuation.setter
    @abstractmethod
    def input_attenuation(self, value: float) -> None:
        pass

    @property
    @abstractmethod
    def trace_mode(self, index: int = 1) -> str:
        pass

    @trace_mode.setter
    @abstractmethod
    def trace_mode(self, mode: str, index: int = 1) -> None:
        pass

    @abstractmethod
    def freq_points(self) -> npt.NDArray[np.float_]:
        pass

    @abstractmethod
    def trace_get(self, idx: int = 1, timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP) -> npt.NDArray[np.float_]:
        pass

    @abstractmethod
    def peak_get(
        self, minimum_power: float = -100.0, idx: int = 1, timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP
    ) -> npt.NDArray[np.float_]:
        pass

    @abstractmethod
    def trace_and_peak_get(
        self, minimum_power: float = -100.0, idx: int = 1, timeout: float = DEFAULT_TIMEOUT_FOR_SWEEP
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        pass

    @abstractmethod
    def max_freq_error_get(self) -> float:
        pass
