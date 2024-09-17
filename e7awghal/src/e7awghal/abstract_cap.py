from abc import ABCMeta, abstractmethod
from collections.abc import Collection
from concurrent.futures import Future
from threading import RLock
from typing import Protocol, TypeVar, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

from e7awghal.abstract_register import AbstractFpgaReg
from e7awghal.capdata import CapIqDataReader, CapIqParser
from e7awghal.classification import ClassificationParam
from e7awghal.common_defs import _DEFAULT_POLLING_PERIOD, _DEFAULT_TIMEOUT


class AbstractCapCtrl(metaclass=ABCMeta):
    @property
    @abstractmethod
    def lock(self) -> RLock:
        pass

    @abstractmethod
    def initialize(self):
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    @abstractmethod
    def modules(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def units(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def units_of_module(self, module: int) -> tuple[int, ...]:
        pass

    @abstractmethod
    def read_reg(self, address: int) -> np.uint32:
        pass

    @abstractmethod
    def read_regs(self, address: int, size: int) -> npt.NDArray[np.uint32]:
        pass

    @abstractmethod
    def write_reg(self, address: int, val: np.uint32) -> None:
        pass

    @abstractmethod
    def write_regs(self, address: int, val: npt.NDArray[np.uint32]) -> None:
        pass

    @abstractmethod
    def _get_master_mask(self, use_cache: bool = True) -> set[int]:
        pass

    @abstractmethod
    def _set_master_mask(self, cap_units: Collection[int], use_cache: bool = True):
        pass

    @abstractmethod
    def start_now(self, cap_mods: Collection[int]) -> None:
        pass

    @abstractmethod
    def clear_done(self, cap_mods: Collection[int]) -> None:
        pass

    @abstractmethod
    def terminate(self, cap_mods: Collection[int]) -> None:
        pass

    @abstractmethod
    def are_awake_any(self, cap_units: Collection[int]) -> bool:
        pass

    @abstractmethod
    def are_busy_any(self, cap_units: Collection[int]) -> bool:
        pass

    @abstractmethod
    def are_done_any(self, cap_units: Collection[int]) -> bool:
        pass

    @abstractmethod
    def are_awake_all(self, cap_units: Collection[int]) -> bool:
        pass

    @abstractmethod
    def are_busy_all(self, cap_units: Collection[int]) -> bool:
        pass

    @abstractmethod
    def are_done_all(self, cap_units: Collection[int]) -> bool:
        pass

    @abstractmethod
    def have_started_all(self, awg_units: Collection[int]) -> bool:
        pass

    @abstractmethod
    def check_error(self, cap_units: Collection[int]):
        pass

    @abstractmethod
    def wait_to_start(
        self,
        units: set[int],
        timeout: float = _DEFAULT_TIMEOUT,
        polling_period: float = _DEFAULT_POLLING_PERIOD,
    ) -> Future[None]:
        pass


class AbstractSimpleMultiTriggerMixin(metaclass=ABCMeta):
    @abstractmethod
    def get_triggering_awgunit_idx(self, module: int) -> Union[int, None]:
        pass

    @abstractmethod
    def set_triggering_awgunit_idx(self, module: int, awgunit_idx: Union[int, None]) -> None:
        pass

    @abstractmethod
    def get_triggerable_units(self) -> set[int]:
        pass

    @abstractmethod
    def clear_triggerable_units(self) -> None:
        pass

    @abstractmethod
    def add_triggerable_unit(self, unit: int) -> None:
        pass

    @abstractmethod
    def add_triggerable_units(self, units: Collection[int]) -> None:
        pass

    @abstractmethod
    def remove_triggerable_unit(self, unit: int) -> None:
        pass

    @abstractmethod
    def remove_triggerable_units(self, units: Collection[int]) -> None:
        pass


@runtime_checkable
class AbstractCapSection(Protocol):
    @property
    def name(self) -> str:
        pass

    @property
    def num_capture_word(self) -> int:
        pass

    @property
    def num_blank_word(self) -> int:
        pass


_AnyCapSection = TypeVar("_AnyCapSection", bound=AbstractCapSection)


@runtime_checkable
class AbstractCapParam(Protocol[_AnyCapSection]):
    @property
    def num_wait_word(self) -> int:
        pass

    @property
    def num_repeat(self) -> int:
        pass

    @property
    def integration_enable(self) -> bool:
        pass

    @property
    def sum_enable(self) -> bool:
        pass

    @property
    def sum_range(self) -> tuple[int, int]:
        pass

    @property
    def complexfir_enable(self) -> bool:
        pass

    @property
    def complexfir_exponent_offset(self) -> int:
        pass

    @property
    def complexfir_coeff(self) -> npt.NDArray[np.complex64]:
        pass

    @property
    def realfirs_enable(self) -> bool:
        pass

    @property
    def realfirs_exponent_offset(self) -> int:
        pass

    @property
    def realfirs_real_coeff(self) -> npt.NDArray[np.float32]:
        pass

    @property
    def realfirs_imag_coeff(self) -> npt.NDArray[np.float32]:
        pass

    @property
    def decimation_enable(self) -> bool:
        pass

    @property
    def window_enable(self) -> bool:
        pass

    @property
    def window_coeff(self) -> npt.NDArray[np.complex128]:
        pass

    @property
    def classification_enable(self) -> bool:
        pass

    @property
    def classification_param(self) -> ClassificationParam:
        pass

    @property
    def num_section(self) -> int:
        pass

    @property
    def sections(self) -> list[_AnyCapSection]:
        pass

    def total_exponent_offset(self) -> int:
        pass

    def get_datasize_in_sample(self) -> int:
        pass

    def get_parser(self) -> "CapIqParser":
        pass


class AbstractCapUnit(metaclass=ABCMeta):
    @property
    @abstractmethod
    def unit_index(self) -> int:
        pass

    @property
    @abstractmethod
    def module_index(self) -> int:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def hard_reset(self) -> None:
        pass

    @abstractmethod
    def is_awake(self) -> bool:
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def check_error(self) -> AbstractFpgaReg:
        pass

    @abstractmethod
    def load_parameter(self, param: AbstractCapParam) -> None:
        pass

    @abstractmethod
    def reload_parameter(self) -> None:
        pass

    @abstractmethod
    def unload_parameter(self) -> None:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abstractmethod
    def get_capture_duration(self) -> int:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def clear_done(self) -> None:
        pass

    @abstractmethod
    def terminate(
        self, timeout: float = _DEFAULT_TIMEOUT, polling_period: float = _DEFAULT_POLLING_PERIOD
    ) -> Future[None]:
        pass

    @abstractmethod
    def get_reader(self) -> CapIqDataReader:
        pass

    @abstractmethod
    def get_num_captured_sample(self) -> int:
        pass
