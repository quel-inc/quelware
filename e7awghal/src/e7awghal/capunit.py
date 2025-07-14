import copy
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import TYPE_CHECKING, Annotated, Any, Final, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, PlainValidator, ValidationInfo, conint, conlist

from e7awghal.abstract_cap import AbstractCapCtrl, AbstractCapParam, AbstractCapUnit
from e7awghal.abstract_register import AbstractFpgaReg, b_1bf_bool, p_1bf_bool
from e7awghal.capctrl import CapCtrlSimpleMulti
from e7awghal.capdata import CapIqDataReader
from e7awghal.capparam import CapSection
from e7awghal.classification import calc_abc_main_sub
from e7awghal.common_defs import _CAP_MINIMUM_ALIGN, _DEFAULT_POLLING_PERIOD, _DEFAULT_TIMEOUT, E7awgHardwareError
from e7awghal.e7awg_memoryobj import E7awgAbstractMemoryManager, E7awgMemoryObj
from e7awghal.hbmctrl import HbmCtrl

logger = logging.getLogger(__name__)


class CapCtrlCtrlReg(AbstractFpgaReg):
    reset: bool = Field(default=False)  # [0]
    start: bool = Field(default=False)  # [1]
    terminate: bool = Field(default=False)  # [2]
    done_clr: bool = Field(default=False)  # [3]

    def _parse(self, v: np.uint32) -> None:
        self.reset = p_1bf_bool(v, 0)
        self.start = p_1bf_bool(v, 1)
        self.terminate = p_1bf_bool(v, 2)
        self.done_clr = p_1bf_bool(v, 3)

    def build(self) -> np.uint32:
        return (
            b_1bf_bool(self.reset, 0)
            | b_1bf_bool(self.start, 1)
            | b_1bf_bool(self.terminate, 2)
            | b_1bf_bool(self.done_clr, 3)
        )

    @classmethod
    def parse(cls, v: np.uint32) -> "CapCtrlCtrlReg":
        r = cls()
        r._parse(v)
        return r


class CapCtrlStatusReg(AbstractFpgaReg):
    wakeup: bool = Field(default=False)  # [0]
    busy: bool = Field(default=False)  # [1]
    done: bool = Field(default=False)  # [2]

    def _parse(self, v: np.uint32) -> None:
        self.wakeup = p_1bf_bool(v, 0)
        self.busy = p_1bf_bool(v, 1)
        self.done = p_1bf_bool(v, 2)

    def build(self) -> np.uint32:
        return b_1bf_bool(self.wakeup, 0) | b_1bf_bool(self.busy, 1) | b_1bf_bool(self.done, 2)

    @classmethod
    def parse(cls, v: np.uint32) -> "CapCtrlStatusReg":
        r = cls()
        r._parse(v)
        return r


class CapCtrlErrorReg(AbstractFpgaReg):
    fifo_overflow: bool = Field(default=False)  # [0]
    xfer_failure: bool = Field(default=False)  # [1]

    def _parse(self, v: np.uint32) -> None:
        self.fifo_overflow = p_1bf_bool(v, 0)
        self.xfer_failure = p_1bf_bool(v, 1)

    def build(self) -> np.uint32:
        return b_1bf_bool(self.fifo_overflow, 0) | b_1bf_bool(self.xfer_failure, 1)

    @classmethod
    def parse(cls, v: np.uint32) -> "CapCtrlErrorReg":
        r = cls()
        r._parse(v)
        return r


class _DspSwitch(AbstractFpgaReg):
    cfir_en: bool = Field(default=False)  # [0]
    deci_en: bool = Field(default=False)  # [1]
    rfirs_en: bool = Field(default=False)  # [2]
    wdw_en: bool = Field(default=False)  # [3]
    sum_en: bool = Field(default=False)  # [4]
    integ_en: bool = Field(default=False)  # [5]
    clsfy_en: bool = Field(default=False)  # [6]

    def _parse(self, v: np.uint32) -> None:
        self.cfir_en = p_1bf_bool(v, 0)
        self.deci_en = p_1bf_bool(v, 1)
        self.rfirs_en = p_1bf_bool(v, 2)
        self.wdw_en = p_1bf_bool(v, 3)
        self.sum_en = p_1bf_bool(v, 4)
        self.integ_en = p_1bf_bool(v, 5)
        self.clsfy_en = p_1bf_bool(v, 6)

    def build(self) -> np.uint32:
        return (
            b_1bf_bool(self.cfir_en, 0)
            | b_1bf_bool(self.deci_en, 1)
            | b_1bf_bool(self.rfirs_en, 2)
            | b_1bf_bool(self.wdw_en, 3)
            | b_1bf_bool(self.sum_en, 4)
            | b_1bf_bool(self.integ_en, 5)
            | b_1bf_bool(self.clsfy_en, 6)
        )

    @classmethod
    def parse(cls, v: np.uint32) -> "_DspSwitch":
        r = cls()
        r._parse(v)
        return r


if TYPE_CHECKING:
    SectionAttrVector = list[int]
    WindowCoeffVector = list[int]
    CfirCoeffVector = list[int]
    RfirsCoeffVector = list[int]
else:
    SectionAttrVector = conlist(conint(ge=0, le=0xFFFF_FFFF), min_length=4096, max_length=4096)
    WindowCoeffVector = conlist(conint(ge=-0x8000_0000, le=0x7FFF_FFFF), min_length=2048, max_length=2048)
    CfirCoeffVector = conlist(conint(ge=-0x8000, le=0x7FFF), min_length=16, max_length=16)
    RfirsCoeffVector = conlist(conint(ge=-0x8000, le=0x7FFF), min_length=8, max_length=8)


class _CapParamSimplifiedMainRegFile(BaseModel, validate_assignment=True):
    capture_delay: int = Field(ge=0, le=0xFFFF_FFFE, default=0)
    capture_address: int = Field(ge=0, le=0xFFFF_FFFF, multiple_of=16, default=0)
    num_capture_sample: int = Field(ge=0, le=0xFFFF_FFFF, default=0)
    num_integ_section: int = Field(ge=1, le=0xFFFF_FFFF, default=1)
    num_sum_section: int = Field(ge=1, le=0x1000, default=1)

    def build(self) -> npt.NDArray[np.uint32]:
        r = np.zeros(8, dtype=np.uint32)
        r[0] = 0
        r[1] = self.capture_delay
        r[2] = self.capture_address
        r[3] = self.num_capture_sample
        r[4] = self.num_integ_section
        r[5] = self.num_sum_section
        r[6] = 0
        r[7] = 0
        return r

    def _parse(self, v: npt.NDArray[np.uint32]) -> None:
        self.capture_delay = v[1]
        self.capture_address = v[2]
        self.num_capture_sample = v[3]
        self.num_integ_section = v[4]
        self.num_sum_section = v[5]

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32]) -> "_CapParamSimplifiedMainRegFile":
        r = cls()
        r._parse(v)
        return r

    @classmethod
    def fromcapparam(cls, cp: AbstractCapParam[CapSection]) -> "_CapParamSimplifiedMainRegFile":
        r = cls()
        r.capture_delay = cp.num_wait_word
        if cp.complexfir_enable:
            raise ValueError("complex fir filter is not supported")
        if cp.decimation_enable:
            raise ValueError("decimation is not supported")
        if cp.window_enable:
            raise ValueError("window function is not supported")
        if cp.integration_enable:
            raise ValueError("integration over repetitions is not supported")
        if cp.sum_enable:
            raise ValueError("summation within a section is not supported")
        r.num_integ_section = cp.num_repeat
        r.num_sum_section = cp.num_section
        return r


class _CapParamMainRegFile(_CapParamSimplifiedMainRegFile):
    dsp_switch: _DspSwitch = Field(default=_DspSwitch())
    sum_begin: int = Field(ge=0, le=0xFFFF_FFFF, default=0)
    sum_end: int = Field(ge=0, le=0xFFFF_FFFF, default=0xFFFF_FFFF)

    def build(self) -> npt.NDArray[np.uint32]:
        if self.sum_begin > self.sum_end:
            raise ValueError(f"sum_begin (= 0x{self.sum_begin:08x}) is greater than sum_end (= 0x{self.sum_end:08x})")
        r = super().build()
        r[0] = self.dsp_switch.build()
        r[6] = self.sum_begin
        r[7] = self.sum_end
        return r

    def _parse(self, v: npt.NDArray[np.uint32]) -> None:
        super()._parse(v)
        self.dsp_switch = _DspSwitch.parse(v[0])
        self.sum_begin = v[6]
        self.sum_end = v[7]

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32]) -> "_CapParamMainRegFile":
        r = cls()
        r._parse(v)
        return r

    @classmethod
    def fromcapparam(cls, cp: AbstractCapParam[CapSection]) -> "_CapParamMainRegFile":
        r = cls()
        r.dsp_switch.cfir_en = cp.complexfir_enable
        r.dsp_switch.deci_en = cp.decimation_enable
        r.dsp_switch.rfirs_en = cp.realfirs_enable
        r.dsp_switch.wdw_en = cp.window_enable
        r.dsp_switch.integ_en = cp.integration_enable
        r.dsp_switch.sum_en = cp.sum_enable
        r.dsp_switch.clsfy_en = cp.classification_enable
        r.capture_delay = cp.num_wait_word
        r.num_integ_section = cp.num_repeat
        r.num_sum_section = cp.num_section
        r.sum_begin = cp.sum_range[0]
        r.sum_end = cp.sum_range[1]
        return r


class _CapParamSectionRegFile(BaseModel, validate_assignment=True):
    nums_capture_word: SectionAttrVector = Field(default=[0 for _ in range(4096)])
    nums_blank_word: SectionAttrVector = Field(default=[1 for _ in range(4096)])

    def build(self, num_section: int = 4096) -> npt.NDArray[np.uint32]:
        if not (1 <= num_section <= 4096):
            raise ValueError(f"invalid number of sections (= {num_section})")
        r = np.zeros(8192, dtype=np.uint32)
        r[0:num_section] = self.nums_capture_word[0:num_section]
        r[4096 : 4096 + num_section] = self.nums_blank_word[0:num_section]
        return r

    def _parse(self, v: npt.NDArray[np.uint32], num_section: int = 4096) -> None:
        if not (1 <= num_section <= 4096):
            raise ValueError(f"invalid number of sections (= {num_section})")
        self.nums_capture_word[0:num_section] = v[0:num_section]
        self.nums_blank_word[0:num_section] = v[4096 : 4096 + num_section]

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32], num_section: int = 4096) -> "_CapParamSectionRegFile":
        r = cls()
        r._parse(v, num_section)
        return r

    @classmethod
    def fromcapparam(cls, cp: AbstractCapParam[CapSection]) -> "_CapParamSectionRegFile":
        r = cls()
        for i, s in enumerate(cp.sections):
            r.nums_capture_word[i] = s.num_capture_word
            r.nums_blank_word[i] = s.num_blank_word
        return r


def _zerolist(n: int) -> list[float]:
    return [0.0 for _ in range(n)]


class _CapParamCfirRegFile(BaseModel, validate_assignment=True):
    real: CfirCoeffVector = Field(default_factory=lambda: _zerolist(16))
    imag: CfirCoeffVector = Field(default_factory=lambda: _zerolist(16))

    def build(self) -> npt.NDArray[np.uint32]:
        r = np.zeros(32, dtype=np.int16)
        r[0:16] = self.real[:]
        r[16:32] = self.imag[:]
        return r.astype(np.uint32)

    def _parse(self, v: npt.NDArray[np.uint32]) -> None:
        vv = v.astype(np.int32)
        self.real[:] = vv[0:16]
        self.imag[:] = vv[16:32]

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32]) -> "_CapParamCfirRegFile":
        if not (isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == 32 and v.dtype == np.uint32):
            raise TypeError("invalid data for complex FIR filter register file")
        r = cls()
        r._parse(v)
        return r

    @classmethod
    def fromcapparam(cls, cp: AbstractCapParam[CapSection]) -> "_CapParamCfirRegFile":
        v = cp.complexfir_coeff[::-1]
        vv = np.round(v * (1 << cp.complexfir_exponent_offset))  # TODO: check this is ok or not
        r = cls(real=vv.real, imag=vv.imag)
        return r


class _CapParamRfirsRegFile(BaseModel, validate_assignment=True):
    real: RfirsCoeffVector = Field(default_factory=lambda: _zerolist(8))
    imag: RfirsCoeffVector = Field(default_factory=lambda: _zerolist(8))

    def build(self) -> npt.NDArray[np.uint32]:
        r = np.zeros(16, dtype=np.int16)
        r[0:8] = self.real[:]
        r[8:16] = self.imag[:]
        return r.astype(np.uint32)

    def _parse(self, v: npt.NDArray[np.uint32]) -> None:
        vv = v.astype(np.int32)
        self.real[:] = vv[0:8]
        self.imag[:] = vv[8:16]

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32]) -> "_CapParamRfirsRegFile":
        if not (isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == 16 and v.dtype == np.uint32):
            raise TypeError("invalid data for real FIR filters register file")
        r = cls()
        r._parse(v)
        return r

    @classmethod
    def fromcapparam(cls, cp: AbstractCapParam[CapSection]) -> "_CapParamRfirsRegFile":
        r = cls(
            real=np.round(cp.realfirs_real_coeff[::-1] * (1 << cp.realfirs_exponent_offset)),
            imag=np.round(cp.realfirs_imag_coeff[::-1] * (1 << cp.realfirs_exponent_offset)),
        )
        return r


class _CapParamWindowRegFile(BaseModel, validate_assignment=True):
    real: WindowCoeffVector = Field(default=[(1 << 30) if i == 0 else 0 for i in range(2048)])
    imag: WindowCoeffVector = Field(default=[0 for i in range(2048)])

    def build(self, window_size=2048) -> npt.NDArray[np.uint32]:
        if not (1 <= window_size <= 2048):
            raise ValueError(f"invalid size of window (= {window_size})")
        r = np.zeros(4096, dtype=np.int32)
        r[0:window_size] = self.real[0:window_size]
        r[2048 : 2048 + window_size] = self.imag[0:window_size]
        return r.astype(np.uint32)

    def _parse(self, v: npt.NDArray[np.uint32], window_size: int = 2048) -> None:
        if not (1 <= window_size <= 2048):
            raise ValueError(f"invalid size of window (= {window_size})")
        vv = v.astype(np.int32)
        self.real[0:window_size] = vv[0:window_size]
        self.imag[0:window_size] = vv[2048 : 2048 + window_size]

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32], window_size: int = 2048) -> "_CapParamWindowRegFile":
        r = cls()
        r._parse(v, window_size)
        return r

    @classmethod
    def fromcapparam(cls, cp: AbstractCapParam[CapSection]) -> "_CapParamWindowRegFile":
        v = cp.window_coeff
        window_size = len(v)
        vv = np.round(v[0:window_size] * (1 << 30))
        r = cls()
        r.real[0:window_size] = vv.real[:]
        r.imag[0:window_size] = vv.imag[:]
        return r


def classification_paramhalf_validation(
    v: Optional[npt.NDArray[np.float32]], info: ValidationInfo
) -> npt.NDArray[np.float32]:
    if v is None:
        v = np.array((0, 32767, 0), dtype=np.float32)
    if not (isinstance(v, np.ndarray) and v.dtype == np.float32 and v.ndim == 1 and v.shape[0] == 3):
        raise TypeError("invalid classification parameter")

    if not (-32768 <= v[0] <= 32767):
        raise ValueError(f"parameter 'a' (= {v[0]}) is out of range")
    if not (-32768 <= v[1] <= 32767):
        raise ValueError(f"parameter 'b' (= {v[1]}) is out of range")
    if not (-0x8000_0000_0000_0000_0000_0000 <= v[2] <= 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF):
        raise ValueError(f"parameter 'c' (= {v[2]}) is out of range")
    return v


ClassificationRegFileHalf = Annotated[
    npt.NDArray[np.float32],
    PlainValidator(classification_paramhalf_validation),
]


class _CapParamClassificationRegFile(BaseModel, validate_assignment=True):
    p0: ClassificationRegFileHalf = Field(default=None, validate_default=True)
    p1: ClassificationRegFileHalf = Field(default=None, validate_default=True)

    def build(self) -> npt.NDArray[np.uint32]:
        t = np.array(np.hstack((self.p0, self.p1)), dtype=np.float32)
        return np.frombuffer(t.tobytes(), dtype=np.uint32)

    def _parse(self, v: npt.NDArray[np.uint32]) -> None:
        self.p0[:] = np.frombuffer(v[0:3].tobytes(), np.float32)
        self.p1[:] = np.frombuffer(v[3:6].tobytes(), np.float32)

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32]) -> "_CapParamClassificationRegFile":
        r = cls()
        r._parse(v)
        return r

    @classmethod
    def fromcapparam(cls, cp: AbstractCapParam[CapSection]) -> "_CapParamClassificationRegFile":
        r = cls()
        r.p0, r.p1 = calc_abc_main_sub(
            cp.classification_param.pivot_x,
            cp.classification_param.pivot_y,
            cp.classification_param.angle_main,
            cp.classification_param.angle_sub,
            cp.total_exponent_offset(),
        )
        return r


class CapUnitSimplified(AbstractCapUnit):
    __slots__ = (
        "_unit_idx",
        "_ctrl_base",
        "_param_base",
        "_hbmctrl",
        "_mm",
        "_pool",
        "_master_lock",
        "_unit_lock",
        "_cancel_lock",
        "_current_param",
        "_current_reader",
    )

    # relative to CTRL_BASE
    _CTRL_CTRL_REG_ADDR: int = 0x0000_0000
    _CTRL_STATUS_REG_ADDR: int = 0x0000_0004
    _CTRL_ERROR_REG_ADDR: int = 0x0000_0008

    # relative to PARAM_BASE
    _PARAM_CAPTURE_ADDRESS_REG_ADDR: int = 0x0000_0008

    def __init__(
        self,
        unit_idx: int,
        capctrl: AbstractCapCtrl,
        hbmctrl: HbmCtrl,
        mm: E7awgAbstractMemoryManager,
        settings: dict[str, Any],
    ):
        if unit_idx not in capctrl.units:
            raise ValueError(f"no cap_unit-#{unit_idx:02d} is available for the given capctrl")
        self._unit_idx: Final[int] = unit_idx
        self._capctrl: AbstractCapCtrl = capctrl
        for k in ("ctrl_base", "param_base"):
            if k not in settings:
                raise ValueError(f"an essential key '{k}' in the settings is missing")
        self._ctrl_base: int = int(settings["ctrl_base"])
        self._param_base: int = int(settings["param_base"])
        self._hbmctrl: HbmCtrl = hbmctrl  # TODO: introduce mechanism to manage captured data
        self._mm: E7awgAbstractMemoryManager = mm  # TODO: introduce mechanism to manage captured data

        # control
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._master_lock = self._capctrl.lock
        self._unit_lock = RLock()
        self._cancel_lock = RLock()
        self._waiting_for_capture: bool = False
        self._capture_cancel_request: bool = False
        self._cancel_duration: int = 0

        # contexts
        self._current_param: Union[AbstractCapParam[CapSection], None] = None
        self._current_reader: Union[CapIqDataReader, None] = None

    def initialize(self):
        # Notes: issue terminate command for the case capture unit has been activated in the previously executed script.
        with self._unit_lock:
            if isinstance(self._capctrl, CapCtrlSimpleMulti):
                with self._master_lock:
                    self._capctrl.remove_triggerable_unit(self._unit_idx)
            self.unload_parameter()
            self._waiting_for_capture = False
            self._terminate()
            self._wait_free(
                _DEFAULT_TIMEOUT,
                _DEFAULT_POLLING_PERIOD,
                f"failed to terminate cap_unit-#{self._unit_idx:02d} due to timeout",
            )
            if self.is_busy():
                raise RuntimeError(f"cap_unit-#{self.unit_index:02d} is still busy after sending a termination request")
            self.clear_done()
            self.check_error()

    @property
    def unit_index(self) -> int:
        return self._unit_idx

    @property
    def module_index(self) -> int:
        with self._master_lock:
            for m in self._capctrl.modules:
                if self.unit_index in self._capctrl.units_of_module(m):
                    return m
            else:
                raise AssertionError("internal inconsistency")

    def _read_ctrl_reg(self, addr) -> np.uint32:
        return self._capctrl.read_reg(self._ctrl_base + addr)

    def _write_ctrl_reg(self, addr, value: np.uint32):
        self._capctrl.write_reg(self._ctrl_base + addr, value)

    def _read_param_reg(self, addr: int) -> np.uint32:
        return self._capctrl.read_reg(self._param_base + addr)

    def _write_param_reg(self, addr: int, val: np.uint32):
        self._capctrl.write_reg(self._param_base + addr, val)

    def _read_param_regs(self, addr: int, size: int) -> npt.NDArray[np.uint32]:
        return self._capctrl.read_regs(self._param_base + addr, size)

    def _write_param_regs(self, addr: int, val: npt.NDArray[np.uint32]):
        self._capctrl.write_regs(self._param_base + addr, val)

    def _set_ctrl(self, ctrl: CapCtrlCtrlReg) -> None:
        self._write_ctrl_reg(self._CTRL_CTRL_REG_ADDR, ctrl.build())

    def _get_status(self) -> CapCtrlStatusReg:
        return CapCtrlStatusReg.parse(self._read_ctrl_reg(self._CTRL_STATUS_REG_ADDR))

    def _get_error(self) -> CapCtrlErrorReg:
        return CapCtrlErrorReg.parse(self._read_ctrl_reg(self._CTRL_ERROR_REG_ADDR))

    def _set_cap_param_address_only(self, mobj: E7awgMemoryObj):
        # Notes: lock should be acquired by caller.
        capture_address = (self._mm._address_offset + mobj.address_top) >> 5
        self._write_param_reg(self._PARAM_CAPTURE_ADDRESS_REG_ADDR, np.uint32(capture_address))

    def _set_cap_param(self, param: AbstractCapParam[CapSection], mobj: E7awgMemoryObj):
        # Notes: lock should be acquired by caller.
        main_regs: _CapParamSimplifiedMainRegFile = _CapParamSimplifiedMainRegFile.fromcapparam(param)
        main_regs.capture_address = (self._mm._address_offset + mobj.address_top) >> 5
        self._write_param_regs(0x0000_0000, main_regs.build())
        sec_regs: _CapParamSectionRegFile = _CapParamSectionRegFile.fromcapparam(param)
        b = sec_regs.build()
        self._write_param_regs(0x0000_1000, b[0 : main_regs.num_sum_section])
        self._write_param_regs(0x0000_5000, b[0x1000 : 0x1000 + main_regs.num_sum_section])

        num_word = 0
        for i in range(main_regs.num_sum_section):
            num_word += sec_regs.nums_blank_word[i] + sec_regs.nums_capture_word[i]
        self._capture_duration = main_regs.capture_delay + num_word * main_regs.num_integ_section

    def is_awake(self) -> bool:
        return self._get_status().wakeup

    def is_busy(self) -> bool:
        return self._get_status().busy

    def is_done(self) -> bool:
        return self._get_status().done

    def check_error(self) -> CapCtrlErrorReg:
        err = self._get_error()
        if err.xfer_failure:
            raise E7awgHardwareError(f"transfer error is detected at cap_unit-#{self._unit_idx:02d}")
        if err.fifo_overflow:
            raise E7awgHardwareError(f"fifo overflow is detected at cap_unit-#{self._unit_idx:02d}")
        return err

    def hard_reset(self) -> None:
        with self._unit_lock, self._master_lock:
            self._mm.reset()
            self._set_ctrl(CapCtrlCtrlReg(reset=True))
            time.sleep(1e-5)
            self._set_ctrl(CapCtrlCtrlReg(reset=False))
            time.sleep(1e-5)
            if not self.is_awake():
                raise E7awgHardwareError(f"cap_unit-#{self._unit_idx:02d} is not awake after resetting")
            if self.is_busy():
                raise E7awgHardwareError(f"cap_unit-#{self._unit_idx:02d} is still busy after resetting")

    def _validate_parameter(self, param: AbstractCapParam[CapSection]) -> None:
        # Notes: validate param
        _CapParamSimplifiedMainRegFile.fromcapparam(param)

    def _allocate_read_buffer(self, **kwargs) -> E7awgMemoryObj:
        if self._current_param is None:
            raise AssertionError("_allocate_read_buffer() requires self._current_param")

        # Notes: lock is acquired by caller
        if self._current_param.classification_enable:
            # Notes: samples_in_word * bytes_in_2bit
            bufsize = np.ceil(self._current_param.get_datasize_in_sample() / 4)
        else:
            # Notes: samples_in_word * bytes_in_complex64
            bufsize = self._current_param.get_datasize_in_sample() * 8
        return self._mm.allocate(bufsize, minimum_align=_CAP_MINIMUM_ALIGN, **kwargs)

    def load_parameter(self, param: AbstractCapParam[CapSection], **kwargs) -> None:
        self._validate_parameter(param)
        with self._unit_lock:
            if self.is_busy():
                raise RuntimeError(f"cap_unit-#{self._unit_idx:02d} is busy")
            self._current_reader = None
            self._current_param = copy.deepcopy(param)
            mobj = self._allocate_read_buffer(**kwargs)
            self._current_reader = CapIqDataReader(self._current_param.get_parser(), mobj, self._mm, self._hbmctrl)
            self._set_cap_param(param, mobj)

    def reload_parameter(self, **kwargs) -> None:
        if self._current_param is None:
            raise RuntimeError(f"capture parameter is not set to cap_unit-#{self._unit_idx:02x} yet")
        with self._unit_lock:
            if self.is_busy():
                raise RuntimeError(f"cap_unit-#{self._unit_idx:02d} is busy")
            self._current_reader = None
            mobj = self._allocate_read_buffer(**kwargs)
            self._current_reader = CapIqDataReader(self._current_param.get_parser(), mobj, self._mm, self._hbmctrl)
            self._set_cap_param_address_only(mobj)

    def unload_parameter(self) -> None:
        with self._unit_lock:
            self._current_reader = None
        self._capture_duration = 0

    def is_loaded(self) -> bool:
        return self._current_reader is not None

    def get_capture_duration(self) -> int:
        return self._capture_duration

    def start(self) -> None:
        v0 = CapCtrlCtrlReg()
        v1 = CapCtrlCtrlReg(start=True)
        with self._unit_lock, self._master_lock:
            # Notes: start is executed at a positive edge
            self._set_ctrl(v0)
            self._set_ctrl(v1)

    def clear_done(self) -> None:
        v0 = CapCtrlCtrlReg()
        v1 = CapCtrlCtrlReg(done_clr=True)
        with self._unit_lock, self._master_lock:
            # Notes: clear_done is executed at a positive edge
            self._set_ctrl(v0)
            self._set_ctrl(v1)

    def _wait_free(self, timeout: float, polling_period: float, timeout_msg: str) -> None:
        # Notes: lock should be acquired by caller.

        # Notes: the timing of check_error() is considered carefully with respect to the efficiency (less register
        #        access is better) and the priority (more important for the users than TimeoutError).
        t1 = time.perf_counter() + timeout
        while time.perf_counter() < t1:
            time.sleep(polling_period)
            with self._master_lock:
                if not self.is_busy():
                    if self.is_done():
                        self.clear_done()
                    self.check_error()
                    break
        else:
            self.check_error()
            raise TimeoutError(timeout_msg)

        return None

    def _wait_free_async(self, timeout: float, polling_period: float, timeout_msg: str) -> Future[None]:
        # Notes: lock should be acquired by caller.
        return self._pool.submit(self._wait_free, timeout, polling_period, timeout_msg)

    def _terminate(self) -> None:
        v0 = CapCtrlCtrlReg()
        v1 = CapCtrlCtrlReg(terminate=True)
        with self._unit_lock, self._master_lock:
            # Notes: terminate is executed at a positive edge
            self._set_ctrl(v0)
            self._set_ctrl(v1)

    def terminate(
        self,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
    ) -> Future[None]:
        timeout_ = timeout or _DEFAULT_TIMEOUT
        polling_period_ = polling_period or _DEFAULT_POLLING_PERIOD
        with self._unit_lock:
            self._terminate()
            return self._wait_free_async(
                timeout_, polling_period_, f"failed to terminate cap_unit-#{self._unit_idx:02d} due to timeout"
            )

    def get_reader(self) -> CapIqDataReader:
        if self._current_reader is None:
            raise RuntimeError(f"fcapunit-#{self._unit_idx} is not configured yet")
        r = self._current_reader
        self.reload_parameter()
        return r

    def get_num_captured_sample(self):
        # TODO: consider more efficient implementation
        with self._unit_lock:
            main_regs = _CapParamMainRegFile.parse(self._read_param_regs(0x0000_0000, 8))
        return main_regs.num_capture_sample

    def _get_cap_param(self) -> dict[str, Any]:
        # Notes: for debug purposes only.
        with self._unit_lock:
            main_regs = _CapParamSimplifiedMainRegFile.parse(self._read_param_regs(0x0000_0000, 8))
            sec_regs = _CapParamSectionRegFile.parse(self._read_param_regs(0x1000, 8192), main_regs.num_sum_section)
            return {
                "main": main_regs,
                "sec": sec_regs,
            }


class CapUnit(CapUnitSimplified):
    # Notes: to be relocated in a right place with some modifications
    def _set_cap_param(self, param: AbstractCapParam[CapSection], mobj: E7awgMemoryObj):
        # Notes: lock should be acquired by caller.
        main_regs: _CapParamMainRegFile = _CapParamMainRegFile.fromcapparam(param)
        main_regs.capture_address = (self._mm._address_offset + mobj.address_top) >> 5
        self._write_param_regs(0x0000_0000, main_regs.build())
        sec_regs: _CapParamSectionRegFile = _CapParamSectionRegFile.fromcapparam(param)
        b = sec_regs.build()
        self._write_param_regs(0x0000_1000, b[0 : main_regs.num_sum_section])
        self._write_param_regs(0x0000_5000, b[0x1000 : 0x1000 + main_regs.num_sum_section])
        if main_regs.dsp_switch.cfir_en:
            c = _CapParamCfirRegFile.fromcapparam(param).build()
            self._write_param_regs(0x0000_9000, c)
        if main_regs.dsp_switch.rfirs_en:
            r = _CapParamRfirsRegFile.fromcapparam(param).build()
            self._write_param_regs(0x0000_A000, r)
        if main_regs.dsp_switch.wdw_en:
            w = _CapParamWindowRegFile.fromcapparam(param).build()
            self._write_param_regs(0x0000_B000, w)
        if main_regs.dsp_switch.clsfy_en:
            f = _CapParamClassificationRegFile.fromcapparam(param).build()
            self._write_param_regs(0x0000_F000, f)

        num_word = 0
        for i in range(main_regs.num_sum_section):
            num_word += sec_regs.nums_capture_word[i] + sec_regs.nums_blank_word[i]
        self._capture_duration = main_regs.capture_delay + num_word * main_regs.num_integ_section

    def _get_cap_param(self) -> dict[str, Any]:
        # Notes: for debug purposes only.
        with self._unit_lock:
            r = super()._get_cap_param()
            r["main"] = _CapParamMainRegFile.parse(self._read_param_regs(0x0000_0000, 8))
            r["cfir"] = _CapParamCfirRegFile.parse(self._read_param_regs(0x0000_9000, 32))
            r["rfirs"] = _CapParamRfirsRegFile.parse(self._read_param_regs(0x0000_A000, 16))
            r["wdw"] = _CapParamWindowRegFile.parse(self._read_param_regs(0x0000_B000, 4096))
            r["clsfy"] = _CapParamClassificationRegFile.parse(self._read_param_regs(0x0000_F000, 6))
            return r

    def _validate_parameter(self, param: AbstractCapParam[CapSection]) -> None:
        # Notes: validate param
        _CapParamMainRegFile.fromcapparam(param)


CapUnitSwitchable = CapUnit
