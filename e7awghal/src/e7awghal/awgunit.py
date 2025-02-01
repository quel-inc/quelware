import logging
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, Callable, Final, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from e7awghal.abstract_register import AbstractFpgaReg, b_1bf_bool, p_1bf_bool
from e7awghal.awgctrl import AwgCtrl
from e7awghal.common_defs import _DEFAULT_POLLING_PERIOD, _DEFAULT_TIMEOUT, E7awgHardwareError
from e7awghal.e7awg_memoryobj import E7awgAbstractMemoryManager
from e7awghal.fwtype import E7FwAuxAttr
from e7awghal.hbmctrl import HbmCtrl
from e7awghal.wavedata import AwgParam, WaveChunk, WaveLibrary

logger = logging.getLogger(__name__)


class AwgUnitCtrlReg(AbstractFpgaReg):
    reset: bool = Field(default=False)  # [0]
    prepare: bool = Field(default=False)  # [1]
    start: bool = Field(default=False)  # [2]
    terminate: bool = Field(default=False)  # [3]
    done_clr: bool = Field(default=False)  # [4]

    def _parse(self, v: np.uint32) -> None:
        self.reset = p_1bf_bool(v, 0)
        self.prepare = p_1bf_bool(v, 1)
        self.start = p_1bf_bool(v, 2)
        self.terminate = p_1bf_bool(v, 3)
        self.done_clr = p_1bf_bool(v, 4)

    def build(self) -> np.uint32:
        return (
            b_1bf_bool(self.reset, 0)
            | b_1bf_bool(self.prepare, 1)
            | b_1bf_bool(self.start, 2)
            | b_1bf_bool(self.terminate, 3)
            | b_1bf_bool(self.done_clr, 4)
        )

    @classmethod
    def parse(cls, v: np.uint32) -> "AwgUnitCtrlReg":
        r = cls()
        r._parse(v)
        return r


class AwgUnitStatusReg(AbstractFpgaReg):
    wakeup: bool = Field(default=False)  # [0]
    busy: bool = Field(default=False)  # [1]
    ready: bool = Field(default=False)  # [2]
    done: bool = Field(default=False)  # [3]

    def _parse(self, v: np.uint32) -> None:
        self.wakeup = p_1bf_bool(v, 0)
        self.busy = p_1bf_bool(v, 1)
        self.ready = p_1bf_bool(v, 2)
        self.done = p_1bf_bool(v, 3)

    def build(self) -> np.uint32:
        return (
            b_1bf_bool(self.wakeup, 0) | b_1bf_bool(self.busy, 1) | b_1bf_bool(self.ready, 2) | b_1bf_bool(self.done, 3)
        )

    @classmethod
    def parse(cls, v: np.uint32) -> "AwgUnitStatusReg":
        r = cls()
        r._parse(v)
        return r


class AwgUnitErrorReg(AbstractFpgaReg):
    xfer_failure: bool = Field(default=False)  # [0]
    sample_shortage: bool = Field(default=False)  # [1]

    def _parse(self, v: np.uint32):
        self.xfer_failure = p_1bf_bool(v, 0)
        self.sample_shortage = p_1bf_bool(v, 1)

    def build(self) -> np.uint32:
        return b_1bf_bool(self.xfer_failure, 0) | b_1bf_bool(self.sample_shortage, 1)

    @classmethod
    def parse(cls, v: np.uint32) -> "AwgUnitErrorReg":
        r = cls()
        r._parse(v)
        return r


class _WaveChunkParameter:
    @classmethod
    def fromwavechunk(cls, wavechunk: WaveChunk, wavelib: WaveLibrary) -> "_WaveChunkParameter":
        name_of_data = wavechunk.name_of_wavedata

        if not wavelib.has_wavedata(name_of_data):
            raise RuntimeError(f"the given library doesn't have wave data `{name_of_data}'")

        return cls(
            dataptr=wavelib.get_pointer_to_wavedata(name_of_data),
            datalen_in_word=wavelib.get_size_in_word_of_wavedata(name_of_data),
            blanklen_in_word=wavechunk.num_blank_word,
            num_repeat=wavechunk.num_repeat,
        )

    def __init__(self, *, dataptr: int, datalen_in_word: int, blanklen_in_word: int, num_repeat: int):
        if dataptr % 16 != 0:
            raise ValueError(f"unaligned data pointer: {dataptr:09x}")
        if datalen_in_word % 16 != 0:
            raise ValueError(f"unaligned length of data pointer: {datalen_in_word:09x}")
        if not 16 <= datalen_in_word <= 67108864:
            raise ValueError(f"invalid unaligned length of data pointer: {datalen_in_word:09x}")

        self._dataptr = dataptr // 16  # Notes: scaled physical address
        self._datalen_in_word = datalen_in_word
        self._blanklen_in_word = blanklen_in_word
        self._num_repeat = num_repeat

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} object with ptr={self._dataptr*16:09x} size={self._datalen_in_word * 4:d} "
            f"blank={self._blanklen_in_word * 4:d} repeat={self._num_repeat:d}>"
        )

    def validate(self):
        # Notes: should be called just before setting me to registers
        self.validate_blanklen_in_word(self._blanklen_in_word)
        self.validate_num_repeat(self._num_repeat)

    @property
    def datalen_in_word(self) -> int:
        return self._datalen_in_word

    @staticmethod
    def validate_blanklen_in_word(v: int):
        if not 0 <= v <= 0xFFFF_FFFF:
            raise ValueError(f"invalid length of blank of a chunk: {v}")

    @property
    def blanklen_in_word(self) -> int:
        return self._blanklen_in_word

    @blanklen_in_word.setter
    def blanklen_in_word(self, v: int):
        self.validate_blanklen_in_word(v)
        self._blanklen_in_word = v

    @staticmethod
    def validate_num_repeat(v: int):
        if not 1 <= v <= 4294967295:
            raise ValueError(f"invalid num_repeat of a chunk: {v}")

    @property
    def num_repeat(self) -> int:
        return self._num_repeat

    @num_repeat.setter
    def num_repeat(self, v: int):
        self.validate_num_repeat(v)
        self._num_repeat = v


class _WaveParamRegFile(BaseModel, validate_assignment=True):
    num_wait_word: int = Field(ge=0, le=0xFFFF_FFFF, default=0)
    num_repeat: int = Field(ge=1, le=0xFFFF_FFFF, default=1)
    num_chunk: int = Field(ge=1, le=16, default=1)
    start_interval: int = Field(ge=1, le=0xFFFF_FFFF, default=1)

    def build(self) -> npt.NDArray[np.uint32]:
        r = np.zeros(4, dtype=np.uint32)
        r[0] = self.num_wait_word
        r[1] = self.num_repeat
        r[2] = self.num_chunk
        r[3] = self.start_interval
        return r

    def _parse(self, v: npt.NDArray[np.uint32]) -> None:
        self.num_wait_word = v[0]
        self.num_repeat = v[1]
        self.num_chunk = v[2]
        self.start_interval = v[3]

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32]) -> "_WaveParamRegFile":
        r = cls()
        r._parse(v)
        return r

    @classmethod
    def fromwaveparam(cls, wp: AwgParam) -> "_WaveParamRegFile":
        r = cls()
        r.num_wait_word = wp.num_wait_word
        r.num_repeat = wp.num_repeat
        r.num_chunk = wp.num_chunk
        r.start_interval = wp.start_interval
        return r


class _WaveChunkRegFile(BaseModel, validate_assignment=True):
    start_address: int = Field(ge=0, le=0x1FFF_FFFF, multiple_of=2, default=0)  # address >> 4
    num_wave_word: int = Field(ge=16, le=0xFFFF_FFFF, multiple_of=16, default=16)
    num_blank_word: int = Field(ge=0, le=0xFFFF_FFFF, default=0)
    num_repeat: int = Field(ge=1, le=0xFFFF_FFFF, default=1)

    def build(self) -> npt.NDArray[np.uint32]:
        r = np.zeros(4, dtype=np.uint32)
        r[0] = self.start_address
        r[1] = self.num_wave_word
        r[2] = self.num_blank_word
        r[3] = self.num_repeat
        return r

    def _parse(self, v: npt.NDArray[np.uint32]) -> None:
        self.start_address = v[0]
        self.num_wave_word = v[1]
        self.num_blank_word = v[2]
        self.num_repeat = v[3]

    @classmethod
    def parse(cls, v: npt.NDArray[np.uint32]) -> "_WaveChunkRegFile":
        r = cls()
        r._parse(v)
        return r

    @classmethod
    def fromwavechunk(cls, c: WaveChunk, lib: WaveLibrary) -> "_WaveChunkRegFile":
        r = cls()
        ptr = lib.get_pointer_to_wavedata(c.name_of_wavedata)
        if ptr & 0x1F != 0:
            raise ValueError(f"misaligned start address of wavedata (= 0x{ptr:09x})")
        r.start_address = lib.get_pointer_to_wavedata(c.name_of_wavedata) >> 4
        r.num_wave_word = lib.get_size_in_word_of_wavedata(c.name_of_wavedata)
        r.num_blank_word = c.num_blank_word
        r.num_repeat = c.num_repeat
        return r


class AwgUnit:
    # relative to CTRL_BASE
    _UNIT_CTRL_REG_ADDR: int = 0x0000_0000
    _UNIT_STATUS_REG_ADDR: int = 0x0000_0004
    _UNIT_ERROR_REG_ADDR: int = 0x0000_0008

    __slots__ = (
        "_unit_idx",
        "_awgctrl",
        "_ctrl_base",
        "_param_base",
        "_chunk_bases",
        "_broken_reset",
        "_lib",
        "_oneshot_initialize_done",
        "_null_wave",
        "_wave_duration",
        "_pool",
        "_master_lock",
        "_unit_lock",
    )

    def __init__(
        self,
        unit_idx: int,
        awgctrl: AwgCtrl,
        hbmctrl: HbmCtrl,
        mm: E7awgAbstractMemoryManager,
        settings: dict[str, Any],
    ):
        awgctrl._validate_unit(unit_idx)
        self._unit_idx: Final[int] = unit_idx
        self._awgctrl: Final[AwgCtrl] = awgctrl

        for k in ("ctrl_base", "param_base", "chunk_bases"):
            if k not in settings:
                raise ValueError(f"an essential key '{k}' in the settings is missing")
        self._ctrl_base: int = int(settings["ctrl_base"])
        self._param_base: int = int(settings["param_base"])
        self._chunk_bases: list[int] = [int(x) for x in settings["chunk_bases"]]
        self._broken_reset: bool = E7FwAuxAttr.BROKEN_AWG_RESET in self._awgctrl._auxattr

        self._lib: Final[WaveLibrary] = WaveLibrary(hbmctrl, mm)
        self._oneshot_initialize_done: bool = False
        self._null_wave: Union[AwgParam, None] = None
        self._wave_duration = 0

        self._pool = ThreadPoolExecutor(max_workers=1)
        self._master_lock = self._awgctrl.lock
        self._unit_lock = RLock()

    @property
    def unit_index(self) -> int:
        return self._unit_idx

    def _read_ctrl_reg(self, addr) -> np.uint32:
        return self._awgctrl.read_reg(self._ctrl_base + addr)

    def _write_ctrl_reg(self, addr, value: np.uint32):
        self._awgctrl.write_reg(self._ctrl_base + addr, value)

    def _read_param_reg(self, addr: int) -> np.uint32:
        return self._awgctrl.read_reg(self._param_base + addr)

    def _write_param_reg(self, addr: int, val: np.uint32):
        self._awgctrl.write_reg(self._param_base + addr, val)

    def _read_param_regs(self, addr: int, size: int) -> npt.NDArray[np.uint32]:
        return self._awgctrl.read_regs(self._param_base + addr, size)

    def _write_param_regs(self, addr: int, vals: npt.NDArray[np.uint32]):
        self._awgctrl.write_regs(self._param_base + addr, vals)

    def _read_chunk_reg(self, chunk_idx: int, addr: int) -> np.uint32:
        return self._awgctrl.read_reg(self._param_base + self._chunk_bases[chunk_idx] + addr)

    def _write_chunk_reg(self, chunk_idx: int, addr: int, val: np.uint32):
        self._awgctrl.write_reg(self._param_base + self._chunk_bases[chunk_idx] + addr, val)

    def _read_chunk_regs(self, chunk_idx: int, addr: int, size: int) -> npt.NDArray[np.uint32]:
        return self._awgctrl.read_regs(self._param_base + self._chunk_bases[chunk_idx] + addr, size)

    def _write_chunk_regs(self, chunk_idx: int, addr: int, vals: npt.NDArray[np.uint32]):
        self._awgctrl.write_regs(self._param_base + self._chunk_bases[chunk_idx] + addr, vals)

    def _set_ctrl(self, ctrl: AwgUnitCtrlReg) -> None:
        self._write_ctrl_reg(self._UNIT_CTRL_REG_ADDR, ctrl.build())

    def _get_status(self) -> AwgUnitStatusReg:
        return AwgUnitStatusReg.parse(self._read_ctrl_reg(self._UNIT_STATUS_REG_ADDR))

    def _get_error(self) -> AwgUnitErrorReg:
        return AwgUnitErrorReg.parse(self._read_ctrl_reg(self._UNIT_ERROR_REG_ADDR))

    def _oneshot_initialize(self):
        if not self._oneshot_initialize_done:
            self.register_wavedata_from_complex64vector(
                "null", np.zeros(64, dtype=np.complex64), address_top=0x000000000
            )
            self._null_wave = AwgParam(num_wait_word=0, num_repeat=1)
            self._null_wave.chunks.append(WaveChunk(name_of_wavedata="null", num_blank_word=0, num_repeat=1))
            self._oneshot_initialize_done = True

    def initialize(self) -> None:
        with self._unit_lock, self._master_lock:
            self.clear_done()
            fut = self.terminate()
        fut.result()

        with self._unit_lock, self._master_lock:
            if self.is_busy():
                raise RuntimeError(f"awg_unit-#{self.unit_index:02d} is still busy after sending a termination request")
            self._oneshot_initialize()
            assert self._null_wave is not None, "internal error"
            self.load_parameter(self._null_wave)

    def is_awake(self) -> bool:
        return self._get_status().wakeup

    def is_busy(self) -> bool:
        return self._get_status().busy

    def is_ready(self) -> bool:
        return self._get_status().ready

    def is_done(self) -> bool:
        return self._get_status().done

    def has_started(self) -> bool:
        v = self._get_status()
        return v.busy or v.done

    def get_names_of_wavedata(self) -> set[str]:
        with self._unit_lock:
            return self._lib.get_names_of_wavedata()

    def register_wavedata_from_complex64vector(self, name: str, iq: npt.NDArray[np.complex64], **kwargs) -> None:
        with self._unit_lock:
            self._lib.register_wavedata_from_complex64vector(name, iq, **kwargs)

    def delete_wavedata(self, name: str) -> None:
        if name == "null":
            raise ValueError("cannot remove 'null' wave data because it is reserved by the system")
        with self._unit_lock:
            self._lib.delete_wavedata(name)

    def has_wavedata(self, name: str) -> bool:
        with self._unit_lock:
            return self._lib.has_wavedata(name)

    def hard_reset(self, suppress_warning: bool = False) -> None:
        if self._broken_reset and not suppress_warning:
            warnings.warn(
                "AwgUnit.hard_reset() may disrupt the JESD204C link due to firmware bug, "
                "it should not be used in nominal situations",
                RuntimeWarning,
            )
        v0 = AwgUnitCtrlReg(reset=True)
        v1 = AwgUnitCtrlReg()
        with self._unit_lock, self._master_lock:
            self._set_ctrl(v0)
            time.sleep(1e-5)
            self._set_ctrl(v1)
            time.sleep(1e-5)
            if not self.is_awake():
                raise E7awgHardwareError(f"awg_unit-#{self._unit_idx:02d} is not awake after resetting")

    def load_parameter(self, wave_param: AwgParam) -> None:
        with self._unit_lock:
            with self._master_lock:
                if self.is_busy():
                    raise RuntimeError(f"stop preloading because awg_uint-#{self._unit_idx:02d} is still busy")
                self._set_wave_param(wave_param)

    @staticmethod
    def _wait_loop(
        check_callback: Callable[[], bool],
        timeout_callback: Callable[[], None],
        timeout: float,
        polling_period: float,
    ) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() < t0 + timeout:
            time.sleep(polling_period)
            if check_callback():
                break
        else:
            timeout_callback()

    def clear_done(self) -> None:
        v0 = AwgUnitCtrlReg()
        v1 = AwgUnitCtrlReg(done_clr=True)
        with self._unit_lock:
            with self._master_lock:
                # Notes: clear_done is executed at a positive edge
                self._set_ctrl(v0)
                self._set_ctrl(v1)

    def _wait_free(self, timeout: float, polling_period: float, timeout_msg: str) -> Future[None]:
        def _wait_free_unit_loop() -> None:
            # Notes: the timing of check_error() is considered carefully with respect to the efficiency (less register
            #        access is better) and the priority (more important for the users than TimeoutError).
            with self._unit_lock:
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

        return self._pool.submit(_wait_free_unit_loop)

    def _terminate(self):
        # Notes: _unit_lock should be taken by the caller
        v0 = AwgUnitCtrlReg()
        v1 = AwgUnitCtrlReg(terminate=True)
        with self._master_lock:
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
            return self._wait_free(
                timeout_, polling_period_, f"failed to terminate awg_unit-#{self._unit_idx:02d} due to timeout"
            )

    def check_error(self) -> None:
        err = self._get_error()
        if err.xfer_failure:
            raise E7awgHardwareError(f"transfer error is detected at awg_unit-#{self._unit_idx:02d}")
        if err.sample_shortage:
            raise E7awgHardwareError(f"sample shortage error is detected at awg_unit-#{self._unit_idx:02d}")

    def _set_wave_param(self, wave_param: AwgParam):
        # Notes: lock should be acquired by caller.
        num_word = 0
        for i, c in enumerate(wave_param.chunks):
            crf = _WaveChunkRegFile.fromwavechunk(c, self._lib)
            num_word += (crf.num_wave_word + crf.num_blank_word) * crf.num_repeat
            self._write_chunk_regs(i, 0, crf.build())

        prf = _WaveParamRegFile.fromwaveparam(wave_param)
        self._wave_duration = prf.num_wait_word + num_word * prf.num_repeat
        self._write_param_regs(0, prf.build())

    def get_wave_duration(self) -> int:
        return self._wave_duration

    def _get_wave_param(self) -> tuple[_WaveParamRegFile, tuple[_WaveChunkRegFile, ...]]:
        # Notes: for debug and test purposes only
        with self._unit_lock:
            prf = _WaveParamRegFile.parse(self._read_param_regs(0, 4))
            crfs = [_WaveChunkRegFile.parse(self._read_chunk_regs(i, 0, 4)) for i in range(prf.num_chunk)]
            return prf, tuple(crfs)
