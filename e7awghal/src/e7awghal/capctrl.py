import copy
import logging
import time
from collections.abc import Collection
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, Final, Union

import numpy as np
import numpy.typing as npt
from pydantic import Field

from e7awghal.abstract_cap import AbstractCapCtrl, AbstractSimpleMultiTriggerMixin
from e7awghal.abstract_register import AbstractFpgaReg, b_1bf_bool, p_1bf_bool
from e7awghal.common_defs import _DEFAULT_POLLING_PERIOD, _DEFAULT_TIMEOUT, E7awgHardwareError
from e7awghal.common_register import E7awgVersion
from e7awghal.e7awg_packet import BasePacketAccess, E7awgOutgoingPacket, E7awgPacketAccess, E7awgPacketMode
from e7awghal.fwtype import E7FwAuxAttr

logger = logging.getLogger(__name__)


class CapMasterCtrlReg(AbstractFpgaReg):
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
    def parse(cls, v: np.uint32) -> "CapMasterCtrlReg":
        r = cls()
        r._parse(v)
        return r


class CapCtrlBase(AbstractCapCtrl):
    __slots__ = (
        "_udprw",
        "_settings",
        "_unit_mask",
        "_units_in_module",
        "_number_of_module",
        "_number_of_unit",
        "_pool",
        "_lock",
    )

    _MASTER_VERSION_REG_ADDR: int = 0x0000_0000
    _MASTER_TGTSEL_REG_ADDR: int = 0x0000_0010
    _MASTER_CTRL_REG_ADDR: int = 0x0000_0014
    _MASTER_STATUS_WAKEUP_REG_ADDR: int = 0x0000_0018
    _MASTER_STATUS_BUSY_REG_ADDR: int = 0x0000_001C
    _MASTER_STATUS_DONE_REG_ADDR: int = 0x0000_0020
    _MASTER_ERROR_FIFO_OVERFLOW_REG_ADDR: int = 0x0000_0024
    _MASTER_ERROR_XFER_FAILURE_REG_ADDR: int = 0x0000_0028

    def __init__(
        self,
        ipaddrs: dict[str, str],
        settings: dict[str, Any],
        udprws: dict[tuple[str, int], BasePacketAccess],
        auxattr: set[E7FwAuxAttr],
    ):
        self._settings: Final[dict[str, Any]] = settings
        for k in ("nic", "port", "units", "ctrl_master_base", "units", "units_in_module"):
            if k not in self._settings:
                raise ValueError(f"an essential key '{k}' in the settings is missing")
        nic, port = settings["nic"], settings["port"]
        udprw = udprws.get((nic, port))
        if udprw is None:
            udprw = E7awgPacketAccess(ipaddrs[nic], port)
            udprws[nic, port] = udprw
        else:
            if not isinstance(udprw, E7awgPacketAccess):
                raise RuntimeError(f"inconsistent network interface configuration for {nic}:{port}")
        self._udprw: Final[E7awgPacketAccess] = udprw
        self._auxattr: Final[set[E7FwAuxAttr]] = auxattr
        self._unit_mask: np.uint32 = np.uint32(0)
        self._units_in_module: tuple[list[int], ...] = tuple(copy.deepcopy(settings["units_in_module"]))
        self._number_of_module: Final[int] = len(self._units_in_module)
        self._number_of_unit: Final[int] = len(settings["units"])
        if self._number_of_unit != sum([len(u_of_m) for u_of_m in self._units_in_module]):
            raise ValueError("mismatched number of units in the settings")
        self._pool = ThreadPoolExecutor()
        self._lock = RLock()

    @property
    def lock(self) -> RLock:
        return self._lock

    def initialize(self):
        with self._lock:
            self._set_master_mask(set(), use_cache=False)

    @property
    def version(self) -> str:
        v = self._read_master_reg(self._MASTER_VERSION_REG_ADDR)
        w = E7awgVersion.parse(v)
        return f"{chr(w.ver_char)}:20{w.ver_year:02d}/{w.ver_month:02d}/{w.ver_day:02d}-{w.ver_id:d}"

    @property
    def num_module(self) -> int:
        return self._number_of_module

    @property
    def modules(self) -> tuple[int, ...]:
        return tuple(range(self._number_of_module))

    @property
    def num_unit(self) -> int:
        return self._number_of_unit

    @property
    def units(self) -> tuple[int, ...]:
        return tuple(range(self._number_of_unit))

    def num_unit_of_module(self, module: int) -> int:
        self._validate_module(module)
        return len(self._units_in_module[module])

    def units_of_module(self, module: int) -> tuple[int, ...]:
        self._validate_module(module)
        return tuple(self._units_in_module[module])

    @staticmethod
    def _set2bitmap(unit_idxs: Collection[int]) -> np.uint32:
        # Notes: unit_idxs should be validated at the caller
        mask = 0
        for u in unit_idxs:
            mask |= 1 << u
        return np.uint32(mask)

    @staticmethod
    def _bitmap2set(unit_mask: np.uint32) -> set[int]:
        r: set[int] = set()
        u: int = int(unit_mask)
        for i in range(32):
            if u & 0x01 == 0x01:
                r.add(i)
            u >>= 1
            if u == 0:
                break
        return r

    def _validate_module(self, mod_idx: int):
        if not 0 <= mod_idx < self._number_of_module:
            raise ValueError(f"invalid cap-module: {mod_idx}")

    def _validate_unit(self, unit_idx: int):
        if not 0 <= unit_idx < self._number_of_unit:
            raise ValueError(f"invalid cap-unit: {unit_idx}")

    def _validate_modules(self, mod_idx: Collection[int]):
        for m in mod_idx:
            self._validate_module(m)

    def _validate_units(self, unit_idxs: Collection[int]):
        for u in unit_idxs:
            self._validate_unit(u)

    def read_reg(self, address: int) -> np.uint32:
        cmd = E7awgOutgoingPacket(
            mode=E7awgPacketMode.CAPTURE_REG_READ,
            address=address,
            num_payload_bytes=4,
        )
        rpl = self._udprw.send_command(cmd)
        val = np.frombuffer(rpl.payload, dtype=np.dtype("<u4"))[0]
        logger.debug(f"{self.__class__.__name__}:_read_reg({address:08x}) --> {val:08x}")
        return val

    def read_regs(self, address: int, size: int) -> npt.NDArray[np.uint32]:
        val = np.zeros(size, dtype=np.uint32)
        idx = 0
        while idx < size:
            s = min(size - idx, 0x140)
            cmd = E7awgOutgoingPacket(
                mode=E7awgPacketMode.CAPTURE_REG_READ,
                address=address + idx * 4,
                num_payload_bytes=4 * s,
            )
            rpl = self._udprw.send_command(cmd)
            val[idx : idx + s] = np.frombuffer(rpl.payload, dtype=np.dtype("<u4"))
            for i in range(s):
                logger.debug(f"{self.__class__.__name__}:_read_reg({address+(idx+i)*4:08x}) --> {val[idx+i]:08x}")
            idx += s

        return val

    def write_reg(self, address: int, val: np.uint32) -> None:
        logger.debug(f"{self.__class__.__name__}:_write_reg({address:08x}, {val:08x})")
        num_payload_bytes = 4
        pkt = E7awgOutgoingPacket(
            mode=E7awgPacketMode.CAPTURE_REG_WRITE,
            address=address,
            num_payload_bytes=num_payload_bytes,
        )
        pkt.payload[0:num_payload_bytes] = int(val).to_bytes(num_payload_bytes, "little")
        _ = self._udprw.send_command(pkt)

    def write_regs(self, address: int, val: npt.NDArray[np.uint32]) -> None:
        view = np.ascontiguousarray(val, "<u4").data
        idx, size = 0, len(view)
        while idx < size:
            s = min(size - idx, 0x140)

            pkt = E7awgOutgoingPacket(
                mode=E7awgPacketMode.CAPTURE_REG_WRITE, address=address + idx * 4, num_payload_bytes=s * 4
            )
            pkt.payload[0 : s * 4] = view[idx : idx + s].tobytes()
            _ = self._udprw.send_command(pkt)
            for i in range(s):
                logger.debug(f"{self.__class__.__name__}:_write_reg({address+(idx+i)*4:08x}, {view[idx+i]:08x})")

            idx += s

    def _read_master_reg(self, addr: int) -> np.uint32:
        return self.read_reg(self._settings["ctrl_master_base"] + addr)

    def _write_master_reg(self, addr: int, val: np.uint32):
        self.write_reg(self._settings["ctrl_master_base"] + addr, val)

    def _get_master_mask(self, use_cache: bool = True) -> set[int]:
        # Notes: awg_units should be validated by caller.
        if not use_cache:
            with self._lock:
                unit_mask = self._read_master_reg(self._MASTER_TGTSEL_REG_ADDR)
                if self._unit_mask != unit_mask:
                    logger.warning(
                        f"cached unit mask (= 0x{self._unit_mask:08x}) is updated "
                        f"with the actual register value (= 0x{unit_mask:08x}"
                    )
                    self._unit_mask = unit_mask
        return self._bitmap2set(self._unit_mask)

    def _set_master_mask(self, cap_units: Collection[int], use_cache: bool = True):
        # Notes: awg_units should be validated by the caller.
        # Notes: lock should be taken by the caller.
        with self._lock:
            unit_mask = self._set2bitmap(cap_units)
            if not use_cache or unit_mask != self._unit_mask:
                self._unit_mask = unit_mask
                self._write_master_reg(self._MASTER_TGTSEL_REG_ADDR, self._unit_mask)

    def _set_master_ctrl(self, ctrl: CapMasterCtrlReg) -> None:
        self._write_master_reg(self._MASTER_CTRL_REG_ADDR, ctrl.build())

    def start_now(self, cap_units: Collection[int]) -> None:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            self._set_master_ctrl(CapMasterCtrlReg())
            self._set_master_ctrl(CapMasterCtrlReg(start=True))
            self._set_master_ctrl(CapMasterCtrlReg())  # Notes: this prevents RTL bad behavior.

    def clear_done(self, cap_units: Collection[int]) -> None:
        # Notes: this also doesn't work well....
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            self._set_master_ctrl(CapMasterCtrlReg())
            self._set_master_ctrl(CapMasterCtrlReg(done_clr=True))
            self._set_master_ctrl(CapMasterCtrlReg())  # Notes: this prevents RTL bad behavior.

    def terminate(self, cap_units: Collection[int]) -> None:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            self._set_master_ctrl(CapMasterCtrlReg())
            self._set_master_ctrl(CapMasterCtrlReg(terminate=True))
            self._set_master_ctrl(CapMasterCtrlReg())  # Notes: this prevents RTL bad behavior.

    def are_awake_any(self, cap_units: Collection[int]) -> bool:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            v = self._read_master_reg(self._MASTER_STATUS_WAKEUP_REG_ADDR)
            return v != 0x00

    def are_busy_any(self, cap_units: Collection[int]) -> bool:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            v = self._read_master_reg(self._MASTER_STATUS_BUSY_REG_ADDR)
            return v != 0x00

    def are_done_any(self, cap_units: Collection[int]) -> bool:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            v = self._read_master_reg(self._MASTER_STATUS_DONE_REG_ADDR)
            return v != 0x00

    def are_awake_all(self, cap_units: Collection[int]) -> bool:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            v = self._read_master_reg(self._MASTER_STATUS_WAKEUP_REG_ADDR)
            return v == self._unit_mask

    def are_busy_all(self, cap_units: Collection[int]) -> bool:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            v = self._read_master_reg(self._MASTER_STATUS_BUSY_REG_ADDR)
            return v == self._unit_mask

    def are_done_all(self, cap_units: Collection[int]) -> bool:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            v = self._read_master_reg(self._MASTER_STATUS_DONE_REG_ADDR)
            return v == self._unit_mask

    def have_started_all(self, cap_units: Collection[int]) -> bool:
        self._validate_units(cap_units)
        with self._lock:
            self._set_master_mask(cap_units)
            v0 = self._read_master_reg(self._MASTER_STATUS_BUSY_REG_ADDR)
            v1 = self._read_master_reg(self._MASTER_STATUS_DONE_REG_ADDR)
            return (v0 | v1) == self._unit_mask

    def check_error(self, cap_units: Collection[int]):
        with self._lock:
            self._set_master_mask(cap_units)
            overflow = self._read_master_reg(self._MASTER_ERROR_FIFO_OVERFLOW_REG_ADDR)
            xfer = self._read_master_reg(self._MASTER_ERROR_XFER_FAILURE_REG_ADDR)
            if overflow != 0:
                idxs = ", ".join([f"#{u:02d}" for u in sorted(self._bitmap2set(overflow))])
                raise E7awgHardwareError(f"fifo overflow error is detected at cap_units-{idxs:s}")
            if xfer != 0:
                idxs = ", ".join([f"#{u:02d}" for u in sorted(self._bitmap2set(xfer))])
                raise E7awgHardwareError(f"transfer error is detected at cap_units-{idxs:s}")

    def _check_done_units(self, units: Collection[int]) -> bool:
        with self._lock:
            if self.are_done_all(units):
                self.clear_done(units)
                self.check_error(units)
                return True
            else:
                return False

    def _wait_to_start(self, units: set[int], timeout: float, polling_period: float, timeout_msg: str) -> Future[None]:
        def _wait_to_start_unit_loop() -> None:
            # Notes: the timing of check_error() is considered carefully with respect to the efficiency (less register
            #        access is better) and the priority (more important for the users than TimeoutError).
            t0 = time.perf_counter()
            while time.perf_counter() < t0 + timeout:
                time.sleep(polling_period)
                with self._lock:
                    if self.have_started_all(units):
                        self.check_error(units)
                        break
            else:
                self.check_error(units)
                raise TimeoutError(timeout_msg)

        return self._pool.submit(_wait_to_start_unit_loop)

    def wait_to_start(
        self,
        units: set[int],
        timeout: float = _DEFAULT_TIMEOUT,
        polling_period: float = _DEFAULT_POLLING_PERIOD,
    ) -> Future[None]:
        self._validate_units(units)
        return self._wait_to_start(
            units,
            timeout,
            polling_period,
            "failed to wait for the completion of some cap_units due to timeout",
        )

    def get_all_capmods(self) -> set[int]:
        return set(self._settings["units_in_module"].keys())

    def get_all_capunts_of_capmod(self, capmod_idx) -> list[int]:
        return list(self._settings["units_in_module"][capmod_idx])


class CapCtrlSimpleMulti(CapCtrlBase, AbstractSimpleMultiTriggerMixin):
    _CTRL_MASTER_TRIGGER_MASK_REG_ADDR: int = 0x0000_000C

    _MODULE_TRIGGER_REG_MAP: dict[int, int] = {}

    __slots__ = ("_triggerable_capunits",)

    def __init__(
        self,
        ipaddrs: dict[str, str],
        settings: dict[str, Any],
        udprws: dict[tuple[str, int], BasePacketAccess],
        auxattr: set[E7FwAuxAttr],
    ):
        super().__init__(ipaddrs, settings, udprws, auxattr)
        self._triggerable_capunits: set[int] = set()

    def initialize(self):
        with self.lock:
            super().initialize()
            # Notes: initialize all trigger settings
            self.clear_triggerable_units()
            for m in self.modules:
                self.set_triggering_awgunit_idx(m, None)

    def set_triggering_awgunit_idx(self, capmod_idx: int, awgunit_idx: Union[int, None]):
        self._validate_module(capmod_idx)
        v = np.uint32(0 if awgunit_idx is None else awgunit_idx + 1)
        self._write_master_reg(self._MODULE_TRIGGER_REG_MAP[capmod_idx], v)

    def get_triggering_awgunit_idx(self, capmod_idx: int) -> Union[int, None]:
        self._validate_module(capmod_idx)
        v = self._read_master_reg(self._MODULE_TRIGGER_REG_MAP[capmod_idx])
        if v == 0:
            return None
        else:
            return int(v - 1)

    def _set_trigger_mask(self) -> None:
        self._write_master_reg(self._CTRL_MASTER_TRIGGER_MASK_REG_ADDR, self._set2bitmap(self._triggerable_capunits))

    def _get_trigger_mask(self) -> set[int]:
        return self._bitmap2set(self._read_master_reg(self._CTRL_MASTER_TRIGGER_MASK_REG_ADDR))

    def get_triggerable_units(self, use_cache: bool = True) -> set[int]:
        if not use_cache:
            with self.lock:
                self._triggerable_capunits.clear()
                self._triggerable_capunits.update(self._get_trigger_mask())

        return set(self._triggerable_capunits)

    def clear_triggerable_units(self) -> None:
        self._triggerable_capunits.clear()
        self._set_trigger_mask()

    def add_triggerable_unit(self, capunit_idx: int) -> None:
        if capunit_idx not in self._triggerable_capunits:
            self._triggerable_capunits.add(capunit_idx)
            self._set_trigger_mask()

    def add_triggerable_units(self, capunit_idxs: Collection[int]) -> None:
        if not self._triggerable_capunits.issuperset(capunit_idxs):
            self._triggerable_capunits.update(capunit_idxs)
            self._set_trigger_mask()

    def remove_triggerable_unit(self, capunit_idx: int) -> None:
        if capunit_idx in self._triggerable_capunits:
            self._triggerable_capunits.remove(capunit_idx)
            self._set_trigger_mask()

    def remove_triggerable_units(self, capunit_idxs: Collection[int]) -> None:
        if not self._triggerable_capunits.isdisjoint(capunit_idxs):
            self._triggerable_capunits.difference_update(capunit_idxs)
            self._set_trigger_mask()


class CapCtrlClassic(CapCtrlSimpleMulti):
    _CTRL_MASTER_TRIGGER_SEL0_REG_ADDR: int = 0x0000_0004
    _CTRL_MASTER_TRIGGER_SEL1_REG_ADDR: int = 0x0000_0008

    _MODULE_TRIGGER_REG_MAP: dict[int, int] = {
        0: _CTRL_MASTER_TRIGGER_SEL0_REG_ADDR,
        1: _CTRL_MASTER_TRIGGER_SEL1_REG_ADDR,
    }


class CapCtrlStandard(CapCtrlSimpleMulti):
    _CTRL_MASTER_TRIGGER_SEL0_REG_ADDR: int = 0x0000_0004
    _CTRL_MASTER_TRIGGER_SEL1_REG_ADDR: int = 0x0000_0008
    _CTRL_MASTER_TRIGGER_SEL2_REG_ADDR: int = 0x0000_002C
    _CTRL_MASTER_TRIGGER_SEL3_REG_ADDR: int = 0x0000_0030

    _MODULE_TRIGGER_REG_MAP: dict[int, int] = {
        0: _CTRL_MASTER_TRIGGER_SEL0_REG_ADDR,
        1: _CTRL_MASTER_TRIGGER_SEL1_REG_ADDR,
        2: _CTRL_MASTER_TRIGGER_SEL2_REG_ADDR,
        3: _CTRL_MASTER_TRIGGER_SEL3_REG_ADDR,
    }

    _CTRL_MASTER_DSP_ENABLE_REG_ADDR: int = 0x0000_0034  # Notes: don't use it, now.

    def __init__(
        self,
        ipaddrs: dict[str, str],
        settings: dict[str, Any],
        udprws: dict[tuple[str, int], BasePacketAccess],
        auxattr: set[E7FwAuxAttr],
    ):
        super().__init__(ipaddrs, settings, udprws, auxattr)


class CapCtrlFeedback(CapCtrlBase):
    # TODO: implment feedback specific APIs

    def __init__(
        self,
        ipaddrs: dict[str, str],
        settings: dict[str, Any],
        udprws: dict[tuple[str, int], BasePacketAccess],
        auxattr: set[E7FwAuxAttr],
    ):
        super().__init__(ipaddrs, settings, udprws, auxattr)
