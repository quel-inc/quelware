import logging
import time
from collections.abc import Collection
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, Final, Optional

import numpy as np
import numpy.typing as npt
from pydantic import Field

from e7awghal.abstract_register import AbstractFpgaReg, b_1bf_bool, p_1bf_bool
from e7awghal.common_defs import _DEFAULT_POLLING_PERIOD, _DEFAULT_TIMEOUT, E7awgHardwareError
from e7awghal.common_register import E7awgVersion
from e7awghal.e7awg_packet import BasePacketAccess, E7awgOutgoingPacket, E7awgPacketAccess, E7awgPacketMode
from e7awghal.fwtype import E7FwAuxAttr

logger = logging.getLogger(__name__)


class AwgMasterCtrlReg(AbstractFpgaReg):
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
    def parse(cls, v: np.uint32) -> "AwgMasterCtrlReg":
        r = cls()
        r._parse(v)
        return r


class AwgCtrl:
    __slots__ = (
        "_udprw",
        "_auxattr",
        "_settings",
        "_number_of_unit",
        "_unit_mask",
        "_pool",
        "_lock",
        "_mask",
    )

    # relative to CTRL_MASTER_BASE
    _MASTER_VERSION_REG_ADDR: int = 0x0000_0000
    _MASTER_TGTSEL_REG_ADDR: int = 0x0000_0004
    _MASTER_CTRL_REG_ADDR: int = 0x0000_0008
    _MASTER_STATUS_WAKEUP_REG_ADDR: int = 0x0000_000C
    _MASTER_STATUS_BUSY_REG_ADDR: int = 0x0000_0010
    _MASTER_STATUS_READY_REG_ADDR: int = 0x0000_0014
    _MASTER_STATUS_DONE_REG_ADDR: int = 0x0000_0018
    _MASTER_ERROR_XFER_FAILURE_REG_ADDR: int = 0x0000_001C
    _MASTER_ERROR_SAMPLE_SHORTAGE_REG_ADDR: int = 0x0000_0020

    def __init__(
        self,
        ipaddrs: dict[str, str],
        settings: dict[str, Any],
        udprws: dict[tuple[str, int], BasePacketAccess],
        auxattr: Collection[E7FwAuxAttr],
    ):
        self._settings: Final[dict[str, Any]] = settings
        for k in ("nic", "port", "units", "ctrl_master_base"):
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
        self._udprw: E7awgPacketAccess = udprw
        self._auxattr: set[E7FwAuxAttr] = set(auxattr)
        self._unit_mask: np.uint32 = np.uint32(0)
        self._number_of_unit: Final[int] = len(settings["units"])
        self._pool = ThreadPoolExecutor()
        self._lock = RLock()

    @property
    def lock(self) -> RLock:
        return self._lock

    @staticmethod
    def set2bitmap(units: Collection[int]) -> np.uint32:
        mask = 0
        for u in units:
            mask |= 1 << u
        return np.uint32(mask)

    @staticmethod
    def bitmap2set(unit_mask: np.uint32) -> set[int]:
        r: set[int] = set()
        u: int = int(unit_mask)
        for i in range(32):
            if u & 0x01 == 0x01:
                r.add(i)
            u >>= 1
            if u == 0:
                break
        return r

    def _validate_unit(self, unit_idx: int):
        if not 0 <= unit_idx < self._number_of_unit:
            raise ValueError(f"invalid awg-unit: {unit_idx}")

    def _validate_units(self, unit_idxs: Collection[int]):
        for u in unit_idxs:
            self._validate_unit(u)

    def read_reg(self, address: int) -> np.uint32:
        cmd = E7awgOutgoingPacket(
            mode=E7awgPacketMode.AWG_REG_READ,
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
                mode=E7awgPacketMode.AWG_REG_READ,
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
            mode=E7awgPacketMode.AWG_REG_WRITE,
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
                mode=E7awgPacketMode.AWG_REG_WRITE, address=address + idx * 4, num_payload_bytes=s * 4
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

    def get_master_mask(self, use_cache: bool = True) -> set[int]:
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
        return self.bitmap2set(self._unit_mask)

    def _set_master_mask(self, awg_units: Collection[int], use_cache: bool = True):
        # Notes: awg_units should be validated by the caller.
        # Notes: lock should be taken by the caller.
        with self._lock:
            unit_mask = self.set2bitmap(awg_units)
            if not use_cache or unit_mask != self._unit_mask:
                self._unit_mask = unit_mask
                self._write_master_reg(self._MASTER_TGTSEL_REG_ADDR, self._unit_mask)

    def _set_master_ctrl(self, ctrl: AwgMasterCtrlReg) -> None:
        self._write_master_reg(self._MASTER_CTRL_REG_ADDR, ctrl.build())

    @property
    def version(self) -> str:
        v = self._read_master_reg(self._MASTER_VERSION_REG_ADDR)
        w = E7awgVersion.parse(v)
        return f"{chr(w.ver_char)}:20{w.ver_year:02d}/{w.ver_month:02d}/{w.ver_day:02d}-{w.ver_id:d}"

    @property
    def num_unit(self) -> int:
        return self._number_of_unit

    @property
    def units(self) -> tuple[int, ...]:
        return tuple(range(self._number_of_unit))

    def initialize(self):
        with self._lock:
            self._set_master_mask(set())

    def prepare(self, awg_units: Collection[int]) -> None:
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            self._set_master_ctrl(AwgMasterCtrlReg())
            self._set_master_ctrl(AwgMasterCtrlReg(prepare=True))
            self._set_master_ctrl(AwgMasterCtrlReg())  # Notes: this prevents RTL bad behavior.

    def start_now(self, awg_units: Collection[int]):
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            self._set_master_ctrl(AwgMasterCtrlReg())
            self._set_master_ctrl(AwgMasterCtrlReg(start=True))
            self._set_master_ctrl(AwgMasterCtrlReg())  # Notes: this prevents RTL bad behavior.

    def clear_done(self, awg_units: Collection[int]) -> None:
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            self._set_master_ctrl(AwgMasterCtrlReg())
            self._set_master_ctrl(AwgMasterCtrlReg(done_clr=True))
            self._set_master_ctrl(AwgMasterCtrlReg())  # Notes: this prevents RTL bad behavior.

    def _terminate(self, awg_units: Collection[int]) -> None:
        # XXX: this doesn't work well, may be some hardware bug??
        with self._lock:
            self._set_master_mask(awg_units)
            self._set_master_ctrl(AwgMasterCtrlReg())
            self._set_master_ctrl(AwgMasterCtrlReg(terminate=True))
            self._set_master_ctrl(AwgMasterCtrlReg())  # Notes: this prevents RTL bad behavior.

    def _wait_free(
        self, awg_units: Collection[int], timeout: float, polling_period: float, timeout_msg: str
    ) -> Future[None]:
        def _wait_free_unit_loop() -> None:
            # Notes: the timing of check_error() is considered carefully with respect to the efficiency (less register
            #        access is better) and the priority (more important for the users than TimeoutError).
            t1 = time.perf_counter() + timeout
            while time.perf_counter() < t1:
                time.sleep(polling_period)
                with self._lock:
                    if not self.are_busy_any(awg_units):
                        if self.are_done_any(awg_units):
                            self.clear_done(awg_units)
                        self.check_error(awg_units)
                        break
            else:
                self.check_error(awg_units)
                raise TimeoutError(timeout_msg)

        return self._pool.submit(_wait_free_unit_loop)

    def terminate(
        self, awg_units: Collection[int], timeout: Optional[float] = None, polling_period: Optional[float] = None
    ) -> Future[None]:
        self._validate_units(awg_units)
        timeout_ = timeout or _DEFAULT_TIMEOUT
        polling_period_ = polling_period or _DEFAULT_POLLING_PERIOD
        with self._lock:
            self._terminate(awg_units)
            return self._wait_free(awg_units, timeout_, polling_period_, "failed to terminate")

    def are_awake_any(self, awg_units: Collection[int]) -> bool:
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            v = self._read_master_reg(self._MASTER_STATUS_WAKEUP_REG_ADDR)
            return v != 0x00

    def are_busy_any(self, awg_units: Collection[int]) -> bool:
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            v = self._read_master_reg(self._MASTER_STATUS_BUSY_REG_ADDR)
            return v != 0x00

    def are_ready_any(self, awg_units: Collection[int]) -> bool:
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            v = self._read_master_reg(self._MASTER_STATUS_READY_REG_ADDR)
            return v != 0x00

    def are_done_any(self, awg_units: Collection[int]) -> bool:
        with self._lock:
            self._set_master_mask(awg_units)
            v = self._read_master_reg(self._MASTER_STATUS_DONE_REG_ADDR)
            return v != 0x00

    def are_awake_all(self, awg_units: Collection[int]) -> bool:
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            v = self._read_master_reg(self._MASTER_STATUS_WAKEUP_REG_ADDR)
            return v == self._unit_mask

    def are_busy_all(self, awg_units: Collection[int]) -> bool:
        with self._lock:
            self._set_master_mask(awg_units)
            v = self._read_master_reg(self._MASTER_STATUS_BUSY_REG_ADDR)
            return v == self._unit_mask

    def are_ready_all(self, awg_units: Collection[int]) -> bool:
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            v = self._read_master_reg(self._MASTER_STATUS_READY_REG_ADDR)
            return v == self._unit_mask

    def are_done_all(self, awg_units: Collection[int]) -> bool:
        with self._lock:
            self._set_master_mask(awg_units)
            v = self._read_master_reg(self._MASTER_STATUS_DONE_REG_ADDR)
            return v == self._unit_mask

    def have_started_all(self, awg_units: Collection[int]) -> bool:
        self._validate_units(awg_units)
        with self._lock:
            self._set_master_mask(awg_units)
            v0 = self._read_master_reg(self._MASTER_STATUS_BUSY_REG_ADDR)
            v1 = self._read_master_reg(self._MASTER_STATUS_DONE_REG_ADDR)
            return (v0 | v1) == self._unit_mask

    def check_error(self, awg_units: Collection[int]):
        with self._lock:
            self._set_master_mask(awg_units)
            shortage = self._read_master_reg(self._MASTER_ERROR_SAMPLE_SHORTAGE_REG_ADDR)
            xfer = self._read_master_reg(self._MASTER_ERROR_XFER_FAILURE_REG_ADDR)
            if shortage != 0:
                idxs = ", ".join([f"#{u:02d}" for u in sorted(self.bitmap2set(shortage))])
                raise E7awgHardwareError(f"sample shortage error is detected at awg_units-{idxs:s}")
            if xfer != 0:
                idxs = ", ".join([f"#{u:02d}" for u in sorted(self.bitmap2set(xfer))])
                raise E7awgHardwareError(f"transfer error is detected at awg_units-{idxs:s}")
