import logging
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from e7awghal.e7awg_packet import BasePacketAccess, E7awgOutgoingPacket, E7awgPacketAccess, E7awgPacketMode
from e7awghal.fwtype import E7FwAuxAttr

logger = logging.getLogger(__name__)


class HbmCtrl:
    _HBM_ALIGNMENT = 32  # bytes
    _HBM_ACCESS_UNIT = 1440  # bytes
    _SIZE_IQ32 = 4
    _SIZE_IQ64 = 8
    _SIZE_U64 = 8

    __slots__ = (
        "_settings",
        "_udprw",
        "_auxattr",
    )

    def __init__(
        self,
        ipaddrs: dict[str, str],
        settings: dict[str, Any],
        udprws: dict[tuple[str, int], BasePacketAccess],
        auxattr: set[E7FwAuxAttr],
    ):
        self._settings: Final[dict[str, Any]] = settings
        for k in ("nic", "port"):
            if k not in self._settings:
                raise ValueError(f"an essential key '{k}' in the settings is missing")
        nic, port = settings["nic"], settings["port"]
        udprw = udprws.get((nic, port))
        if udprw is None:
            udprw = E7awgPacketAccess(ipaddrs[nic], port)
            udprws[nic, port] = udprw
        else:
            if not isinstance(udprw, E7awgPacketAccess):
                raise RuntimeError(f"inconsistent packet accesses for {nic}:{port}")
        self._udprw: Final[E7awgPacketAccess] = udprw
        self._auxattr: Final[set[E7FwAuxAttr]] = auxattr

    def initialize(self):
        # Notes: nothing to do
        return

    def _validate_address_range(self, addr: int, size: int) -> None:
        if not (
            self._settings["address_top"] <= addr <= self._settings["address_bottom"]
            and self._settings["address_top"] <= addr + size - 1 <= self._settings["address_bottom"]
        ):
            raise ValueError(f"invalid address range: {addr:09x} -- {addr+size: 09x}")

    def _read_simple(self, addr: int, size: int) -> memoryview:
        assert addr % self._HBM_ALIGNMENT == 0
        assert size % self._HBM_ALIGNMENT == 0
        assert size <= self._HBM_ACCESS_UNIT
        cmd = E7awgOutgoingPacket(
            mode=E7awgPacketMode.WAVE_RAM_READ,
            address=addr,
            num_payload_bytes=size,
        )
        return self._udprw.send_command(cmd).payload

    def _write_simple(self, addr: int, size: int, v: memoryview):
        # logger.info(f"addr = {addr:09x}, size = {size:d}, len(v) = {len(v):d}")
        assert addr % self._HBM_ALIGNMENT == 0
        assert size % self._HBM_ALIGNMENT == 0
        assert size <= self._HBM_ACCESS_UNIT
        cmd = E7awgOutgoingPacket(
            mode=E7awgPacketMode.WAVE_RAM_WRITE,
            address=addr,
            num_payload_bytes=size,
        )
        cmd.payload[:] = v
        _ = self._udprw.send_command(cmd)

    def _update_simple(self, addr0: int, size0: int, addr: int, size: int, update: memoryview):
        assert addr0 % self._HBM_ALIGNMENT == 0
        assert size0 % self._HBM_ALIGNMENT == 0
        assert size0 <= self._HBM_ACCESS_UNIT

        cmd = E7awgOutgoingPacket(
            mode=E7awgPacketMode.WAVE_RAM_WRITE,
            address=addr0,
            num_payload_bytes=size0,
        )
        cmd.payload[:] = self._read_simple(addr0, size0)  # TODO: optimize it (?)
        cmd.payload[addr - addr0 : addr - addr0 + size] = update
        _ = self._udprw.send_command(cmd)

    def _read_start_and_end(self, addr: int, size: int, addr0: int, size0: int, retbuf: memoryview) -> None:
        # logger.debug(f"addr={addr:09x}, size={size:d}, addr0={addr0:09x}, size0={size0:d}")
        # logger.debug(f"len(retbuf) = {len(retbuf)}")
        assert size < self._HBM_ACCESS_UNIT and size0 <= self._HBM_ACCESS_UNIT
        retbuf[0:size] = self._read_simple(addr0, size0)[addr - addr0 : addr - addr0 + size]

    def _read_start(self, addr: int, addr0: int, size0: int, retbuf: memoryview) -> tuple[int, int, int]:
        assert self._HBM_ACCESS_UNIT <= size0
        size = self._HBM_ACCESS_UNIT - (addr - addr0)
        retbuf[0:size] = self._read_simple(addr0, self._HBM_ACCESS_UNIT)[addr - addr0 :]
        return addr0 + self._HBM_ACCESS_UNIT, size0 - self._HBM_ACCESS_UNIT, size

    def _read_mid_or_end(self, addr0: int, size0: int, retbuf: memoryview, idx: int, size1: int):
        size = min(size0, self._HBM_ACCESS_UNIT)
        size_w = min(size, size1 - idx)
        retbuf[idx : idx + size_w] = self._read_simple(addr0, size)[:size_w]
        return addr0 + size, size0 - size, idx + size

    def _read_generic_array(self, addr: int, size: int, elem_size: int, elem_typename: str) -> memoryview:
        if addr % elem_size != 0:
            raise ValueError(f"unaligned access of {elem_typename}, addr (= {addr}) is not a multiple of {elem_size}")
        if size <= 0:
            raise ValueError(f"invalid size to read: {size}")
        self._validate_address_range(addr, size * elem_size)

        addr0 = (addr // self._HBM_ALIGNMENT) * self._HBM_ALIGNMENT
        size1 = size * elem_size
        size0 = ((size1 + (addr - addr0) + self._HBM_ALIGNMENT - 1) // self._HBM_ALIGNMENT) * self._HBM_ALIGNMENT

        retbuf = memoryview(bytearray(size * elem_size))
        if size1 < self._HBM_ACCESS_UNIT and size0 <= self._HBM_ACCESS_UNIT:
            self._read_start_and_end(addr, size1, addr0, size0, retbuf)
        else:
            addr0, size0, idx = self._read_start(addr, addr0, size0, retbuf)
            while size0 > 0:
                addr0, size0, idx = self._read_mid_or_end(addr0, size0, retbuf, idx, size1)
        return retbuf

    def read_iq32(self, addr: int, size: int) -> npt.NDArray[np.int16]:
        retbuf = self._read_generic_array(addr, size, self._SIZE_IQ32, "complex32")
        return np.frombuffer(retbuf, dtype=np.int16).reshape((size, 2))

    def read_iq64(self, addr: int, size: int) -> npt.NDArray[np.complex64]:
        retbuf = self._read_generic_array(addr, size, self._SIZE_IQ64, "complex64")
        return np.frombuffer(retbuf, dtype=np.complex64).reshape((size,))

    def read_u64(self, addr: int, size: int) -> npt.NDArray[np.uint64]:
        retbuf = self._read_generic_array(addr, size, self._SIZE_U64, "uint64")
        return np.frombuffer(retbuf, dtype=np.uint64).reshape((size,))

    def _write_start_and_end(self, addr: int, size: int, addr0: int, size0: int, value: memoryview) -> None:
        # logger.info(f"addr = {addr:09x}, size = {size:d}, addr0 = {addr0:09x}, size0 = {size0:d}")
        assert size < self._HBM_ACCESS_UNIT and size0 <= self._HBM_ACCESS_UNIT
        if size != size0:
            self._update_simple(addr0, size0, addr, size, value)
        else:
            self._write_simple(addr0, size0, value)

    def _write_start(self, addr: int, addr0: int, size0: int, value: memoryview) -> tuple[int, int, int]:
        assert self._HBM_ACCESS_UNIT <= size0
        size = self._HBM_ACCESS_UNIT - (addr - addr0)
        if size != self._HBM_ACCESS_UNIT:
            self._update_simple(addr0, self._HBM_ACCESS_UNIT, addr, size, value[:size])
        else:
            self._write_simple(addr0, self._HBM_ACCESS_UNIT, value[:size])
        return addr0 + self._HBM_ACCESS_UNIT, size0 - self._HBM_ACCESS_UNIT, size

    def _write_mid(self, addr0: int, size0: int, value: memoryview, idx: int) -> tuple[int, int, int]:
        assert self._HBM_ACCESS_UNIT <= size0
        self._write_simple(addr0, self._HBM_ACCESS_UNIT, value[idx : idx + self._HBM_ACCESS_UNIT])
        return addr0 + self._HBM_ACCESS_UNIT, size0 - self._HBM_ACCESS_UNIT, idx + self._HBM_ACCESS_UNIT

    def _write_end(self, addr: int, addr0: int, size0: int, value: memoryview, idx: int, size1: int) -> None:
        # logger.info(
        #     f"addr = {addr:09x}, addr0 = {addr0:09x}, size0 = {size0:d}, "
        #     f"len(value) = {len(value):d}, idx = {idx:d}, size1:{size1:d}"
        # )
        assert size0 <= self._HBM_ACCESS_UNIT
        if size1 - idx != size0:
            self._update_simple(addr0, size0, addr + idx, size1 - idx, value[idx:size1])
        else:
            self._write_simple(addr0, size0, value[idx:size1])

    def _write_generic_array(
        self, addr: int, size: int, elem_size: int, memview: memoryview, elem_typename: str
    ) -> None:
        if addr % elem_size != 0:
            raise ValueError(f"unaligned access of {elem_typename}, addr (= {addr}) is not a multiple of {elem_size}")
        if size <= 0:
            raise ValueError(f"invalid size to write: {size}")
        self._validate_address_range(addr, size * elem_size)

        addr0 = (addr // self._HBM_ALIGNMENT) * self._HBM_ALIGNMENT
        size1 = size * elem_size
        size0 = ((size1 + (addr - addr0) + self._HBM_ALIGNMENT - 1) // self._HBM_ALIGNMENT) * self._HBM_ALIGNMENT

        if size1 < self._HBM_ACCESS_UNIT and size0 <= self._HBM_ACCESS_UNIT:
            self._write_start_and_end(addr, size1, addr0, size0, memview)
        else:
            addr0, size0, idx = self._write_start(addr, addr0, size0, memview)
            while size0 > self._HBM_ACCESS_UNIT:
                addr0, size0, idx = self._write_mid(addr0, size0, memview, idx)
            if size1 - idx > 0:
                self._write_end(addr, addr0, size0, memview, idx, size1)

    def write_iq32(self, addr: int, size: int, value: npt.NDArray[np.int16]) -> None:
        if len(value.shape) != 2 or value.shape[1] != 2:
            raise ValueError(f"invalid shape of the vector: {value.shape}")
        if value.shape[0] < size:
            raise ValueError(f"the vector to be written is too short ({value.shape[0]} < {size}")

        memview = memoryview(value.tobytes())  # Notes: COPY!
        self._write_generic_array(addr, size, self._SIZE_IQ32, memview, "complex32")

    def write_iq64(self, addr: int, size: int, value: npt.NDArray[np.complex64]) -> None:
        if len(value.shape) != 1:
            raise ValueError(f"invalid shape of the vector: {value.shape}")
        if value.shape[0] < size:
            raise ValueError(f"the vector to be written is too short ({value.shape[0]} < {size}")

        memview = memoryview(value.tobytes())  # Notes: COPY!
        self._write_generic_array(addr, size, self._SIZE_IQ64, memview, "complex64")
