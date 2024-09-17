import logging
from threading import RLock
from typing import Any, Final, Union

import numpy as np

from e7awghal.e7awg_packet import (
    BasePacketAccess,
    E7awgSimple32OutgoingPacket,
    E7awgSimple32PacketAccess,
    E7awgSimple32PacketMode,
)
from e7awghal.fwtype import E7FwAuxAttr

logger = logging.getLogger(__name__)


class ClockcounterCtrl:
    CLOCK_FREQUENCY: Final[float] = 125e6  # [Hz]

    __slots__ = (
        "_settings",
        "_udprw",
        "_auxattr",
        "_sysref_latch",
        "_lock",
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
            udprw = E7awgSimple32PacketAccess(ipaddrs[nic], port)
            udprws[nic, port] = udprw
        else:
            if not isinstance(udprw, E7awgSimple32PacketAccess):
                raise RuntimeError(f"inconsistent network interface configuration for {nic}:{port}")
        self._udprw: Final[E7awgSimple32PacketAccess] = udprw
        self._auxattr: Final[set[E7FwAuxAttr]] = auxattr
        self._sysref_latch: Final[bool] = E7FwAuxAttr.NO_SYSREF_LATCH not in auxattr
        self._lock = RLock()

    def initialize(self):
        # Notes: nothing to do
        return

    def read_counter(self) -> tuple[int, Union[int, None]]:
        mode = E7awgSimple32PacketMode.CLK_CNTR_READ
        cmd = E7awgSimple32OutgoingPacket(mode=mode)
        rpl = self._udprw.send_command(cmd, mode.is_network_endian)
        val = np.frombuffer(rpl.payload, dtype=np.dtype(">u8"))
        if self._sysref_latch:
            logger.debug(f"{self.__class__.__name__}:read_counter --> {val[0]:08x}, {val[1]:08x}")
            return val[0], val[1]
        else:
            logger.debug(f"{self.__class__.__name__}:read_counter --> {val[0]:08x}")
            return val[0], None
