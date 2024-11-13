import logging
import struct
from threading import RLock
from typing import Any, Final

from e7awghal.e7awg_packet import (
    BasePacketAccess,
    E7awgSimple64OutgoingPacket,
    E7awgSimple64PacketAccess,
    E7awgSimple64PacketMode,
)
from e7awghal.fwtype import E7FwAuxAttr
from e7awghal.syncdata import _SyncInterface

logger = logging.getLogger(__name__)


class ClockmasterCtrl:
    __slots__ = (
        "_settings",
        "_udprw",
        "_auxattr",
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
            udprw = E7awgSimple64PacketAccess(ipaddrs[nic], port)
            udprws[nic, port] = udprw
        else:
            if not isinstance(udprw, E7awgSimple64PacketAccess):
                raise RuntimeError(f"inconsistent network interface configuration for {nic}:{port}")
        self._udprw: Final[E7awgSimple64PacketAccess] = udprw
        self._auxattr: Final[set[E7FwAuxAttr]] = auxattr
        self._lock = RLock()

    def initialize(self):
        # Notes: nothing to do
        return

    def read_counter(self) -> int:
        mode = E7awgSimple64PacketMode.MCLK_CNTR_READ
        cmd = E7awgSimple64OutgoingPacket(mode=mode, num_payload_bytes=0)
        # Notes: null payload is fine.
        rpl = self._udprw.send_command(cmd)
        return struct.unpack("<Q", rpl.payload)[0]

    def kick_sync(self, boxes: set[_SyncInterface]) -> None:
        mode = E7awgSimple64PacketMode.MCLK_SYNC_KICK
        cmd = E7awgSimple64OutgoingPacket(mode=mode, num_payload_bytes=len(boxes) * 8)
        for i, box in enumerate(boxes):
            cmd.payload[i * 8 : (i + 1) * 8] = struct.pack(">LL", int(box.ipaddr), box.port)
        self._udprw.send_command(cmd)
