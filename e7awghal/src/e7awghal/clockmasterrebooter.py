import logging
from threading import RLock
from typing import Any, Final

from e7awghal.e7awg_packet import (
    BasePacketAccess,
    E7awgSimple64OutgoingPacket,
    E7awgSimple64PacketAccess,
    E7awgSimple64PacketMode,
)
from e7awghal.fwtype import E7FwAuxAttr

logger = logging.getLogger(__name__)


class ClockmasterRebooter:
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
        # Notes: nothing to do.
        return

    def reboot(self) -> None:
        mode = E7awgSimple64PacketMode.CLOCKMASTER_REBOOT
        cmd = E7awgSimple64OutgoingPacket(mode=mode, num_payload_bytes=8)
        nance: bytes = b"abcdefgh"  # any word of any length is OK.
        cmd.payload[0:8] = nance
        rpl = self._udprw.send_command(cmd)
        if rpl.payload != nance:
            raise RuntimeError("failed reboot of clockmaster")
