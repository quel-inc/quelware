import logging
from threading import RLock
from typing import Any, Final

from e7awghal.e7awg_packet import BasePacketAccess, E7awgOutgoingPacket, E7awgPacketAccess, E7awgPacketMode
from e7awghal.fwtype import E7FwAuxAttr

logger = logging.getLogger(__name__)


class Au50Rebooter:
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
        self._lock = RLock()

    def initialize(self):
        # Notes: nothing to do.
        return

    def reset(self):
        # Notes: nothing to do.
        return

    def reboot(self) -> None:
        pkt = E7awgOutgoingPacket(
            mode=E7awgPacketMode.REBOOTER_FPGA_REBOOT,
            address=0,
            num_payload_bytes=0,
        )
        _ = self._udprw.send_command(pkt)
