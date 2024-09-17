from typing import Any, Final, Optional, Union

from e7awghal.clockmasterctrl import ClockmasterCtrl
from e7awghal.clockmasterrebooter import ClockmasterRebooter
from e7awghal.e7awg_packet import BasePacketAccess
from e7awghal.fwtype import E7FwAuxAttr


class AbstractClockmasterHal:
    _SETTINGS: dict[str, dict[str, Any]]

    def __init__(self, name: str, ipaddrs: dict[str, str], auxattr: Optional[set[E7FwAuxAttr]] = None):
        self._name: Final[str] = name
        self._udprws: Final[dict[tuple[str, int], BasePacketAccess]] = {}
        self._ctrl: Final[ClockmasterCtrl] = ClockmasterCtrl(
            ipaddrs, self._SETTINGS["ctrl"], self._udprws, auxattr or set()
        )
        self._rebooter: Final[ClockmasterRebooter] = ClockmasterRebooter(
            ipaddrs, self._SETTINGS["rebooter"], self._udprws, auxattr or set()
        )

    def initialize(self):
        self._ctrl.initialize()
        self._rebooter.initialize()

    @property
    def name(self) -> str:
        return self._name

    @property
    def ctrl(self) -> ClockmasterCtrl:
        return self._ctrl

    @property
    def rebooter(self) -> ClockmasterRebooter:
        return self._rebooter


class ClockmasterAu200Hal(AbstractClockmasterHal):
    _SETTINGS = {
        "ctrl": {
            "nic": "A",
            "port": 16384,
        },
        "rebooter": {
            "nic": "A",
            "port": 16385,
        },
    }

    def __init__(self, *, name: Union[str, None] = None, ipaddr: str):
        if name is None:
            name = ipaddr
        super().__init__(name=name, ipaddrs={"A": ipaddr})
