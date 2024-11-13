import logging
from typing import Any, Callable, Final, Optional, Union

from e7awghal.clockmasterctrl import ClockmasterCtrl
from e7awghal.clockmasterrebooter import ClockmasterRebooter
from e7awghal.e7awg_packet import BasePacketAccess
from e7awghal.fwtype import E7FwAuxAttr

logger = logging.getLogger(__name__)


class AbstractClockmasterHal:
    _SETTINGS: dict[str, dict[str, Any]]

    def __init__(
        self,
        name: str,
        ipaddrs: dict[str, str],
        auxattr: Optional[set[E7FwAuxAttr]] = None,
        auth_callback: Optional[Callable[[], bool]] = None,
    ):
        self._name: Final[str] = name
        self._udprws: Final[dict[tuple[str, int], BasePacketAccess]] = {}
        self._auth_callback: Union[Callable[[], bool], None] = auth_callback
        self._ctrl: Final[ClockmasterCtrl] = ClockmasterCtrl(
            ipaddrs, self._SETTINGS["ctrl"], self._udprws, auxattr or set()
        )
        self._rebooter: Final[ClockmasterRebooter] = ClockmasterRebooter(
            ipaddrs, self._SETTINGS["rebooter"], self._udprws, auxattr or set()
        )
        # Notes: injecting auth_callback to all created udprws!
        for udprw in self._udprws.values():
            logger.debug(f"udprw {udprw._dest_addrport[0]}:{udprw._dest_addrport[1]} is created")
            udprw.inject_auth_callback(self._auth_callback)

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

    def __init__(
        self, *, name: Union[str, None] = None, ipaddr: str, auth_callback: Optional[Callable[[], bool]] = None
    ):
        if name is None:
            name = ipaddr
        super().__init__(name=name, ipaddrs={"A": ipaddr}, auth_callback=auth_callback)
