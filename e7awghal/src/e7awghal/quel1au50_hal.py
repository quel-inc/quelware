import logging
import socket
import time
from abc import ABCMeta
from ipaddress import IPv4Address
from typing import Any, Callable, Final, Optional

from e7awghal.au50rebooter import Au50Rebooter
from e7awghal.awgctrl import AwgCtrl
from e7awghal.awgunit import AwgUnit
from e7awghal.capctrl import CapCtrlBase, CapCtrlClassic, CapCtrlStandard
from e7awghal.capunit import AbstractCapUnit, CapUnit, CapUnitSimplified, CapUnitSwitchable
from e7awghal.clockcounterctrl import ClockcounterCtrl
from e7awghal.e7awg_memoryobj import E7awgAbstractMemoryManager, E7awgPrimitiveMemoryManager
from e7awghal.e7awg_packet import BasePacketAccess
from e7awghal.fwtype import E7FwAuxAttr, E7FwType
from e7awghal.hbmctrl import HbmCtrl
from e7awghal.simplemulti import SimplemultiSequencer
from e7awghal.syncdata import _SyncInterface
from e7awghal.versionchecker import Quel1Au50HalVersionChecker

logger = logging.getLogger(__name__)

_settings_hbm_common: dict[str, Any] = {
    "nic": "A",
    "port": 16384,
    "address_top": 0x0_0000_0000,
    "address_bottom": 0x1_FFFF_FFFF,
}

_settings_awg_common: dict[str, Any] = {
    "nic": "A",
    "port": 16385,
    "ctrl_master_base": 0x0000_0000,
    "ctrl_cls": AwgCtrl,
    "units": [
        {
            "aucls": AwgUnit,
            "reg": {
                "ctrl_base": 0x0000_0080 + u * 0x0000_0080,
                "param_base": 0x0000_1000 + u * 0x0000_0400,
                "chunk_bases": [0x0000_0040 + c * 0x0000_0010 for c in range(16)],
            },
            "mm": {
                "name": f"awgmm-#{u:02d}",
                "address_top": 0x0_0000_0000 + 0x0_2000_0000 * u,
                "size": 0x0_1000_0000,
            },
        }
        for u in range(16)
    ],
}

_settings_cap_classic: dict[str, Any] = {
    "nic": "A",
    "port": 16385,
    "ctrl_master_base": 0x0000_0000,
    "ctrl_cls": CapCtrlClassic,
    "unit_in_module": [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ],
    "units": [
        {
            "cucls": CapUnit,
            "reg": {
                "ctrl_base": 0x0000_0100 + u * 0x0000_0100,
                "param_base": 0x0001_0000 + u * 0x0001_0000,
            },
            "mm": {
                "name": f"capmm-#{u:02d}",
                "address_top": 0x0_1000_0000 + 0x0_2000_0000 * u,
                "size": 0x0_1000_0000,
            },
        }
        for u in range(8)
    ],
}

_settings_cap_standard: dict[str, Any] = {
    "nic": "A",
    "port": 16385,
    "ctrl_master_base": 0x0000_0000,
    "ctrl_cls": CapCtrlStandard,
    "units_in_module": [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8],
        [9],
    ],
    "units": [
        {
            "cucls": CapUnitSwitchable if u < 8 else CapUnitSimplified,
            "reg": {
                "ctrl_base": 0x0000_0100 + u * 0x0000_0100,
                "param_base": 0x0001_0000 + u * 0x0001_0000,
            },
            "mm": {
                "name": f"capmm-#{u:02d}",
                "address_top": (0x0_1000_0000 + 0x0_2000_0000 * u) if u < 8 else (0x0_5000_0000 + 0x0_2000_0000 * u),
                "size": 0x0_1000_0000,
            },
        }
        for u in range(10)
    ],
}

_settings_seq_simplemulti: dict[str, Any] = {
    "nic": "B",
    "port": 16384,
    "ctrl_cls": SimplemultiSequencer,
}

_settings_au50rebooter: dict[str, Any] = {
    "nic": "B",
    "port": 16384,
}


_settings_clock_standard: dict[str, Any] = {
    "nic": "B",
    "port": 16385,
}


class AbstractQuel1Au50Hal(metaclass=ABCMeta):
    __slots__ = (
        "_name",
        "_udprws",
        "_auxattr",
        "_awgctrl",
        "_capctrl",
        "_hbmctrl",
        "_sqrctrl",
        "_syncintf",
        "_clkcntr",
        "_mms",
        "_awgunits",
        "_capunits",
        "_au50rebooter",
        "_auth_callback",
    )
    _SETTINGS: dict[str, dict[str, Any]]
    _FW_TYPE: E7FwType

    @classmethod
    def fw_type(self) -> E7FwType:
        if hasattr(self, "_FW_TYPE"):
            return self._FW_TYPE
        else:
            raise TypeError("cannot resolve firmware type for abstract class")

    def __init__(
        self,
        name: str,
        ipaddrs: dict[str, str],
        mmcls: type,
        auxattr: set[E7FwAuxAttr],
        auth_callback: Optional[Callable[[], bool]] = None,
    ):
        self._name: Final[str] = name
        self._udprws: Final[dict[tuple[str, int], BasePacketAccess]] = {}
        self._auxattr: Final[set[E7FwAuxAttr]] = auxattr
        self._auth_callback: Optional[Callable[[], bool]] = auth_callback

        awgctrl_cls = self._SETTINGS["awg"]["ctrl_cls"]
        if not issubclass(awgctrl_cls, AwgCtrl):
            raise ValueError(f"invalid class of AwgCtrl: {awgctrl_cls}")
        self._awgctrl: Final[AwgCtrl] = awgctrl_cls(ipaddrs, self._SETTINGS["awg"], self._udprws, self._auxattr)

        capctrl_cls = self._SETTINGS["cap"]["ctrl_cls"]
        if not issubclass(capctrl_cls, CapCtrlBase):
            raise ValueError(f"invalid class of CapCtrl: {capctrl_cls}")
        self._capctrl: Final[CapCtrlBase] = capctrl_cls(ipaddrs, self._SETTINGS["cap"], self._udprws, self._auxattr)

        self._hbmctrl = HbmCtrl(ipaddrs, self._SETTINGS["hbm"], self._udprws, self._auxattr)

        sqrctrl_cls = self._SETTINGS["sqr"]["ctrl_cls"]
        # TODO: allow the other classes to support Feedback, probably.
        if not issubclass(sqrctrl_cls, SimplemultiSequencer):
            raise TypeError(f"invalid type of sequencer: {sqrctrl_cls}")
        self._sqrctrl: Final[SimplemultiSequencer] = sqrctrl_cls(
            ipaddrs, self._SETTINGS["sqr"], self._udprws, self._awgctrl, self._auxattr
        )
        self._au50rebooter: Final[Au50Rebooter] = Au50Rebooter(
            ipaddrs, self._SETTINGS["rbt"], self._udprws, self._auxattr
        )

        self._clkcntr: Final[ClockcounterCtrl] = ClockcounterCtrl(
            ipaddrs, self._SETTINGS["clk"], self._udprws, self._auxattr
        )
        self._syncintf: Final[_SyncInterface] = _SyncInterface.create(ipaddrs, self._SETTINGS["clk"])

        # Notes: injecting auth_callback to all created udprws!
        for udprw in self._udprws.values():
            logger.debug(f"udprw {udprw._dest_addrport[0]}:{udprw._dest_addrport[1]} is created")
            udprw.inject_auth_callback(self._auth_callback)

        if not issubclass(mmcls, E7awgAbstractMemoryManager):
            raise TypeError(f"invalid memory manager class: {mmcls}")

        aus: list[AwgUnit] = []
        for i, unit_settings in enumerate(self._SETTINGS["awg"]["units"]):
            aucls = unit_settings["aucls"]
            if issubclass(aucls, AwgUnit):
                aus.append(
                    aucls(
                        unit_idx=i,
                        awgctrl=self._awgctrl,
                        hbmctrl=self._hbmctrl,
                        mm=mmcls(**unit_settings["mm"]),
                        settings=unit_settings["reg"],
                    )
                )
            else:
                ValueError(f"invalid class (= {unit_settings['aucls']}) for AwgUnit")
        self._awgunits: Final[tuple[AwgUnit, ...]] = tuple(aus)

        cus: list[AbstractCapUnit] = []
        for i, unit_settings in enumerate(self._SETTINGS["cap"]["units"]):
            cucls = unit_settings["cucls"]
            if issubclass(cucls, AbstractCapUnit):
                cus.append(
                    cucls(
                        unit_idx=i,
                        capctrl=self._capctrl,
                        hbmctrl=self._hbmctrl,
                        mm=mmcls(**unit_settings["mm"]),
                        settings=unit_settings["reg"],
                    )
                )
            else:
                ValueError(f"invalid class (= {unit_settings['cucls']}) for CapUnit")
        self._capunits: Final[tuple[AbstractCapUnit, ...]] = tuple(cus)

    def __repr__(self) -> str:
        return f"<{str(self.__class__.__name__)} -- {self._name}>"

    @property
    def name(self) -> str:
        return self._name

    @property
    def fwversion(self) -> str:
        return self._awgctrl.version

    @property
    def auxattr(self) -> set[E7FwAuxAttr]:
        return set(self._auxattr)

    @property
    def awgctrl(self) -> AwgCtrl:
        return self._awgctrl

    @property
    def capctrl(self) -> CapCtrlBase:
        return self._capctrl

    @property
    def hbmctrl(self) -> HbmCtrl:
        return self._hbmctrl

    @property
    def clkcntr(self) -> ClockcounterCtrl:
        return self._clkcntr

    # TODO: reconsider APi for a firmware with feedback sequencer
    @property
    def sqrctrl(self) -> SimplemultiSequencer:
        if hasattr(self, "_sqrctrl"):
            return self._sqrctrl
        else:
            raise ValueError("no sqrctrl is available")

    @property
    def au50rebooter(self) -> Au50Rebooter:
        return self._au50rebooter

    @property
    def syncintf(self) -> _SyncInterface:
        return self._syncintf

    def awgunit(self, unit_idx: int):
        if unit_idx < len(self._awgunits):
            return self._awgunits[unit_idx]
        else:
            raise ValueError(f"invalid index of AWG unit: {unit_idx}")

    def capunit(self, unit_idx: int):
        if unit_idx < len(self._capunits):
            return self._capunits[unit_idx]
        else:
            raise ValueError(f"invalid index of AWG unit: {unit_idx}")

    def initialize(self):
        self._hbmctrl.initialize()
        self._clkcntr.initialize()
        self._sqrctrl.initialize()
        self._awgctrl.initialize()
        for au in self._awgunits:
            au.initialize()
        self._capctrl.initialize()
        for cu in self._capunits:
            cu.initialize()
        self._au50rebooter.initialize()


# Notes: not used but kept as an example. it'll be removed after another Hal class is available.
class Quel1Au50SimplemultiClassicHal(AbstractQuel1Au50Hal):
    _SETTINGS: dict[str, dict[str, Any]] = {
        "awg": _settings_awg_common,
        "cap": _settings_cap_classic,  # Notes: <---
        "hbm": _settings_hbm_common,
        "clk": _settings_clock_standard,
        "sqr": _settings_seq_simplemulti,
        "rbt": _settings_au50rebooter,
    }

    _FW_TYPE = E7FwType.SIMPLEMULTI_CLASSIC

    def __init__(
        self,
        *,
        name: str,
        ipaddrs: dict[str, str],
        mmcls: type = E7awgPrimitiveMemoryManager,
        auxattr: Optional[set[E7FwAuxAttr]] = None,
        auth_callback: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(name=name, ipaddrs=ipaddrs, mmcls=mmcls, auxattr=auxattr or set(), auth_callback=auth_callback)


class Quel1Au50SimplemultiStandardHal(AbstractQuel1Au50Hal):
    _SETTINGS: dict[str, dict[str, Any]] = {
        "awg": _settings_awg_common,
        "cap": _settings_cap_standard,  # Notes: <---
        "hbm": _settings_hbm_common,
        "clk": _settings_clock_standard,
        "sqr": _settings_seq_simplemulti,
        "rbt": _settings_au50rebooter,
    }

    _FW_TYPE = E7FwType.SIMPLEMULTI_STANDARD

    def __init__(
        self,
        *,
        name: str,
        ipaddrs: dict[str, str],
        mmcls: type = E7awgPrimitiveMemoryManager,
        auxattr: Optional[set[E7FwAuxAttr]] = None,
        auth_callback: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(name=name, ipaddrs=ipaddrs, mmcls=mmcls, auxattr=auxattr or set(), auth_callback=auth_callback)


def create_quel1au50hal(
    *,
    name: Optional[str] = None,
    ipaddr_wss: str,
    ipaddr_sss: Optional[str] = None,
    mmcls: type = E7awgPrimitiveMemoryManager,
    auth_callback: Optional[Callable[[], bool]] = None,
) -> AbstractQuel1Au50Hal:
    vc = Quel1Au50HalVersionChecker(ipaddr_wss, 16385)
    # Notes: liveness check of e7awghal endpoint was done here with using ping.
    vc._udprw.inject_auth_callback(auth_callback=auth_callback)

    for i in range(3):
        try:
            if i > 0:
                time.sleep(2.5)
            fw_type, fw_auxattr, _ = vc.resolve_fwtype()
            break
        except socket.timeout:
            pass
    else:
        raise RuntimeError(f"failed to acquire firmware information from {ipaddr_wss}")

    for cls in AbstractQuel1Au50Hal.__subclasses__():
        assert issubclass(cls, AbstractQuel1Au50Hal)
        if cls.fw_type() == fw_type:
            if name is None:
                name = ipaddr_wss
            if ipaddr_sss is None:
                ipaddr_sss = str(IPv4Address(ipaddr_wss) + 0x00010000)
            proxy: AbstractQuel1Au50Hal = cls(
                name=name,
                ipaddrs={"A": ipaddr_wss, "B": ipaddr_sss},
                mmcls=mmcls,
                auxattr=fw_auxattr,
                auth_callback=auth_callback,
            )
            break
    else:
        raise AssertionError(f"internal error, unexpected firmware type: {fw_type}")

    return proxy
