import copy
from ipaddress import IPv4Address
from typing import Any, Callable, Optional

from e7awghal.e7awg_memoryobj import E7awgPrimitiveMemoryManager
from e7awghal.fwtype import E7FwAuxAttr, E7FwType
from e7awghal.quel1au50_hal import (
    AbstractQuel1Au50Hal,
    _settings_au50rebooter,
    _settings_awg_common,
    _settings_cap_standard,
    _settings_clock_standard,
    _settings_hbm_common,
    _settings_seq_simplemulti,
)
from e7awghal.versionchecker import Quel1Au50HalVersionChecker
from testlibs.awgctrl_with_hlapi import AwgCtrlHL, AwgUnitHL
from testlibs.capunit_with_hlapi import CapUnitHL, CapUnitSimplifiedHL

_settings_awg_common_for_test: dict[str, Any] = copy.deepcopy(_settings_awg_common)
_settings_awg_common_for_test["ctrl_cls"] = AwgCtrlHL
for i, settings in enumerate(_settings_awg_common_for_test["units"]):
    settings["aucls"] = AwgUnitHL

_settings_cap_standard_for_test: dict[str, Any] = copy.deepcopy(_settings_cap_standard)
for i, settings in enumerate(_settings_cap_standard_for_test["units"]):
    settings["cucls"] = CapUnitHL if i < 8 else CapUnitSimplifiedHL


class Quel1Au50SimplemultiStandardHalForTest(AbstractQuel1Au50Hal):
    _SETTINGS: dict[str, dict[str, Any]] = {
        "awg": _settings_awg_common_for_test,
        "cap": _settings_cap_standard_for_test,
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


def create_quel1au50hal_for_test(
    *,
    name: Optional[str] = None,
    ipaddr_wss: str,
    ipaddr_sss: Optional[str] = None,
    mmcls: type = E7awgPrimitiveMemoryManager,
    auth_callback: Optional[Callable[[], bool]] = None,
) -> AbstractQuel1Au50Hal:
    vc = Quel1Au50HalVersionChecker(ipaddr_wss, 16385)
    if not vc.ping():
        raise RuntimeError(f"failed to communicate with {ipaddr_wss}")
    vc._udprw.inject_auth_callback(auth_callback=auth_callback)
    fw_type, fw_auxattr, _ = vc.resolve_fwtype()
    for cls in (Quel1Au50SimplemultiStandardHalForTest,):
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
