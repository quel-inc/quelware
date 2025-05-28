import logging
from typing import Any, Callable, Collection, Dict, Mapping, Set, Tuple, Union

from packaging.version import Version

from quel_ic_config.exstickge_coap_client import Quel1seBoard, _ExstickgeCoapClientBase
from quel_ic_config.exstickge_coap_tempctrl_client import _ExstickgeCoapClientQuel1seTempctrlBase
from quel_ic_config.exstickge_proxy import LsiKindId
from quel_ic_config.quel1_config_subsystem_tempctrl import Quel1seConfigSubsystemTempctrlDebugMixin
from quel_ic_config.quel1_thermistor import Quel1seOnboardThermistor, Quel1Thermistor
from quel_ic_config.quel1se_config_subsystem import _Quel1seConfigSubsystemBase
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger(__name__)


class ExstickgeCoapClientAdda(_ExstickgeCoapClientQuel1seTempctrlBase):
    _VALID_BOXTYPE: Set[str] = {"quel1se-riken8", "quel1se-fujitsu11-a", "quel1se-fujitsu11-b"}

    _VERSION_SPEC: Tuple[Version, Version, Set[Version]] = Version("0.0.0"), Version("999.0.0"), set()

    _URI_MAPPINGS: Mapping[Tuple[LsiKindId, int], str] = {
        (LsiKindId.AD9082, 0): "adda/mxfe_0",
        (LsiKindId.AD9082, 1): "adda/mxfe_1",
        (LsiKindId.LMX2594, 0): "adda/pll_0",
        (LsiKindId.LMX2594, 1): "adda/pll_1",
        (LsiKindId.AD7490, 0): "tmp/ad_tc0",
    }

    _AVAILABLE_BOARDS: Tuple[Quel1seBoard, ...] = ()

    _TEMPCTRL_AD7490_NAME: Tuple[str, ...] = ("adda",)

    # Notes: no read is available for AD5328
    _READ_REG_PATHS: Mapping[LsiKindId, Callable[[int], str]] = {
        LsiKindId.AD9082: lambda addr: f"/reg/{addr:04x}",
        LsiKindId.LMX2594: lambda addr: f"/reg/{addr:04x}",
        LsiKindId.AD7490: lambda addr: "/ctrl",
    }

    _WRITE_REG_PATHS_AND_PAYLOADS: Mapping[LsiKindId, Callable[[int, int], Tuple[str, str]]] = {
        LsiKindId.AD9082: lambda addr, value: (f"/reg/{addr:04x}", f"{value:02x}"),
        LsiKindId.LMX2594: lambda addr, value: (f"/reg/{addr:04x}", f"{value:04x}"),
        LsiKindId.AD7490: lambda addr, value: ("/ctrl", f"{value:04x}"),
    }

    def __init__(
        self,
        target_addr: str,
        target_port: int = _ExstickgeCoapClientQuel1seTempctrlBase.DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientQuel1seTempctrlBase.DEFAULT_RESPONSE_TIMEOUT,
    ):
        super().__init__(target_addr, target_port, timeout)


class Quel1seAddaConfigSubsystem(_Quel1seConfigSubsystemBase, Quel1seConfigSubsystemTempctrlDebugMixin):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE = "quel-1se-adda.json"
    _NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 2,
        "ad7490": 1,
        "powerboard_pwm": 0,
    }

    _PROXY_CLASSES: Tuple[type, ...] = (ExstickgeCoapClientAdda,)

    _GROUPS: Set[int] = {0, 1}

    _DAC_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (0, 2),
        (0, 3): (0, 3),
        (1, 0): (1, 3),
        (1, 1): (1, 2),
        (1, 2): (1, 1),
        (1, 3): (1, 0),
    }

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (0, "m"): (0, 2),
        (1, "r"): (1, 3),
        (1, "m"): (1, 2),
    }

    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]] = {
        (0, "r"): ((0, 5),),
        (0, "m"): ((0, 4),),
        (1, "r"): ((1, 5),),
        (1, "m"): ((1, 4),),
    }

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {}
    _MIXER_IDX: Dict[Tuple[int, int], int] = {}
    _VATT_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {}
    _RFSWITCH_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {}
    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]] = {}

    _DEFAULT_TEMPCTRL_AUTO_START_AT_LINKUP: bool = False

    _THERMISTORS: Dict[Tuple[int, int], Quel1Thermistor] = {
        (0, 0): Quel1seOnboardThermistor("adda_lmx2594_0"),
        (0, 1): Quel1seOnboardThermistor("adda_lmx2594_1"),
    }

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeCoapClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        _Quel1seConfigSubsystemBase.__init__(self, css_addr, boxtype, port, timeout, sender_limit_by_binding)
        self._construct_tempctrl_debug()

    def configure_peripherals(
        self,
        param: Dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        pass


class Quel2ProtoAddaConfigSubsystem(Quel1seAddaConfigSubsystem):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE = "quel-2-proto-adda.json"
