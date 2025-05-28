import logging
from typing import Any, Callable, Collection, Dict, Mapping, Set, Tuple, Union

from packaging.version import Version

from quel_ic_config.exstickge_coap_client import (
    AbstractSyncAsyncCoapClient,
    Quel1seBoard,
    SyncAsyncCoapClientWithFileLock,
    _ExstickgeCoapClientBase,
)
from quel_ic_config.exstickge_coap_tempctrl_client import _ExstickgeCoapClientQuel1seTempctrlBase
from quel_ic_config.exstickge_proxy import LsiKindId
from quel_ic_config.quel1_config_subsystem_tempctrl import (
    Quel1seConfigSubsystemTempctrlDebugMixin,
    Quel1seConfigSubsystemTempctrlMixin,
)
from quel_ic_config.quel1_thermistor import Quel1seExternalThermistor, Quel1seOnboardThermistor, Quel1Thermistor
from quel_ic_config.quel1se_config_subsystem import _Quel1seConfigSubsystemBase
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger(__name__)


class _ExstickgeCoapClientQuel1seFujitsu11Base(_ExstickgeCoapClientQuel1seTempctrlBase):
    _VALID_BOXTYPE: Set[str] = {"quel1se-fujitsu11-a", "quel1se-fujitsu11-b"}

    _URI_MAPPINGS: Mapping[Tuple[LsiKindId, int], str] = {
        (LsiKindId.AD9082, 0): "adda/mxfe_0",
        (LsiKindId.AD9082, 1): "adda/mxfe_1",
        (LsiKindId.LMX2594, 0): "adda/pll_0",
        (LsiKindId.LMX2594, 1): "adda/pll_1",
        (LsiKindId.LMX2594, 2): "mx0/pll_0",
        (LsiKindId.LMX2594, 3): "mx0/pll_1",
        (LsiKindId.LMX2594, 4): "mx0/pll_2",
        (LsiKindId.LMX2594, 5): "mx0/pll_3",
        (LsiKindId.LMX2594, 6): "mx1/pll_0",
        (LsiKindId.LMX2594, 7): "mx1/pll_1",
        (LsiKindId.LMX2594, 8): "mx1/pll_2",
        (LsiKindId.LMX2594, 9): "mx1/pll_3",
        (LsiKindId.ADRF6780, 0): "mx0/mix_0",
        (LsiKindId.ADRF6780, 1): "mx0/mix_1",
        (LsiKindId.ADRF6780, 2): "mx0/mix_2",
        (LsiKindId.ADRF6780, 3): "mx0/mix_3",
        (LsiKindId.ADRF6780, 4): "mx1/mix_0",
        (LsiKindId.ADRF6780, 5): "mx1/mix_1",
        (LsiKindId.ADRF6780, 6): "mx1/mix_2",
        (LsiKindId.ADRF6780, 7): "mx1/mix_3",
        (LsiKindId.AD5328, 0): "mx0/da_rc0",
        (LsiKindId.AD5328, 1): "mx1/da_rc0",
        (LsiKindId.AD7490, 0): "tmp/ad_tc0",
        (LsiKindId.AD7490, 1): "pwr/ad_tc0",
        (LsiKindId.AD7490, 2): "mx0/ad_tc0",
        (LsiKindId.AD7490, 3): "mx1/ad_tc0",
        (LsiKindId.AD7490, 4): "ps0/ad_tc0",
        (LsiKindId.AD7490, 5): "ps0/ad_tc1",
        (LsiKindId.AD7490, 6): "ps1/ad_tc0",
        (LsiKindId.AD7490, 7): "ps1/ad_tc1",
        (LsiKindId.MIXERBOARD_GPIO, 0): "mx0/xbar/gpio",
        (LsiKindId.MIXERBOARD_GPIO, 1): "mx1/xbar/gpio",
        (LsiKindId.PATHSELECTORBOARD_GPIO, 0): "ps0/rfsw",
        (LsiKindId.PATHSELECTORBOARD_GPIO, 1): "ps1/rfsw",
        (LsiKindId.POWERBOARD_PWM, 0): "pwr/xbar/pwm",
    }

    _AVAILABLE_BOARDS: Tuple[Quel1seBoard, ...] = (Quel1seBoard.POWER, Quel1seBoard.MIXER0, Quel1seBoard.MIXER1)

    _TEMPCTRL_AD7490_NAME: Tuple[str, ...] = ("adda", "pwr", "mx0", "mx1", "ps0a", "ps0b", "ps1a", "ps1b")

    def read_reset(self, kind: LsiKindId, idx: int) -> Union[int, None]:
        v = super().read_reset(kind, idx)
        if v is not None:
            return v
        elif kind == LsiKindId.ADRF6780:
            # TODO: consider better way
            bdname, icname = tuple(self._URI_MAPPINGS[kind, idx].split("/"))
            bdidx = int(bdname[-1])
            icidx = int(icname.split("_")[-1])

            adrf6780_rstn = self.read_reg(LsiKindId.MIXERBOARD_GPIO, bdidx, 2)
            if adrf6780_rstn is None:
                raise RuntimeError("failed to acquire reset status of ADRF6780s")
            return (adrf6780_rstn >> icidx) & 0x1
        else:
            return None

    def write_reset(self, kind: LsiKindId, idx: int, value: int) -> bool:
        if super().write_reset(kind, idx, value):
            return True
        elif kind == LsiKindId.ADRF6780:
            # TODO: consider better way
            bdname, icname = tuple(self._URI_MAPPINGS[kind, idx].split("/"))
            bdidx = int(bdname[-1])
            icidx = int(icname.split("_")[-1])

            adrf6780_rstn = self.read_reg(LsiKindId.MIXERBOARD_GPIO, bdidx, 2)
            if adrf6780_rstn is None:
                raise RuntimeError("failed to acquire reset status of ADRF6780s")

            # Notes: value is already validated in the base method.
            if value == 1:
                adrf6780_rstn |= 1 << icidx
            else:
                adrf6780_rstn &= ~(1 << icidx)
            if self.write_reg(LsiKindId.MIXERBOARD_GPIO, bdidx, 2, adrf6780_rstn):
                return True
            else:
                raise RuntimeError("failed to set reset status of ADRF6780s")
        else:
            return False


class ExstickgeCoapClientQuel1seFujitsu11(_ExstickgeCoapClientQuel1seFujitsu11Base):
    _VERSION_SPEC: Tuple[Version, Version, Set[Version]] = Version("1.0.0"), Version("1.2.1"), set()

    # Notes: no read is available for AD5328
    _READ_REG_PATHS: Mapping[LsiKindId, Callable[[int], str]] = {
        LsiKindId.AD9082: lambda addr: f"/reg/{addr:04x}",
        LsiKindId.ADRF6780: lambda addr: f"/reg/{addr:04x}",
        LsiKindId.LMX2594: lambda addr: f"/reg/{addr:04x}",
        LsiKindId.MIXERBOARD_GPIO: lambda addr: f"/{addr:04x}",
        LsiKindId.PATHSELECTORBOARD_GPIO: lambda addr: "",
        LsiKindId.AD7490: lambda addr: "/ctrl",
        LsiKindId.POWERBOARD_PWM: lambda addr: f"/{addr:04x}",
    }

    _WRITE_REG_PATHS_AND_PAYLOADS: Mapping[LsiKindId, Callable[[int, int], Tuple[str, str]]] = {
        LsiKindId.AD9082: lambda addr, value: (f"/reg/{addr:04x}", f"{value:02x}"),
        LsiKindId.ADRF6780: lambda addr, value: (f"/reg/{addr:04x}", f"{value:04x}"),
        LsiKindId.LMX2594: lambda addr, value: (f"/reg/{addr:04x}", f"{value:04x}"),
        LsiKindId.AD5328: lambda addr, value: ("/any_wr", f"{((addr & 0xF) << 12) | (value & 0xFFF):04x}"),
        LsiKindId.MIXERBOARD_GPIO: lambda addr, value: (f"/{addr:04x}", f"{value:04x}"),
        LsiKindId.PATHSELECTORBOARD_GPIO: lambda addr, value: ("", f"{value:02x}"),
        LsiKindId.AD7490: lambda addr, value: ("/ctrl", f"{value:04x}"),
        LsiKindId.POWERBOARD_PWM: lambda addr, value: (f"/{addr:04x}", f"{value:04x}"),
    }

    def __init__(
        self,
        target_addr: str,
        target_port: int = _ExstickgeCoapClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase.DEFAULT_RESPONSE_TIMEOUT,
    ):
        super().__init__(target_addr, target_port, timeout)

    def _creating_core(self) -> AbstractSyncAsyncCoapClient:
        return SyncAsyncCoapClientWithFileLock(self._target)


class _Quel1seFujitsu11ConfigSubsystemBase(_Quel1seConfigSubsystemBase):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE = "quel-1se-fujitsu11.json"
    _NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 10,
        "adrf6780": 8,
        "ad5328": 2,
        "ad7490": 8,
        "mixerboard_gpio": 2,
        "pathselectorboard_gpio": 2,
        "powerboard_pwm": 1,
    }

    _PROXY_CLASSES: Tuple[type, ...] = (ExstickgeCoapClientQuel1seFujitsu11,)

    _GROUPS: Set[int] = {0, 1}

    _DAC_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (0, 2),
        (0, 3): (0, 3),
        (1, 0): (1, 0),
        (1, 1): (1, 1),
        (1, 2): (1, 2),
        (1, 3): (1, 3),
    }

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, 0): (2, 0),
        (0, 1): (3, 0),
        (0, 2): (4, 0),
        (0, 3): (5, 0),
        (1, 0): (6, 0),
        (1, 1): (7, 0),
        (1, 2): (8, 0),
        (1, 3): (9, 0),
        (0, "r"): (2, 1),
        (0, "m"): (3, 1),
        (1, "r"): (6, 1),
        (1, "m"): (7, 1),
    }

    _MIXER_IDX: Dict[Tuple[int, int], int] = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 4,
        (1, 1): 5,
        (1, 2): 6,
        (1, 3): 7,
    }

    _VATT_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (0, 2),
        (0, 3): (0, 3),
        (1, 0): (1, 0),
        (1, 1): (1, 1),
        (1, 2): (1, 2),
        (1, 3): (1, 3),
    }

    _DEFAULT_TEMPCTRL_AUTO_START_AT_LINKUP: bool = False


class _Quel1seFujitsu11TypeAConfigSubsystemBase(_Quel1seFujitsu11ConfigSubsystemBase):

    __slots__ = ()

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

    _RFSWITCH_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, "r"): (0, 0),
        (0, 0): (0, 0),
        (0, 1): (0, 2),
        (0, 2): (0, 1),
        (0, 3): (0, 3),
        (0, "m"): (0, 4),
        (1, "r"): (1, 0),
        (1, 0): (1, 0),
        (1, 1): (1, 2),
        (1, 2): (1, 1),
        (1, 3): (1, 3),
        (1, "m"): (1, 4),
    }

    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]] = {
        (0, 0): (0, "r"),
        (1, 0): (1, "r"),
    }

    _THERMISTORS: Dict[Tuple[int, int], Quel1Thermistor] = {
        (0, 0): Quel1seOnboardThermistor("adda_lmx2594_1"),
        (0, 1): Quel1seOnboardThermistor("adda_lmx2594_0"),
        (1, 0): Quel1seExternalThermistor("rxmil_00"),  # rename it with 0r or 0m
        (1, 1): Quel1seExternalThermistor("rxmil_01"),  # rename it with 0r or 0m
        (1, 2): Quel1seExternalThermistor("rxmil_10"),  # rename it with 1r or 1m
        (1, 3): Quel1seExternalThermistor("rxmil_11"),  # rename it with 1r or 1m
        (1, 5): Quel1seExternalThermistor("front_panel"),
        (1, 7): Quel1seExternalThermistor("rear_panel"),
        (2, 0): Quel1seOnboardThermistor("mx0_adrf6780_0"),
        (2, 1): Quel1seOnboardThermistor("mx0_adrf6780_1"),
        (2, 2): Quel1seOnboardThermistor("mx0_adrf6780_2"),
        (2, 3): Quel1seOnboardThermistor("mx0_adrf6780_3"),
        (2, 4): Quel1seOnboardThermistor("mx0_lmx2594_0"),
        (2, 5): Quel1seOnboardThermistor("mx0_lmx2594_1"),
        (2, 6): Quel1seOnboardThermistor("mx0_lmx2594_2"),
        (2, 7): Quel1seOnboardThermistor("mx0_lmx2594_3"),
        (3, 0): Quel1seOnboardThermistor("mx1_adrf6780_0"),
        (3, 1): Quel1seOnboardThermistor("mx1_adrf6780_1"),
        (3, 2): Quel1seOnboardThermistor("mx1_adrf6780_2"),
        (3, 3): Quel1seOnboardThermistor("mx1_adrf6780_3"),
        (3, 4): Quel1seOnboardThermistor("mx1_lmx2594_0"),
        (3, 5): Quel1seOnboardThermistor("mx1_lmx2594_1"),
        (3, 6): Quel1seOnboardThermistor("mx1_lmx2594_2"),
        (3, 7): Quel1seOnboardThermistor("mx1_lmx2594_3"),
        (4, 0): Quel1seOnboardThermistor("ps0_sw_monitorout"),
        (4, 1): Quel1seOnboardThermistor("ps0_sw_monitorin"),
        (4, 3): Quel1seOnboardThermistor("ps0_sw0_path_d"),
        (4, 4): Quel1seOnboardThermistor("ps0_sw1_path_d"),
        (4, 5): Quel1seOnboardThermistor("ps0_sw2_path_d"),
        (4, 6): Quel1seOnboardThermistor("ps0_sw0_path_c"),
        (4, 7): Quel1seOnboardThermistor("ps0_sw1_path_c"),
        (4, 11): Quel1seOnboardThermistor("ps0_lna_readout"),
        (4, 12): Quel1seOnboardThermistor("ps0_lna_path_b"),
        (4, 13): Quel1seOnboardThermistor("ps0_lna_path_c"),
        (4, 14): Quel1seOnboardThermistor("ps0_lna_path_d"),
        (4, 15): Quel1seOnboardThermistor("ps0_sw_monitorloop"),
        (5, 0): Quel1seOnboardThermistor("ps0_lna_readin"),
        (5, 1): Quel1seOnboardThermistor("ps0_sw0_readin"),
        (5, 2): Quel1seOnboardThermistor("ps0_sw1_readin"),
        (5, 3): Quel1seOnboardThermistor("ps0_sw2_readin"),
        (5, 4): Quel1seOnboardThermistor("ps0_sw_readloop"),
        (5, 9): Quel1seOnboardThermistor("ps0_sw0_path_b"),
        (5, 10): Quel1seOnboardThermistor("ps0_sw1_path_b"),
        (5, 11): Quel1seOnboardThermistor("ps0_sw0_readout"),
        (5, 12): Quel1seOnboardThermistor("ps0_sw1_readout"),
        (5, 13): Quel1seOnboardThermistor("ps0_sw2_readout"),
        (5, 14): Quel1seOnboardThermistor("ps0_sw2_path_b"),
        (5, 15): Quel1seOnboardThermistor("ps0_sw2_path_c"),
        (6, 0): Quel1seOnboardThermistor("ps1_sw_monitorout"),
        (6, 1): Quel1seOnboardThermistor("ps1_sw_monitorin"),
        (6, 3): Quel1seOnboardThermistor("ps1_sw0_path_d"),
        (6, 4): Quel1seOnboardThermistor("ps1_sw1_path_d"),
        (6, 5): Quel1seOnboardThermistor("ps1_sw2_path_d"),
        (6, 6): Quel1seOnboardThermistor("ps1_sw0_path_c"),
        (6, 7): Quel1seOnboardThermistor("ps1_sw1_path_c"),
        (6, 11): Quel1seOnboardThermistor("ps1_lna_readout"),
        (6, 12): Quel1seOnboardThermistor("ps1_lna_path_b"),
        (6, 13): Quel1seOnboardThermistor("ps1_lna_path_c"),
        (6, 14): Quel1seOnboardThermistor("ps1_lna_path_d"),
        (6, 15): Quel1seOnboardThermistor("ps1_sw_monitorloop"),
        (7, 0): Quel1seOnboardThermistor("ps1_lna_readin"),
        (7, 1): Quel1seOnboardThermistor("ps1_sw0_readin"),
        (7, 2): Quel1seOnboardThermistor("ps1_sw1_readin"),
        (7, 3): Quel1seOnboardThermistor("ps1_sw2_readin"),
        (7, 4): Quel1seOnboardThermistor("ps1_sw_readloop"),
        (7, 9): Quel1seOnboardThermistor("ps1_sw0_path_b"),
        (7, 10): Quel1seOnboardThermistor("ps1_sw1_path_b"),
        (7, 11): Quel1seOnboardThermistor("ps1_sw0_readout"),
        (7, 12): Quel1seOnboardThermistor("ps1_sw1_readout"),
        (7, 13): Quel1seOnboardThermistor("ps1_sw2_readout"),
        (7, 14): Quel1seOnboardThermistor("ps1_sw2_path_b"),
        (7, 15): Quel1seOnboardThermistor("ps1_sw2_path_c"),
    }

    _ACTUATORS: Dict[str, Tuple[str, int]] = {
        "adda_lmx2594_0": ("fan", 0),  # Notes: both fan-#0 and fan-#1 are driven with the same value.
        "adda_lmx2594_1": ("fan", 1),  # Notes: so, this mapping shows proximal pair of thermistor and actuator.
        "mx0_adrf6780_0": ("heater", 0),
        "mx0_adrf6780_1": ("heater", 1),
        "mx0_adrf6780_2": ("heater", 2),
        "mx0_adrf6780_3": ("heater", 3),
        "mx0_lmx2594_0": ("heater", 4),
        "mx0_lmx2594_1": ("heater", 5),
        "mx0_lmx2594_2": ("heater", 6),
        "mx0_lmx2594_3": ("heater", 7),
        "rxmil_00": ("heater", 8),
        "rxmil_01": ("heater", 9),
        "ps0_lna_readin": ("heater", 10),
        "ps0_lna_readout": ("heater", 11),
        "ps0_lna_path_b": ("heater", 12),
        "ps0_lna_path_c": ("heater", 13),
        "ps0_lna_path_d": ("heater", 14),
        "ps0_sw_monitorloop": ("heater", 15),
        "mx1_adrf6780_0": ("heater", 20),
        "mx1_adrf6780_1": ("heater", 21),
        "mx1_adrf6780_2": ("heater", 22),
        "mx1_adrf6780_3": ("heater", 23),
        "mx1_lmx2594_0": ("heater", 24),
        "mx1_lmx2594_1": ("heater", 25),
        "mx1_lmx2594_2": ("heater", 26),
        "mx1_lmx2594_3": ("heater", 27),
        "rxmil_10": ("heater", 28),
        "rxmil_11": ("heater", 29),
        "ps1_lna_readin": ("heater", 30),
        "ps1_lna_readout": ("heater", 31),
        "ps1_lna_path_b": ("heater", 32),
        "ps1_lna_path_c": ("heater", 33),
        "ps1_lna_path_d": ("heater", 34),
        "ps1_sw_monitorloop": ("heater", 35),
    }


class _Quel1seFujitsu11TypeBConfigSubsystemBase(_Quel1seFujitsu11ConfigSubsystemBase):

    __slots__ = ()

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "m"): (0, 2),
        (1, "m"): (1, 2),
    }

    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]] = {
        (0, "m"): ((0, 4),),
        (1, "m"): ((1, 4),),
    }

    _RFSWITCH_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (0, 2),
        (0, 2): (0, 1),
        (0, 3): (0, 3),
        (0, "m"): (0, 4),
        (1, 0): (1, 0),
        (1, 1): (1, 2),
        (1, 2): (1, 1),
        (1, 3): (1, 3),
        (1, "m"): (1, 4),
    }

    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]] = {}

    _THERMISTORS: Dict[Tuple[int, int], Quel1Thermistor] = {
        (0, 0): Quel1seOnboardThermistor("adda_lmx2594_1"),
        (0, 1): Quel1seOnboardThermistor("adda_lmx2594_0"),
        (1, 1): Quel1seExternalThermistor("rxmil_01"),  # rename it with 0m
        (1, 3): Quel1seExternalThermistor("rxmil_11"),  # rename it with 1m
        (1, 5): Quel1seExternalThermistor("front_panel"),
        (1, 7): Quel1seExternalThermistor("rear_panel"),
        (2, 0): Quel1seOnboardThermistor("mx0_adrf6780_0"),
        (2, 1): Quel1seOnboardThermistor("mx0_adrf6780_1"),
        (2, 2): Quel1seOnboardThermistor("mx0_adrf6780_2"),
        (2, 3): Quel1seOnboardThermistor("mx0_adrf6780_3"),
        (2, 4): Quel1seOnboardThermistor("mx0_lmx2594_0"),
        (2, 5): Quel1seOnboardThermistor("mx0_lmx2594_1"),
        (2, 6): Quel1seOnboardThermistor("mx0_lmx2594_2"),
        (2, 7): Quel1seOnboardThermistor("mx0_lmx2594_3"),
        (3, 0): Quel1seOnboardThermistor("mx1_adrf6780_0"),
        (3, 1): Quel1seOnboardThermistor("mx1_adrf6780_1"),
        (3, 2): Quel1seOnboardThermistor("mx1_adrf6780_2"),
        (3, 3): Quel1seOnboardThermistor("mx1_adrf6780_3"),
        (3, 4): Quel1seOnboardThermistor("mx1_lmx2594_0"),
        (3, 5): Quel1seOnboardThermistor("mx1_lmx2594_1"),
        (3, 6): Quel1seOnboardThermistor("mx1_lmx2594_2"),
        (3, 7): Quel1seOnboardThermistor("mx1_lmx2594_3"),
        (4, 0): Quel1seOnboardThermistor("ps0_sw_monitorout"),
        (4, 1): Quel1seOnboardThermistor("ps0_sw_monitorin"),
        (4, 3): Quel1seOnboardThermistor("ps0_sw0_path_d"),
        (4, 4): Quel1seOnboardThermistor("ps0_sw1_path_d"),
        (4, 5): Quel1seOnboardThermistor("ps0_sw2_path_d"),
        (4, 6): Quel1seOnboardThermistor("ps0_sw0_path_c"),
        (4, 7): Quel1seOnboardThermistor("ps0_sw1_path_c"),
        (4, 11): Quel1seOnboardThermistor("ps0_lna_readout"),
        (4, 12): Quel1seOnboardThermistor("ps0_lna_path_b"),
        (4, 13): Quel1seOnboardThermistor("ps0_lna_path_c"),
        (4, 14): Quel1seOnboardThermistor("ps0_lna_path_d"),
        (4, 15): Quel1seOnboardThermistor("ps0_sw_monitorloop"),
        (5, 0): Quel1seOnboardThermistor("ps0_lna_readin"),
        (5, 1): Quel1seOnboardThermistor("ps0_sw0_readin"),
        (5, 2): Quel1seOnboardThermistor("ps0_sw1_readin"),
        (5, 3): Quel1seOnboardThermistor("ps0_sw2_readin"),
        (5, 4): Quel1seOnboardThermistor("ps0_sw_readloop"),
        (5, 9): Quel1seOnboardThermistor("ps0_sw0_path_b"),
        (5, 10): Quel1seOnboardThermistor("ps0_sw1_path_b"),
        (5, 11): Quel1seOnboardThermistor("ps0_sw0_readout"),
        (5, 12): Quel1seOnboardThermistor("ps0_sw1_readout"),
        (5, 13): Quel1seOnboardThermistor("ps0_sw2_readout"),
        (5, 14): Quel1seOnboardThermistor("ps0_sw2_path_b"),
        (5, 15): Quel1seOnboardThermistor("ps0_sw2_path_c"),
        (6, 0): Quel1seOnboardThermistor("ps1_sw_monitorout"),
        (6, 1): Quel1seOnboardThermistor("ps1_sw_monitorin"),
        (6, 3): Quel1seOnboardThermistor("ps1_sw0_path_d"),
        (6, 4): Quel1seOnboardThermistor("ps1_sw1_path_d"),
        (6, 5): Quel1seOnboardThermistor("ps1_sw2_path_d"),
        (6, 6): Quel1seOnboardThermistor("ps1_sw0_path_c"),
        (6, 7): Quel1seOnboardThermistor("ps1_sw1_path_c"),
        (6, 11): Quel1seOnboardThermistor("ps1_lna_readout"),
        (6, 12): Quel1seOnboardThermistor("ps1_lna_path_b"),
        (6, 13): Quel1seOnboardThermistor("ps1_lna_path_c"),
        (6, 14): Quel1seOnboardThermistor("ps1_lna_path_d"),
        (6, 15): Quel1seOnboardThermistor("ps1_sw_monitorloop"),
        (7, 0): Quel1seOnboardThermistor("ps1_lna_readin"),
        (7, 1): Quel1seOnboardThermistor("ps1_sw0_readin"),
        (7, 2): Quel1seOnboardThermistor("ps1_sw1_readin"),
        (7, 3): Quel1seOnboardThermistor("ps1_sw2_readin"),
        (7, 4): Quel1seOnboardThermistor("ps1_sw_readloop"),
        (7, 9): Quel1seOnboardThermistor("ps1_sw0_path_b"),
        (7, 10): Quel1seOnboardThermistor("ps1_sw1_path_b"),
        (7, 11): Quel1seOnboardThermistor("ps1_sw0_readout"),
        (7, 12): Quel1seOnboardThermistor("ps1_sw1_readout"),
        (7, 13): Quel1seOnboardThermistor("ps1_sw2_readout"),
        (7, 14): Quel1seOnboardThermistor("ps1_sw2_path_b"),
        (7, 15): Quel1seOnboardThermistor("ps1_sw2_path_c"),
    }

    _ACTUATORS: Dict[str, Tuple[str, int]] = {
        "adda_lmx2594_0": ("fan", 0),  # Notes: both fan-#0 and fan-#1 are driven with the same value.
        "adda_lmx2594_1": ("fan", 1),  # Notes: so, this mapping shows proximal pair of thermistor and actuator.
        "mx0_adrf6780_0": ("heater", 0),
        "mx0_adrf6780_1": ("heater", 1),
        "mx0_adrf6780_2": ("heater", 2),
        "mx0_adrf6780_3": ("heater", 3),
        "mx0_lmx2594_0": ("heater", 4),
        "mx0_lmx2594_1": ("heater", 5),
        "mx0_lmx2594_2": ("heater", 6),
        "mx0_lmx2594_3": ("heater", 7),
        "rxmil_01": ("heater", 9),
        "ps0_lna_readin": ("heater", 10),
        "ps0_lna_readout": ("heater", 11),
        "ps0_lna_path_b": ("heater", 12),
        "ps0_lna_path_c": ("heater", 13),
        "ps0_lna_path_d": ("heater", 14),
        "ps0_sw_monitorloop": ("heater", 15),
        "mx1_adrf6780_0": ("heater", 20),
        "mx1_adrf6780_1": ("heater", 21),
        "mx1_adrf6780_2": ("heater", 22),
        "mx1_adrf6780_3": ("heater", 23),
        "mx1_lmx2594_0": ("heater", 24),
        "mx1_lmx2594_1": ("heater", 25),
        "mx1_lmx2594_2": ("heater", 26),
        "mx1_lmx2594_3": ("heater", 27),
        "rxmil_11": ("heater", 29),
        "ps1_lna_readin": ("heater", 30),
        "ps1_lna_readout": ("heater", 31),
        "ps1_lna_path_b": ("heater", 32),
        "ps1_lna_path_c": ("heater", 33),
        "ps1_lna_path_d": ("heater", 34),
        "ps1_sw_monitorloop": ("heater", 35),
    }


class Quel1seFujitsu11TypeAConfigSubsystem(
    _Quel1seFujitsu11TypeAConfigSubsystemBase,
    Quel1seConfigSubsystemTempctrlMixin,
):
    _DEFAULT_TEMPCTRL_AUTO_START_AT_LINKUP: bool = True

    __slots__ = ()

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeCoapClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        _Quel1seFujitsu11TypeAConfigSubsystemBase.__init__(
            self, css_addr, boxtype, port, timeout, sender_limit_by_binding
        )

    def initialize(self) -> None:
        super().initialize()
        self._construct_tempctrl()

    def configure_peripherals(
        self,
        param: Dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        _Quel1seFujitsu11TypeAConfigSubsystemBase.configure_peripherals(
            self,
            param,
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )
        self.init_tempctrl(param)


class Quel1seFujitsu11TypeADebugConfigSubsystem(
    _Quel1seFujitsu11TypeAConfigSubsystemBase,
    Quel1seConfigSubsystemTempctrlDebugMixin,
):
    __slots__ = ()

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeCoapClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        _Quel1seFujitsu11TypeAConfigSubsystemBase.__init__(
            self, css_addr, boxtype, port, timeout, sender_limit_by_binding
        )

    def initialize(self) -> None:
        super().initialize()
        self._construct_tempctrl_debug()

    def configure_peripherals(
        self,
        param: Dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        _Quel1seFujitsu11TypeAConfigSubsystemBase.configure_peripherals(
            self,
            param,
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )
        self.init_tempctrl_debug(param)


class Quel1seFujitsu11TypeBConfigSubsystem(
    _Quel1seFujitsu11TypeBConfigSubsystemBase,
    Quel1seConfigSubsystemTempctrlMixin,
):
    _DEFAULT_TEMPCTRL_AUTO_START_AT_LINKUP: bool = True

    __slots__ = ()

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeCoapClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        _Quel1seFujitsu11TypeBConfigSubsystemBase.__init__(
            self, css_addr, boxtype, port, timeout, sender_limit_by_binding
        )

    def initialize(self) -> None:
        super().initialize()
        self._construct_tempctrl()

    def configure_peripherals(
        self,
        param: Dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        _Quel1seFujitsu11TypeBConfigSubsystemBase.configure_peripherals(
            self,
            param,
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )
        self.init_tempctrl(param)


class Quel1seFujitsu11TypeBDebugConfigSubsystem(
    _Quel1seFujitsu11TypeBConfigSubsystemBase,
    Quel1seConfigSubsystemTempctrlDebugMixin,
):
    __slots__ = ()

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeCoapClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        _Quel1seFujitsu11TypeBConfigSubsystemBase.__init__(
            self, css_addr, boxtype, port, timeout, sender_limit_by_binding
        )

    def initialize(self) -> None:
        super().initialize()
        self._construct_tempctrl_debug()

    def configure_peripherals(
        self,
        param: Dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        _Quel1seFujitsu11TypeBConfigSubsystemBase.configure_peripherals(
            self,
            param,
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )
        self.init_tempctrl_debug(param)
