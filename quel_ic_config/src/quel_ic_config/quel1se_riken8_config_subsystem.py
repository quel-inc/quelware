import logging
import time
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Mapping, Set, Tuple, Union, cast

from packaging.version import Version

from quel_ic_config.exstickge_coap_client import Quel1seBoard, _ExstickgeCoapClientBase, get_exstickge_server_info
from quel_ic_config.exstickge_coap_tempctrl_client import _ExstickgeCoapClientQuel1seTempctrlBase
from quel_ic_config.exstickge_proxy import LsiKindId
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemMixerboardGpioMixin,
    Quel1ConfigSubsystemPathselectorboardGpioMixin,
    Quel1ConfigSubsystemRoot,
)
from quel_ic_config.quel1_config_subsystem_tempctrl import (
    Quel1seConfigSubsystemTempctrlDebugMixin,
    Quel1seConfigSubsystemTempctrlMixin,
)
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption, Quel1Feature
from quel_ic_config.thermistor import Quel1seExternalThermistor, Quel1seOnboardThermistor, Thermistor

logger = logging.getLogger(__name__)


class _ExstickgeCoapClientQuel1seRiken8Base(_ExstickgeCoapClientQuel1seTempctrlBase):
    _VALID_BOXTYPE: Set[str] = {"quel1se-riken8"}

    _URI_MAPPINGS: Mapping[Tuple[LsiKindId, int], str] = {
        (LsiKindId.AD9082, 0): "adda/mxfe_0",
        (LsiKindId.AD9082, 1): "adda/mxfe_1",
        (LsiKindId.LMX2594, 0): "adda/pll_0",
        (LsiKindId.LMX2594, 1): "adda/pll_1",
        (LsiKindId.LMX2594, 2): "mx0/pll_0",
        (LsiKindId.LMX2594, 3): "mx0/pll_2",
        (LsiKindId.LMX2594, 4): "mx0/pll_4",
        (LsiKindId.ADRF6780, 0): "mx0/mix_0",
        (LsiKindId.ADRF6780, 1): "mx0/mix_2",
        (LsiKindId.AD5328, 0): "mx0/da_rc0",
        (LsiKindId.AD7490, 0): "tmp/ad_tc0",
        (LsiKindId.AD7490, 1): "pwr/ad_tc0",
        (LsiKindId.AD7490, 2): "mx0/ad_tc0",
        (LsiKindId.AD7490, 3): "mx1/ad_tc0",
        (LsiKindId.AD7490, 4): "ps0/ad_tc0",
        (LsiKindId.AD7490, 5): "ps0/ad_tc1",
        (LsiKindId.AD7490, 6): "ps1/ad_tc0",
        (LsiKindId.AD7490, 7): "ps1/ad_tc1",
        (LsiKindId.MIXERBOARD_GPIO, 0): "mx0/xbar/gpio",
        (LsiKindId.PATHSELECTORBOARD_GPIO, 0): "ps0/rfsw",
        (LsiKindId.PATHSELECTORBOARD_GPIO, 1): "ps1/rfsw",
        (LsiKindId.POWERBOARD_PWM, 0): "pwr/xbar/pwm",
    }

    _AVAILABLE_BOARDS: Set[Quel1seBoard] = {Quel1seBoard.POWER, Quel1seBoard.MIXER0}

    _TEMPCTRL_AD7490_NAME: Tuple[str, ...] = ("adda", "pwr", "mx0", "mx1", "ps0a", "ps0b", "ps1a", "ps1b")

    def read_reset(self, kind: LsiKindId, idx: int) -> Union[int, None]:
        v = super().read_reset(kind, idx)
        if v is not None:
            return v
        elif kind == LsiKindId.ADRF6780:
            # TODO: stop calling read_reg()
            adrf6780_rstn = self.read_reg(LsiKindId.MIXERBOARD_GPIO, 0, 2)
            if adrf6780_rstn is None:
                raise RuntimeError("failed to acquire reset status of ADRF6780s")
            # TODO: consider better way
            pos = int(self._URI_MAPPINGS[kind, idx].split("_")[-1])
            return (adrf6780_rstn >> pos) & 0x1
        else:
            return None

    def write_reset(self, kind: LsiKindId, idx: int, value: int) -> bool:
        if super().write_reset(kind, idx, value):
            return True
        elif kind == LsiKindId.ADRF6780:
            # TODO: stop calling read_reg()
            adrf6780_rstn = self.read_reg(LsiKindId.MIXERBOARD_GPIO, 0, 2)
            if adrf6780_rstn is None:
                raise RuntimeError("failed to acquire reset status of ADRF6780s")
            # TODO: consider better way
            pos = int(self._URI_MAPPINGS[kind, idx].split("_")[-1])
            # Notes: value is already validated in the base method.
            if value == 1:
                adrf6780_rstn |= 1 << pos
            else:
                adrf6780_rstn &= ~(1 << pos)
            if self.write_reg(LsiKindId.MIXERBOARD_GPIO, 0, 2, adrf6780_rstn):
                return True
            else:
                raise RuntimeError("failed to set reset status of ADRF6780s")
        else:
            return False


class ExstickgeCoapClientQuel1seRiken8Dev1(_ExstickgeCoapClientQuel1seRiken8Base):
    _VERSION_SPEC: Tuple[Version, Version, Set[Version]] = Version("0.0.1"), Version("0.0.1"), set()

    # Notes: no read is available for AD5328
    _READ_REG_PATHS: Mapping[LsiKindId, Callable[[int], str]] = {
        LsiKindId.AD9082: lambda addr: f"/reg/0x{addr:04x}",
        LsiKindId.ADRF6780: lambda addr: f"/reg/0x{addr:04x}",
        LsiKindId.LMX2594: lambda addr: f"/reg/0x{addr:04x}",
        LsiKindId.MIXERBOARD_GPIO: lambda addr: f"/0x{addr:04x}",
        LsiKindId.PATHSELECTORBOARD_GPIO: lambda addr: "",
        LsiKindId.AD7490: lambda addr: "/ctrl",
        LsiKindId.POWERBOARD_PWM: lambda addr: f"/0x{addr:04x}",
    }

    _WRITE_REG_PATHS_AND_PAYLOADS: Mapping[LsiKindId, Callable[[int, int], Tuple[str, str]]] = {
        LsiKindId.AD9082: lambda addr, value: (f"/reg/0x{addr:04x}", f"0x{value:02x}"),
        LsiKindId.ADRF6780: lambda addr, value: (f"/reg/0x{addr:04x}", f"0x{value:04x}"),
        LsiKindId.LMX2594: lambda addr, value: (f"/reg/0x{addr:04x}", f"0x{value:04x}"),
        LsiKindId.AD5328: lambda addr, value: ("/any_wr", f"0x{((addr & 0xF) << 12) | (value & 0xFFF):04x}"),
        LsiKindId.MIXERBOARD_GPIO: lambda addr, value: (f"/0x{addr:04x}", f"0x{value:04x}"),
        LsiKindId.PATHSELECTORBOARD_GPIO: lambda addr, value: ("", f"0x{value:02x}"),
        LsiKindId.AD7490: lambda addr, value: ("/ctrl", f"0x{value:04x}"),
        LsiKindId.POWERBOARD_PWM: lambda addr, value: (f"/0x{addr:04x}", f"0x{value:04x}"),
    }

    def __init__(
        self,
        target_addr: str,
        target_port: int = _ExstickgeCoapClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase._DEFAULT_RESPONSE_TIMEOUT,
    ):
        super().__init__(target_addr, target_port, timeout)

    def read_boxtype(self) -> str:
        # Notes: "v0.0.1" firmware doesn't implement read_boxtype API
        return "quel1se-riken8"

    def read_current_config(self) -> str:
        # Notes: dummy functions
        return ""

    def write_current_config(self, cfg: str) -> None:
        # Notes: dummy functions
        logger.warning("config subsystem firmware doesn't support current_config API, just ignored")
        return


class ExstickgeCoapClientQuel1seRiken8Dev2(_ExstickgeCoapClientQuel1seRiken8Base):
    _VERSION_SPEC: Tuple[Version, Version, Set[Version]] = Version("0.1.0"), Version("0.1.0"), set()

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
        target_port: int = _ExstickgeCoapClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase._DEFAULT_RESPONSE_TIMEOUT,
    ):
        super().__init__(target_addr, target_port, timeout)

    def read_boxtype(self) -> str:
        # Notes: "v0.1.0" firmware doesn't implement read_boxtype API
        return "quel1se-riken8"

    def read_current_config(self) -> str:
        # Notes: dummy functions
        return ""

    def write_current_config(self, cfg: str) -> None:
        # Notes: dummy functions
        logger.warning("config subsystem firmware doesn't support current_config API, just ignored")
        return


class ExstickgeCoapClientQuel1seRiken8(_ExstickgeCoapClientQuel1seRiken8Base):
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
        target_port: int = _ExstickgeCoapClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase._DEFAULT_RESPONSE_TIMEOUT,
    ):
        super().__init__(target_addr, target_port, timeout)


class _Quel1seRiken8ConfigSubsystemBase(
    Quel1ConfigSubsystemRoot,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemMixerboardGpioMixin,
    Quel1ConfigSubsystemPathselectorboardGpioMixin,
):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE = "quel-1se-riken8.json"
    _NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 5,
        "adrf6780": 2,
        "ad5328": 1,
        "ad7490": 8,
        "mixerboard_gpio": 1,
        "pathselectorboard_gpio": 2,
        "powerboard_pwm": 1,
    }

    _GROUPS: Set[int] = {0, 1}

    _MXFE_IDXS: Set[int] = {0, 1}

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

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (0, "m"): (0, 2),
        (1, "m"): (1, 2),
    }

    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {
        (0, "r"): (5,),
        (0, "m"): (4,),
        (1, "m"): (4,),
    }

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, 0): (2, 0),
        (0, 2): (3, 0),
        (0, "r"): (2, 1),
        (0, "m"): (4, 0),
        (1, "m"): (4, 1),
    }

    _MIXER_IDX: Dict[Tuple[int, int], int] = {
        (0, 0): 0,
        (0, 2): 1,
    }

    _VATT_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 2): (0, 2),
    }

    _RFSWITCH_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, "r"): (0, 0),
        (0, 0): (0, 0),
        (0, 1): (0, 0),  # combined with (0, 0) internally
        (0, 2): (0, 2),
        (0, 3): (0, 3),
        (0, "m"): (0, 4),
        (1, 0): (1, 0),
        (1, 1): (1, 1),
        (1, 2): (1, 2),
        (1, 3): (1, 3),
        (1, "m"): (1, 4),
    }

    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]] = {
        (0, 0): (0, "r"),
    }

    _DEFAULT_TEMPCTRL_AUTO_START_AT_LINKUP: bool = False

    _THERMISTORS: Dict[Tuple[int, int], Thermistor] = {
        (0, 0): Quel1seOnboardThermistor("adda_lmx2594_1"),
        (0, 1): Quel1seOnboardThermistor("adda_lmx2594_0"),
        (1, 5): Quel1seExternalThermistor("front_panel"),
        (1, 7): Quel1seExternalThermistor("rear_panel"),
        (2, 0): Quel1seOnboardThermistor("mx0_adrf6780_0"),
        (2, 1): Quel1seOnboardThermistor("mx0_amp_1"),
        (2, 2): Quel1seOnboardThermistor("mx0_adrf6780_2"),
        (2, 3): Quel1seOnboardThermistor("mx0_amp_3"),
        (2, 4): Quel1seOnboardThermistor("mx0_lmx2594_0"),
        (2, 5): Quel1seOnboardThermistor("mx0_lmx2594_2"),
        (2, 6): Quel1seOnboardThermistor("mx0_hmc8193_r"),  # TODO: check correspondence of hmc8193s
        (2, 7): Quel1seOnboardThermistor("mx0_hmc8193_m0"),
        (2, 8): Quel1seOnboardThermistor("mx0_hmc8193_m1"),
        (2, 9): Quel1seOnboardThermistor("mx0_lmx2594_4"),
        (3, 0): Quel1seOnboardThermistor("mx1_amp_0"),  # TODO: check them
        (3, 1): Quel1seOnboardThermistor("mx1_amp_1"),
        (3, 2): Quel1seOnboardThermistor("mx1_amp_2"),
        (3, 3): Quel1seOnboardThermistor("mx1_amp_3"),
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

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        features: Union[Collection[Quel1Feature], None] = None,
        config_path: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,  # TODO: should be elaborated.
        port: int = _ExstickgeCoapClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase._DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        Quel1ConfigSubsystemRoot.__init__(
            self, css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )
        self._construct_ad9082()
        self._construct_lmx2594()
        self._construct_adrf6780()
        self._construct_ad5328()
        self._construct_mixerboard_gpio()
        self._construct_pathselectorboard_gpio()

    def _create_exstickge_proxy(
        self, port: int, timeout: float, sender_limit_by_binding: bool
    ) -> _ExstickgeCoapClientBase:
        # Notes: port will be available later.
        # Notes: sender_limit_by_binding may be available later.

        is_coap_firmware, coap_firmware_version, coap_boxtype = get_exstickge_server_info(self._css_addr)
        if is_coap_firmware:
            for proxy_cls in (
                ExstickgeCoapClientQuel1seRiken8Dev1,
                ExstickgeCoapClientQuel1seRiken8Dev2,
                ExstickgeCoapClientQuel1seRiken8,
            ):
                if proxy_cls.matches(coap_boxtype, coap_firmware_version):
                    return proxy_cls(self._css_addr, port, timeout)
            else:
                raise RuntimeError(
                    f"unsupported CoAP firmware is running on exstickge: {coap_boxtype}:{coap_firmware_version}"
                )
        else:
            raise RuntimeError("no CoAP firmware is running on exstickge")

    def configure_peripherals(
        self,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = set()
        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = set()

        # Notes: close all RF switches at first
        for i in (0, 1):
            self.init_pathselectorboard_gpio(i)

        # Notes: release reset of CPLDs on all the peripheral board
        proxy = cast(_ExstickgeCoapClientBase, self._proxy)
        for board in (Quel1seBoard.MIXER0, Quel1seBoard.POWER):
            if not proxy.read_board_active(board):
                logger.info(f"releasing reset of board '{board.value}'")
                proxy.write_board_active(board, True)
            else:
                logger.info(f"board '{board.value}' is already activated")

        # Notes: initialize ICs on mixer board 0 for RF
        self.init_ad5328(0)

        for i in (0, 1):
            proxy.write_reset(LsiKindId.ADRF6780, i, 1)
            self.init_adrf6780(i, ignore_id_mismatch=i in ignore_access_failure_of_adrf6780)

        for i in (2, 3, 4):
            self.init_lmx2594(i, ignore_lock_failure=i in ignore_lock_failure_of_lmx2594)

    def configure_all_mxfe_clocks(self, ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None) -> None:
        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        for group in range(2):
            lmx2594_idx = 0 + group
            self.init_lmx2594(lmx2594_idx, ignore_lock_failure=lmx2594_idx in ignore_lock_failure_of_lmx2594)

    def get_ad9082_hard_reset(self, mxfe_idx: int) -> bool:
        # Notes: re-consider better way
        proxy = cast(_ExstickgeCoapClientBase, self._proxy)
        v = proxy.read_reset(LsiKindId.AD9082, mxfe_idx)
        if v == 0:
            return True
        elif v == 1:
            return False
        else:
            raise RuntimeError(f"invlid value {v} for state of AD9082[{mxfe_idx}]'s hard_reset")

    def set_ad9082_hard_reset(self, mxfe_idx: int, value: bool) -> None:
        # Notes: re-consider better way
        proxy = cast(_ExstickgeCoapClientBase, self._proxy)
        proxy.write_reset(LsiKindId.AD9082, mxfe_idx, 0 if value else 1)

    def configure_mxfe(
        self,
        mxfe_idx: int,
        *,
        hard_reset: bool = False,
        soft_reset: bool = False,
        mxfe_init: bool = False,
        use_204b: bool = True,
        use_bg_cal: bool = False,
        ignore_crc_error: bool = False,
    ) -> bool:
        self._validate_group(mxfe_idx)
        if hard_reset:
            logger.info(f"asserting a reset pin of {self._css_addr}:AD9082-{mxfe_idx}")
            self.set_ad9082_hard_reset(mxfe_idx, True)
            time.sleep(0.01)

        if self.get_ad9082_hard_reset(mxfe_idx):
            logger.info(f"negating a reset pin of {self._css_addr}:AD9082-{mxfe_idx}")
            self.set_ad9082_hard_reset(mxfe_idx, False)

        self.ad9082[mxfe_idx].initialize(
            reset=soft_reset, link_init=mxfe_init, use_204b=use_204b, use_bg_cal=use_bg_cal
        )
        return self.check_link_status(mxfe_idx, mxfe_init, ignore_crc_error)

    def dump_channel(self, group: int, line: int, channel: int) -> Dict[str, Any]:
        """dumping the current configuration of a transmitter channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the target channel belongs to.
        :param channel: a line-local index of the channel.
        :return: the current configuration information of the channel.
        """
        self._validate_channel(group, line, channel)
        return {
            "fnco_freq": self.get_dac_fnco(group, line, channel),
        }

    def dump_line(self, group: int, line: int) -> Dict[str, Any]:
        """dumping the current configuration of a transmitter line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: the current configuration information of the line.
        """
        self._validate_line(group, line)
        r: Dict[str, Any] = {}
        r["channels"] = {
            ch: self.dump_channel(group, line, ch) for ch in range(self.get_num_channels_of_line(group, line))
        }
        r["cnco_freq"] = self.get_dac_cnco(group, line)
        r["fullscale_current"] = self.get_fullscale_current(group, line)
        if (group, line) in self._LO_IDX:
            d_ratio = self.get_divider_ratio(group, line)
            if d_ratio > 0:
                r["lo_freq"] = self.get_lo_multiplier(group, line) * 100_000_000 // d_ratio
            else:
                r["lo_freq"] = 0
        if (group, line) in self._VATT_IDX:
            r["sideband"] = self.get_sideband(group, line)
            vatt = self.get_vatt_carboncopy(group, line)
            if vatt is not None:
                r["vatt"] = vatt
        if (group, line) in self._RFSWITCH_IDX:
            r["rfswitch"] = "block" if self.is_blocked_line(group, line) else "pass"
        return r

    def dump_rchannel(self, group: int, rline: str, rchannel: int) -> Dict[str, Any]:
        """dumping the current configuration of the receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param rline: a group-local index of a line which the target channel belongs to.
        :param rchannel: a line-local index of the target channel.
        :return: the current configuration information of the channel.
        """
        self._validate_rchannel(group, rline, rchannel)
        return {
            "fnco_freq": self.get_adc_fnco(group, rline, rchannel),
        }

    def dump_rline(self, group: int, rline: str) -> Dict[str, Any]:
        """dumping the current configuration of a receiver line.

        :param group: an index of a group which the line belongs to.
        :param rline: a group-local index of the line.
        :return: the current configuration information of the line.
        """
        self._validate_rline(group, rline)
        r: Dict[str, Any] = {}
        if (group, rline) in self._LO_IDX:
            d_ratio = self.get_divider_ratio(group, rline)
            if d_ratio > 0:
                r["lo_freq"] = self.get_lo_multiplier(group, rline) * 100_000_000 // d_ratio
            else:
                r["lo_freq"] = 0
        r["cnco_freq"] = self.get_adc_cnco(group, rline)
        r["channels"] = {
            rch: self.dump_rchannel(group, rline, rch) for rch in range(self.get_num_rchannels_of_rline(group, rline))
        }
        if (group, rline) in self._RFSWITCH_IDX:
            r["rfswitch"] = "loop" if self.is_blocked_line(group, rline) else "open"
        return r


class Quel1seRiken8ConfigSubsystem(
    _Quel1seRiken8ConfigSubsystemBase,
    Quel1seConfigSubsystemTempctrlMixin,
):
    _DEFAULT_TEMPCTRL_AUTO_START_AT_LINKUP: bool = True

    __slots__ = ()

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        features: Union[Collection[Quel1Feature], None] = None,
        config_path: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,  # TODO: should be elaborated.
        port: int = _ExstickgeCoapClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase._DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        _Quel1seRiken8ConfigSubsystemBase.__init__(
            self, css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )
        self._construct_tempctrl()

    def configure_peripherals(
        self,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        _Quel1seRiken8ConfigSubsystemBase.configure_peripherals(
            self, ignore_access_failure_of_adrf6780, ignore_lock_failure_of_lmx2594
        )
        self.init_tempctrl()


class Quel1seRiken8DebugConfigSubsystem(
    _Quel1seRiken8ConfigSubsystemBase,
    Quel1seConfigSubsystemTempctrlDebugMixin,
):
    _HEATER_MAP: Dict[int, str] = {
        0: "mx0_adrf6780_0",
        1: "mx0_amp_1",
        2: "mx0_adrf6780_2",
        3: "mx0_amp_3",
        4: "mx0_lmx2594_0",
        6: "mx0_lmx2594_2",
        10: "ps0_lna_readin",
        11: "ps0_lna_readout",
        12: "ps0_lna_path_b",
        13: "ps0_lna_path_c",
        14: "ps0_lna_path_d",
        15: "ps0_sw_monitorloop",
        16: "mx0_hmc8193_r",
        17: "mx0_hmc8193_m0",
        18: "mx0_hmc8193_m1",
        19: "mx0_lmx2594_4",
        20: "mx1_amp_0",
        21: "mx1_amp_1",
        22: "mx1_amp_2",
        23: "mx1_amp_3",
        30: "ps1_lna_readin",
        31: "ps1_lna_readout",
        32: "ps1_lna_path_b",
        33: "ps1_lna_path_c",
        34: "ps1_lna_path_d",
        35: "ps1_sw_monitorloop",
    }
    _HEADTER_RVMAP: Dict[str, int] = {v: k for k, v in _HEATER_MAP.items()}
    _HEATERS: Set[int] = set(_HEATER_MAP.keys())

    __slots__ = ()

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        features: Union[Collection[Quel1Feature], None] = None,
        config_path: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,  # TODO: should be elaborated.
        port: int = _ExstickgeCoapClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase._DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        _Quel1seRiken8ConfigSubsystemBase.__init__(
            self, css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )
        self._construct_tempctrl_debug()

    def configure_peripherals(
        self,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        _Quel1seRiken8ConfigSubsystemBase.configure_peripherals(
            self, ignore_access_failure_of_adrf6780, ignore_lock_failure_of_lmx2594
        )
        self.init_tempctrl_debug()