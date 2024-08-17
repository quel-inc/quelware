import logging
import socket
from pathlib import Path
from typing import Any, Collection, Dict, Mapping, Set, Tuple, Union

from quel_ic_config.exstickge_sock_client import LsiKindId, _ExstickgeSockClientBase
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemRoot,
)
from quel_ic_config.quel1_config_subsystem_tempctrl import Quel1ConfigSubsystemTempctrlMixin
from quel_ic_config.quel1se_proto_adda_config_subsystem import Quel1seProtoAddaConfigSubsystemGpioMixin
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption, Quel1Feature

logger = logging.getLogger(__name__)


# Notes: this class is corresponding to e-trees/exstickge_spi_device_ctrl/source/top_poc.v
class ExstickgeSockClientQuel1seProto8(_ExstickgeSockClientBase):
    _AD9082_IF_0 = LsiKindId.AD9082
    _ADRF6780_IF_0 = LsiKindId.ADRF6780
    _LMX2594_IF_0 = LsiKindId.LMX2594
    _LMX2594_IF_1 = LsiKindId.LMX2594 + 1
    _AD5328_IF_0 = LsiKindId.AD5328
    _GPIO_IF_0 = LsiKindId.GPIO
    _AD5328_IF_1 = LsiKindId.AD5328 + 2
    _GPIO_IF_1 = LsiKindId.GPIO + 2

    _SPIIF_MAPPINGS: Mapping[LsiKindId, Mapping[int, Tuple[int, int]]] = {
        LsiKindId.AD9082: {
            0: (_AD9082_IF_0, 0),
            1: (_AD9082_IF_0, 1),
        },
        LsiKindId.LMX2594: {
            0: (_LMX2594_IF_0, 0),  # for Readout/Readin
            1: (_LMX2594_IF_0, 1),  # for two Monitors  (RF-A: J616,  RF-B: J517)
            2: (_LMX2594_IF_0, 2),  # for Pump
            3: (_LMX2594_IF_1, 3),  # for AD9082_0 clock
            4: (_LMX2594_IF_1, 4),  # for AD9098_1 clock
        },
        LsiKindId.ADRF6780: {
            0: (_ADRF6780_IF_0, 0),  # for ReadOut
            1: (_ADRF6780_IF_0, 1),  # for Pump
        },
        LsiKindId.AD5328: {
            0: (_AD5328_IF_0, 0),  # ChA: Readout-mixer, ChC: Pump-mixer, ChE-ChH: heater 1-{4,5,6,7}
            1: (_AD5328_IF_1, 0),  # to heaters
        },
        LsiKindId.GPIO: {
            0: (_GPIO_IF_0, 0),  # not used
            1: (_GPIO_IF_1, 0),  # [1:0] AD9082 reset
        },
    }

    def __init__(
        self,
        target_address,
        target_port=_ExstickgeSockClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase._DEFAULT_RESPONSE_TIMEOUT,
        receiver_limit_by_binding: bool = False,
        sock: Union[socket.socket, None] = None,
    ):
        super().__init__(target_address, target_port, timeout, receiver_limit_by_binding, sock)


class Quel1seProto8ConfigSubsystem(
    Quel1ConfigSubsystemRoot,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1seProtoAddaConfigSubsystemGpioMixin,
    Quel1ConfigSubsystemTempctrlMixin,
):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE: str = "quel-1se-proto8.json"
    _NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 5,
        "adrf6780": 2,
        "ad5328": 2,
        "gpio": 2,
    }

    _GROUPS: Set[int] = {0, 1}

    _MXFE_IDXS: Set[int] = {0, 1}

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

    # Notes: (1, "r") is not used in this configuration.
    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (0, "m"): (0, 2),
        (1, "m"): (1, 2),
    }

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 2): (2, 0),
        (0, "r"): (0, 1),
        (0, "m"): (1, 0),
        (1, "m"): (1, 1),
    }

    _MIXER_IDX: Dict[Tuple[int, int], int] = {
        (0, 0): 0,
        (0, 2): 1,
    }

    _VATT_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 2): (0, 2),
    }

    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {
        (0, "r"): (5,),
        (0, "m"): (4,),
        (1, "m"): (4,),
    }

    _GPIO_FOR_AD9082_HARDRESET: Dict[int, Tuple[int, str]] = {
        0: (1, "b00"),
        1: (1, "b01"),
    }

    _RFSWITCH_NAME: Dict[Tuple[int, Union[int, str]], Tuple[int, str]] = {}

    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]] = {}

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        features: Union[Collection[Quel1Feature], None] = None,
        config_path: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        port: int = _ExstickgeSockClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase._DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        Quel1ConfigSubsystemRoot.__init__(
            self, css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )
        self._construct_ad9082()
        self._construct_lmx2594()
        self._construct_adrf6780()
        self._construct_ad5328()
        self._construct_gpio()

    def _create_exstickge_proxy(
        self, port: int, timeout: float, sender_limit_by_binding: bool
    ) -> _ExstickgeSockClientBase:
        return ExstickgeSockClientQuel1seProto8(self._css_addr, port, timeout, sender_limit_by_binding)

    def configure_peripherals(
        self,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = {}

        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        for i in range(2):
            self.init_gpio(i)

        for i in range(2):
            self.init_ad5328(i)

        for i in range(2):
            self.init_adrf6780(i, ignore_id_mismatch=i in ignore_access_failure_of_adrf6780)

        for i in range(3):
            self.init_lmx2594(i, ignore_lock_failure=i in ignore_lock_failure_of_lmx2594)

    def configure_all_mxfe_clocks(self, ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None) -> None:
        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        for group in range(2):
            lmx2594_idx = 3 + group
            self.init_lmx2594(lmx2594_idx, ignore_lock_failure=lmx2594_idx in ignore_lock_failure_of_lmx2594)

    def configure_mxfe(
        self,
        mxfe_idx: int,
        *,
        soft_reset: bool = False,
        hard_reset: bool = False,
        mxfe_init: bool = False,
        use_204b: bool = True,
        use_bg_cal: bool = False,
        ignore_crc_error: bool = False,
    ) -> bool:
        self._validate_group(mxfe_idx)
        if hard_reset:
            logger.info(f"asserting a reset pin of {self._css_addr}:AD9082-{mxfe_idx}")
            self.set_ad9082_hard_reset(mxfe_idx, True)

        if self.get_ad9082_hard_reset(mxfe_idx):
            logger.info(f"negating a reset pin of {self._css_addr}:AD9082-{mxfe_idx}")
            self.set_ad9082_hard_reset(mxfe_idx, False)

        self.ad9082[mxfe_idx].initialize(
            reset=soft_reset, link_init=mxfe_init, use_204b=use_204b, use_bg_cal=use_bg_cal
        )
        return self.check_link_status(mxfe_idx, mxfe_init, ignore_crc_error)

    def dump_channel(self, group: int, line: int, channel: int) -> Dict[str, Any]:
        self._validate_channel(group, line, channel)
        return {
            "fnco_freq": self.get_dac_fnco(group, line, channel),
        }

    def dump_line(self, group: int, line: int) -> Dict[str, Any]:
        self._validate_line(group, line)
        r: Dict[str, Any] = {}
        r["channels"] = {
            ch: self.dump_channel(group, line, ch) for ch in range(self.get_num_channels_of_line(group, line))
        }
        r["cnco_freq"] = self.get_dac_cnco(group, line)
        if (group, line) in self._LO_IDX:
            r["lo_freq"] = self.get_lo_multiplier(group, line) * 100_000_000
        if (group, line) in self._VATT_IDX:
            r["sideband"] = self.get_sideband(group, line)
            vatt = self.get_vatt_carboncopy(group, line)
            if vatt is not None:
                r["vatt"] = vatt
        # if (group, line) in self._RFSWITCH_NAME:
        #     r["rfswitch"] = "block" if self.is_blocked_line(group, line) else "pass"
        return r

    def dump_rchannel(self, group: int, rline: str, rchannel: int) -> Dict[str, Any]:
        self._validate_rchannel(group, rline, rchannel)
        return {
            "fnco_freq": self.get_adc_fnco(group, rline, rchannel),
        }

    def dump_rline(self, group: int, rline: str) -> Dict[str, Any]:
        self._validate_rline(group, rline)
        r: Dict[str, Any] = {}
        # if (group, rline) in self._RFSWITCH_NAME:
        #     r["rfswitch"] = "loop" if self.is_loopedback_read(group) else "open"
        if (group, rline) in self._LO_IDX:
            r["lo_freq"] = self.get_lo_multiplier(group, rline) * 100_000_000
        r["cnco_freq"] = (self.get_adc_cnco(group, rline),)
        r["channels"] = {
            rch: self.dump_rchannel(group, rline, rch) for rch in range(self.get_num_rchannels_of_rline(group, rline))
        }
        return r
