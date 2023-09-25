import logging
import socket
from pathlib import Path
from typing import Collection, Dict, Mapping, Tuple, Union

from quel_ic_config.exstickge_proxy import LsiKindId, _ExstickgeProxyBase
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemRfswitch,
    Quel1ConfigSubsystemRoot,
)
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption

logger = logging.getLogger(__name__)


class ExstickgeProxyQuel1(_ExstickgeProxyBase):
    _AD9082_IF_0 = LsiKindId.AD9082
    _ADRF6780_IF_0 = LsiKindId.ADRF6780
    _ADRF6780_IF_1 = LsiKindId.ADRF6780 + 1
    _LMX2594_IF_0 = LsiKindId.LMX2594
    _LMX2594_IF_1 = LsiKindId.LMX2594 + 1
    _AD5328_IF_0 = LsiKindId.AD5328
    _GPIO_IF_0 = LsiKindId.GPIO

    _SPIIF_MAPPINGS: Mapping[LsiKindId, Mapping[int, Tuple[int, int]]] = {
        LsiKindId.AD9082: {
            0: (_AD9082_IF_0, 0),
            1: (_AD9082_IF_0, 1),
        },
        LsiKindId.ADRF6780: {
            0: (_ADRF6780_IF_0, 0),
            1: (_ADRF6780_IF_0, 1),
            2: (_ADRF6780_IF_0, 2),
            3: (_ADRF6780_IF_0, 3),
            4: (_ADRF6780_IF_1, 0),
            5: (_ADRF6780_IF_1, 1),
            6: (_ADRF6780_IF_1, 2),
            7: (_ADRF6780_IF_1, 3),
        },
        LsiKindId.LMX2594: {
            0: (_LMX2594_IF_0, 0),
            1: (_LMX2594_IF_0, 1),
            2: (_LMX2594_IF_0, 2),
            3: (_LMX2594_IF_0, 3),
            4: (_LMX2594_IF_0, 4),
            5: (_LMX2594_IF_1, 0),
            6: (_LMX2594_IF_1, 1),
            7: (_LMX2594_IF_1, 2),
            8: (_LMX2594_IF_1, 3),
            9: (_LMX2594_IF_1, 4),
        },
        LsiKindId.AD5328: {
            0: (_AD5328_IF_0, 0),
        },
        LsiKindId.GPIO: {
            0: (_GPIO_IF_0, 0),
        },
    }

    def __init__(
        self,
        target_address,
        target_port=16384,
        timeout: float = 2.0,
        receiver_limit_by_binding: bool = False,
        sock: Union[socket.socket, None] = None,
    ):
        super().__init__(target_address, target_port, timeout, receiver_limit_by_binding, sock)


class Quel1ConfigSubsystem(
    Quel1ConfigSubsystemRoot,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemRfswitch,
):
    __slots__ = ()

    DEFAULT_CONFIG_JSONFILE: str = "quel-1.json"
    NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 10,
        "adrf6780": 8,
        "ad5328": 1,
        "gpio": 1,
    }

    _DAC_IDX: Dict[Tuple[int, int], int] = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 3,
        (1, 1): 2,
        (1, 2): 1,
        (1, 3): 0,
    }

    _ADC_IDX: Dict[Tuple[int, str], int] = {
        (0, "r"): 3,
        (0, "m"): 2,
        (1, "r"): 3,
        (1, "m"): 2,
    }

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Union[int, None]] = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 7,
        (1, 1): 6,
        (1, 2): 5,
        (1, 3): 4,
        (0, "r"): 0,
        (0, "m"): 1,
        (1, "r"): 7,
        (1, "m"): 6,
    }

    _MIXER_IDX: Dict[Tuple[int, int], Union[int, None]] = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 7,
        (1, 1): 6,
        (1, 2): 5,
        (1, 3): 4,
    }

    _VATT_IDX: Dict[Tuple[int, int], Union[Tuple[int, int], None]] = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (0, 2),
        (0, 3): (0, 3),
        (1, 0): (0, 7),
        (1, 1): (0, 6),
        (1, 2): (0, 5),
        (1, 3): (0, 4),
    }

    _ADC_CH_IDX: Dict[Tuple[int, str], int] = {
        (0, "r"): 5,
        (0, "m"): 4,
        (1, "r"): 5,
        (1, "m"): 4,
    }

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        config_path: Path,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        port: int = 16384,
        timeout: float = 0.5,
        sender_limit_by_binding: bool = False,
    ):
        Quel1ConfigSubsystemRoot.__init__(
            self, css_addr, boxtype, config_path, config_options, port, timeout, sender_limit_by_binding
        )
        self._construct_ad9082()
        self._construct_lmx2594()
        self._construct_adrf6780()
        self._construct_ad5328()
        self._construct_rfswitch(0)

    def _create_exstickge_proxy(self, port: int, timeout: float, sender_limit_by_binding: bool) -> _ExstickgeProxyBase:
        return ExstickgeProxyQuel1(self._css_addr, port, timeout, sender_limit_by_binding)

    def configure_peripherals(self) -> None:
        self.init_rfswitch()

        self.init_ad5328(0)
        for i in range(0, 8):
            self.init_adrf6780(i)
            is_locked = self.init_lmx2594(i)
            if not is_locked:
                raise RuntimeError(f"failed to lock PLL of {self._css_addr}:LMX2594-{i}")

    def configure_all_mxfe_clocks(self) -> None:
        for group in range(2):
            lmx2594_idx = 8 + group
            is_locked = self.init_lmx2594(lmx2594_idx)
            if not is_locked:
                raise RuntimeError(f"failed to lock PLL of {self._css_addr}:LMX2594-{lmx2594_idx}")

    def configure_mxfe(
        self,
        group: int,
        soft_reset: bool = True,
        hard_reset: bool = False,
        configure_clock: bool = False,
        ignore_crc_error: bool = False,
    ) -> bool:
        self._validate_group_and_line_out(group, 0)

        if configure_clock:
            lmx2594_idx = 8 + group
            is_locked = self.init_lmx2594(lmx2594_idx)
            if not is_locked:
                raise RuntimeError(f"failed to lock PLL of {self._css_addr}:LMX2594-{lmx2594_idx}")

        if hard_reset:
            logger.warning(
                f"QuEL-1 ({self._css_addr}) does not support hardware reset of AD9082-{group}, "
                "conducts software reset instead."
            )
            soft_reset = True

        self.ad9082[group].initialize(reset=soft_reset)
        link_valid = self.ad9082[group].check_link_status(ignore_crc_error=ignore_crc_error)
        if not link_valid:
            logger.warning("{self._css_addr}:AD9082-{group} failed to link-up")
        return link_valid
