import logging
import socket
from pathlib import Path
from typing import Collection, Dict, Mapping, Set, Tuple, Union

from quel_ic_config.exstickge_sock_client import LsiKindId, _ExstickgeSockClientBase
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemGpioMixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemRoot,
)
from quel_ic_config.quel1_config_subsystem_tempctrl import Quel1ConfigSubsystemTempctrlMixin
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption, Quel1Feature

logger = logging.getLogger(__name__)


class ExstickgeSockClientQuel1seProto11(_ExstickgeSockClientBase):
    _AD9082_IF_0 = LsiKindId.AD9082
    _ADRF6780_IF_0 = LsiKindId.ADRF6780
    _ADRF6780_IF_1 = LsiKindId.ADRF6780 + 1
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
        # Notes: ExStickGE names ADRF6780 and LMX2594 as in the same way as the conventional MEE board.
        LsiKindId.ADRF6780: {
            0: (_ADRF6780_IF_0, 0),
            1: (_ADRF6780_IF_0, 1),
            2: (_ADRF6780_IF_0, 2),
            3: (_ADRF6780_IF_0, 3),
            6: (_ADRF6780_IF_1, 0),
            7: (_ADRF6780_IF_1, 1),
            5: (_ADRF6780_IF_1, 2),
            4: (_ADRF6780_IF_1, 3),
        },
        LsiKindId.LMX2594: {
            0: (_LMX2594_IF_0, 0),
            1: (_LMX2594_IF_0, 1),
            2: (_LMX2594_IF_0, 2),
            3: (_LMX2594_IF_0, 3),
            6: (_LMX2594_IF_0, 4),
            7: (_LMX2594_IF_1, 0),
            5: (_LMX2594_IF_1, 1),
            4: (_LMX2594_IF_1, 2),
            8: (_LMX2594_IF_1, 3),
            9: (_LMX2594_IF_1, 4),
        },
        LsiKindId.AD5328: {
            0: (_AD5328_IF_0, 0),
            1: (_AD5328_IF_1, 0),
        },
        LsiKindId.GPIO: {0: (_GPIO_IF_0, 0), 1: (_GPIO_IF_1, 1)},
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


class Quel1seProto11ConfigSubsystem(
    Quel1ConfigSubsystemRoot,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemGpioMixin,
    Quel1ConfigSubsystemTempctrlMixin,
):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE: str = "quel-1se-proto11.json"
    _NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 10,
        "adrf6780": 8,
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
        (1, 0): (1, 0),
        (1, 1): (1, 1),
        (1, 2): (1, 2),
        (1, 3): (1, 3),
    }

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (0, "m"): (0, 2),
        (1, "r"): (1, 3),
        (1, "m"): (1, 2),
    }

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (1, 0),
        (0, 2): (2, 0),
        (0, 3): (3, 0),
        (1, 0): (4, 0),
        (1, 1): (5, 0),
        (1, 2): (6, 0),
        (1, 3): (7, 0),
        (0, "r"): (0, 1),
        (0, "m"): (1, 1),
        (1, "r"): (4, 1),
        (1, "m"): (5, 1),
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

    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {
        (0, "r"): (5,),
        (0, "m"): (4,),
        (1, "r"): (5,),
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
        return ExstickgeSockClientQuel1seProto11(self._css_addr, port, timeout, sender_limit_by_binding)

    def configure_peripherals(
        self,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
        available_mixer_boards: Union[Collection[int], None] = None,
    ) -> None:
        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = {}

        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        if available_mixer_boards is None:
            available_mixer_boards = {0, 1}

        for i in range(2):
            self.init_gpio(i)
        self.gpio_helper[1].write_field(0, b02=True, b03=True)  # release reset of ADRF6780s
        self.gpio_helper[1].flush()

        for i in range(2):
            if (i == 0 and 0 in available_mixer_boards) or (i == 1 and 1 in available_mixer_boards):
                self.init_ad5328(i)

        for i in range(0, 8):
            if (i in {0, 1, 2, 3} and 0 in available_mixer_boards) or (
                i in {4, 5, 6, 7} and 1 in available_mixer_boards
            ):
                self.init_adrf6780(i, ignore_id_mismatch=i in ignore_access_failure_of_adrf6780)
                self.init_lmx2594(i, ignore_lock_failure=i in ignore_lock_failure_of_lmx2594)

    def configure_all_mxfe_clocks(self, ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None) -> None:
        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        for group in range(2):
            lmx2594_idx = 8 + group
            self.init_lmx2594(lmx2594_idx, ignore_lock_failure=lmx2594_idx in ignore_lock_failure_of_lmx2594)

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
            logger.warning(
                f"QuEL-1 ({self._css_addr}) does not support hardware reset of AD9082-{mxfe_idx}, "
                "conducts software reset instead."
            )
            soft_reset = True

        self.ad9082[mxfe_idx].initialize(
            reset=soft_reset, link_init=mxfe_init, use_204b=use_204b, use_bg_cal=use_bg_cal
        )
        return self.check_link_status(mxfe_idx, mxfe_init, ignore_crc_error)
