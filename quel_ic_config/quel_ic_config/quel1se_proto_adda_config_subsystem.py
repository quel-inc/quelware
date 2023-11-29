import logging
import socket
from pathlib import Path
from typing import Collection, Dict, Mapping, Set, Tuple, Union

from quel_ic_config.exstickge_proxy import LsiKindId, _ExstickgeProxyBase
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemGpioMixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemRoot,
)
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption

logger = logging.getLogger(__name__)


class ExstickgeProxyQuel1SeProtoAdda(_ExstickgeProxyBase):
    _AD9082_IF_0 = LsiKindId.AD9082
    _LMX2594_IF_1 = LsiKindId.LMX2594 + 1
    _GPIO_IF_0 = LsiKindId.GPIO
    _GPIO_IF_1 = LsiKindId.GPIO + 2

    _SPIIF_MAPPINGS: Mapping[LsiKindId, Mapping[int, Tuple[int, int]]] = {
        LsiKindId.AD9082: {
            0: (_AD9082_IF_0, 0),
            1: (_AD9082_IF_0, 1),
        },
        LsiKindId.LMX2594: {
            0: (_LMX2594_IF_1, 3),
            1: (_LMX2594_IF_1, 4),
        },
        LsiKindId.GPIO: {
            0: (_GPIO_IF_0, 0),
            1: (_GPIO_IF_1, 0),
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


class Quel1SeProtoAddaConfigSubsystemGpioMixin(Quel1ConfigSubsystemGpioMixin):
    _GPIO_FOR_AD9082_HARDRESET: Dict[int, Tuple[int, str]]

    def _get_idx_for_ad9082_hard_reset(self, mxfe_idx: int) -> Tuple[int, str]:
        if mxfe_idx in self._GPIO_FOR_AD9082_HARDRESET:
            return self._GPIO_FOR_AD9082_HARDRESET[mxfe_idx]
        else:
            raise ValueError(f"no AD9082 hard-reset pin for group:{mxfe_idx}")

    def get_ad9082_hard_reset(self, mxfe_idx: int) -> bool:
        blk_idx, bit_name = self._get_idx_for_ad9082_hard_reset(mxfe_idx)
        return not getattr(self.gpio_helper[blk_idx].read_reg(0), bit_name)

    def set_ad9082_hard_reset(self, mxfe_idx: int, reset_state: bool) -> None:
        """
        :param mxfe_idx: index of MxFE IC or of group.
        :param reset_state: True for asserting reset.
        :return: None
        """
        blk_idx, bit_name = self._get_idx_for_ad9082_hard_reset(mxfe_idx)
        self.gpio_helper[blk_idx].write_field(0, **{bit_name: not reset_state})
        self.gpio_helper[blk_idx].flush()


class Quel1SeProtoAddaConfigSubsystem(
    Quel1ConfigSubsystemRoot,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1SeProtoAddaConfigSubsystemGpioMixin,
):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE = "quel-1se-proto-adda.json"
    _NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 2,
        "gpio": 2,
    }

    _GROUPS: Set[int] = {0, 1}

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

    _LO_IDX: Dict[Tuple[int, Union[int, str]], int] = {}
    _MIXER_IDX: Dict[Tuple[int, int], int] = {}
    _VATT_IDX: Dict[Tuple[int, int], int] = {}
    _RFSWITCH_NAME: Dict[Tuple[int, Union[int, str]], Tuple[int, str]] = {}

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        config_path: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,  # TODO: should be elaborated.
        port: int = 16384,
        timeout: float = 0.5,
        sender_limit_by_binding: bool = False,
    ):
        Quel1ConfigSubsystemRoot.__init__(
            self, css_addr, boxtype, config_path, config_options, port, timeout, sender_limit_by_binding
        )
        self._construct_ad9082()
        self._construct_lmx2594()
        self._construct_gpio()

    def _create_exstickge_proxy(self, port: int, timeout: float, sender_limit_by_binding: bool) -> _ExstickgeProxyBase:
        return ExstickgeProxyQuel1SeProtoAdda(self._css_addr, port, timeout, sender_limit_by_binding)

    def configure_peripherals(self) -> None:
        for i in range(self._NUM_IC["gpio"]):
            self.init_gpio(i)

    def configure_all_mxfe_clocks(self) -> None:
        for group in range(2):
            lmx2594_id = 0 + group
            is_locked = self.init_lmx2594(lmx2594_id)
            if not is_locked:
                raise RuntimeError(f"failed to lock PLL of {self._css_addr}:LMX2594-{lmx2594_id}")

    def configure_mxfe(
        self,
        group: int,
        *,
        hard_reset: bool = False,
        soft_reset: bool = False,
        mxfe_init: bool = False,
        use_204b: bool = True,
        ignore_crc_error: bool = False,
    ) -> bool:
        self._validate_group(group)
        if hard_reset:
            logger.info(f"asserting a reset pin of {self._css_addr}:AD9082-{group}")
            self.set_ad9082_hard_reset(group, True)

        if self.get_ad9082_hard_reset(group):
            logger.info(f"negating a reset pin of {self._css_addr}:AD9082-{group}")
            self.set_ad9082_hard_reset(group, False)

        self.ad9082[group].initialize(reset=soft_reset, link_init=mxfe_init, use_204b=use_204b)
        link_valid = self.ad9082[group].check_link_status(ignore_crc_error=ignore_crc_error)
        if not link_valid:
            if mxfe_init:
                logger.warning(f"{self._css_addr}:AD9082-#{group} link-up failure")
            else:
                logger.warning(f"{self._css_addr}:AD9082-#{group} is not linked up yet")

        return link_valid
