import logging
from pathlib import Path
from typing import Collection, Dict, Mapping, Set, Tuple, Union, cast

from quel_ic_config.exstickge_coap_client import _ExstickgeCoapClientBase
from quel_ic_config.exstickge_proxy import LsiKindId
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemRoot,
)
from quel_ic_config.quel1_config_subsystem_tempctrl import Quel1ConfigSubsystemTempctrlMixin
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption, Quel1Feature

logger = logging.getLogger(__name__)


class ExstickgeCoapClientAdda(_ExstickgeCoapClientBase):
    _URI_MAPPINGS: Mapping[Tuple[LsiKindId, int], str] = {
        (LsiKindId.AD9082, 0): "adda/mxfe_0",
        (LsiKindId.AD9082, 1): "adda/mxfe_1",
        (LsiKindId.LMX2594, 0): "adda/pll_0",
        (LsiKindId.LMX2594, 1): "adda/pll_1",
    }

    def __init__(
        self,
        target_addr: str,
        target_port: int = _ExstickgeCoapClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase._DEFAULT_RESPONSE_TIMEOUT,
    ):
        super().__init__(target_addr, target_port, timeout)


class Quel1seAddaConfigSubsystem(
    Quel1ConfigSubsystemRoot,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemTempctrlMixin,
):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE = "quel-1se-adda.json"
    _NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 2,
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

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (0, "m"): (0, 2),
        (1, "r"): (1, 3),
        (1, "m"): (1, 2),
    }

    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {
        (0, "r"): (5,),
        (0, "m"): (4,),
        (1, "r"): (5,),
        (1, "m"): (4,),
    }

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {}
    _MIXER_IDX: Dict[Tuple[int, int], int] = {}
    _VATT_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {}
    _RFSWITCH_NAME: Dict[Tuple[int, Union[int, str]], Tuple[int, str]] = {}
    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]] = {}

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

    def _create_exstickge_proxy(
        self, port: int, timeout: float, sender_limit_by_binding: bool
    ) -> _ExstickgeCoapClientBase:
        # Notes: port will be available later.
        # Notes: sender_limit_by_binding may be available later.
        return ExstickgeCoapClientAdda(self._css_addr, port, timeout)

    def configure_peripherals(
        self,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        _ = ignore_access_failure_of_adrf6780  # not used
        _ = ignore_lock_failure_of_lmx2594  # not used

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

        if self.get_ad9082_hard_reset(mxfe_idx):
            logger.info(f"negating a reset pin of {self._css_addr}:AD9082-{mxfe_idx}")
            self.set_ad9082_hard_reset(mxfe_idx, False)

        self.ad9082[mxfe_idx].initialize(
            reset=soft_reset, link_init=mxfe_init, use_204b=use_204b, use_bg_cal=use_bg_cal
        )
        return self.check_link_status(mxfe_idx, mxfe_init, ignore_crc_error)
