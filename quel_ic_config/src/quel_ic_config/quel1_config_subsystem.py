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
    Quel1ConfigSubsystemNoRfswitch,
    Quel1ConfigSubsystemRfswitch,
    Quel1ConfigSubsystemRoot,
)
from quel_ic_config.quel1_config_subsystem_tempctrl import Quel1ConfigSubsystemTempctrlMixin
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption, Quel1Feature

logger = logging.getLogger(__name__)


class ExstickgeSockClientQuel1(_ExstickgeSockClientBase):
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
        target_port=_ExstickgeSockClientBase._DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase._DEFAULT_RESPONSE_TIMEOUT,
        receiver_limit_by_binding: bool = False,
        sock: Union[socket.socket, None] = None,
    ):
        super().__init__(target_address, target_port, timeout, receiver_limit_by_binding, sock)


class QuelMeeBoardConfigSubsystem(
    Quel1ConfigSubsystemRoot,
    Quel1ConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemTempctrlMixin,
):
    __slots__ = ()

    _DEFAULT_CONFIG_JSONFILE: str = "quel-1.json"

    # TODO: move "gpio" to the appropriate place.
    _NUM_IC: Dict[str, int] = {
        "ad9082": 2,
        "lmx2594": 10,
        "adrf6780": 8,
        "ad5328": 1,
        "gpio": 1,
    }

    _GROUPS: Set[int] = {0, 1}
    _MXFE_IDXS: Set[int] = {0, 1}

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

    def _create_exstickge_proxy(
        self, port: int, timeout: float, sender_limit_by_binding: bool
    ) -> _ExstickgeSockClientBase:
        return ExstickgeSockClientQuel1(self._css_addr, port, timeout, sender_limit_by_binding)

    def configure_peripherals(
        self,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = {}

        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        self.init_ad5328(0)
        for i in range(0, 8):
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
        hard_reset: bool = False,
        soft_reset: bool = False,
        mxfe_init: bool = False,
        use_204b: bool = False,
        use_bg_cal: bool = True,
        ignore_crc_error: bool = False,
    ) -> bool:
        self._validate_mxfe(mxfe_idx)

        if hard_reset:
            logger.warning(
                f"QuEL-1 ({self._css_addr}) does not support hardware reset of AD9082-#{mxfe_idx}, "
                "conducts software reset instead."
            )
            soft_reset = True

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
        return r


class QubeConfigSubsystem(
    QuelMeeBoardConfigSubsystem,
):
    __slots__ = ()

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

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (1, 0),
        (0, 2): (2, 0),
        (0, 3): (3, 0),
        (1, 0): (7, 0),
        (1, 1): (6, 0),
        (1, 2): (5, 0),
        (1, 3): (4, 0),
        (0, "r"): (0, 1),
        (0, "m"): (1, 1),
        (1, "r"): (7, 1),
        (1, "m"): (6, 1),
    }

    _MIXER_IDX: Dict[Tuple[int, int], int] = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 7,
        (1, 1): 6,
        (1, 2): 5,
        (1, 3): 4,
    }

    _VATT_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (0, 2),
        (0, 3): (0, 3),
        (1, 0): (0, 7),
        (1, 1): (0, 6),
        (1, 2): (0, 5),
        (1, 3): (0, 4),
    }

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
        super().__init__(
            css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )


class Quel1ConfigSubsystem(QubeConfigSubsystem, Quel1ConfigSubsystemRfswitch):
    __slots__ = ()

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
        QubeConfigSubsystem.__init__(
            self, css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )
        self._construct_rfswitch(0)

    def configure_peripherals(
        self,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        # Notes: init_rfswitch() should be called at the head of configure_peripherals() to avoid the leakage of sprious
        #        during the initialization of RF components.
        self.init_rfswitch()
        super().configure_peripherals(ignore_access_failure_of_adrf6780, ignore_lock_failure_of_lmx2594)

    def dump_line(self, group: int, line: int) -> Dict[str, Any]:
        r = super().dump_line(group, line)
        if (group, line) in self._RFSWITCH_NAME:
            r["rfswitch"] = "block" if self.is_blocked_line(group, line) else "pass"
        return r

    def dump_rline(self, group: int, rline: str) -> Dict[str, Any]:
        r = super().dump_rline(group, rline)
        if (group, rline) in self._RFSWITCH_NAME:
            r["rfswitch"] = "loop" if self.is_blocked_line(group, rline) else "open"
        return r


class QubeOuTypeAConfigSubsystem(QubeConfigSubsystem, Quel1ConfigSubsystemNoRfswitch):
    __slots__ = ()

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (1, "r"): (1, 3),
    }

    # TODO: will be replaced with a parser method
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {
        (0, "r"): (5,),
        (1, "r"): (5,),
    }

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
        if boxtype != Quel1BoxType.QuBE_OU_TypeA:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(
            css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )


class QubeOuTypeBConfigSubsystem(QubeConfigSubsystem, Quel1ConfigSubsystemNoRfswitch):
    __slots__ = ()

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {}

    # TODO: will be replaced with a parser method
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {}

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
        if boxtype != Quel1BoxType.QuBE_OU_TypeB:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(
            css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )


class Quel1TypeAConfigSubsystem(Quel1ConfigSubsystem):
    __slots__ = ()

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (0, "m"): (0, 2),
        (1, "r"): (1, 3),
        (1, "m"): (1, 2),
    }

    # TODO: will be replaced with a parser method
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {
        (0, "r"): (5,),
        (0, "m"): (4,),
        (1, "r"): (5,),
        (1, "m"): (4,),
    }

    _RFSWITCH_NAME: Dict[Tuple[int, Union[int, str]], Tuple[int, str]] = {
        (0, 0): (0, "path0"),
        (0, 1): (0, "path1"),
        (0, 2): (0, "path2"),
        (0, 3): (0, "path3"),
        (0, "r"): (0, "path0"),
        (0, "m"): (0, "monitor"),
        (1, 0): (1, "path0"),
        (1, 1): (1, "path1"),
        (1, 2): (1, "path2"),
        (1, 3): (1, "path3"),
        (1, "r"): (1, "path0"),
        (1, "m"): (1, "monitor"),
    }

    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]] = {
        (0, 0): (0, "r"),
        (1, 0): (1, "r"),
    }

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
        if boxtype not in {Quel1BoxType.QuBE_RIKEN_TypeA, Quel1BoxType.QuEL1_TypeA}:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(
            css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )


class Quel1TypeBConfigSubsystem(Quel1ConfigSubsystem):
    __slots__ = ()

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "m"): (0, 2),
        (1, "m"): (1, 2),
    }

    # TODO: will be replaced with a parser method
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {
        (0, "m"): (4,),
        (1, "m"): (4,),
    }

    _RFSWITCH_NAME: Dict[Tuple[int, Union[int, str]], Tuple[int, str]] = {
        (0, 0): (0, "path0"),
        (0, 1): (0, "path1"),
        (0, 2): (0, "path2"),
        (0, 3): (0, "path3"),
        (0, "m"): (0, "monitor"),
        (1, 0): (1, "path0"),
        (1, 1): (1, "path1"),
        (1, 2): (1, "path2"),
        (1, 3): (1, "path3"),
        (1, "m"): (1, "monitor"),
    }

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
        if boxtype not in {Quel1BoxType.QuBE_RIKEN_TypeB, Quel1BoxType.QuEL1_TypeB}:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(
            css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )


class Quel1NecConfigSubsystem(
    QuelMeeBoardConfigSubsystem,
    Quel1ConfigSubsystemNoRfswitch,
):
    __slots__ = ()

    _GROUPS: Set[int] = {0, 1, 2, 3}

    _DAC_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (0, 2),
        (1, 0): (0, 1),
        (1, 1): (0, 3),
        (2, 0): (1, 2),
        (2, 1): (1, 0),
        (3, 0): (1, 3),
        (3, 1): (1, 1),
    }

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (2, 0),
        (1, 0): (1, 0),
        (1, 1): (3, 0),
        (2, 0): (6, 0),
        (2, 1): (4, 0),
        (3, 0): (7, 0),
        (3, 1): (5, 0),
        (0, "r"): (0, 1),
        (1, "r"): (1, 1),
        (2, "r"): (6, 1),
        (3, "r"): (7, 1),
    }

    _MIXER_IDX: Dict[Tuple[int, int], int] = {
        (0, 0): 0,
        (0, 1): 2,
        (1, 0): 1,
        (1, 1): 3,
        (2, 0): 6,
        (2, 1): 4,
        (3, 0): 7,
        (3, 1): 5,
    }

    _VATT_IDX: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (0, 0),
        (0, 1): (0, 2),
        (1, 0): (0, 1),
        (1, 1): (0, 3),
        (2, 0): (0, 6),
        (2, 1): (0, 4),
        (3, 0): (0, 7),
        (3, 1): (0, 5),
    }

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (1, "r"): (0, 2),
        (2, "r"): (1, 3),
        (3, "r"): (1, 2),
    }

    # TODO: will be replaced with a parser method
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]] = {
        (0, "r"): (5,),
        (1, "r"): (4,),
        (2, "r"): (5,),
        (3, "r"): (4,),
    }

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
        if boxtype != Quel1BoxType.QuEL1_NEC:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(
            css_addr, boxtype, features, config_path, config_options, port, timeout, sender_limit_by_binding
        )
