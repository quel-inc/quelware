import logging
import time
from pathlib import Path
from typing import Any, Collection, Dict, Final, Mapping, Set, Tuple, Union

from quel_ic_config.exstickge_sock_client import (
    AbstractLockKeeper,
    DummyLockKeeper,
    FileLockKeeper,
    LsiKindId,
    _ExstickgeSockClientBase,
)
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemNoRfswitch,
    Quel1ConfigSubsystemRfswitch,
    Quel1ConfigSubsystemRoot,
    Quel1GenericConfigSubsystemAd9082Mixin,
)
from quel_ic_config.quel1_config_subsystem_tempctrl import Quel1ConfigSubsystemTempctrlMixin
from quel_ic_config.quel_config_common import _DEFAULT_LOCK_DIRECTORY, Quel1BoxType
from quel_ic_config.quel_ic import Ad9082Generic

_DEFAULT_SOCKET_SERVER_DETECTION_TIMEOUT: Final[float] = 15.0  # [s]

logger = logging.getLogger(__name__)


class AbstractExstickgeSockClientQuel1(_ExstickgeSockClientBase):
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
        target_port=_ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        receiver_limit_by_binding: bool = False,
    ):
        super().__init__(target_address, target_port, timeout, receiver_limit_by_binding)


class ExstickgeSockClientQuel1WithDummyLock(AbstractExstickgeSockClientQuel1):
    def _create_lockkeeper(self) -> AbstractLockKeeper:
        t = DummyLockKeeper(target=self._target)
        t.activate()
        return t


class ExstickgeSockClientQuel1WithFileLock(AbstractExstickgeSockClientQuel1):
    def __init__(
        self,
        target_address,
        target_port=_ExstickgeSockClientBase.DEFAULT_PORT,
        lock_directory: Path = _DEFAULT_LOCK_DIRECTORY,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        receiver_limit_by_binding: bool = False,
    ):
        self._lock_directory: Path = lock_directory
        super().__init__(target_address, target_port, timeout, receiver_limit_by_binding)

    def _create_lockkeeper(self) -> AbstractLockKeeper:
        t = FileLockKeeper(target=self._target, lock_directory=self._lock_directory)
        t.activate()
        return t


class Ad9082Quel1(Ad9082Generic):
    def _reset_pin_ctrl_cb(self, level: int) -> Tuple[bool]:
        logger.warning("hardware reset of AD9082 is not implemented")
        return (False,)


class Quel1ConfigSubsystemAd9082Mixin(Quel1GenericConfigSubsystemAd9082Mixin):
    def _construct_ad9082(self):
        self._ad9082: tuple[Ad9082Generic, ...] = tuple(
            Ad9082Quel1(self._proxy, idx) for idx in range(self._NUM_IC["ad9082"])
        )
        self.allow_dual_modulus_nco = True


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

    _GROUPS: Set[int]

    _MXFE_IDXS: Set[int] = {0, 1}
    _LMX2594_OF_MXFES: Tuple[int, ...] = (8, 9)

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        Quel1ConfigSubsystemRoot.__init__(self, css_addr, boxtype, port, timeout, sender_limit_by_binding)

    def initialize(self) -> None:
        super().initialize()
        self._construct_ad9082()
        self._construct_lmx2594()
        self._construct_adrf6780()
        self._construct_ad5328()

    def _create_exstickge_proxy(
        self, port: int, timeout: float, sender_limit_by_binding: bool
    ) -> _ExstickgeSockClientBase:
        # TODO: consider to accept user defined lock directory or not

        proxy = ExstickgeSockClientQuel1WithFileLock(
            self._css_addr, port, _DEFAULT_LOCK_DIRECTORY, timeout, sender_limit_by_binding
        )
        proxy.initialize()
        # Notes: check the availablity of some end-points of the server after taking the lock.
        #        MEE board has LMX2594[0] definitely.
        # Notes: lock should be acquired at the end of ExstickgeSockClientQuel1WithFileLock.__init__() if available.
        if proxy.has_lock:
            t0 = time.perf_counter()
            while time.perf_counter() < t0 + _DEFAULT_SOCKET_SERVER_DETECTION_TIMEOUT:
                if proxy.read_reg(LsiKindId.LMX2594, 0, 0x0000) is not None:
                    break
                time.sleep(3.0)
            else:
                raise RuntimeError(f"socket server is not available on {self._css_addr}")
        else:
            # Notes: BoxLockError will raise at _create_css_object()
            pass

        return proxy

    def configure_peripherals(
        self,
        param: Dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = {}

        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        self.init_ad5328(0, param["ad5328"][0])
        for i in range(0, 8):
            self.init_adrf6780(i, param["adrf6780"][i], ignore_id_mismatch=i in ignore_access_failure_of_adrf6780)
            self.init_lmx2594(i, param["lmx2594"][i], ignore_lock_failure=i in ignore_lock_failure_of_lmx2594)

    def configure_all_mxfe_clocks(
        self, param: Dict[str, Any], *, ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None
    ) -> None:
        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        for group in range(2):
            lmx2594_idx = 8 + group
            self.init_lmx2594(
                lmx2594_idx,
                param["lmx2594"][lmx2594_idx],
                ignore_lock_failure=lmx2594_idx in ignore_lock_failure_of_lmx2594,
            )

    def configure_mxfe(
        self,
        mxfe_idx: int,
        param: Dict[str, Any],
        *,
        hard_reset: bool = False,
        soft_reset: bool = False,
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
            hard_reset = False
            soft_reset = True

        self.ad9082[mxfe_idx].configure(
            param["ad9082"][mxfe_idx],
            hard_reset=hard_reset,
            soft_reset=soft_reset,
            use_204b=use_204b,
            use_bg_cal=use_bg_cal,
        )
        health, lglv, diag = self.check_link_status(mxfe_idx, True, ignore_crc_error)
        logger.log(lglv, diag)
        return health

    def reconnect_mxfe(
        self,
        mxfe_idx: int,
        *,
        ignore_crc_error: bool = False,
    ) -> bool:
        self._validate_mxfe(mxfe_idx)
        pll = self.lmx2594[self._LMX2594_OF_MXFES[mxfe_idx]]
        dr = pll.get_divider_ratio()[0]  # Notes: assuming that OUTA is used.
        se = pll.get_sync_enable()
        if dr == 0:
            raise RuntimeError(f"PLL of AD9082-#{mxfe_idx} is not activated")
        ref_clk = pll.get_lo_multiplier() * 100_000_000 // dr
        if se:
            ref_clk *= 4

        self.ad9082[mxfe_idx].reconnect(ref_clk)
        health, lglv, diag = self.check_link_status(mxfe_idx, False, ignore_crc_error)
        logger.log(lglv, diag)
        return health

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
        port: int = _ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        super().__init__(css_addr, boxtype, port, timeout, sender_limit_by_binding)


class Quel1ConfigSubsystem(QubeConfigSubsystem, Quel1ConfigSubsystemRfswitch):
    __slots__ = ()

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        QubeConfigSubsystem.__init__(self, css_addr, boxtype, port, timeout, sender_limit_by_binding)

    def initialize(self) -> None:
        super().initialize()
        self._construct_rfswitch(0)

    def configure_peripherals(
        self,
        param: dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        # Notes: init_rfswitch() should be called at the head of configure_peripherals() to avoid the leakage of sprious
        #        during the initialization of RF components.
        self.init_rfswitch(param["gpio"][0])
        super().configure_peripherals(
            param,
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )

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
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]] = {
        (0, "r"): ((0, 5),),
        (1, "r"): ((1, 5),),
    }

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        if boxtype != Quel1BoxType.QuBE_OU_TypeA:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(css_addr, boxtype, port, timeout, sender_limit_by_binding)


class QubeOuTypeBConfigSubsystem(QubeConfigSubsystem, Quel1ConfigSubsystemNoRfswitch):
    __slots__ = ()

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {}

    # TODO: will be replaced with a parser method
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]] = {}

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        if boxtype != Quel1BoxType.QuBE_OU_TypeB:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(css_addr, boxtype, port, timeout, sender_limit_by_binding)


class Quel1TypeAConfigSubsystem(Quel1ConfigSubsystem):
    __slots__ = ()

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "r"): (0, 3),
        (0, "m"): (0, 2),
        (1, "r"): (1, 3),
        (1, "m"): (1, 2),
    }

    # TODO: will be replaced with a parser method
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]] = {
        (0, "r"): ((0, 5),),
        (0, "m"): ((0, 4),),
        (1, "r"): ((1, 5),),
        (1, "m"): ((1, 4),),
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
        port: int = _ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        if boxtype not in {Quel1BoxType.QuBE_RIKEN_TypeA, Quel1BoxType.QuEL1_TypeA}:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(css_addr, boxtype, port, timeout, sender_limit_by_binding)


class Quel1TypeBConfigSubsystem(Quel1ConfigSubsystem):
    __slots__ = ()

    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]] = {
        (0, "m"): (0, 2),
        (1, "m"): (1, 2),
    }

    # TODO: will be replaced with a parser method
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]] = {
        (0, "m"): ((0, 4),),
        (1, "m"): ((1, 4),),
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
        port: int = _ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        if boxtype not in {Quel1BoxType.QuBE_RIKEN_TypeB, Quel1BoxType.QuEL1_TypeB}:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(css_addr, boxtype, port, timeout, sender_limit_by_binding)


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
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]] = {
        (0, "r"): ((0, 5),),
        (1, "r"): ((0, 4),),
        (2, "r"): ((1, 5),),
        (3, "r"): ((1, 4),),
    }

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeSockClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeSockClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        if boxtype != Quel1BoxType.QuEL1_NEC:
            raise ValueError(f"invalid boxtype: {boxtype} for {self.__class__.__name__}")
        super().__init__(css_addr, boxtype, port, timeout, sender_limit_by_binding)
