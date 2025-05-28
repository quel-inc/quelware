import logging
from typing import Any, Collection, Dict, Set, Tuple, Union, cast

from quel_ic_config.exstickge_coap_client import _ExstickgeCoapClientBase, get_exstickge_server_info
from quel_ic_config.exstickge_proxy import LsiKindId
from quel_ic_config.quel1_config_subsystem_common import (
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemMixerboardGpioMixin,
    Quel1ConfigSubsystemPathselectorboardGpioMixin,
    Quel1ConfigSubsystemRoot,
    Quel1GenericConfigSubsystemAd9082Mixin,
)
from quel_ic_config.quel_config_common import Quel1BoxType
from quel_ic_config.quel_ic import Ad9082Generic

logger = logging.getLogger(__name__)


class Ad9082Quel1se(Ad9082Generic):
    def _reset_pin_ctrl_cb(self, level: int) -> Tuple[bool]:
        proxy = cast(_ExstickgeCoapClientBase, self.proxy)
        proxy.write_reset(LsiKindId.AD9082, self.idx, level)
        return (True,)


class Quel1seConfigSubsystemAd9082Mixin(Quel1GenericConfigSubsystemAd9082Mixin):
    def _construct_ad9082(self):
        self._ad9082: tuple[Ad9082Generic, ...] = tuple(
            Ad9082Quel1se(self._proxy, idx) for idx in range(self._NUM_IC["ad9082"])
        )
        self.allow_dual_modulus_nco = True


class _Quel1seConfigSubsystemBase(
    Quel1ConfigSubsystemRoot,
    Quel1seConfigSubsystemAd9082Mixin,
    Quel1ConfigSubsystemLmx2594Mixin,
    Quel1ConfigSubsystemAd6780Mixin,
    Quel1ConfigSubsystemAd5328Mixin,
    Quel1ConfigSubsystemMixerboardGpioMixin,
    Quel1ConfigSubsystemPathselectorboardGpioMixin,
):
    __slots__ = ()

    _PROXY_CLASSES: Tuple[type, ...]

    _MXFE_IDXS: Set[int] = {0, 1}
    _LMX2594_OF_MXFES: Tuple[int, ...] = (0, 1)

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = _ExstickgeCoapClientBase.DEFAULT_PORT,
        timeout: float = _ExstickgeCoapClientBase.DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        Quel1ConfigSubsystemRoot.__init__(self, css_addr, boxtype, port, timeout, sender_limit_by_binding)

    def initialize(self) -> None:
        super().initialize()
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
            for proxy_cls in self._PROXY_CLASSES:
                if not issubclass(proxy_cls, _ExstickgeCoapClientBase):
                    raise AssertionError(f"invalid proxy class {proxy_cls.__name__} is in the list")
                if proxy_cls.matches(coap_boxtype, coap_firmware_version):
                    return proxy_cls(self._css_addr, port, timeout)
            else:
                raise RuntimeError(
                    f"unsupported CoAP firmware is running on {self._css_addr}: {coap_boxtype}:{coap_firmware_version}"
                )
        else:
            raise RuntimeError(f"CoAP server is not available on {self._css_addr}")

    def configure_peripherals(
        self,
        param: dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = set()
        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = set()

        # Notes: close all RF switches at first
        for i in range(self._NUM_IC["pathselectorboard_gpio"]):
            self.init_pathselectorboard_gpio(i, param["pathselectorboard_gpio"][i])

        # Notes: release reset of CPLDs on all the peripheral board
        proxy = cast(_ExstickgeCoapClientBase, self._proxy)
        for board in proxy.available_boards_with_cpld:
            if not proxy.read_board_active(board):
                logger.info(f"releasing reset of board '{board.value}'")
                proxy.write_board_active(board, True)
            else:
                logger.info(f"board '{board.value}' is already activated")

        # Notes: initialize ICs on mixer board 0 for RF
        for i in range(self._NUM_IC["ad5328"]):
            self.init_ad5328(i, param["ad5328"][i])

        for i in range(self._NUM_IC["adrf6780"]):
            proxy.write_reset(LsiKindId.ADRF6780, i, 1)
            self.init_adrf6780(i, param["adrf6780"][i], ignore_id_mismatch=i in ignore_access_failure_of_adrf6780)

        for i in range(2, self._NUM_IC["lmx2594"]):
            self.init_lmx2594(i, param["lmx2594"][i], ignore_lock_failure=i in ignore_lock_failure_of_lmx2594)

    def configure_all_mxfe_clocks(
        self, param: dict[str, Any], ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None
    ) -> None:
        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        for group in range(2):
            lmx2594_idx = 0 + group
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
        use_204b: bool = True,
        use_bg_cal: bool = False,
        ignore_crc_error: bool = False,
    ) -> bool:
        self._validate_group(mxfe_idx)

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
