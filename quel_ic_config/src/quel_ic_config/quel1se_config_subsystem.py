import logging
import time
from typing import Any, Collection, Dict, Set, Tuple, Union, cast

from quel_ic_config.exstickge_coap_client import _ExstickgeCoapClientBase, get_exstickge_server_info
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
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger(__name__)


class _Quel1seConfigSubsystemBase(
    Quel1ConfigSubsystemRoot,
    Quel1ConfigSubsystemAd9082Mixin,
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
                    f"unsupported CoAP firmware is running on exstickge: {coap_boxtype}:{coap_firmware_version}"
                )
        else:
            raise RuntimeError("no CoAP firmware is running on exstickge")

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

    def get_ad9082_hard_reset(self, mxfe_idx: int) -> bool:
        # Notes: re-consider better way
        proxy = cast(_ExstickgeCoapClientBase, self._proxy)
        v = proxy.read_reset(LsiKindId.AD9082, mxfe_idx)
        if v == 0:
            return True
        elif v == 1:
            return False
        else:
            raise RuntimeError(f"invalid value {v} for state of AD9082[{mxfe_idx}]'s hard_reset")

    def set_ad9082_hard_reset(self, mxfe_idx: int, value: bool) -> None:
        # Notes: re-consider better way
        proxy = cast(_ExstickgeCoapClientBase, self._proxy)
        proxy.write_reset(LsiKindId.AD9082, mxfe_idx, 0 if value else 1)

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

        if hard_reset:
            logger.info(f"asserting a reset pin of {self._css_addr}:AD9082-{mxfe_idx}")
            self.set_ad9082_hard_reset(mxfe_idx, True)
            time.sleep(0.01)

        if self.get_ad9082_hard_reset(mxfe_idx):
            logger.info(f"negating a reset pin of {self._css_addr}:AD9082-{mxfe_idx}")
            self.set_ad9082_hard_reset(mxfe_idx, False)

        self.ad9082[mxfe_idx].configure(
            param["ad9082"][mxfe_idx], reset=soft_reset, use_204b=use_204b, use_bg_cal=use_bg_cal
        )
        return self.check_link_status(mxfe_idx, True, ignore_crc_error)

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
        return self.check_link_status(mxfe_idx, False, ignore_crc_error)

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
