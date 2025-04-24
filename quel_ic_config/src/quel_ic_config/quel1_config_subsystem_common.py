import copy
import logging
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Collection, Dict, Final, List, Mapping, Set, Tuple, Union

from quel_ic_config.ad5328 import Ad5328ConfigHelper
from quel_ic_config.ad9082 import LinkStatus, NcoFtw
from quel_ic_config.adrf6780 import Adrf6780ConfigHelper, Adrf6780LoSideband
from quel_ic_config.exstickge_sock_client import _ExstickgeProxyBase
from quel_ic_config.lmx2594 import Lmx2594ConfigHelper
from quel_ic_config.mixerboard_gpio import MixerboardGpioConfigHelper
from quel_ic_config.pathselectorboard_gpio import PathselectorboardGpioConfigHelper
from quel_ic_config.powerboard_pwm import PowerboardPwmConfigHelper
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1RuntimeOption
from quel_ic_config.quel_ic import (
    Ad5328,
    Ad7490,
    Ad9082Generic,
    Adrf6780,
    Lmx2594,
    MixerboardGpio,
    PathselectorboardGpio,
    PowerboardPwm,
    QubeRfSwitchArray,
    Quel1TypeARfSwitchArray,
    Quel1TypeBRfSwitchArray,
)
from quel_ic_config.rfswitcharray import AbstractRfSwitchArrayMixin, RfSwitchArrayConfigHelper

logger = logging.getLogger(__name__)


class Quel1ConfigSubsystemBaseSlot(metaclass=ABCMeta):
    __slots__ = (
        "_css_addr",
        "_boxtype",
        "_runtime_options",
        "_proxy",
        "_ad9082",
        "_lmx2594",
        "_lmx2594_helper",
        "_adrf6780",
        "_adrf6780_helper",
        "_ad5328",
        "_ad5328_helper",
        "_ad7490",
        "_rfswitch",
        "_rfswitch_gpio_idx",
        "_rfswitch_helper",
        "_mixerboard_gpio",
        "_mixerboard_gpio_helper",
        "_pathselectorboard_gpio",
        "_pathselectorboard_gpio_helper",
        "_powerboard_pwm",
        "_powerboard_pwm_helper",
        "_tempctrl_watcher",
        "_tempctrl_auto_start_at_linkup",
    )

    _VALID_IC_NAME: Set[str] = {
        "ad9082",
        "lmx2594",
        "adrf6780",
        "ad5328",
        "ad7490",
        "gpio",
        "mixerboard_gpio",
        "pathselectorboard_gpio",
        "powerboard_pwm",
    }

    _DEFAULT_CONFIG_JSONFILE: str
    _NUM_IC: Dict[str, int]
    _GROUPS: Set[int]
    _MXFE_IDXS: Set[int]
    _DAC_IDX: Dict[Tuple[int, int], Tuple[int, int]]
    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]]
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[Tuple[int, int], ...]]

    def __init__(self):
        # variable defined in the importer class
        self._css_addr: str
        self._boxtype: Quel1BoxType
        self._runtime_options: Set[Quel1RuntimeOption]
        self._proxy: _ExstickgeProxyBase

        # TODO: types of IC proxy will be reconsidered when controler proxy (e.g. _ExstickgeProxyBase) is generalized.
        self._ad9082: Tuple[Ad9082Generic, ...] = ()
        self._lmx2594: Tuple[Lmx2594, ...] = ()
        self._lmx2594_helper: Tuple[Lmx2594ConfigHelper, ...] = ()
        self._adrf6780: Tuple[Adrf6780, ...] = ()
        self._adrf6780_helper: Tuple[Adrf6780ConfigHelper, ...] = ()
        self._ad5328: Tuple[Ad5328, ...] = ()
        self._ad5328_helper: Tuple[Ad5328ConfigHelper, ...] = ()
        self._rfswitch_gpio_idx: int = 0
        self._rfswitch: Union[AbstractRfSwitchArrayMixin, None] = None
        self._rfswitch_helper: Union[RfSwitchArrayConfigHelper, None] = None

    @property
    def has_lock(self) -> bool:
        return self._proxy.has_lock

    def get_num_ic(self, ic_name: str) -> int:
        if ic_name in self._VALID_IC_NAME:
            return self._NUM_IC.get(ic_name, 0)
        else:
            raise ValueError(f"invalid name of ic: '{ic_name}'")

    def _set_runtime_option(self, option: Quel1RuntimeOption, value: bool) -> None:
        if value:
            self._runtime_options.add(option)
        else:
            if option in self._runtime_options:
                self._runtime_options.remove(option)

    def _get_runtime_option(self, option: Quel1RuntimeOption) -> bool:
        return option in self._runtime_options

    def get_all_groups(self) -> Set[int]:
        return self._GROUPS

    def _validate_group(self, group: int) -> None:
        if group not in self._GROUPS:
            raise ValueError(f"an invalid group: {group}")

    def get_all_mxfes(self) -> Set[int]:
        return self._MXFE_IDXS

    def _validate_mxfe(self, mxfe_idx: int) -> None:
        if mxfe_idx not in self._MXFE_IDXS:
            raise ValueError(f"an invalid mxfe: {mxfe_idx}")

    def get_all_any_lines(self) -> Set[Tuple[int, Union[int, str]]]:
        any_lines: Set[Tuple[int, Union[int, str]]] = set()
        for g, l in self._DAC_IDX:
            any_lines.add((g, l))
        for g, rl in self._ADC_IDX:
            any_lines.add((g, rl))
        return any_lines

    def get_all_lines_of_group(self, group: int) -> Set[int]:
        self._validate_group(group)
        lines: Set[int] = set()
        for g, l in self._DAC_IDX:
            if g == group:
                lines.add(l)
        return lines

    def get_all_rlines_of_group(self, group: int) -> Set[str]:
        rlines: Set[str] = set()
        for g, l in self._ADC_IDX:
            if g == group:
                rlines.add(l)
        return rlines

    def _validate_line(self, group: int, line: int) -> None:
        if (group, line) not in self._DAC_IDX:
            raise ValueError(f"an invalid pair of group:{group} and line:{line}")

    def _validate_rline(self, group: int, rline: str) -> None:
        if (group, rline) not in self._ADC_IDX:
            raise ValueError(f"an invalid pair of group:{group} and rline:{rline}")

    def _validate_line_or_rline(self, group: int, line: Union[int, str]) -> None:
        if not ((group, line) in self._DAC_IDX or (group, line) in self._ADC_IDX):
            raise ValueError(f"an invalid pair of group:{group} and line/rline:{line}")

    @abstractmethod
    def get_num_channels_of_line(self, group: int, line: int):
        # a class managing MxFE should implement this.
        pass

    @abstractmethod
    def get_num_rchannels_of_rline(self, group: int, rline: str):
        # a class managing MxFE should implement this.
        pass

    def _validate_channel(self, group: int, line: int, channel: int) -> None:
        num_ch = self.get_num_channels_of_line(group, line)
        if not (isinstance(channel, int) and 0 <= channel < num_ch):
            raise ValueError(f"an invalid combination of group:{group}, line:{line}, and channel:{channel}")

    def _validate_rchannel(self, group: int, rline: str, rchannel: int) -> None:
        num_rch = self.get_num_rchannels_of_rline(group, rline)
        if not (isinstance(rchannel, int) and 0 <= rchannel < num_rch):
            raise ValueError(f"an invalid combination of group:{group}, rline:{rline}, and rchannel:{rchannel}")


class Quel1ConfigSubsystemRoot(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        port: int = 0,
        timeout: float = 0.0,
        sender_limit_by_binding: bool = False,
    ):
        super(Quel1ConfigSubsystemRoot, self).__init__()
        self._css_addr: str = css_addr
        self._boxtype: Quel1BoxType = boxtype
        self._runtime_options: Set[Quel1RuntimeOption] = set()
        self._proxy: _ExstickgeProxyBase = self._create_exstickge_proxy(port, timeout, sender_limit_by_binding)

    def __del__(self):
        if hasattr(self, "_proxy"):
            self._proxy.terminate()

    def initialize(self) -> None:
        pass

    def get_default_config_filename(self) -> Path:
        return Path(self._DEFAULT_CONFIG_JSONFILE)

    def get_num_ics(self) -> Dict[str, int]:
        return copy.copy(self._NUM_IC)

    @abstractmethod
    def _create_exstickge_proxy(self, port: int, timeout: float, sender_limit_by_binding: bool) -> _ExstickgeProxyBase:
        pass

    @property
    def boxtype(self) -> Quel1BoxType:
        # Notes: at the css layer, boxtype is Quel1BoxType.
        #        at the box layer, boxtype is its alias in string.
        return self._boxtype

    @property
    def ipaddr_css(self) -> str:
        return self._css_addr

    @abstractmethod
    def configure_peripherals(
        self,
        param: Dict[str, Any],
        *,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        """configure ICs other than MxFEs and PLLs for their reference clock.

        :param param: configuration parameters.
        :param ignore_access_failure_of_adrf6780: a collection of index of ADRF6780 to ignore access failure during the
                                                  initialization process.
        :param ignore_lock_failure_of_lmx2594: a collection of index of LMX2594 to ignore PLL lock failure during the
                                               initialization process.
        :return: None
        """
        pass

    @abstractmethod
    def configure_all_mxfe_clocks(
        self, param: Dict[str, Any], *, ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None
    ) -> None:
        """configure PLLs for clocks of MxFEs.

        :param param: configuration parameters.
        :param ignore_lock_failure_of_lmx2594: a collection of index of LMX2594 to ignore PLL lock failure during the
                                               initialization process.
        :return: None
        """
        pass

    @abstractmethod
    def configure_mxfe(
        self,
        mxfe_idx: int,
        param: dict[str, Any],
        *,
        hard_reset: bool,
        soft_reset: bool,
        use_204b: bool,
        use_bg_cal: bool,
        ignore_crc_error: bool,
    ) -> bool:
        """configure an MxFE and its related data objects. PLLs for their clock must be set up in advance.

        :param mxfe_idx: an index of a group which the target MxFE belongs to.
        :param param: configuration parameters.
        :param hard_reset: enabling hard reset of the MxFE before the initialization if available.
        :param soft_reset: enabling soft reset of the MxFE before the initialization.
        :param use_204b: using a workaround method to link the MxFE up during the initialization if True.
        :param ignore_crc_error: ignoring CRC error flag at the validation of the link status if True.
        :return: True if the target MxFE is available.
        """
        pass

    @abstractmethod
    def reconnect_mxfe(
        self,
        mxfe_idx: int,
        *,
        ignore_crc_error: bool,
    ) -> bool:
        """configure an MxFE and its related data objects. PLLs for their clock must be set up in advance.

        :param mxfe_idx: an index of a group which the target MxFE belongs to.
        :param ignore_crc_error: ignoring CRC error flag at the validation of the link status if True.
        :return: True if the target MxFE is available.
        """
        pass


class Quel1GenericConfigSubsystemAd9082Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    @abstractmethod
    def _construct_ad9082(self): ...

    def validate_chip_id(self, mxfe_idx: int):
        self._validate_mxfe(mxfe_idx)
        chip_id = self.ad9082[mxfe_idx].device_chip_id_get()
        if chip_id.prod_id != 0x9082 or chip_id.dev_revision < 3:
            raise RuntimeError(
                f"unexpected chip_id of mxfe-#{mxfe_idx}: prod_id = {chip_id.prod_id:04x}, rev = {chip_id.dev_revision}"
            )

    @property
    def allow_dual_modulus_nco(self) -> bool:
        return self._get_runtime_option(Quel1RuntimeOption.ALLOW_DUAL_MODULUS_NCO)

    @allow_dual_modulus_nco.setter
    def allow_dual_modulus_nco(self, v: bool) -> None:
        self._set_runtime_option(Quel1RuntimeOption.ALLOW_DUAL_MODULUS_NCO, v)

    @property
    def ad9082(self) -> Tuple[Ad9082Generic, ...]:
        return self._ad9082

    def check_link_status(
        self, mxfe_idx: int, mxfe_init: bool = False, ignore_crc_error: bool = False
    ) -> Tuple[bool, int, str]:
        link_status, crc_error = self.ad9082[mxfe_idx].get_link_status()
        link_status_strs = ["("]
        # Notes: be aware that assuming that any elements of LinkStatus start with "LINK_STATUS_".
        # TODO: should define str() for LinkStatus.
        link_status_strs.append(f"link status is {str(link_status).split('.')[-1][12:]}, ")
        if crc_error == 0:
            link_status_strs.append("No CRC error")
        else:
            link_status_strs.append("CRC error is detected")
        link_status_strs.append(")")
        link_status_str = "".join(link_status_strs)
        mxfe_name = f"{self._css_addr}:AD9082-#{mxfe_idx}"
        judge: bool = False
        if link_status == LinkStatus.LINK_STATUS_LOCKED:
            if crc_error == 0:
                judge = True
            elif crc_error == 1 and ignore_crc_error:
                judge = True

        if judge:
            if crc_error == 0:
                lglv: int = logging.INFO
                if mxfe_init:
                    diag: str = f"{mxfe_name} links up successfully {link_status_str}"
                else:
                    diag = f"{mxfe_name} has linked up healthy {link_status_str}"
            else:
                lglv = logging.WARNING
                if mxfe_init:
                    diag = f"{mxfe_name} links up successfully with ignored crc error {link_status_str}"
                else:
                    diag = f"{mxfe_name} has linked up but crc error is detected thereafter {link_status_str}"
        else:
            lglv = logging.WARNING
            if mxfe_init:
                diag = f"{mxfe_name} fails to link up {link_status_str}"
            else:
                diag = f"{mxfe_name} has not linked up yet {link_status_str}"

        return judge, lglv, diag

    def clear_crc_error(self, mxfe_idx: int) -> None:
        self.ad9082[mxfe_idx].clear_crc_error()

    def _validate_frequency_info(
        self, mxfe_idx: int, freq_type: str, freq_in_hz: Union[float, None], ftw: Union[NcoFtw, None]
    ) -> Tuple[float, NcoFtw]:
        # Notes: Assuming that group is already validated.
        # Notes: decimation rates can be different among DACs, however,
        #        QuEL-1 doesn't allow such configuration currently.
        if (freq_in_hz is not None and ftw is not None) or (freq_in_hz is None and ftw is None):
            raise ValueError("either of freq_in_hz or ftw is required")
        elif freq_in_hz is not None:
            # TODO: consider to use calc_*_*_ftw_rational().
            if freq_type == "dac_cnco":
                ftw = self.ad9082[mxfe_idx].calc_dac_cnco_ftw(freq_in_hz, fractional_mode=self.allow_dual_modulus_nco)
            elif freq_type == "dac_fnco":
                ftw = self.ad9082[mxfe_idx].calc_dac_fnco_ftw(freq_in_hz, fractional_mode=self.allow_dual_modulus_nco)
            elif freq_type == "adc_cnco":
                ftw = self.ad9082[mxfe_idx].calc_adc_cnco_ftw(freq_in_hz, fractional_mode=self.allow_dual_modulus_nco)
            elif freq_type == "adc_fnco":
                ftw = self.ad9082[mxfe_idx].calc_adc_fnco_ftw(freq_in_hz, fractional_mode=self.allow_dual_modulus_nco)
            else:
                raise ValueError(f"invalid freq_type: {freq_in_hz}")
        elif ftw is not None:
            if not isinstance(ftw, NcoFtw):
                raise TypeError("unexpected ftw is given")
            # Notes: freq_in_hz is computed back from ftw for logger messages.
            if freq_type == "dac_cnco":
                freq_in_hz = self.ad9082[mxfe_idx].calc_dac_cnco_freq(ftw)
            elif freq_type == "dac_fnco":
                freq_in_hz = self.ad9082[mxfe_idx].calc_dac_fnco_freq(ftw)
            elif freq_type == "adc_cnco":
                raise NotImplementedError
            elif freq_type == "adc_fnco":
                raise NotImplementedError
            else:
                raise ValueError(f"invalid freq_type: {freq_type}")
        else:
            raise AssertionError  # never happens

        return freq_in_hz, ftw

    def get_dac_idx(self, group: int, line: int) -> Tuple[int, int]:
        self._validate_line(group, line)
        return self._DAC_IDX[(group, line)]

    def get_fduc_idx(self, group: int, line: int, channel: int) -> Tuple[int, int]:
        self._validate_channel(group, line, channel)
        mxfe_idx, dac_idx = self.get_dac_idx(group, line)
        return mxfe_idx, self.ad9082[mxfe_idx].get_fduc_of_dac(dac_idx)[channel]

    def get_num_channels_of_line(self, group: int, line: int) -> int:
        mxfe_idx, dac_idx = self.get_dac_idx(group, line)
        return len(self.ad9082[mxfe_idx].get_fduc_of_dac(dac_idx))

    def get_adc_idx(self, group: int, rline: str) -> Tuple[int, int]:
        self._validate_rline(group, rline)
        return self._ADC_IDX[(group, rline)]

    def get_rline_from_adc_idx(self, mxfe_idx: int, adc_idx: int) -> Union[Tuple[int, str], None]:
        self._validate_mxfe(mxfe_idx)
        for gl, (m, c) in self._ADC_IDX.items():
            if m == mxfe_idx and c == adc_idx:
                return gl
        else:
            return None

    def get_num_rchannels_of_rline(self, group: int, rline: str) -> int:
        self._validate_rline(group, rline)
        return len(self._ADC_CH_IDX[group, rline])

    def get_fddc_idx(self, group: int, rline: str, rchannel: int) -> Tuple[int, int]:
        self._validate_rchannel(group, rline, rchannel)
        return self._ADC_CH_IDX[group, rline][rchannel]

    def set_dac_cnco(
        self, group: int, line: int, freq_in_hz: Union[float, None] = None, ftw: Union[NcoFtw, None] = None
    ) -> None:
        """setting CNCO frequency of a transmitter line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :param freq_in_hz: the CNCO frequency of the line in Hz.
        :param ftw: an FTW can be passed instead of freq_in_hz if necessary.
        :return: None
        """
        mxfe_idx, dac_idx = self.get_dac_idx(group, line)
        freq_in_hz_, ftw_ = self._validate_frequency_info(mxfe_idx, "dac_cnco", freq_in_hz, ftw)
        logger.info(f"DAC-CNCO{dac_idx} of MxFE{mxfe_idx} of {self._css_addr} is set to {freq_in_hz_}Hz (ftw = {ftw_})")
        self.ad9082[mxfe_idx].set_dac_cnco({dac_idx}, ftw_)

    def get_dac_cnco(self, group: int, line: int) -> float:
        """getting CNCO frequency of a transmitter line.

        :param group: an index of group which the line belongs to.
        :param line: a group-local index of the line.
        :return: the current CNCO frequency of the line in Hz.
        """
        mxfe_idx, dac_idx = self.get_dac_idx(group, line)
        ftw = self.ad9082[mxfe_idx].get_dac_cnco(dac_idx)
        return self.ad9082[mxfe_idx].calc_dac_cnco_freq(ftw)

    def is_equivalent_dac_cnco(self, group: int, line: int, freq0: float, freq1: float) -> bool:
        mxfe_idx, _ = self.get_dac_idx(group, line)
        return self.ad9082[mxfe_idx].is_equivalent_dac_cnco(freq0, freq1, self.allow_dual_modulus_nco)

    def set_dac_fnco(
        self,
        group: int,
        line: int,
        channel: int,
        freq_in_hz: Union[float, None] = None,
        ftw: Union[NcoFtw, None] = None,
    ) -> None:
        """setting FNCO frequency of the transmitter channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :param freq_in_hz: the FNCO frequency of the channel in Hz.
        :param ftw: an FTW can be passed instead of freq_in_hz if necessary.
        :return: None
        """
        mxfe_idx, fnco_idx = self.get_fduc_idx(group, line, channel)
        freq_in_hz_, ftw_ = self._validate_frequency_info(mxfe_idx, "dac_fnco", freq_in_hz, ftw)
        logger.info(
            f"DAC-FNCO{fnco_idx} of MxFE{mxfe_idx} of {self._css_addr} is set to {freq_in_hz_}Hz (ftw = {ftw_})"
        )
        self.ad9082[mxfe_idx].set_dac_fnco({fnco_idx}, ftw_)

    def get_dac_fnco(self, group: int, line: int, channel: int) -> float:
        """getting FNCO frequency of a transmitter channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :return: the current FNCO frequency of the channel in Hz.
        """
        mxfe_idx, fnco_idx = self.get_fduc_idx(group, line, channel)
        ftw = self.ad9082[mxfe_idx].get_dac_fnco(fnco_idx)
        return self.ad9082[mxfe_idx].calc_dac_fnco_freq(ftw)

    def is_equivalent_dac_fnco(self, group: int, line: int, freq0: float, freq1: float) -> bool:
        mxfe_idx, _ = self.get_dac_idx(group, line)
        return self.ad9082[mxfe_idx].is_equivalent_dac_fnco(freq0, freq1, self.allow_dual_modulus_nco)

    def set_adc_cnco(
        self, group: int, rline: str, freq_in_hz: Union[float, None] = None, ftw: Union[NcoFtw, None] = None
    ):
        """setting CNCO frequency of a receiver line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :param freq_in_hz: the CNCO frequency of the line in Hz.
        :param ftw: an FTW can be passed instead of freq_in_hz if necessary.
        :return: None
        """
        mxfe_idx, adc_idx = self.get_adc_idx(group, rline)
        freq_in_hz_, ftw_ = self._validate_frequency_info(mxfe_idx, "adc_cnco", freq_in_hz, ftw)
        logger.info(f"ADC-CNCO{adc_idx} of MxFE{mxfe_idx} of {self._css_addr} is set to {freq_in_hz_}Hz (ftw = {ftw_})")
        self.ad9082[mxfe_idx].set_adc_cnco({adc_idx}, ftw_)

    def get_adc_cnco(self, group: int, rline: str) -> float:
        """getting CNCO frequency of a receiver line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: the current CNCO frequency of the line in Hz.
        """
        mxfe_idx, adc_idx = self.get_adc_idx(group, rline)
        ftw = self.ad9082[mxfe_idx].get_adc_cnco(adc_idx)
        return self.ad9082[mxfe_idx].calc_adc_cnco_freq(ftw)

    def is_equivalent_adc_cnco(self, group: int, rline: str, freq0: float, freq1: float) -> bool:
        mxfe_idx, _ = self.get_adc_idx(group, rline)
        return self.ad9082[mxfe_idx].is_equivalent_adc_cnco(freq0, freq1, self.allow_dual_modulus_nco)

    def set_adc_fnco(
        self,
        group: int,
        rline: str,
        rchannel: int,
        freq_in_hz: Union[float, None] = None,
        ftw: Union[NcoFtw, None] = None,
    ):
        """setting FNCO frequency of a receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :param freq_in_hz: the FNCO frequency of the channel in Hz.
        :param ftw: an FTW can be passed instead of freq_in_hz if necessary.
        :return: None
        """
        mxfe_idx, fnco_idx = self.get_fddc_idx(group, rline, rchannel)
        freq_in_hz_, ftw_ = self._validate_frequency_info(mxfe_idx, "adc_fnco", freq_in_hz, ftw)
        logger.info(
            f"ADC-FNCO{fnco_idx} of MxFE{mxfe_idx} of {self._css_addr} is set to {freq_in_hz_}Hz (ftw = {ftw_})"
        )
        self.ad9082[mxfe_idx].set_adc_fnco({fnco_idx}, ftw_)

    def get_adc_fnco(self, group: int, rline: str, rchannel: int) -> float:
        """getting FNCO frequency of a receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :return: the current FNCO frequency of the channel in Hz.
        """
        mxfe_idx, fnco_idx = self.get_fddc_idx(group, rline, rchannel)
        ftw = self.ad9082[mxfe_idx].get_adc_fnco(fnco_idx)
        return self.ad9082[mxfe_idx].calc_adc_fnco_freq(ftw)

    def is_equivalent_adc_fnco(self, group: int, rline: str, freq0: float, freq1: float) -> bool:
        mxfe_idx, _ = self.get_adc_idx(group, rline)
        return self.ad9082[mxfe_idx].is_equivalent_adc_fnco(freq0, freq1, self.allow_dual_modulus_nco)

    def set_pair_cnco(self, group_dac: int, line_dac: int, group_adc: int, rline_adc: str, freq_in_hz: float) -> None:
        dac_mxfe_idx, dac_idx = self.get_dac_idx(group_dac, line_dac)
        adc_mxfe_idx, adc_idx = self.get_adc_idx(group_adc, rline_adc)

        dac_clk = self.ad9082[dac_mxfe_idx].device.dev_info.dac_freq_hz
        adc_clk = self.ad9082[adc_mxfe_idx].device.dev_info.adc_freq_hz
        # TODO: relax constrains by-need basis.
        if dac_clk % adc_clk != 0:
            raise RuntimeError(f"ratio of dac_clk (= {dac_clk}Hz) and adc_clk (= {adc_clk}Hz) is not N:1")
        ratio: int = dac_clk // adc_clk

        freq_in_hz_, dac_ftw = self._validate_frequency_info(dac_mxfe_idx, "dac_cnco", freq_in_hz, None)
        adc_ftw = dac_ftw.multiply(ratio)
        logger.info(
            f"DAC-CNCO{dac_idx} of MxFE{dac_mxfe_idx} and ADC-CNCO{adc_idx} of MxFE{adc_mxfe_idx} of {self._css_addr} "
            f"are set to {freq_in_hz_}Hz (dac_ftw = {dac_ftw}, adc_ftw = {adc_ftw})"
        )
        self.ad9082[dac_mxfe_idx].set_dac_cnco({dac_idx}, dac_ftw)
        self.ad9082[adc_mxfe_idx].set_adc_cnco({adc_idx}, adc_ftw)

    def get_link_status(self, mxfe_idx: int) -> Tuple[LinkStatus, int]:
        """getting the status of the datalink between a MxFE in a group and the FPGA.

        :param mxfe_idx: an index of the group which the target MxFE belongs to.
        :return: the content of the registers of the MxFE at the addresses of 0x55E and 0x5BB.
        """
        self._validate_mxfe(mxfe_idx)
        return self._ad9082[mxfe_idx].get_link_status()

    def set_fullscale_current(self, group: int, line: int, fsc: int) -> None:
        """setting the full-scale current of a DAC of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line
        :param fsc: full-scale current of the DAC of the line in uA.
        :return: None
        """
        mxfe_idx, dac_idx = self.get_dac_idx(group, line)
        self.ad9082[mxfe_idx].set_fullscale_current(1 << dac_idx, fsc)

    def get_fullscale_current(self, group: int, line: int) -> int:
        """geting the current full scale current of a DAC of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line
        :return: the current full-scale current of the DAC in uA
        """
        mxfe_idx, dac_idx = self.get_dac_idx(group, line)
        return self.ad9082[mxfe_idx].get_fullscale_current(dac_idx)

    def is_equal_fullscale_current(self, group: int, line: int, fsc0: int, fsc1: int) -> bool:
        mxfe_idx, _ = self.get_dac_idx(group, line)
        return self.ad9082[mxfe_idx].is_equal_fullscale_current(fsc0, fsc1)

    def get_main_interpolation_rate(self, mxfe_idx: int) -> int:
        """getting the current main interpolation rate of a MxFE of a group.

        :param mxfe_idx: an index of the group which the target MxFE belongs to.
        :return: main interpolation rate
        """
        self._validate_mxfe(mxfe_idx)
        return self.ad9082[mxfe_idx].get_main_interpolation_rate()

    def get_channel_interpolation_rate(self, mxfe_idx: int) -> int:
        """getting the current channel interpolation rate of a MxFE of a group.

        :param mxfe_idx: a group which the target MxFE belongs to.
        :return: channel interpolation rate
        """
        self._validate_mxfe(mxfe_idx)
        return self.ad9082[mxfe_idx].get_channel_interpolation_rate()

    def get_virtual_adc_select(self, mxfe_idx: int) -> List[int]:
        """getting converter select matrix of ADC virtual converters after FDDC.

        :param mxfe_idx: a group which the target MxFE belongs to.
        :return: converter select matrix in list
        """
        self._validate_mxfe(mxfe_idx)
        return self.ad9082[mxfe_idx].get_virtual_adc_select()

    def get_crc_error_counts(self, mxfe_idx: int) -> List[int]:
        return self.ad9082[mxfe_idx].get_crc_error_counts()

    def get_mxfe_temperature_range(self, mxfe_idx: int) -> Tuple[int, int]:
        """getting the current die temperatures of a MxFE of a group.

        :param mxfe_idx: an index of the group which the target MxFE belongs to.
        :return: a pair of maximum and minimum temperatures in the die.
        """
        self._validate_mxfe(mxfe_idx)
        temp_max, temp_min = self.ad9082[mxfe_idx].get_temperatures()
        return temp_max, temp_min


class Quel1ConfigSubsystemLmx2594Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]]  # (group, line) --> (idx_of_ic, idx_of_output)

    def _construct_lmx2594(self):
        self._lmx2594: Tuple[Lmx2594, ...] = tuple(Lmx2594(self._proxy, idx) for idx in range(self._NUM_IC["lmx2594"]))
        self._lmx2594_helper: Tuple[Lmx2594ConfigHelper, ...] = tuple(Lmx2594ConfigHelper(ic) for ic in self._lmx2594)

    @property
    def lmx2594(self) -> Tuple[Lmx2594, ...]:
        return self._lmx2594

    @property
    def lmx2594_helper(self) -> Tuple[Lmx2594ConfigHelper, ...]:
        return self._lmx2594_helper

    def init_lmx2594(
        self,
        idx: int,
        param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]],
        ignore_lock_failure: bool = False,
    ) -> bool:
        # Notes: ignore_lock_failure is prepared for experimental purposes.
        ic, helper = self._lmx2594[idx], self._lmx2594_helper[idx]
        ic.soft_reset()
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()
        is_locked = ic.calibrate()
        if not is_locked:
            if ignore_lock_failure:
                logger.error(
                    f"failed to lock PLL of {self._css_addr}:LMX2594-#{idx}, keep running with an unlocked PLL"
                )
            else:
                raise RuntimeError(f"failed to lock PLL of {self._css_addr}:LMX2594-#{idx}")
        return is_locked

    def set_lo_multiplier(self, group: int, line: Union[int, str], freq_multiplier: int) -> bool:
        """setting the frequency multiplier of a PLL of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the PLL belongs to.
        :param freq_multiplier: the frequency multiplier to set.
        :return: True if the frequency calibration is completed successfully.
        """
        self._validate_line_or_rline(group, line)
        if (group, line) not in self._LO_IDX:
            raise ValueError(f"no LO is available for (group:{group}, line:{line})")
        lo_idx, _ = self._LO_IDX[(group, line)]
        ic = self._lmx2594[lo_idx]
        ic.set_lo_multiplier(freq_multiplier)
        logger.info(f"updating LO frequency[{lo_idx}] of {self._css_addr} to {freq_multiplier * 100}MHz")
        return ic.calibrate()

    def get_lo_multiplier(self, group: int, line: Union[int, str]) -> int:
        """get the current frequency multiplier of a PLL of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the PLL belongs to.
        :return: the current frequency multiplier of the PLL.
        """
        self._validate_line_or_rline(group, line)
        if (group, line) not in self._LO_IDX:
            raise ValueError(f"no LO is available for (group:{group}, line:{line})")
        lo_idx, _ = self._LO_IDX[(group, line)]
        return self._lmx2594[lo_idx].get_lo_multiplier()

    def set_divider_ratio(self, group: int, line: Union[int, str], divide_ratio: int) -> None:
        """setting the output frequency divide ration of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the PLL belongs to.
        :param divide_ratio: a ratio of the frequency output divider.
        """
        self._validate_line_or_rline(group, line)
        if (group, line) not in self._LO_IDX:
            raise ValueError(f"no LO is available for (group:{group}, line:{line})")
        if divide_ratio < 1:
            raise ValueError(f"invalid divide ratio {divide_ratio} for (group:{group}, line:{line})")

        lo_idx, outpin = self._LO_IDX[(group, line)]
        ic = self._lmx2594[lo_idx]
        current_ratios = ic.get_divider_ratio()
        modified_ratios = list(current_ratios)
        # Notes: both output pin should generate the same freq (an unused outpin is kept as is.)
        for i, ratio in enumerate(current_ratios):
            if ratio != 0:
                modified_ratios[i] = divide_ratio
        ic.set_divider_ratio(*modified_ratios)
        logger.info(
            f"updating frequency divide ratio of (group:{group}, line:{line}) of {self._css_addr} to {divide_ratio}"
        )

    def get_divider_ratio(self, group: int, line: Union[int, str]) -> int:
        """get the current output frequency divide ratio of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the PLL belongs to.
        :return: the current frequency divide ratio of the output
        """
        self._validate_line_or_rline(group, line)
        if (group, line) not in self._LO_IDX:
            raise ValueError(f"no LO is available for group:{group}, line:{line}")
        lo_idx, outpin = self._LO_IDX[(group, line)]

        ic = self._lmx2594[lo_idx]
        if outpin not in {0, 1}:
            raise AssertionError(f"invalid outpin '{outpin}' is specified in class definition")
        return ic.get_divider_ratio()[outpin]


class Quel1ConfigSubsystemAd6780Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _MIXER_IDX: Dict[Tuple[int, int], int]

    def _construct_adrf6780(self):
        self._adrf6780: Tuple[Adrf6780, ...] = tuple(
            Adrf6780(self._proxy, idx) for idx in range(self._NUM_IC.get("adrf6780", 0))
        )
        self._adrf6780_helper: Tuple[Adrf6780ConfigHelper, ...] = tuple(
            Adrf6780ConfigHelper(ic) for ic in self._adrf6780
        )

    @property
    def adrf6780(self) -> Tuple[Adrf6780, ...]:
        return self._adrf6780

    @property
    def adrf6780_helper(self) -> Tuple[Adrf6780ConfigHelper, ...]:
        return self._adrf6780_helper

    def init_adrf6780(
        self, idx, param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]], ignore_id_mismatch: bool = False
    ) -> bool:
        # Notes: ignore_id_mismatch is a workaround for some ill-designed boards.
        ic, helper = self._adrf6780[idx], self._adrf6780_helper[idx]
        ic.soft_reset(parity_enable=False)
        matched, (chip_id, chip_rev) = ic.check_id()
        if matched:
            logger.info(f"chip revision of ADRF6780[{idx}] of {self._css_addr} is ({chip_id}, {chip_rev})")
        else:
            if ignore_id_mismatch:
                logger.error(
                    f"unexpected chip revision of ADRF6780[{idx}] of {self._css_addr}: ({chip_id}, {chip_rev}), "
                    f"keep running with an unstable mixer"
                )
            else:
                raise RuntimeError(
                    f"unexpected chip revision of ADRF6780[{idx}] of {self._css_addr}: ({chip_id}, {chip_rev})"
                )
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()
        return matched

    def _validate_sideband(self, sideband: str):
        if not isinstance(sideband, str):
            raise TypeError(f"unexpected sideband: {sideband}")
        if sideband not in {"L", "U"}:
            raise ValueError(f"unexpected sideband: {sideband}")

    def _get_mixer_idx(self, group, line):
        if (group, line) in self._MIXER_IDX:
            return self._MIXER_IDX[(group, line)]
        raise ValueError(f"no configurable mixer is available for group:{group}, line:{line}")

    def set_sideband(self, group: int, line: int, sideband: str) -> None:
        """setting the active sideband of a mixer of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the mixer belongs to.
        :param sideband: the chosen active sideband, either of "U" for upper or "L" for lower.
        :return: None
        """
        mixer_idx = self._get_mixer_idx(group, line)
        self._validate_sideband(sideband)
        if sideband == "L":
            logger.info(f"ADRF6780[{mixer_idx}] of {self._css_addr} is set to LSB mode")
            ic = self._adrf6780[mixer_idx]
            ic.set_lo_sideband(Adrf6780LoSideband.Lsb)
        elif sideband == "U":
            logger.info(f"ADRF6780[{mixer_idx}] of {self._css_addr} is set to USB mode")
            ic = self._adrf6780[mixer_idx]
            ic.set_lo_sideband(Adrf6780LoSideband.Usb)
        else:
            raise AssertionError  # never happens

    def get_sideband(self, group: int, line: int) -> str:
        """getting the active sideband of a mixer of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the mixer belongs to.
        :return: the current sideband of the mixer.
        """
        mixer_idx = self._get_mixer_idx(group, line)
        if mixer_idx is None:
            raise ValueError(f"no configurable mixer is available for group:{group}, line:{line}")

        ic = self._adrf6780[mixer_idx]
        ssb = ic.get_lo_sideband()
        if ssb == Adrf6780LoSideband.Usb:
            return "U"
        elif ssb == Adrf6780LoSideband.Lsb:
            return "L"
        else:
            raise AssertionError


class Quel1ConfigSubsystemAd5328Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    MAX_VATT: Final[int] = 0xC9B

    _VATT_IDX: Dict[Tuple[int, int], Tuple[int, int]]

    def _construct_ad5328(self):
        self._ad5328: Tuple[Ad5328, ...] = tuple(
            Ad5328(self._proxy, idx) for idx in range(self._NUM_IC.get("ad5328", 0))
        )
        self._ad5328_helper: Tuple[Ad5328ConfigHelper, ...] = tuple(Ad5328ConfigHelper(ic) for ic in self._ad5328)

    @property
    def ad5328(self) -> Tuple[Ad5328, ...]:
        return self._ad5328

    @property
    def ad5328_helper(self) -> Tuple[Ad5328ConfigHelper, ...]:
        return self._ad5328_helper

    def init_ad5328(self, idx: int, param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]]) -> None:
        ic, helper = self._ad5328[idx], self._ad5328_helper[idx]
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()
        ic.update_dac()

    def _validate_vatt(self, vatt: int) -> int:
        if not (0 <= vatt < 4096):
            raise ValueError("invalid vatt value")
        if vatt > self.MAX_VATT:
            vatt = self.MAX_VATT
            logger.warning(f"vatt is clipped to {self.MAX_VATT}")
        return vatt

    def _get_vatt_idx(self, group: int, line: int):
        if (group, line) in self._VATT_IDX:
            return self._VATT_IDX[(group, line)]
        else:
            raise ValueError(f"no variable attenuator is available for group:{group}, line:{line}")

    def set_vatt(self, group: int, line: int, vatt: int) -> None:
        """setting the VATT control voltage of a mixer of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the mixer belongs to.
        :param vatt: a control voltage of the VATT (in unit of 3.3V / 4096, see data sheet of ADRF6780 for details).
        :return: None
        """
        ic_idx, vatt_idx = self._get_vatt_idx(group, line)
        vatt = self._validate_vatt(vatt)
        logger.info(f"VATT{vatt_idx} of {self._css_addr} is set to 0x{vatt:03x}")
        ic = self._ad5328[ic_idx]
        ic.set_output(vatt_idx, vatt, update_dac=True)

    def get_vatt_carboncopy(self, group: int, line: int) -> Union[int, None]:
        """getting the VATT control voltage of a mixer of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the mixer belongs to.
        :return: the latest copy of the control voltage of the VATT of the line passed to set_vatt().
        """
        ic_idx, vatt_idx = self._get_vatt_idx(group, line)
        ic = self._ad5328[ic_idx]
        return ic.get_output_carboncopy(vatt_idx)


class NoRfSwitchError(Exception):
    pass


class NoLoopbackPathError(Exception):
    pass


class Quel1ConfigSubsystemNoRfswitch(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def is_subordinate_rfswitch(self, group: int, line: Union[int, str]) -> bool:
        return False

    def pass_line(self, group: int, line: Union[int, str]) -> None:
        logger.info(f"making (group:{group}, line:{line}) passing")

    def block_line(self, group: int, line: Union[int, str]) -> None:
        raise NoRfSwitchError()

    def is_blocked_line(self, group: int, line: Union[int, str]) -> bool:
        return False

    def is_passed_line(self, group: int, line: Union[int, str]) -> bool:
        return True

    def activate_monitor_loop(self, group: int) -> None:
        raise NoLoopbackPathError()

    def deactivate_monitor_loop(self, group: int) -> None:
        logger.info(f"deactivate monitor loop (group:{group})")

    def is_loopedback_monitor(self, group: int) -> bool:
        return False

    def activate_read_loop(self, group: int) -> None:
        raise NoLoopbackPathError()

    def deactivate_read_loop(self, group: int) -> None:
        logger.info(f"deactivate read loop (group:{group})")

    def is_loopedback_read(self, group: int) -> bool:
        return False


class Quel1ConfigSubsystemRfswitch(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _RFSWITCH_NAME: Dict[Tuple[int, Union[int, str]], Tuple[int, str]]
    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]]

    # TODO: this is not good with respect to the hierarchical structure.
    def _construct_rfswitch(self, gpio_idx: int):
        if self._boxtype in {Quel1BoxType.QuEL1_TypeA, Quel1BoxType.QuEL1_NTT}:
            gpio_tmp: AbstractRfSwitchArrayMixin = Quel1TypeARfSwitchArray(self._proxy, gpio_idx)
        elif self._boxtype == Quel1BoxType.QuEL1_TypeB:
            gpio_tmp = Quel1TypeBRfSwitchArray(self._proxy, gpio_idx)
        elif self._boxtype in {
            Quel1BoxType.QuBE_RIKEN_TypeA,
            Quel1BoxType.QuBE_RIKEN_TypeB,
        }:
            gpio_tmp = QubeRfSwitchArray(self._proxy, gpio_idx)
        else:
            raise ValueError(f"invalid boxtype: {self._boxtype}")
        self._rfswitch_gpio_idx = gpio_idx
        self._rfswitch: AbstractRfSwitchArrayMixin = gpio_tmp
        self._rfswitch_helper: RfSwitchArrayConfigHelper = RfSwitchArrayConfigHelper(self._rfswitch)

    @property
    def rfswitch(self) -> AbstractRfSwitchArrayMixin:
        return self._rfswitch

    @property
    def rfswitch_helper(self) -> RfSwitchArrayConfigHelper:
        return self._rfswitch_helper

    def init_rfswitch(self, param: Dict[str, Dict[str, Dict[str, Union[int, bool]]]]) -> None:
        # XXX: = self._param["gpio"][self._rfswitch_gpio_idx]
        helper = self._rfswitch_helper
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()

    def _alternate_rfswitch(self, group: int, **sw_update: bool) -> None:
        # Note: assuming that group is already validated.
        ic, helper = self._rfswitch, self._rfswitch_helper
        current_sw: int = ic.read_reg(0)
        helper.write_field(f"Group{group}", **sw_update)
        helper.flush()
        altered_sw: int = ic.read_reg(0)
        time.sleep(0.01)
        logger.debug(
            f"alter the state of RF switch array of {self._css_addr} from {current_sw:014b} to {altered_sw:014b}"
        )

    def _get_rfswitch_name(self, group: int, line: Union[int, str]) -> Tuple[int, str]:
        if (group, line) in self._RFSWITCH_NAME:
            return self._RFSWITCH_NAME[(group, line)]
        else:
            raise ValueError(f"no switch available for group:{group}, line:{line}")

    def is_subordinate_rfswitch(self, group: int, line: Union[int, str]) -> bool:
        return (group, line) in self._RFSWITCH_SUBORDINATE_OF

    def pass_line(self, group: int, line: Union[int, str]) -> None:
        """allowing a line to emit RF signal from its corresponding SMA connector.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: None
        """
        logger.info(f"making (group:{group}, line:{line}) passing")
        swgroup, swname = self._get_rfswitch_name(group, line)
        self._alternate_rfswitch(swgroup, **{swname: False})

    def block_line(self, group: int, line: Union[int, str]) -> None:
        """blocking a line to emit RF signal from its corresponding SMA connector.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: None
        """
        logger.info(f"making (group:{group}, line:{line}) blocked")
        swgroup, swname = self._get_rfswitch_name(group, line)
        self._alternate_rfswitch(swgroup, **{swname: True})

    def is_blocked_line(self, group: int, line: Union[int, str]) -> bool:
        """checking if the emission of RF signal of a line is blocked or not.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: True if the RF signal emission from its corresponding SMA connector is blocked.
        """
        swgroup, swname = self._get_rfswitch_name(group, line)
        return getattr(self._rfswitch_helper.read_reg(swgroup), swname)

    def is_passed_line(self, group: int, line: Union[int, str]) -> bool:
        """checking if the emission of RF signal from a line is passed or not.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: True if the RF signal emission from its corresponding SMA connector is passed.
        """
        return not self.is_blocked_line(group, line)

    def activate_monitor_loop(self, group: int) -> None:
        """enabling an internal loop-back of a monitor path of a group, from monitor-out to monitor-in.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"activate monitor loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "m")
        self._alternate_rfswitch(swgroup, **{swname: True})

    def deactivate_monitor_loop(self, group: int) -> None:
        """disabling an internal loop-back of a monitor path of a group.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"deactivate monitor loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "m")
        self._alternate_rfswitch(swgroup, **{swname: False})

    def is_loopedback_monitor(self, group: int) -> bool:
        """checking if a monitor loop-back path of a group is activated or not.

        :param group: an index of the group.
        :return: True if the monitor loop-back path of the group is activated.
        """
        swgroup, swname = self._get_rfswitch_name(group, "m")
        return getattr(self._rfswitch_helper.read_reg(swgroup), swname)

    def activate_read_loop(self, group: int) -> None:
        """enabling an internal loop-back of a read path of a group, from monitor-out to monitor-in.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"activate read loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "r")
        self._alternate_rfswitch(swgroup, **{swname: True})

    def deactivate_read_loop(self, group: int) -> None:
        """disabling an internal loop-back of a read path of a group.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"deactivate read loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "r")
        self._alternate_rfswitch(swgroup, **{swname: False})

    def is_loopedback_read(self, group: int) -> bool:
        """checking if a read loop-back path of a group is activated or not.

        :param group: an index of the group.
        :return: True if the read loop-back path of the group is activated.
        """
        swgroup, swname = self._get_rfswitch_name(group, "r")
        return getattr(self._rfswitch_helper.read_reg(swgroup), swname)


class Quel1ConfigSubsystemMixerboardGpioMixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_mixerboard_gpio(self):
        self._mixerboard_gpio: Tuple[MixerboardGpio, ...] = tuple(
            MixerboardGpio(self._proxy, idx) for idx in range(self._NUM_IC.get("mixerboard_gpio", 0))
        )

        self._mixerboard_gpio_helper: Tuple[MixerboardGpioConfigHelper, ...] = tuple(
            MixerboardGpioConfigHelper(ic) for ic in self._mixerboard_gpio
        )

    @property
    def mixerboard_gpio(self) -> Tuple[MixerboardGpio, ...]:
        return self._mixerboard_gpio

    @property
    def mixerboard_gpio_helper(self) -> Tuple[MixerboardGpioConfigHelper, ...]:
        return self._mixerboard_gpio_helper

    def init_mixerboard_gpio(self, idx: int, param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]]) -> None:
        # XXX: = self._param["mixerboard_gpio"][idx]
        helper = self.mixerboard_gpio_helper[idx]
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()


# TODO: reconsider the design, this looks too wet.
class Quel1ConfigSubsystemPathselectorboardGpioMixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _RFSWITCH_IDX: Dict[Tuple[int, Union[int, str]], Tuple[int, int]]
    _RFSWITCH_SUBORDINATE_OF: Dict[Tuple[int, Union[int, str]], Tuple[int, Union[int, str]]]

    def _construct_pathselectorboard_gpio(self):
        self._pathselectorboard_gpio: Tuple[PathselectorboardGpio, ...] = tuple(
            PathselectorboardGpio(self._proxy, idx) for idx in range(self._NUM_IC.get("pathselectorboard_gpio", 0))
        )

        self._pathselectorboard_gpio_helper: Tuple[PathselectorboardGpioConfigHelper, ...] = tuple(
            PathselectorboardGpioConfigHelper(ic) for ic in self._pathselectorboard_gpio
        )

    @property
    def pathselectorboard_gpio(self) -> Tuple[PathselectorboardGpio, ...]:
        return self._pathselectorboard_gpio

    @property
    def pathselectorboard_gpio_helper(self) -> Tuple[PathselectorboardGpioConfigHelper, ...]:
        return self._pathselectorboard_gpio_helper

    def init_pathselectorboard_gpio(
        self, idx: int, param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]]
    ) -> None:
        # XXX: = self._param["pathselectorboard_gpio"][idx]
        helper = self.pathselectorboard_gpio_helper[idx]
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()

    def is_subordinate_rfswitch(self, group: int, line: Union[int, str]) -> bool:
        return (group, line) in self._RFSWITCH_SUBORDINATE_OF

    def _alternate_rfswitch(self, idx: int, **sw_update: bool) -> None:
        # Note: assuming that group is already validated.
        ic, helper = self._pathselectorboard_gpio, self._pathselectorboard_gpio_helper
        current_sw: int = ic[idx].read_reg(0)
        helper[idx].write_field(0, **sw_update)
        helper[idx].flush()
        altered_sw: int = ic[idx].read_reg(0)
        time.sleep(0.01)
        logger.debug(
            f"alter the state of RF switch array {idx} of {self._css_addr} from {current_sw:06b} to {altered_sw:06b}"
        )

    def _get_rfswitch_name(self, group: int, line: Union[int, str]) -> Tuple[int, str]:
        if (group, line) in self._RFSWITCH_IDX:
            idx, bit = self._RFSWITCH_IDX[(group, line)]
            return idx, f"b{bit:02}"
        else:
            raise ValueError(f"no switch available for group:{group}, line:{line}")

    def pass_line(self, group: int, line: Union[int, str]) -> None:
        """allowing a line to emit RF signal from its corresponding SMA connector.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: None
        """
        logger.info(f"making (group:{group}, line:{line}) passing")
        swgroup, swname = self._get_rfswitch_name(group, line)
        self._alternate_rfswitch(swgroup, **{swname: False})

    def block_line(self, group: int, line: Union[int, str]) -> None:
        """blocking a line to emit RF signal from its corresponding SMA connector.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: None
        """
        logger.info(f"making (group:{group}, line:{line}) blocked")
        swgroup, swname = self._get_rfswitch_name(group, line)
        self._alternate_rfswitch(swgroup, **{swname: True})

    def is_blocked_line(self, group: int, line: Union[int, str]) -> bool:
        """checking if the emission of RF signal of a line is blocked or not.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: True if the RF signal emission from its corresponding SMA connector is blocked.
        """
        swgroup, swname = self._get_rfswitch_name(group, line)
        return getattr(self._pathselectorboard_gpio_helper[swgroup].read_reg(0), swname)

    def is_passed_line(self, group: int, line: Union[int, str]) -> bool:
        """checking if the emission of RF signal from a line is passed or not.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: True if the RF signal emission from its corresponding SMA connector is passed.
        """
        return not self.is_blocked_line(group, line)

    def activate_monitor_loop(self, group: int) -> None:
        """enabling an internal loop-back of a monitor path of a group, from monitor-out to monitor-in.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"activate monitor loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "m")
        self._alternate_rfswitch(swgroup, **{swname: True})

    def deactivate_monitor_loop(self, group: int) -> None:
        """disabling an internal loop-back of a monitor path of a group.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"deactivate monitor loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "m")
        self._alternate_rfswitch(swgroup, **{swname: False})

    def is_loopedback_monitor(self, group: int) -> bool:
        """checking if a monitor loop-back path of a group is activated or not.

        :param group: an index of the group.
        :return: True if the monitor loop-back path of the group is activated.
        """
        swgroup, swname = self._get_rfswitch_name(group, "m")
        return getattr(self._pathselectorboard_gpio_helper[swgroup].read_reg(0), swname)

    def activate_read_loop(self, group: int) -> None:
        """enabling an internal loop-back of a read path of a group, from monitor-out to monitor-in.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"activate read loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "r")
        self._alternate_rfswitch(swgroup, **{swname: True})

    def deactivate_read_loop(self, group: int) -> None:
        """disabling an internal loop-back of a read path of a group.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"deactivate read loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "r")
        self._alternate_rfswitch(swgroup, **{swname: False})

    def is_loopedback_read(self, group: int) -> bool:
        """checking if a read loop-back path of a group is activated or not.

        :param group: an index of the group.
        :return: True if the read loop-back path of the group is activated.
        """
        swgroup, swname = self._get_rfswitch_name(group, "r")
        return getattr(self._pathselectorboard_gpio_helper[swgroup].read_reg(0), swname)


class Quel1ConfigSubsystemAd7490Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_ad7490(self):
        self._ad7490: Tuple[Ad7490, ...] = tuple(
            Ad7490(self._proxy, idx) for idx in range(self._NUM_IC.get("ad7490", 0))
        )

    @property
    def ad7490(self) -> Tuple[Ad7490, ...]:
        return self._ad7490

    def init_ad7490(self, idx: int, param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]]) -> None:
        # XXX: = self._param["ad7490"][idx]
        self._ad7490[idx].set_default_config(**param["registers"]["Config"])
        # Notes: default_config is not applied to the IC now because it'll be applied just before reading channels.


class Quel1ConfigSubsystemPowerboardPwmMixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_powerboard_pwm(self):
        self._powerboard_pwm: Tuple[PowerboardPwm, ...] = tuple(
            PowerboardPwm(self._proxy, idx) for idx in range(self._NUM_IC.get("powerboard_pwm", 0))
        )

        self._powerboard_pwm_helper: Tuple[PowerboardPwmConfigHelper, ...] = tuple(
            PowerboardPwmConfigHelper(ic) for ic in self._powerboard_pwm
        )

    @property
    def powerboard_pwm(self) -> Tuple[PowerboardPwm, ...]:
        return self._powerboard_pwm

    @property
    def powerboard_pwm_helper(self) -> Tuple[PowerboardPwmConfigHelper, ...]:
        return self._powerboard_pwm_helper
