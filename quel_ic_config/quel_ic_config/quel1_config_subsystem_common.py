import copy
import json
import logging
import os.path as osp
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Collection, Dict, Final, List, Mapping, Sequence, Set, Tuple, Union

from pydantic.v1.utils import deep_update

from quel_ic_config.ad5328 import Ad5328ConfigHelper
from quel_ic_config.ad9082_v106 import NcoFtw
from quel_ic_config.adrf6780 import Adrf6780ConfigHelper, Adrf6780LoSideband
from quel_ic_config.exstickge_proxy import _ExstickgeProxyBase
from quel_ic_config.generic_gpio import GenericGpioConfigHelper
from quel_ic_config.lmx2594 import Lmx2594ConfigHelper
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption
from quel_ic_config.quel_ic import (
    Ad5328,
    Ad9082V106,
    Adrf6780,
    GenericGpio,
    Lmx2594,
    QubeRfSwitchArray,
    Quel1TypeARfSwitchArray,
    Quel1TypeBRfSwitchArray,
)
from quel_ic_config.rfswitcharray import AbstractRfSwitchArrayMixin, RfSwitchArrayConfigHelper

logger = logging.getLogger(__name__)


class Quel1ConfigSubsystemBaseSlot(metaclass=ABCMeta):
    __slots__ = (
        "_css_addr",
        "_param",
        "_boxtype",
        "_config_path",
        "_config_options",
        "_proxy",
        "_ad9082",
        "_lmx2594",
        "_lmx2594_helper",
        "_adrf6780",
        "_adrf6780_helper",
        "_ad5328",
        "_ad5328_helper",
        "_rfswitch",
        "_rfswitch_gpio_idx",
        "_rfswitch_helper",
        "_gpio",
        "_gpio_helper",
    )

    _DEFAULT_CONFIG_JSONFILE: str
    _NUM_IC: Dict[str, int]
    _GROUPS: Set[int]
    _DAC_IDX: Dict[Tuple[int, int], int]
    _ADC_IDX: Dict[Tuple[int, str], int]
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]]

    def __init__(self):
        # variable defined in the importer class
        self._css_addr: str
        self._boxtype: Quel1BoxType
        self._config_path: Path
        self._config_options: Collection[Quel1ConfigOption]
        self._param: Dict[str, List[Dict[str, Any]]]  # TODO: clean up around here!
        self._proxy: _ExstickgeProxyBase

        # TODO: types of IC proxy will be reconsidered when controler proxy (e.g. _ExstickgeProxyBase) is generalized.
        self._ad9082: Tuple[Ad9082V106, ...] = ()
        self._lmx2594: Tuple[Lmx2594, ...] = ()
        self._lmx2594_helper: Tuple[Lmx2594ConfigHelper, ...] = ()
        self._adrf6780: Tuple[Adrf6780, ...] = ()
        self._adrf6780_helper: Tuple[Adrf6780ConfigHelper, ...] = ()
        self._ad5328: Tuple[Ad5328, ...] = ()
        self._ad5328_helper: Tuple[Ad5328ConfigHelper, ...] = ()
        self._rfswitch_gpio_idx: int = 0
        self._rfswitch: Union[AbstractRfSwitchArrayMixin, None] = None
        self._rfswitch_helper: Union[RfSwitchArrayConfigHelper, None] = None
        self._gpio: Tuple[GenericGpio, ...] = ()
        self._gpio_helper: Tuple[GenericGpioConfigHelper, ...] = ()

    def get_all_groups(self) -> Set[int]:
        return self._GROUPS

    def _validate_group(self, group: int) -> None:
        if group not in self._GROUPS:
            raise ValueError("an invalid group: {group}")

    def get_all_lines_of_group(self, group: int) -> Set[int]:
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
    DEFAULT_CONFIG_PATH: Final[Path] = Path(osp.dirname(__file__)) / "settings"

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        config_path: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        port: int = 16384,
        timeout: float = 0.5,
        sender_limit_by_binding: bool = False,
    ):
        super(Quel1ConfigSubsystemRoot, self).__init__()
        self._css_addr: str = css_addr
        self._boxtype: Quel1BoxType = boxtype
        self._config_path: Path = config_path if config_path is not None else self.DEFAULT_CONFIG_PATH
        self._config_options: Collection[Quel1ConfigOption] = config_options if config_options is not None else set()
        self._param: Dict[str, Any] = self._load_config()  # TODO: Dict[str, Any] is tentative.
        self._proxy: _ExstickgeProxyBase = self._create_exstickge_proxy(port, timeout, sender_limit_by_binding)

    @abstractmethod
    def _create_exstickge_proxy(self, port: int, timeout: float, sender_limit_by_binding: bool) -> _ExstickgeProxyBase:
        pass

    def _remove_comments(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        s1: Dict[str, Any] = {}
        for k, v in settings.items():
            if not k.startswith("#"):
                if isinstance(v, dict):
                    s1[k] = self._remove_comments(v)
                else:
                    s1[k] = v
        return s1

    def _match_conditional_include(self, directive: Mapping[str, Union[str, Sequence[str]]]) -> bool:
        flag: bool = True
        file: str = ""
        for k, v in directive.items():
            if k == "file":
                if isinstance(v, str):
                    file = v
                else:
                    raise TypeError(f"invalid type of 'file': {k}")
            elif k == "boxtype":
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise TypeError(f"invalid type of 'boxtype': {k}")
                for bt1 in v:
                    if bt1 not in self._boxtype.value:
                        flag = False
            elif k == "option":
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise TypeError(f"invalid type of 'option': {k}")
                for op1 in v:
                    if Quel1ConfigOption(op1) not in self._config_options:
                        flag = False
            else:
                raise ValueError(f"invalid key of conditional include: {k}")

        if file == "":
            raise ValueError(f"no file is specified in conditional include: {directive}")
        return flag

    def _include_config(
        self, directive: Union[str, Mapping[str, str], Sequence[Union[str, Mapping[str, str]]]]
    ) -> Tuple[Dict[str, Any], Set[Quel1ConfigOption]]:
        fired_options: Set[Quel1ConfigOption] = set()

        if isinstance(directive, str) or isinstance(directive, dict):
            directive = [directive]

        config: Dict[str, Any] = {}
        for d1 in directive:
            if isinstance(d1, str):
                with open(self._config_path / d1) as f:
                    logger.info(f"basic config applied: {d1}")
                    config = deep_update(config, json.load(f))
            elif isinstance(d1, dict):
                if self._match_conditional_include(d1):
                    with open(self._config_path / d1["file"]) as f:
                        logger.info(f"conditional config applied: {d1}")
                        config = deep_update(config, json.load(f))
                        if "option" in d1:
                            option = d1["option"]
                            if isinstance(option, str):
                                fired_options.add(Quel1ConfigOption(option))
                            elif isinstance(option, list):
                                fired_options.update({Quel1ConfigOption(o) for o in d1["option"]})
                            else:
                                raise AssertionError
            else:
                raise TypeError(f"malformed template: '{d1}'")
        if "meta" in config:
            del config["meta"]
        return self._remove_comments(config), fired_options

    def _load_config(self) -> Dict[str, Any]:
        logger.info(f"loading configuration settings from '{self._config_path / self._DEFAULT_CONFIG_JSONFILE}'")
        fired_options: Set[Quel1ConfigOption] = set()

        with open(self._config_path / self._DEFAULT_CONFIG_JSONFILE) as f:
            root: Dict[str, Any] = self._remove_comments(json.load(f))

        config = copy.copy(root)
        for k0, directive0 in root.items():
            if k0 == "meta":
                pass
            elif k0 in self._NUM_IC.keys():
                for idx, directive1 in enumerate(directive0):
                    if idx >= self._NUM_IC[k0]:
                        raise ValueError(f"too many {k0.upper()}s are found")
                    config[k0][idx], fired_options_1 = self._include_config(directive1)
                    fired_options.update(fired_options_1)
            else:
                raise ValueError(f"invalid name of IC: '{k0}'")

        for k1, n1 in self._NUM_IC.items():
            if len(config[k1]) != n1:
                raise ValueError(
                    f"lacking config, there should be {n1} instances of '{k1}', "
                    f"but actually have {len(config[k1])} ones"
                )

        for option in self._config_options:
            if option not in fired_options:
                logger.warning(f"config option '{str(option)}' is not applicable")

        return config

    @property
    def ipaddr_css(self) -> str:
        return self._css_addr

    @abstractmethod
    def configure_peripherals(self) -> None:
        """configure ICs other than MxFEs and PLLs for their reference clock.

        :return: None
        """
        pass

    @abstractmethod
    def configure_all_mxfe_clocks(self) -> None:
        """configure PLLs for clocks of MxFEs.

        :return: None
        """
        pass

    @abstractmethod
    def configure_mxfe(
        self,
        group: int,
        *,
        hard_reset: bool,
        soft_reset: bool,
        mxfe_init: bool,
        use_204b: bool,
        ignore_crc_error: bool,
    ) -> bool:
        """configure an MxFE and its related data objects. PLLs for their clock must be set up in advance.

        :param group: an index of a group which the target MxFE belongs to.
        :param hard_reset: enabling hard reset of the MxFE before the initialization if available.
        :param soft_reset: enabling soft reset of the MxFE before the initialization.
        :param mxfe_init: enabling the initialization of the MxFE's, not just for initializing the host-side data
                          objects.
        :param use_204b: using a workaround method to link the MxFE up during the initialization if True.
        :param ignore_crc_error: ignoring CRC error flag at the validation of the link status if True.
        :return: True if the target MxFE is available.
        """
        pass


class Quel1ConfigSubsystemAd9082Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_ad9082(self):
        self._ad9082 = tuple(Ad9082V106(self._proxy, idx, p) for idx, p in enumerate(self._param["ad9082"]))

    @property
    def ad9082(self) -> Tuple[Ad9082V106, ...]:
        return self._ad9082

    def _validate_frequency_info(
        self, group: int, freq_type: str, freq_in_hz: Union[int, None], ftw: Union[NcoFtw, None]
    ) -> Tuple[int, NcoFtw]:
        # Notes: Assuming that group is already validated.
        # Notes: decimation rates can be different among DACs, however,
        #        QuEL-1 doesn't allow such configuration currently.
        if (freq_in_hz is not None and ftw is not None) or (freq_in_hz is None and ftw is None):
            raise ValueError("either of freq_in_hz or ftw is required")
        elif freq_in_hz is not None:
            # TODO: consider to use calc_*_*_ftw_rational().
            if freq_type == "dac_cnco":
                ftw = self.ad9082[group].calc_dac_cnco_ftw(freq_in_hz)
            elif freq_type == "dac_fnco":
                ftw = self.ad9082[group].calc_dac_fnco_ftw(freq_in_hz)
            elif freq_type == "adc_cnco":
                ftw = self.ad9082[group].calc_adc_cnco_ftw(freq_in_hz)
            elif freq_type == "adc_fnco":
                ftw = self.ad9082[group].calc_adc_fnco_ftw(freq_in_hz)
            else:
                raise ValueError(f"invalid freq_type: {freq_in_hz}")
        elif ftw is not None:
            if not isinstance(ftw, NcoFtw):
                raise TypeError("unexpected ftw is given")
            # Notes: freq_in_hz is computed back from ftw for logger messages.
            if freq_type == "dac_cnco":
                freq_in_hz_ = self.ad9082[group].calc_dac_cnco_freq(ftw)
            elif freq_type == "dac_fnco":
                freq_in_hz_ = self.ad9082[group].calc_dac_fnco_freq(ftw)
            elif freq_type == "adc_cnco":
                raise NotImplementedError
            elif freq_type == "adc_fnco":
                raise NotImplementedError
            else:
                raise ValueError(f"invalid freq_type: {freq_type}")
            freq_in_hz = int(freq_in_hz_ + 0.5)
        else:
            raise AssertionError  # never happens

        return freq_in_hz, ftw

    def _get_dac_idx(self, group: int, line: int) -> int:
        self._validate_line(group, line)
        return self._DAC_IDX[(group, line)]

    def _get_channels_of_line(self, group: int, line: int) -> Tuple[int, ...]:
        dac_idx = self._get_dac_idx(group, line)
        dac_ch_idxes = tuple(self._param["ad9082"][group]["tx"]["channel_assign"][f"dac{dac_idx}"])
        return dac_ch_idxes

    def get_num_channels_of_line(self, group: int, line: int) -> int:
        return len(self._get_channels_of_line(group, line))

    def _get_dac_ch_idx(self, group: int, line: int, channel: int) -> int:
        self._validate_channel(group, line, channel)
        return self._get_channels_of_line(group, line)[channel]

    def _get_adc_idx(self, group: int, rline: str) -> int:
        self._validate_rline(group, rline)
        return self._ADC_IDX[(group, rline)]

    def _get_rchannels_of_rline(self, group: int, rline: str) -> Tuple[int, ...]:
        # TODO: determine rchannel based on self._param.
        self._validate_rline(group, rline)
        return self._ADC_CH_IDX[group, rline]

    def get_num_rchannels_of_rline(self, group: int, rline: str) -> int:
        return len(self._get_rchannels_of_rline(group, rline))

    def _get_adc_rch_idx(self, group: int, rline: str, rchannel: int) -> int:
        self._validate_rchannel(group, rline, rchannel)
        return self._get_rchannels_of_rline(group, rline)[rchannel]

    def set_dac_cnco(
        self, group: int, line: int, freq_in_hz: Union[int, None] = None, ftw: Union[NcoFtw, None] = None
    ) -> None:
        """setting CNCO frequency of a transmitter line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :param freq_in_hz: the CNCO frequency of the line in Hz.
        :param ftw: an FTW can be passed instead of freq_in_hz if necessary.
        :return: None
        """
        dac_idx = self._get_dac_idx(group, line)
        freq_in_hz_, ftw_ = self._validate_frequency_info(group, "dac_cnco", freq_in_hz, ftw)
        logger.info(
            f"DAC-CNCO{dac_idx} of MxFE{group} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.delta_a}, {ftw_.modulus_b})"
        )
        self.ad9082[group].set_dac_cnco({dac_idx}, ftw_)

    def get_dac_cnco(self, group: int, line: int) -> float:
        """getting CNCO frequency of a transmitter line.

        :param group: an index of group which the line belongs to.
        :param line: a group-local index of the line.
        :return: the current CNCO frequency of the line in Hz.
        """
        dac_idx = self._get_dac_idx(group, line)
        ftw = self.ad9082[group].get_dac_cnco(dac_idx)
        return self.ad9082[group].calc_dac_cnco_freq(ftw)

    def set_dac_fnco(
        self, group: int, line: int, channel: int, freq_in_hz: Union[int, None] = None, ftw: Union[NcoFtw, None] = None
    ) -> None:
        """setting FNCO frequency of the transmitter channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :param freq_in_hz: the FNCO frequency of the channel in Hz.
        :param ftw: an FTW can be passed instead of freq_in_hz if necessary.
        :return: None
        """
        fnco_idx = self._get_dac_ch_idx(group, line, channel)
        freq_in_hz_, ftw_ = self._validate_frequency_info(group, "dac_fnco", freq_in_hz, ftw)
        logger.info(
            f"DAC-FNCO{fnco_idx} of MxFE{group} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.delta_a}, {ftw_.modulus_b})"
        )
        self.ad9082[group].set_dac_fnco({fnco_idx}, ftw_)

    def get_dac_fnco(self, group: int, line: int, channel: int) -> float:
        """getting FNCO frequency of a transmitter channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :return: the current FNCO frequency of the channel in Hz.
        """
        fnco_idx = self._get_dac_ch_idx(group, line, channel)
        ftw = self.ad9082[group].get_dac_fnco(fnco_idx)
        return self.ad9082[group].calc_dac_fnco_freq(ftw)

    def set_adc_cnco(
        self, group: int, rline: str, freq_in_hz: Union[int, None] = None, ftw: Union[NcoFtw, None] = None
    ):
        """setting CNCO frequency of a receiver line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :param freq_in_hz: the CNCO frequency of the line in Hz.
        :param ftw: an FTW can be passed instead of freq_in_hz if necessary.
        :return: None
        """
        adc_idx = self._get_adc_idx(group, rline)
        freq_in_hz_, ftw_ = self._validate_frequency_info(group, "adc_cnco", freq_in_hz, ftw)
        logger.info(
            f"ADC-CNCO{adc_idx} of MxFE{group} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.delta_a}, {ftw_.modulus_b})"
        )
        self.ad9082[group].set_adc_cnco({adc_idx}, ftw_)

    def get_adc_cnco(self, group: int, rline: str) -> float:
        """getting CNCO frequency of a receiver line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: the current CNCO frequency of the line in Hz.
        """
        adc_idx = self._get_adc_idx(group, rline)
        ftw = self.ad9082[group].get_adc_cnco(adc_idx)
        return self.ad9082[group].calc_adc_cnco_freq(ftw)

    def set_adc_fnco(
        self,
        group: int,
        rline: str,
        rchannel: int,
        freq_in_hz: Union[int, None] = None,
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
        fnco_idx = self._get_adc_rch_idx(group, rline, rchannel)
        freq_in_hz_, ftw_ = self._validate_frequency_info(group, "adc_fnco", freq_in_hz, ftw)
        logger.info(
            f"ADC-FNCO{fnco_idx} of MxFE{group} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.delta_a}, {ftw_.modulus_b})"
        )
        self.ad9082[group].set_adc_fnco({fnco_idx}, ftw_)

    def get_adc_fnco(self, group: int, rline: str, rchannel: int) -> float:
        """getting FNCO frequency of a receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :return: the current FNCO frequency of the channel in Hz.
        """
        fnco_idx = self._get_adc_rch_idx(group, rline, rchannel)
        ftw = self.ad9082[group].get_adc_fnco(fnco_idx)
        return self.ad9082[group].calc_adc_fnco_freq(ftw)

    def get_link_status(self, group: int) -> Tuple[int, int]:
        """getting the status of the datalink between a MxFE in a group and the FPGA.

        :param group: an index of the group which the target MxFE belongs to.
        :return: the content of the registers of the MxFE at the addresses of 0x55E and 0x5BB.
        """
        self._validate_group(group)
        return self._ad9082[group].get_link_status()

    def set_fullscale_current(self, group: int, line: int, fsc: int) -> None:
        """setting the full-scale current of a DAC of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line
        :param fsc: full-scale current of the DAC of the line in uA.
        :return: None
        """
        dac_idx = self._get_dac_idx(group, line)
        self.ad9082[group].set_fullscale_current(1 << dac_idx, fsc)

    def get_fullscale_current(self, group: int, line: int) -> int:
        """geting the current full scale current of a DAC of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line
        :return: the current full-scale current of the DAC in uA
        """
        dac_idx = self._get_dac_idx(group, line)
        return self.ad9082[group].get_fullscale_current(dac_idx)

    def get_main_interpolation_rate(self, group: int) -> int:
        """getting the current main interpolation rate of a MxFE of a group.

        :param group: an index of the group which the target MxFE belongs to.
        :return: main interpolation rate
        """
        self._validate_group(group)
        return self.ad9082[group].get_main_interpolation_rate()

    def get_channel_interpolation_rate(self, group: int) -> int:
        """getting the current channel interpolation rate of a MxFE of a group.

        :param group: a group which the target MxFE belongs to.
        :return: channel interpolation rate
        """
        self._validate_group(group)
        return self.ad9082[group].get_channel_interpolation_rate()

    def get_virtual_adc_select(self, group: int) -> List[int]:
        """getting converter select matrix of ADC virtual converters after FDDC.

        :param group: a group which the target MxFE belongs to.
        :return: converter select matrix in list
        """
        self._validate_group(group)
        return self.ad9082[group].get_virtual_adc_select()

    def get_ad9082_temperatures(self, group: int) -> Tuple[int, int]:
        """getting the current die temperatures of a MxFE of a group.

        :param group: an index of the group which the target MxFE belongs to.
        :return: a pair of maximum and minimum temperatures in the die.
        """
        self._validate_group(group)
        temp_max, temp_min = self.ad9082[group].get_temperatures()
        return temp_max, temp_min


class Quel1ConfigSubsystemLmx2594Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    MIN_FREQ_MULTIPLIER: Final[int] = 75
    MAX_FREQ_MULTIPLIER: Final[int] = 150

    _LO_IDX: Dict[Tuple[int, Union[int, str]], int]

    def _construct_lmx2594(self):
        self._lmx2594: Tuple[Lmx2594, ...] = tuple(Lmx2594(self._proxy, idx) for idx in range(self._NUM_IC["lmx2594"]))
        self._lmx2594_helper: Tuple[Lmx2594ConfigHelper, ...] = tuple(Lmx2594ConfigHelper(ic) for ic in self._lmx2594)

    @property
    def lmx2594(self) -> Tuple[Lmx2594, ...]:
        return self._lmx2594

    @property
    def lmx2594_helper(self) -> Tuple[Lmx2594ConfigHelper, ...]:
        return self._lmx2594_helper

    def init_lmx2594(self, idx) -> bool:
        ic, helper = self._lmx2594[idx], self._lmx2594_helper[idx]
        param: Dict[str, Dict[str, Union[int, bool]]] = self._param["lmx2594"][idx]
        ic.soft_reset()
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()
        return ic.calibrate()

    def _validate_freq_multiplier(self, freq_multiplier: int):
        if not isinstance(freq_multiplier, int):
            raise TypeError(f"unexpected frequency multiplier: {freq_multiplier}")
        if not (self.MIN_FREQ_MULTIPLIER <= freq_multiplier <= self.MAX_FREQ_MULTIPLIER):
            raise TypeError(f"invalid frequency multiplier: {freq_multiplier}")

    def set_lo_multiplier(self, group: int, line: Union[int, str], freq_multiplier: int) -> bool:
        """setting the frequency multiplier of a PLL of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the PLL belongs to.
        :param freq_multiplier: the frequency multiplier to set.
        :return: True if the frequency calibration is completed successfully.
        """
        self._validate_line_or_rline(group, line)
        if (group, line) not in self._LO_IDX:
            raise ValueError("no LO is available for (group{group}, line{line})")
        lo_idx: int = self._LO_IDX[(group, line)]
        self._validate_freq_multiplier(freq_multiplier)

        logger.info(f"updating LO frequency[{lo_idx}] of {self._css_addr} to {freq_multiplier * 100}MHz")
        ic, helper = self._lmx2594[lo_idx], self._lmx2594_helper[lo_idx]
        helper.write_field("R34", pll_n_18_16=(freq_multiplier >> 16) & 0x0007)
        helper.write_field("R36", pll_n=(freq_multiplier & 0xFFFF))
        helper.flush()
        return ic.calibrate()

    def get_lo_multiplier(self, group: int, line: Union[int, str]) -> int:
        """get the current frequency multiplier of a PLL of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line which the PLL belongs to.
        :return: the current frequency multiplier of the PLL.
        """
        self._validate_line_or_rline(group, line)
        if (group, line) not in self._LO_IDX:
            raise ValueError("no LO is available for (group{group}, line{line})")
        lo_idx: int = self._LO_IDX[(group, line)]

        helper = self._lmx2594_helper[lo_idx]
        return (getattr(helper.read_reg("R34"), "pll_n_18_16") << 16) + getattr(helper.read_reg("R36"), "pll_n")


class Quel1ConfigSubsystemAd6780Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _MIXER_IDX: Dict[Tuple[int, int], int]

    def _construct_adrf6780(self):
        self._adrf6780: Tuple[Adrf6780, ...] = tuple(
            Adrf6780(self._proxy, idx) for idx in range(self._NUM_IC["adrf6780"])
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

    def init_adrf6780(self, idx, skip_revision_check: bool = False):
        # Notes: skip_revision_check is a workaround for ill-designed board (prototype 2-8G mixer board).
        ic, helper = self._adrf6780[idx], self._adrf6780_helper[idx]
        param: Dict[str, Dict[str, Union[int, bool]]] = self._param["adrf6780"][idx]
        ic.soft_reset(parity_enable=False)
        if not skip_revision_check:
            rev = ic.check_id()
            logger.info(f"chip revision of ADRF6780[{idx}] of {self._css_addr} is {rev}")
        else:
            logger.info("chip revision check is skipped")
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()

    def _validate_sideband(self, sideband: str):
        if not isinstance(sideband, str):
            raise TypeError(f"unexpected sideband: {sideband}")
        if sideband not in {"L", "U"}:
            raise ValueError(f"unexpected sideband: {sideband}")

    def _get_mixer_idx(self, group, line):
        if (group, line) in self._MIXER_IDX:
            return self._MIXER_IDX[(group, line)]
        raise ValueError(f"no mixer is available for (group:{group}, line:{line})")

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
            raise ValueError(f"no mixer is available for (group{group}, line{line})")

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
        self._ad5328: Tuple[Ad5328, ...] = tuple(Ad5328(self._proxy, idx) for idx in range(self._NUM_IC["ad5328"]))
        self._ad5328_helper: Tuple[Ad5328ConfigHelper, ...] = tuple(Ad5328ConfigHelper(ic) for ic in self._ad5328)

    @property
    def ad5328(self) -> Tuple[Ad5328, ...]:
        return self._ad5328

    @property
    def ad5328_helper(self) -> Tuple[Ad5328ConfigHelper, ...]:
        return self._ad5328_helper

    def init_ad5328(self, idx):
        ic, helper = self._ad5328[idx], self._ad5328_helper[idx]
        param: Dict[str, Dict[str, Union[int, bool]]] = self._param["ad5328"][idx]
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
            raise ValueError(f"no variable attenuator is available for (group:{group}, line:{line})")

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


class Quel1ConfigSubsystemRfswitch(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _RFSWITCH_NAME: Dict[Tuple[int, Union[int, str]], Tuple[int, str]]

    def _construct_rfswitch(self, gpio_idx: int):
        if self._boxtype == Quel1BoxType.QuEL1_TypeA:
            gpio_tmp: AbstractRfSwitchArrayMixin = Quel1TypeARfSwitchArray(self._proxy, gpio_idx)
        elif self._boxtype == Quel1BoxType.QuEL1_TypeB:
            gpio_tmp = Quel1TypeBRfSwitchArray(self._proxy, gpio_idx)
        elif self._boxtype in {Quel1BoxType.QuBE_TypeA, Quel1BoxType.QuBE_TypeB}:
            gpio_tmp = QubeRfSwitchArray(self._proxy, gpio_idx)
        else:
            raise ValueError("invalid boxtype: {boxtype}")
        self._rfswitch_gpio_idx = gpio_idx
        self._rfswitch: AbstractRfSwitchArrayMixin = gpio_tmp
        self._rfswitch_helper: RfSwitchArrayConfigHelper = RfSwitchArrayConfigHelper(self._rfswitch)

    @property
    def rfswitch(self) -> AbstractRfSwitchArrayMixin:
        return self._rfswitch

    @property
    def rfswitch_helper(self) -> RfSwitchArrayConfigHelper:
        return self._rfswitch_helper

    def init_rfswitch(self):
        helper = self._rfswitch_helper
        param: Dict[str, Dict[str, Dict[str, Union[int, bool]]]] = self._param["gpio"][self._rfswitch_gpio_idx]
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()

    def _alternate_loop_rfswitch(self, group: int, **sw_update: bool) -> None:
        # Note: assuming that group is already validated.
        ic, helper = self._rfswitch, self._rfswitch_helper
        current_sw: int = ic.read_reg(0)
        helper.write_field(f"Group{group}", **sw_update)
        helper.flush()
        altered_sw: int = ic.read_reg(0)
        time.sleep(0.01)
        logger.info(
            f"alter the state of RF switch array of {self._css_addr} from {current_sw:014b} to {altered_sw:014b}"
        )

    def _get_rfswitch_name(self, group: int, line: Union[int, str]) -> Tuple[int, str]:
        if (group, line) in self._RFSWITCH_NAME:
            return self._RFSWITCH_NAME[(group, line)]
        else:
            raise ValueError("no switch available for group:{group}, line:{line}")

    def pass_line(self, group: int, line: Union[int, str]) -> None:
        """allowing a line to emit RF signal from its corresponding SMA connector.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: None
        """
        logger.info(f"making (group:{group}, line:{line}) passing")
        swgroup, swname = self._get_rfswitch_name(group, line)
        self._alternate_loop_rfswitch(swgroup, **{swname: False})

    def block_line(self, group: int, line: Union[int, str]) -> None:
        """blocking a line to emit RF signal from its corresponding SMA connector.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: None
        """
        logger.info(f"making (group:{group}, line:{line}) blocked")
        swgroup, swname = self._get_rfswitch_name(group, line)
        self._alternate_loop_rfswitch(swgroup, **{swname: True})

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
        self._alternate_loop_rfswitch(swgroup, **{swname: True})

    def deactivate_monitor_loop(self, group: int) -> None:
        """disabling an internal loop-back of a monitor path of a group.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"deactivate monitor loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "m")
        self._alternate_loop_rfswitch(swgroup, **{swname: False})

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
        self._alternate_loop_rfswitch(swgroup, **{swname: True})

    def deactivate_read_loop(self, group: int) -> None:
        """disabling an internal loop-back of a read path of a group.

        :param group: an index of the group.
        :return: None
        """
        logger.info(f"deactivate read loop (group:{group})")
        swgroup, swname = self._get_rfswitch_name(group, "r")
        self._alternate_loop_rfswitch(swgroup, **{swname: False})

    def is_loopedback_read(self, group: int) -> bool:
        """checking if a read loop-back path of a group is activated or not.

        :param group: an index of the group.
        :return: True if the read loop-back path of the group is activated.
        """
        swgroup, swname = self._get_rfswitch_name(group, "r")
        return getattr(self._rfswitch_helper.read_reg(swgroup), swname)


class Quel1ConfigSubsystemGpioMixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_gpio(self):
        self._gpio: Tuple[GenericGpio, ...] = tuple(
            GenericGpio(self._proxy, idx) for idx in range(self._NUM_IC["gpio"])
        )
        self._gpio_helper: Tuple[GenericGpioConfigHelper, ...] = tuple(GenericGpioConfigHelper(ic) for ic in self._gpio)

    @property
    def gpio(self) -> Tuple[GenericGpio, ...]:
        return self._gpio

    @property
    def gpio_helper(self) -> Tuple[GenericGpioConfigHelper, ...]:
        return self._gpio_helper

    def init_gpio(self, idx):
        helper = self.gpio_helper[idx]
        param: Dict[str, Dict[str, Union[int, bool]]] = self._param["gpio"][idx]
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()
