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
from quel_ic_config.exstickge_sock_client import _ExstickgeProxyBase
from quel_ic_config.generic_gpio import GenericGpioConfigHelper
from quel_ic_config.lmx2594 import Lmx2594ConfigHelper
from quel_ic_config.mixerboard_gpio import MixerboardGpioConfigHelper
from quel_ic_config.pathselectorboard_gpio import PathselectorboardGpioConfigHelper
from quel_ic_config.powerboard_pwm import PowerboardPwmConfigHelper
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption, Quel1Feature
from quel_ic_config.quel_ic import (
    Ad5328,
    Ad7490,
    Ad9082V106,
    Adrf6780,
    GenericGpio,
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
        "_param",
        "_boxtype",
        "_features",
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
        "_ad7490",
        "_rfswitch",
        "_rfswitch_gpio_idx",
        "_rfswitch_helper",
        "_gpio",
        "_gpio_helper",
        "_mixerboard_gpio",
        "_mixerboard_gpio_helper",
        "_pathselectorboard_gpio",
        "_pathselectorboard_gpio_helper",
        "_powerboard_pwm",
        "_powerboard_pwm_helper",
        "_tempctrl_watcher",
        "_tempctrl_auto_start_at_linkup",
    )

    _DEFAULT_CONFIG_JSONFILE: str
    _NUM_IC: Dict[str, int]
    _GROUPS: Set[int]
    _MXFE_IDXS: Set[int]
    _DAC_IDX: Dict[Tuple[int, int], Tuple[int, int]]
    _ADC_IDX: Dict[Tuple[int, str], Tuple[int, int]]
    _ADC_CH_IDX: Dict[Tuple[int, str], Tuple[int, ...]]

    def __init__(self):
        # variable defined in the importer class
        self._css_addr: str
        self._boxtype: Quel1BoxType
        self._features: Collection[Quel1Feature]
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

    def get_all_mxfes(self) -> Set[int]:
        return self._MXFE_IDXS

    def _validate_mxfe(self, mxfe_idx: int) -> None:
        if mxfe_idx not in self._MXFE_IDXS:
            raise ValueError("an invalid mxfe: {mxfe_idx}")

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
    DEFAULT_CONFIG_PATH: Final[Path] = Path(osp.dirname(__file__)) / "settings"

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        features: Union[Collection[Quel1Feature], None] = None,
        config_path: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        port: int = _ExstickgeProxyBase._DEFAULT_PORT,
        timeout: float = _ExstickgeProxyBase._DEFAULT_RESPONSE_TIMEOUT,
        sender_limit_by_binding: bool = False,
    ):
        super(Quel1ConfigSubsystemRoot, self).__init__()
        self._css_addr: str = css_addr
        self._boxtype: Quel1BoxType = boxtype
        self._features: Set[Quel1Feature] = set(features) if features is not None else {Quel1Feature.SINGLE_ADC}
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
            elif k == "boxtype":  # OR
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise TypeError(f"invalid type of 'boxtype': {k}")
                if self.boxtype.value[1] not in v:
                    flag = False
            elif k == "option":  # AND
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise TypeError(f"invalid type of 'option': {k}")
                for op1 in v:
                    if Quel1ConfigOption(op1) not in self._config_options:
                        flag = False
            elif k == "feature":  # AND
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise TypeError(f"invalid type of 'option': {k}")
                for ft1 in v:
                    if Quel1Feature(ft1) not in self._features:
                        flag = False
            elif k == "otherwise":
                if not isinstance(v, str):
                    raise TypeError(f"invalid type of 'otherwise': {k}")
            else:
                raise ValueError(f"invalid key of conditional include: {k}")

        if file == "":
            raise ValueError(f"no file is specified in conditional include: {directive}")
        return flag

    def _include_config(
        self, directive: Union[str, Mapping[str, str], Sequence[Union[str, Mapping[str, str]]]], label_for_log: str
    ) -> Tuple[Dict[str, Any], Set[Quel1ConfigOption]]:
        fired_options: Set[Quel1ConfigOption] = set()

        if isinstance(directive, str) or isinstance(directive, dict):
            directive = [directive]

        config: Dict[str, Any] = {}
        for d1 in directive:
            if isinstance(d1, str):
                with open(self._config_path / d1) as f:
                    logger.info(f"basic config applied to {label_for_log}: {d1}")
                    config = deep_update(config, json.load(f))
            elif isinstance(d1, dict):
                if self._match_conditional_include(d1):
                    with open(self._config_path / d1["file"]) as f:
                        logger.info(f"conditional config applied to {label_for_log}: {d1}")
                        config = deep_update(config, json.load(f))
                        if "option" in d1:
                            option = d1["option"]
                            if isinstance(option, str):
                                fired_options.add(Quel1ConfigOption(option))
                            elif isinstance(option, list):
                                fired_options.update({Quel1ConfigOption(o) for o in option})
                            else:
                                raise AssertionError
                elif "otherwise" in d1:
                    with open(self._config_path / d1["otherwise"]) as f:
                        logger.info(f"'otherwise' part of conditional config applied to {label_for_log}: {d1}")
                        config = deep_update(config, json.load(f))
            else:
                raise TypeError(f"malformed template at {label_for_log}: '{d1}'")
        if "meta" in config:
            del config["meta"]
        return self._remove_comments(config), fired_options

    def _load_config(self) -> Dict[str, Any]:
        logger.info(f"loading configuration settings from '{self._config_path / self._DEFAULT_CONFIG_JSONFILE}'")
        logger.info(f"boxtype = {self.boxtype}")
        logger.info(f"config_options = {self._config_options}")
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
                    config[k0][idx], fired_options_1 = self._include_config(directive1, label_for_log=f"{k0}[{idx}]")
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
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ) -> None:
        """configure ICs other than MxFEs and PLLs for their reference clock.

        :param ignore_access_failure_of_adrf6780: a collection of index of ADRF6780 to ignore access failure during the
                                                  initialization process
        :param ignore_lock_failure_of_lmx2594: a collection of index of LMX2594 to ignore PLL lock failure during the
                                               initialization process
        :return: None
        """
        pass

    @abstractmethod
    def configure_all_mxfe_clocks(self, ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None) -> None:
        """configure PLLs for clocks of MxFEs.

        :param ignore_lock_failure_of_lmx2594: a collection of index of LMX2594 to ignore PLL lock failure during the
                                               initialization process
        :return: None
        """
        pass

    @abstractmethod
    def configure_mxfe(
        self,
        mxfe_idx: int,
        *,
        hard_reset: bool,
        soft_reset: bool,
        mxfe_init: bool,
        use_204b: bool,
        use_bg_cal: bool,
        ignore_crc_error: bool,
    ) -> bool:
        """configure an MxFE and its related data objects. PLLs for their clock must be set up in advance.

        :param mxfe_idx: an index of a group which the target MxFE belongs to.
        :param hard_reset: enabling hard reset of the MxFE before the initialization if available.
        :param soft_reset: enabling soft reset of the MxFE before the initialization.
        :param mxfe_init: enabling the initialization of the MxFE's, not just for initializing the host-side data
                          objects.
        :param use_204b: using a workaround method to link the MxFE up during the initialization if True.
        :param ignore_crc_error: ignoring CRC error flag at the validation of the link status if True.
        :return: True if the target MxFE is available.
        """
        pass

    def terminate(self):
        self._proxy.terminate()


class Quel1ConfigSubsystemAd9082Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_ad9082(self):
        self._ad9082 = tuple(Ad9082V106(self._proxy, idx, p) for idx, p in enumerate(self._param["ad9082"]))

    @property
    def ad9082(self) -> Tuple[Ad9082V106, ...]:
        return self._ad9082

    def check_link_status(self, mxfe_idx: int, mxfe_init: bool = False, ignore_crc_error: bool = False) -> bool:
        link_status, crc_flag = self.ad9082[mxfe_idx].get_link_status()
        judge: bool = False
        if link_status == 0xE0:
            if crc_flag == 0x01:
                judge = True
            elif crc_flag == 0x11 and ignore_crc_error:
                judge = True

        if judge:
            if crc_flag == 0x01:
                if mxfe_init:
                    logger.info(
                        f"{self._css_addr}:AD9082-#{mxfe_idx} links up successfully "
                        f"(link status = 0x{link_status:02x}, crc_flag = 0x{crc_flag:02x})"
                    )
                else:
                    logger.info(
                        f"{self._css_addr}:AD9082-#{mxfe_idx} has linked up healthy "
                        f"(link status = 0x{link_status:02x}, crc_flag = 0x{crc_flag:02x})"
                    )
            else:
                if mxfe_init:
                    logger.warning(
                        f"{self._css_addr}:AD9082-#{mxfe_idx} links up successfully with ignored crc error "
                        f"(link status = 0x{link_status:02x}, crc_flag = 0x{crc_flag:02x})"
                    )
                else:
                    logger.warning(
                        f"{self._css_addr}:AD9082-#{mxfe_idx} has linked up with ignored crc error "
                        f"(link status = 0x{link_status:02x}, crc_flag = 0x{crc_flag:02x})"
                    )
        else:
            if mxfe_init:
                logger.warning(
                    f"{self._css_addr}:AD9082-#{mxfe_idx} fails to link up "
                    f"(link_status = 0x{link_status:02x}, crc_flag = 0x{crc_flag:02x})"
                )
            else:
                logger.warning(
                    f"{self._css_addr}:AD9082-#{mxfe_idx} has not linked up yet "
                    f"(link_status = 0x{link_status:02x}, crc_flag = 0x{crc_flag:02x})"
                )

        return judge

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
                ftw = self.ad9082[mxfe_idx].calc_dac_cnco_ftw(freq_in_hz)
            elif freq_type == "dac_fnco":
                ftw = self.ad9082[mxfe_idx].calc_dac_fnco_ftw(freq_in_hz)
            elif freq_type == "adc_cnco":
                ftw = self.ad9082[mxfe_idx].calc_adc_cnco_ftw(freq_in_hz)
            elif freq_type == "adc_fnco":
                ftw = self.ad9082[mxfe_idx].calc_adc_fnco_ftw(freq_in_hz)
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

    def _get_dac_idx(self, group: int, line: int) -> Tuple[int, int]:
        self._validate_line(group, line)
        return self._DAC_IDX[(group, line)]

    def _get_channels_of_line(self, group: int, line: int) -> Tuple[int, ...]:
        mxfe_idx, dac_idx = self._get_dac_idx(group, line)
        dac_ch_idxes = tuple(self._param["ad9082"][mxfe_idx]["dac"]["channel_assign"][f"dac{dac_idx}"])
        return dac_ch_idxes

    def get_num_channels_of_line(self, group: int, line: int) -> int:
        return len(self._get_channels_of_line(group, line))

    def _get_dac_ch_idx(self, group: int, line: int, channel: int) -> int:
        self._validate_channel(group, line, channel)
        return self._get_channels_of_line(group, line)[channel]

    def _get_adc_idx(self, group: int, rline: str) -> Tuple[int, int]:
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
        self, group: int, line: int, freq_in_hz: Union[float, None] = None, ftw: Union[NcoFtw, None] = None
    ) -> None:
        """setting CNCO frequency of a transmitter line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :param freq_in_hz: the CNCO frequency of the line in Hz.
        :param ftw: an FTW can be passed instead of freq_in_hz if necessary.
        :return: None
        """
        mxfe_idx, dac_idx = self._get_dac_idx(group, line)
        freq_in_hz_, ftw_ = self._validate_frequency_info(mxfe_idx, "dac_cnco", freq_in_hz, ftw)
        logger.info(
            f"DAC-CNCO{dac_idx} of MxFE{mxfe_idx} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.delta_a}, {ftw_.modulus_b})"
        )
        self.ad9082[mxfe_idx].set_dac_cnco({dac_idx}, ftw_)

    def get_dac_cnco(self, group: int, line: int) -> float:
        """getting CNCO frequency of a transmitter line.

        :param group: an index of group which the line belongs to.
        :param line: a group-local index of the line.
        :return: the current CNCO frequency of the line in Hz.
        """
        mxfe_idx, dac_idx = self._get_dac_idx(group, line)
        ftw = self.ad9082[mxfe_idx].get_dac_cnco(dac_idx)
        return self.ad9082[mxfe_idx].calc_dac_cnco_freq(ftw)

    def is_equivalent_dac_cnco(self, group: int, line: int, freq0: float, freq1: float) -> bool:
        mxfe_idx, _ = self._get_dac_idx(group, line)
        return self.ad9082[mxfe_idx].is_equivalent_dac_cnco(freq0, freq1)

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
        mxfe_idx, _ = self._get_dac_idx(group, line)
        fnco_idx = self._get_dac_ch_idx(group, line, channel)
        freq_in_hz_, ftw_ = self._validate_frequency_info(mxfe_idx, "dac_fnco", freq_in_hz, ftw)
        logger.info(
            f"DAC-FNCO{fnco_idx} of MxFE{mxfe_idx} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.delta_a}, {ftw_.modulus_b})"
        )
        self.ad9082[mxfe_idx].set_dac_fnco({fnco_idx}, ftw_)

    def get_dac_fnco(self, group: int, line: int, channel: int) -> float:
        """getting FNCO frequency of a transmitter channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :return: the current FNCO frequency of the channel in Hz.
        """
        mxfe_idx, _ = self._get_dac_idx(group, line)
        fnco_idx = self._get_dac_ch_idx(group, line, channel)
        ftw = self.ad9082[mxfe_idx].get_dac_fnco(fnco_idx)
        return self.ad9082[mxfe_idx].calc_dac_fnco_freq(ftw)

    def is_equivalent_dac_fnco(self, group: int, line: int, freq0: float, freq1: float) -> bool:
        mxfe_idx, _ = self._get_dac_idx(group, line)
        return self.ad9082[mxfe_idx].is_equivalent_dac_fnco(freq0, freq1)

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
        mxfe_idx, adc_idx = self._get_adc_idx(group, rline)
        freq_in_hz_, ftw_ = self._validate_frequency_info(mxfe_idx, "adc_cnco", freq_in_hz, ftw)
        logger.info(
            f"ADC-CNCO{adc_idx} of MxFE{mxfe_idx} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.delta_a}, {ftw_.modulus_b})"
        )
        self.ad9082[mxfe_idx].set_adc_cnco({adc_idx}, ftw_)

    def get_adc_cnco(self, group: int, rline: str) -> float:
        """getting CNCO frequency of a receiver line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :return: the current CNCO frequency of the line in Hz.
        """
        mxfe_idx, adc_idx = self._get_adc_idx(group, rline)
        ftw = self.ad9082[mxfe_idx].get_adc_cnco(adc_idx)
        return self.ad9082[mxfe_idx].calc_adc_cnco_freq(ftw)

    def is_equivalent_adc_cnco(self, group: int, rline: str, freq0: float, freq1: float) -> bool:
        mxfe_idx, _ = self._get_adc_idx(group, rline)
        return self.ad9082[mxfe_idx].is_equivalent_adc_cnco(freq0, freq1)

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
        mxfe_idx, _ = self._get_adc_idx(group, rline)
        fnco_idx = self._get_adc_rch_idx(group, rline, rchannel)
        freq_in_hz_, ftw_ = self._validate_frequency_info(mxfe_idx, "adc_fnco", freq_in_hz, ftw)
        logger.info(
            f"ADC-FNCO{fnco_idx} of MxFE{mxfe_idx} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.delta_a}, {ftw_.modulus_b})"
        )
        self.ad9082[mxfe_idx].set_adc_fnco({fnco_idx}, ftw_)

    def get_adc_fnco(self, group: int, rline: str, rchannel: int) -> float:
        """getting FNCO frequency of a receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the chennel.
        :return: the current FNCO frequency of the channel in Hz.
        """
        mxfe_idx, _ = self._get_adc_idx(group, rline)
        fnco_idx = self._get_adc_rch_idx(group, rline, rchannel)
        ftw = self.ad9082[mxfe_idx].get_adc_fnco(fnco_idx)
        return self.ad9082[mxfe_idx].calc_adc_fnco_freq(ftw)

    def is_equivalent_adc_fnco(self, group: int, rline: str, freq0: float, freq1: float) -> bool:
        mxfe_idx, _ = self._get_adc_idx(group, rline)
        return self.ad9082[mxfe_idx].is_equivalent_adc_fnco(freq0, freq1)

    def set_pair_cnco(self, group_dac: int, line_dac: int, group_adc: int, rline_adc: str, freq_in_hz: float) -> None:
        dac_mxfe_idx, dac_idx = self._get_dac_idx(group_dac, line_dac)
        adc_mxfe_idx, adc_idx = self._get_adc_idx(group_adc, rline_adc)

        dac_clk = self.ad9082[dac_mxfe_idx].device.dev_info.dac_freq_hz
        adc_clk = self.ad9082[adc_mxfe_idx].device.dev_info.adc_freq_hz
        # TODO: relax constrains by-need basis.
        if dac_clk % adc_clk != 0:
            raise RuntimeError("ratio of dac_clk (= {dac_clk}Hz) and adc_clk (= {adc_clk}Hz) is not N:1")
        ratio: int = dac_clk // adc_clk

        freq_in_hz_, dac_ftw = self._validate_frequency_info(dac_mxfe_idx, "dac_cnco", freq_in_hz, None)
        adc_ftw = NcoFtw()
        adc_ftw.ftw = dac_ftw.ftw * ratio
        logger.info(
            f"DAC-CNCO{dac_idx} of MxFE{dac_mxfe_idx} and ADC-CNCO{adc_idx} of MxFE{adc_mxfe_idx} of {self._css_addr} "
            f"are set to {freq_in_hz_}Hz "
            f"(dac_ftw = {dac_ftw.ftw}, {dac_ftw.delta_a}, {dac_ftw.modulus_b})"
            f"(adc_ftw = {adc_ftw.ftw}, {adc_ftw.delta_a}, {adc_ftw.modulus_b})"
        )
        self.ad9082[dac_mxfe_idx].set_dac_cnco({dac_idx}, dac_ftw)
        self.ad9082[adc_mxfe_idx].set_adc_cnco({adc_idx}, adc_ftw)

    def get_link_status(self, mxfe_idx: int) -> Tuple[int, int]:
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
        mxfe_idx, dac_idx = self._get_dac_idx(group, line)
        self.ad9082[mxfe_idx].set_fullscale_current(1 << dac_idx, fsc)

    def get_fullscale_current(self, group: int, line: int) -> int:
        """geting the current full scale current of a DAC of a line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line
        :return: the current full-scale current of the DAC in uA
        """
        mxfe_idx, dac_idx = self._get_dac_idx(group, line)
        return self.ad9082[mxfe_idx].get_fullscale_current(dac_idx)

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

    def get_ad9082_temperatures(self, mxfe_idx: int) -> Tuple[int, int]:
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

    def init_lmx2594(self, idx: int, ignore_lock_failure: bool = False) -> bool:
        # Notes: ignore_lock_failure is prepared for experimental purposes.
        ic, helper = self._lmx2594[idx], self._lmx2594_helper[idx]
        param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]] = self._param["lmx2594"][idx]
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
            raise ValueError("no LO is available for (group:{group}, line:{line})")
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

    def init_adrf6780(self, idx, ignore_id_mismatch: bool = False) -> bool:
        # Notes: ignore_id_mismatch is a workaround for some ill-designed boards.
        ic, helper = self._adrf6780[idx], self._adrf6780_helper[idx]
        param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]] = self._param["adrf6780"][idx]
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
            raise ValueError(f"no mixer is available for (group:{group}, line:{line})")

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

    def init_ad5328(self, idx) -> None:
        ic, helper = self._ad5328[idx], self._ad5328_helper[idx]
        param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]] = self._param["ad5328"][idx]
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

    def init_rfswitch(self) -> None:
        helper = self._rfswitch_helper
        param: Dict[str, Dict[str, Dict[str, Union[int, bool]]]] = self._param["gpio"][self._rfswitch_gpio_idx]
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
            raise ValueError("no switch available for group:{group}, line:{line}")

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

    def init_gpio(self, idx) -> None:
        helper = self.gpio_helper[idx]
        param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]] = self._param["gpio"][idx]
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()


class Quel1ConfigSubsystemMixerboardGpioMixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_mixerboard_gpio(self):
        self._mixerboard_gpio: Tuple[MixerboardGpio, ...] = tuple(
            MixerboardGpio(self._proxy, idx) for idx in range(self._NUM_IC["mixerboard_gpio"])
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

    def init_mixerboard_gpio(self, idx) -> None:
        helper = self.mixerboard_gpio_helper[idx]
        param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]] = self._param["mixerboard_gpio"][idx]
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
            PathselectorboardGpio(self._proxy, idx) for idx in range(self._NUM_IC["pathselectorboard_gpio"])
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

    def init_pathselectorboard_gpio(self, idx) -> None:
        helper = self.pathselectorboard_gpio_helper[idx]
        param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]] = self._param["pathselectorboard_gpio"][idx]
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
            raise ValueError("no switch available for group:{group}, line:{line}")

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
        self._ad7490: Tuple[Ad7490, ...] = tuple(Ad7490(self._proxy, idx) for idx in range(self._NUM_IC["ad7490"]))

    @property
    def ad7490(self) -> Tuple[Ad7490, ...]:
        return self._ad7490

    def init_ad7490(self, idx) -> None:
        param: Mapping[str, Mapping[str, Mapping[str, Union[int, bool]]]] = self._param["ad7490"][idx]
        self._ad7490[idx].set_default_config(**param["registers"]["Config"])
        # Notes: default_config is not applied to the IC now because it'll be applied just before reading channels.


class Quel1ConfigSubsystemPowerboardPwmMixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_powerboard_pwm(self):
        self._powerboard_pwm: Tuple[PowerboardPwm, ...] = tuple(
            PowerboardPwm(self._proxy, idx) for idx in range(self._NUM_IC["powerboard_pwm"])
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
