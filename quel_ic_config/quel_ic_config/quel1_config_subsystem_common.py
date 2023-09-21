import copy
import json
import logging
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Final, List, Mapping, Sequence, Set, Tuple, Union

from pydantic.utils import deep_update

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

    DEFAULT_CONFIG_JSONFILE: str = "non_existent"
    NUM_IC: Dict[str, int]

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

    _validate_group_and_line_out: Callable[[int, int], None]


class Quel1ConfigSubsystemRoot(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()
    NUM_IC: Dict[str, int] = {}

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        config_path: Path,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        port: int = 16384,
        timeout: float = 0.5,
        sender_limit_by_binding: bool = False,
    ):
        super(Quel1ConfigSubsystemRoot, self).__init__()
        self._css_addr: str = css_addr
        self._boxtype: Quel1BoxType = boxtype
        self._config_path: Path = config_path
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
    ) -> Dict[str, Any]:
        if isinstance(directive, str) or isinstance(directive, dict):
            directive = [directive]

        config: Dict[str, Any] = {}
        for d1 in directive:
            if isinstance(d1, str):
                with open(self._config_path / d1) as f:
                    config = deep_update(config, json.load(f))
            elif isinstance(d1, dict):
                if self._match_conditional_include(d1):
                    with open(self._config_path / d1["file"]) as f:
                        config = deep_update(config, json.load(f))
            else:
                raise TypeError(f"malformed template: '{d1}'")
        if "meta" in config:
            del config["meta"]
        return self._remove_comments(config)

    def _load_config(self) -> Dict[str, Any]:
        with open(self._config_path / self.DEFAULT_CONFIG_JSONFILE) as f:
            root: Dict[str, Any] = self._remove_comments(json.load(f))

        config = copy.copy(root)
        for k0, directive0 in root.items():
            if k0 == "meta":
                pass
            elif k0 in self.NUM_IC.keys():
                for idx, directive1 in enumerate(directive0):
                    if idx >= self.NUM_IC[k0]:
                        raise ValueError(f"too many {k0.upper()}s are found")
                    config[k0][idx] = self._include_config(directive1)
            else:
                raise ValueError(f"invalid name of IC: '{k0}'")

        for k1, n1 in self.NUM_IC.items():
            if len(config[k1]) != n1:
                raise ValueError(
                    f"lacking config, there should be {n1} instances of '{k1}', "
                    f"but actually have {len(config[k1])} ones"
                )

        return config

    def _validate_group_and_line_out(self, group: int, line: int) -> None:
        if group not in {0, 1}:
            raise ValueError(f"invalid group: {group}")
        if line not in {0, 1, 2, 3}:
            raise ValueError(f"invalid line: {line}")

    @property
    def ipaddr_css(self) -> str:
        return self._css_addr

    @abstractmethod
    def configure_peripherals(self) -> None:
        pass

    @abstractmethod
    def configure_all_mxfe_clocks(self) -> None:
        pass

    @abstractmethod
    def configure_mxfe(
        self, group: int, soft_reset: bool, hard_reset: bool, configure_clock: bool, ignore_crc_error: bool
    ) -> bool:
        pass


class Quel1ConfigSubsystemAd9082Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _DAC_IDX: Dict[Tuple[int, int], int]
    _ADC_IDX: Dict[Tuple[int, str], int]
    _ADC_CH_IDX: Dict[Tuple[int, str], int]

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

    def set_dac_cnco(self, group: int, line: int, freq_in_hz: Union[int, None] = None, ftw: Union[NcoFtw, None] = None):
        dac_idx = self._DAC_IDX[(group, line)]
        freq_in_hz_, ftw_ = self._validate_frequency_info(group, "dac_cnco", freq_in_hz, ftw)
        logger.info(
            f"DAC-CNCO{dac_idx} of MxFE{group} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.modulus_a}, {ftw_.modulus_b})"
        )
        self.ad9082[group].set_dac_cnco({dac_idx}, ftw_)

    def set_dac_fnco(
        self, group: int, line: int, channel: int, freq_in_hz: Union[int, None] = None, ftw: Union[NcoFtw, None] = None
    ):
        dac_idx = self._DAC_IDX[(group, line)]
        dac_ch_idxes = tuple(self._param["ad9082"][group]["tx"]["channel_assign"][f"dac{dac_idx}"])  # (!)
        if channel < 0 or len(dac_ch_idxes) <= channel:
            raise ValueError(f"invalid channelizer index of DAC{dac_idx} of MxFE{group}")
        fnco_idx = dac_ch_idxes[channel]
        freq_in_hz_, ftw_ = self._validate_frequency_info(group, "dac_fnco", freq_in_hz, ftw)
        logger.info(
            f"DAC-FNCO{fnco_idx} of MxFE{group} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.modulus_a}, {ftw_.modulus_b})"
        )
        self.ad9082[group].set_dac_fnco({fnco_idx}, ftw_)

    def set_adc_cnco(self, group: int, line: str, freq_in_hz: Union[int, None] = None, ftw: Union[NcoFtw, None] = None):
        adc_idx = self._ADC_IDX[(group, line)]
        freq_in_hz_, ftw_ = self._validate_frequency_info(group, "adc_cnco", freq_in_hz, ftw)
        logger.info(
            f"ADC-CNCO{adc_idx} of MxFE{group} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.modulus_a}, {ftw_.modulus_b})"
        )
        self.ad9082[group].set_adc_cnco({adc_idx}, ftw_)

    def set_adc_fnco(self, group: int, line: str, freq_in_hz: Union[int, None] = None, ftw: Union[NcoFtw, None] = None):
        adc_ch_idx = self._ADC_CH_IDX[(group, line)]
        freq_in_hz_, ftw_ = self._validate_frequency_info(group, "adc_fnco", freq_in_hz, ftw)
        logger.info(
            f"ADC-FNCO{adc_ch_idx} of MxFE{group} of {self._css_addr} is set to {freq_in_hz_}Hz "
            f"(ftw = {ftw_.ftw}, {ftw_.modulus_a}, {ftw_.modulus_b})"
        )
        self.ad9082[group].set_adc_fnco({adc_ch_idx}, ftw_)

    def get_ad9082_temperatures(self, group: int) -> Tuple[int, int]:
        temp_max, temp_min = self.ad9082[group].get_temperatures()
        return temp_max, temp_min


class Quel1ConfigSubsystemLmx2594Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _LO_IDX: Dict[Tuple[int, Union[int, str]], Union[int, None]]

    def _construct_lmx2594(self):
        self._lmx2594: Tuple[Lmx2594, ...] = tuple(Lmx2594(self._proxy, idx) for idx in range(self.NUM_IC["lmx2594"]))
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
        if not (80 <= freq_multiplier <= 150):
            raise TypeError(f"invalid frequency multiplier: {freq_multiplier}")

    def set_lo_multiplier(self, group: int, line: int, freq_multiplier: int) -> bool:
        lo_idx = self._LO_IDX[(group, line)]
        if lo_idx is None:
            raise ValueError("no LO is available for (group{group}, line{line})")
        self._validate_freq_multiplier(freq_multiplier)

        logger.info(f"updating LO frequency[{lo_idx}] of {self._css_addr} to {freq_multiplier * 100}MHz")
        ic, helper = self._lmx2594[lo_idx], self._lmx2594_helper[lo_idx]
        helper.write_field("R34", pll_n_18_16=0)
        helper.write_field("R36", pll_n=freq_multiplier)
        helper.flush()
        return ic.calibrate()


class Quel1ConfigSubsystemAd6780Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    _MIXER_IDX: Dict[Tuple[int, int], Union[int, None]]

    def _construct_adrf6780(self):
        self._adrf6780: Tuple[Adrf6780, ...] = tuple(
            Adrf6780(self._proxy, idx) for idx in range(self.NUM_IC["adrf6780"])
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

    def set_sideband(self, group: int, line: int, sideband: str):
        mixer_idx = self._MIXER_IDX[(group, line)]
        if mixer_idx is None:
            raise ValueError(f"no mixer is available for (group{group}, line{line})")

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


class Quel1ConfigSubsystemAd5328Mixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    MAX_VATT: Final[int] = 0xC9B

    _VATT_IDX: Dict[Tuple[int, int], Union[Tuple[int, int], None]]

    def _construct_ad5328(self):
        self._ad5328: Tuple[Ad5328, ...] = tuple(Ad5328(self._proxy, idx) for idx in range(self.NUM_IC["ad5328"]))
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

    def set_vatt(self, group: int, line: int, vatt: int):
        idxes = self._VATT_IDX[(group, line)]
        if idxes is None:
            raise ValueError(f"no VATT is available for (group{group}, line{line})")
        ic_idx, vatt_idx = idxes
        vatt = self._validate_vatt(vatt)
        logger.info(f"VATT{vatt_idx} of {self._css_addr} is set to 0x{vatt:03x}")
        ic = self._ad5328[ic_idx]
        ic.set_output(vatt_idx, vatt, update_dac=True)


class Quel1ConfigSubsystemRfswitch(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

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

    def pass_line(self, group: int, line: int) -> None:
        self._validate_group_and_line_out(group, line)
        self._alternate_loop_rfswitch(group, **{f"path{line}": False})

    def pass_lines(self, group: int, lines: Set[int]) -> None:
        target: Dict[str, bool] = {}
        for line in lines:
            self._validate_group_and_line_out(group, line)
            target[f"path{line}"] = False
        self._alternate_loop_rfswitch(group, **target)

    def block_line(self, group: int, line: int) -> None:
        self._validate_group_and_line_out(group, line)
        self._alternate_loop_rfswitch(group, **{f"path{line}": True})

    def block_lines(self, group: int, lines: Set[int]) -> None:
        target: Dict[str, bool] = {}
        for line in lines:
            self._validate_group_and_line_out(group, line)
            target[f"path{line}"] = True
        self._alternate_loop_rfswitch(group, **target)

    def open_monitor(self, group: int) -> None:
        self._validate_group_and_line_out(group, 1)
        self._alternate_loop_rfswitch(group, monitor=False)

    def activate_monitor_loop(self, group: int) -> None:
        self._validate_group_and_line_out(group, 1)
        self._alternate_loop_rfswitch(group, monitor=True)

    def block_monitor(self, group: int) -> None:
        self.activate_monitor_loop(group)

    def open_read(self, group: int) -> None:
        self._validate_group_and_line_out(group, 0)
        self._alternate_loop_rfswitch(group, path0=False)

    def activate_read_loop(self, group: int) -> None:
        self._validate_group_and_line_out(group, 0)
        self._alternate_loop_rfswitch(group, path0=True)

    def block_read(self, group: int) -> None:
        self.activate_read_loop(group)


class Quel1ConfigSubsystemGpioMixin(Quel1ConfigSubsystemBaseSlot):
    __slots__ = ()

    def _construct_gpio(self):
        self._gpio: Tuple[GenericGpio, ...] = tuple(GenericGpio(self._proxy, idx) for idx in range(self.NUM_IC["gpio"]))
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
