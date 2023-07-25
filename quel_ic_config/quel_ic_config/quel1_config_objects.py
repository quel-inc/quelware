import copy
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Dict, Final, Mapping, Sequence, Tuple, Union

from pydantic.utils import deep_update

from quel_ic_config.ad5328 import Ad5328ConfigHelper
from quel_ic_config.adrf6780 import Adrf6780ConfigHelper
from quel_ic_config.lmx2594 import Lmx2594ConfigHelper
from quel_ic_config.quel_ic import (
    Ad5328,
    Ad9082V106,
    Adrf6780,
    ExstickgeProxyQuel1,
    Lmx2594,
    QubeRfSwitchArray,
    Quel1TypeARfSwitchArray,
    Quel1TypeBRfSwitchArray,
)
from quel_ic_config.rfswitcharray import AbstractRfSwitchArrayMixin, RfSwitchArrayConfigHelper

logger = logging.getLogger(__name__)


class Quel1BoxType(Enum):
    QuBE_TypeA = ("qube", "type-a")  # TODO: not tested yet
    QuBE_TypeB = ("qube", "type-b")  # TODO: not tested yet
    QuEL1_TypeA = ("quel-1", "type-a")
    QuEL1_TypeB = ("quel-1", "type-b")  # TODO: not tested yet
    QuEL1_NTT = ("quel-1", "ntt")  # TODO: not supported yet

    @classmethod
    def fromstr(cls, label: str) -> "Quel1BoxType":
        return _QuelBoxTypeMap[label]


_QuelBoxTypeMap: Dict[str, Quel1BoxType] = {
    "qube-a": Quel1BoxType.QuEL1_TypeA,
    "qube-b": Quel1BoxType.QuEL1_TypeB,
    "quel1-a": Quel1BoxType.QuEL1_TypeA,
    "quel1-b": Quel1BoxType.QuEL1_TypeB,
}


class QuelConfigOption(str, Enum):
    USE_READ_IN_MXFE0 = "use_read_in_mxfe0"  # TODO: this is removed when dynamic alternation of input port is realized
    USE_MONITOR_IN_MXFE0 = "use_monitor_in_mxfe0"  # TODO: same as above
    USE_READ_IN_MXFE1 = "use_read_in_mxfe1"  # TODO: same as above
    USE_MONITOR_IN_MXFE1 = "use_monitor_in_mxfe1"  # TODO: same as above


class Quel1ConfigObjects:
    __slots__ = [
        "_css_addr",
        "_param",
        "_boxtype",
        "_config_path",
        "_config_options",
        "_proxy",
        "_ad9082",
        "_lmx2594",
        "_lmx2594_helper",
        "_ad5328",
        "_ad5328_helper",
        "_adrf6780",
        "_adrf6780_helper",
        "_gpio",
        "_gpio_helper",
    ]
    NUM_IC: Final[Dict[str, int]] = {
        "ad9082": 2,
        "lmx2594": 10,
        "adrf6780": 8,
        "ad5328": 1,
        "gpio": 1,
    }

    def __init__(
        self,
        css_addr: str,
        boxtype: Quel1BoxType,
        config_path: Path,
        config_options: Union[Collection[QuelConfigOption], None] = None,  # TODO: should be elaborated.
        port: int = 16384,
        timeout: float = 0.5,
        sender_limit_by_binding: bool = False,
    ):
        self._css_addr: Final[str] = css_addr
        self._boxtype: Final[Quel1BoxType] = boxtype
        self._config_path: Final[Path] = config_path
        self._config_options: Final[Collection[QuelConfigOption]] = (
            config_options if config_options is not None else set()
        )
        self._param: Dict[str, Any] = self._load_config()  # TODO: Dict[str, Any] is tentative.
        self._proxy: Final[ExstickgeProxyQuel1] = ExstickgeProxyQuel1(
            self._css_addr, port, timeout, sender_limit_by_binding
        )
        self._ad9082: Final[Tuple[Ad9082V106, ...]] = tuple(
            Ad9082V106(self._proxy, idx, self._param["ad9082"][idx]) for idx in range(self.NUM_IC["ad9082"])
        )
        self._lmx2594: Final[Tuple[Lmx2594, ...]] = tuple(
            Lmx2594(self._proxy, idx) for idx in range(self.NUM_IC["lmx2594"])
        )
        self._adrf6780: Final[Tuple[Adrf6780, ...]] = tuple(
            Adrf6780(self._proxy, idx) for idx in range(self.NUM_IC["adrf6780"])
        )
        self._ad5328: Final[Tuple[Ad5328, ...]] = tuple(
            Ad5328(self._proxy, idx) for idx in range(self.NUM_IC["ad5328"])
        )
        if boxtype == Quel1BoxType.QuEL1_TypeA:
            gpio_tmp: Tuple[AbstractRfSwitchArrayMixin, ...] = tuple(
                Quel1TypeARfSwitchArray(self._proxy, idx) for idx in range(self.NUM_IC["gpio"])
            )
        elif boxtype == Quel1BoxType.QuEL1_TypeB:
            gpio_tmp = tuple(Quel1TypeBRfSwitchArray(self._proxy, idx) for idx in range(self.NUM_IC["gpio"]))
        elif boxtype in {Quel1BoxType.QuBE_TypeA, Quel1BoxType.QuBE_TypeB}:
            gpio_tmp = tuple(QubeRfSwitchArray(self._proxy, idx) for idx in range(self.NUM_IC["gpio"]))
        else:
            raise ValueError("invalid boxtype: {boxtype}")
        self._gpio: Final[Tuple[AbstractRfSwitchArrayMixin, ...]] = gpio_tmp

        self._lmx2594_helper: Final[Tuple[Lmx2594ConfigHelper, ...]] = tuple(
            Lmx2594ConfigHelper(ic) for ic in self._lmx2594
        )
        self._adrf6780_helper: Final[Tuple[Adrf6780ConfigHelper, ...]] = tuple(
            Adrf6780ConfigHelper(ic) for ic in self._adrf6780
        )
        self._ad5328_helper: Final[Tuple[Ad5328ConfigHelper, ...]] = tuple(
            Ad5328ConfigHelper(ic) for ic in self._ad5328
        )
        self._gpio_helper: Final[Tuple[RfSwitchArrayConfigHelper, ...]] = tuple(
            RfSwitchArrayConfigHelper(ic) for ic in self._gpio
        )

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
                    if QuelConfigOption(op1) not in self._config_options:
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
        with open(self._config_path / "quel-1.json") as f:
            root: Dict[str, Any] = self._remove_comments(json.load(f))

        config = copy.copy(root)
        for k0, directive0 in root.items():
            if k0 == "meta":
                pass
            elif k0 in {"ad9082", "lmx2594", "adrf6780", "ad5328", "gpio"}:
                for idx, directive1 in enumerate(directive0):
                    if idx >= self.NUM_IC[k0]:
                        raise ValueError(f"too many {k0.upper()}s are found")
                    config[k0][idx] = self._include_config(directive1)
            else:
                raise ValueError(f"invalid name of IC: '{k0}'")

        return config

    @property
    def ipaddr_css(self) -> str:
        return self._css_addr

    @property
    def ad9082(self) -> Tuple[Ad9082V106, ...]:
        return self._ad9082

    @property
    def lmx2594(self) -> Tuple[Tuple[Lmx2594, Lmx2594ConfigHelper], ...]:
        return tuple((self._lmx2594[idx], self._lmx2594_helper[idx]) for idx in range(self.NUM_IC["lmx2594"]))

    @property
    def adrf6780(self) -> Tuple[Tuple[Adrf6780, Adrf6780ConfigHelper], ...]:
        return tuple((self._adrf6780[idx], self._adrf6780_helper[idx]) for idx in range(self.NUM_IC["adrf6780"]))

    @property
    def ad5328(self) -> Tuple[Tuple[Ad5328, Ad5328ConfigHelper], ...]:
        return tuple((self._ad5328[idx], self._ad5328_helper[idx]) for idx in range(self.NUM_IC["ad5328"]))

    @property
    def gpio(self) -> Tuple[Tuple[AbstractRfSwitchArrayMixin, RfSwitchArrayConfigHelper], ...]:
        return tuple((self._gpio[idx], self._gpio_helper[idx]) for idx in range(self.NUM_IC["gpio"]))

    def init_lmx2594(self, idx):
        ic, helper = self.lmx2594[idx]
        param: Dict[str, Dict[str, Union[int, bool]]] = self._param["lmx2594"][idx]
        ic.soft_reset()
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()
        ic.calibrate()

    def init_adrf6780(self, idx):
        ic, helper = self.adrf6780[idx]
        param: Dict[str, Dict[str, Union[int, bool]]] = self._param["adrf6780"][idx]
        ic.soft_reset(parity_enable=False)
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()

    def init_ad5328(self, idx):
        ic, helper = self.ad5328[idx]
        param: Dict[str, Dict[str, Union[int, bool]]] = self._param["ad5328"][idx]
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()
        ic.update_dac()

    def init_gpio(self, idx):
        _, helper = self.gpio[idx]
        param: Dict[str, Dict[str, Union[int, bool]]] = self._param["gpio"][idx]
        for name, fields in param["registers"].items():
            helper.write_field(name, **fields)
        helper.flush()

    def _alternate_loop_rfswitch(self, group: int, **sw_update: bool) -> None:
        ic, helper = self.gpio[0]
        current_sw: int = ic.read_reg(0)
        if group not in {0, 1}:
            raise ValueError(f"invalid group: {group}")
        helper.write_field(f"Group{group}", **sw_update)
        helper.flush()
        altered_sw: int = ic.read_reg(0)
        time.sleep(0.01)
        logger.info(
            f"alter the state of RF switch array of {self.ipaddr_css} from {current_sw:014b} to {altered_sw:014b}"
        )

    def open_monitor(self, group: int) -> None:
        self._alternate_loop_rfswitch(group, monitor=False)

    def activate_monitor_loop(self, group: int) -> None:
        self._alternate_loop_rfswitch(group, monitor=True)

    def open_read(self, group: int) -> None:
        self._alternate_loop_rfswitch(group, path0=False)

    def activate_read_loop(self, group: int) -> None:
        self._alternate_loop_rfswitch(group, path0=True)

    def set_lo_freq(self, group: int, line: int, freq_multiplier: int):
        if group not in {0, 1}:
            raise ValueError(f"invalid group: {group}")
        if line not in {0, 1, 2, 3}:
            raise ValueError(f"invalid line: {line}")
        if not (80 <= freq_multiplier <= 150):
            raise ValueError(f"invalid lo_freq_multiplier: {freq_multiplier}")

        idx = line if group == 0 else 7 - line
        logger.info(f"updating LO frequency[{idx}] to {freq_multiplier * 100}MHz")
        ic, helper = self.lmx2594[idx]
        helper.write_field("R34", pll_n_18_16=0)
        helper.write_field("R36", pll_n=freq_multiplier)
        helper.flush()
        ic.calibrate()
