import logging
from dataclasses import dataclass, field
from typing import Dict, Set, Union

from quel_ic_config.abstract_ic import (
    AbstractIcConfigHelper,
    AbstractIcMixin,
    AbstractIcReg,
    b_1bf_bool,
    b_nbf,
    p_1bf_bool,
    p_nbf,
)

logger = logging.getLogger(__name__)


@dataclass
class Ad5328ConfigReg(AbstractIcReg):
    gain_e_h: bool = field(default=False)  # [5]
    gain_a_d: bool = field(default=False)  # [4]
    buf_e_h: bool = field(default=False)  # [3]
    buf_a_d: bool = field(default=False)  # [2]
    vdd_e_h: bool = field(default=False)  # [1]
    vdd_a_d: bool = field(default=False)  # [0]

    def parse(self, v: int) -> None:
        self.gain_e_h = p_1bf_bool(v, 5)
        self.gain_a_d = p_1bf_bool(v, 4)
        self.buf_e_h = p_1bf_bool(v, 3)
        self.buf_a_d = p_1bf_bool(v, 2)
        self.vdd_e_h = p_1bf_bool(v, 1)
        self.vdd_a_d = p_1bf_bool(v, 0)

    def build(self) -> int:
        return (
            b_1bf_bool(self.gain_e_h, 5)
            | b_1bf_bool(self.gain_a_d, 4)
            | b_1bf_bool(self.buf_e_h, 3)
            | b_1bf_bool(self.buf_a_d, 2)
            | b_1bf_bool(self.vdd_e_h, 1)
            | b_1bf_bool(self.vdd_a_d, 0)
        )


@dataclass
class Ad5328DacX(AbstractIcReg):
    data: int = field(default=0)  # [11:0]

    def parse(self, v: int) -> None:
        self.data = p_nbf(v, 11, 0)

    def build(self) -> int:
        return b_nbf(self.data, 11, 0)


@dataclass
class Ad5328PowerDownReg(AbstractIcReg):
    h: bool = field(default=False)  # [7]
    g: bool = field(default=False)  # [6]
    f: bool = field(default=False)  # [5]
    e: bool = field(default=False)  # [4]
    d: bool = field(default=False)  # [3]
    c: bool = field(default=False)  # [2]
    b: bool = field(default=False)  # [1]
    a: bool = field(default=False)  # [0]

    def parse(self, v: int) -> None:
        self.h = p_1bf_bool(v, 7)
        self.g = p_1bf_bool(v, 6)
        self.f = p_1bf_bool(v, 5)
        self.e = p_1bf_bool(v, 4)
        self.d = p_1bf_bool(v, 3)
        self.c = p_1bf_bool(v, 2)
        self.b = p_1bf_bool(v, 1)
        self.a = p_1bf_bool(v, 0)

    def build(self) -> int:
        return (
            b_1bf_bool(self.h, 7)
            | b_1bf_bool(self.g, 6)
            | b_1bf_bool(self.f, 5)
            | b_1bf_bool(self.e, 4)
            | b_1bf_bool(self.d, 3)
            | b_1bf_bool(self.c, 2)
            | b_1bf_bool(self.b, 1)
            | b_1bf_bool(self.a, 0)
        )


Ad5328Regs: Dict[int, type] = {
    0: Ad5328DacX,
    1: Ad5328DacX,
    2: Ad5328DacX,
    3: Ad5328DacX,
    4: Ad5328DacX,
    5: Ad5328DacX,
    6: Ad5328DacX,
    7: Ad5328DacX,
    8: Ad5328ConfigReg,
    12: Ad5328PowerDownReg,
}


Ad5328RegNames: Dict[str, int] = {
    "DacA": 0,
    "DacB": 1,
    "DacC": 2,
    "DacD": 3,
    "DacE": 4,
    "DacF": 5,
    "DacG": 6,
    "DacH": 7,
    "Config": 8,
    "PowerDown": 12,
}


class Ad5328Mixin(AbstractIcMixin):
    Regs = Ad5328Regs
    RegNames = Ad5328RegNames
    _UPDATE = 10
    _RESET = 15  # both data and control are reset

    def __init__(self, name: str):
        super().__init__(name)
        self._carbon_copy: Dict[int, int] = {}
        self._carbon_copy_not_updated: Dict[int, int] = {}

    def dump_regs(self) -> Dict[int, int]:
        # Notes: AD5328 doesn't support register read.
        raise NotImplementedError

    def soft_reset(self) -> None:
        """commiting soft reset.

        :return: None
        """
        for i in range(8):
            self._carbon_copy[i] = 0
        self.write_reg(self._RESET, 0x000)

    @staticmethod
    def _validate_channel(channel: int) -> None:
        if not (0 <= channel <= 7):
            raise ValueError(f"invalid channel: {channel}")

    def update_dac(self) -> None:
        """update the output of the DACs once.
        :return: None
        """
        self._carbon_copy.update(self._carbon_copy_not_updated)
        self.write_reg(self._UPDATE, 0x002)

    def set_output(self, channel: int, value: int, update_dac=False) -> None:
        """setting the output value of a DAC channel.

        :param channel: an index of a DAC channel to be set. The output voltage of the DAC will be updated when
                        update_dac() is called.
        :param value: an output value of the DAC
        :param update_dac: conduct update_dac() if True. Its default value is False.
        :return: None
        """
        self._validate_channel(channel)
        if not (0 <= value <= 4095):
            raise ValueError(f"invalid output value: {value} for channel {channel} of {self.name}")
        self.write_reg(channel, value)
        if update_dac:
            self.update_dac()

    def get_output_carboncopy(self, channel: int) -> Union[int, None]:
        self._validate_channel(channel)
        if channel in self._carbon_copy:
            return self._carbon_copy[channel]
        else:
            return None


class Ad5328ConfigHelper(AbstractIcConfigHelper):
    """Helper class for programming AD5328 with convenient notations. It also work as a storage of register values since
    AD5328 doesn't support register read.
    """

    def __init__(self, ic: Ad5328Mixin):
        super().__init__(ic, no_read=True)
        self.modified: Set[int] = set()

    def _write_reg(self, addr: int, data: Union[int, AbstractIcReg]) -> None:
        super()._write_reg(addr, data)
        self.modified.add(addr)

    def discard(self):
        super().discard()
        self.modified.clear()

    def flush(self, discard_after_flush=False):
        for addr in (8, 12, 0, 1, 2, 3, 4, 5, 6, 7):
            if addr in self.updated and addr in self.modified:
                self.ic.write_reg(addr, self.updated[addr])
        if discard_after_flush:
            self.discard()
        else:
            self.modified.clear()
