import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union

from quel_ic_config.abstract_ic import AbstractIcMixin, AbstractIcReg, b_1bf_bool, b_nbf, p_1bf_bool, p_nbf

logger = logging.getLogger(__name__)


@dataclass
class Ad7490ConfigReg(AbstractIcReg):
    write: bool = field(default=True)  # [11+4]
    seq: bool = field(default=True)  # [10+4]
    addr: int = field(default=0xF)  # [9+4:6+4]
    pm: int = field(default=0x3)  # [5+4:4:4]
    shadow: bool = field(default=True)  # [3+4]
    weak: bool = field(default=False)  # [2+4]
    range: bool = field(default=True)  # [1+4], 0 for 2V_ref, 1 for V_ref
    coding: bool = field(default=True)  # [0+4], 0 for signed, 1 for unsigned

    def parse(self, v: int) -> None:
        self.write = p_1bf_bool(v, 15)
        self.seq = p_1bf_bool(v, 14)
        self.addr = p_nbf(v, 13, 10)
        self.pm = p_nbf(v, 9, 8)
        self.shadow = p_1bf_bool(v, 7)
        self.weak = p_1bf_bool(v, 6)
        self.range = p_1bf_bool(v, 5)
        self.coding = p_1bf_bool(v, 4)

    def build(self) -> int:
        return (
            b_1bf_bool(self.write, 15)
            | b_1bf_bool(self.seq, 14)
            | b_nbf(self.addr, 13, 10)
            | b_nbf(self.pm, 9, 8)
            | b_1bf_bool(self.shadow, 7)
            | b_1bf_bool(self.weak, 6)
            | b_1bf_bool(self.range, 5)
            | b_1bf_bool(self.coding, 4)
        )


class Ad7490Mixin(AbstractIcMixin):
    Regs: Dict[int, type] = {
        0: Ad7490ConfigReg,
    }
    RegNames: Dict[str, int] = {
        "Config": 0,
    }

    def __init__(self, name: str):
        super().__init__(name)
        config = Ad7490ConfigReg()
        self._default_config: int = config.build()
        self._num_read_iter: int = config.addr + 1

    def dump_regs(self) -> Dict[int, int]:
        raise NotImplementedError

    def set_default_config(self, **kwargs):
        config = Ad7490ConfigReg(**kwargs)
        self._default_config = config.build()
        self._num_read_iter = config.addr + 1

    def apply_default_config(self):
        # TODO: check the return value
        self.write_reg(0, self._default_config)

    def read_adcs_raw(self) -> Dict[int, int]:
        self.apply_default_config()
        adcs: Dict[int, int] = {}
        for i in range(self._num_read_iter):
            adcs[i] = self.read_reg(0)
        return adcs

    def read_adcs(self) -> Union[List[int], None]:
        self.apply_default_config()
        adcs: List[int] = []
        for i in range(self._num_read_iter):
            v = self.read_reg(0)
            if (v >> 12) & 0xF == i:
                adcs.append(v & 0xFFF)
            else:
                logger.error(f"broken data from {self.name} at channel {i}: {v:04x}")
                return None
        return adcs


# Notes: No helper class is implemented because AD7490 is simple enough.
