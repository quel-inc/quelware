import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

from quel_ic_config.abstract_ic import AbstractIcConfigHelper, AbstractIcMixin, AbstractIcReg, b_1bf_bool, p_1bf_bool

logger = logging.getLogger(__name__)


@dataclass
class Gpio14(AbstractIcReg):
    b13: bool = field(default=False)
    b12: bool = field(default=False)
    b11: bool = field(default=False)
    b10: bool = field(default=False)
    b09: bool = field(default=False)
    b08: bool = field(default=False)
    b07: bool = field(default=False)
    b06: bool = field(default=False)
    b05: bool = field(default=False)
    b04: bool = field(default=False)
    b03: bool = field(default=False)
    b02: bool = field(default=False)
    b01: bool = field(default=False)
    b00: bool = field(default=False)

    def parse(self, v: int) -> None:
        self.b13 = p_1bf_bool(v, 13)
        self.b12 = p_1bf_bool(v, 12)
        self.b11 = p_1bf_bool(v, 11)
        self.b10 = p_1bf_bool(v, 10)
        self.b09 = p_1bf_bool(v, 9)
        self.b08 = p_1bf_bool(v, 8)
        self.b07 = p_1bf_bool(v, 7)
        self.b06 = p_1bf_bool(v, 6)
        self.b05 = p_1bf_bool(v, 5)
        self.b04 = p_1bf_bool(v, 4)
        self.b03 = p_1bf_bool(v, 3)
        self.b02 = p_1bf_bool(v, 2)
        self.b01 = p_1bf_bool(v, 1)
        self.b00 = p_1bf_bool(v, 0)

    def build(self) -> int:
        return (
            b_1bf_bool(self.b13, 13)
            | b_1bf_bool(self.b12, 12)
            | b_1bf_bool(self.b11, 11)
            | b_1bf_bool(self.b10, 10)
            | b_1bf_bool(self.b09, 9)
            | b_1bf_bool(self.b08, 8)
            | b_1bf_bool(self.b07, 7)
            | b_1bf_bool(self.b06, 6)
            | b_1bf_bool(self.b05, 5)
            | b_1bf_bool(self.b04, 4)
            | b_1bf_bool(self.b03, 3)
            | b_1bf_bool(self.b02, 2)
            | b_1bf_bool(self.b01, 1)
            | b_1bf_bool(self.b00, 0)
        )


GenericGpioRegs: Dict[int, type] = {
    0: Gpio14,
}


GenericGpioRegNames: Dict[str, int] = {
    "Bits": 0,
}


class GenericGpioMixin(AbstractIcMixin):
    Regs = GenericGpioRegs
    RegNames = GenericGpioRegNames

    def __init__(self, name):
        super().__init__(name)

    def dump_regs(self) -> Dict[int, int]:
        """dump all the available registers.
        :return: a mapping between an address and a value of the registers
        """
        regs = {}
        for addr in (0,):
            regs[addr] = self.read_reg(addr)  # actually addr is ignored by exstickge
        return regs

    def _read_and_parse_reg(self, regname: str) -> Tuple[int, AbstractIcReg]:
        addr = self.RegNames[regname]
        regcls = self.Regs[addr]
        reg = regcls()
        reg.parse(self.read_reg(addr))
        return addr, reg

    def _build_and_write_reg(self, addr: int, regobj: AbstractIcReg) -> None:
        self.write_reg(addr, regobj.build())


class GenericGpioConfigHelper(AbstractIcConfigHelper):
    """Helper class for programming GPIO with convenient notations. It also provides caching capability that
    keep modifications on the registers to write them at once with flash_updated() in a right order.
    """

    def __init__(self, ic: GenericGpioMixin):
        super().__init__(ic)

    def flush(self, discard_after_flush=True):
        for addr in (0,):
            if addr in self.updated:
                self.ic.write_reg(addr, self.updated[addr])
        if discard_after_flush:
            self.discard()
