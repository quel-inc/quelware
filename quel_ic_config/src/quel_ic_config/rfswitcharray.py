import logging
from dataclasses import dataclass, field
from typing import Dict, cast

from quel_ic_config.abstract_ic import AbstractIcConfigHelper, AbstractIcMixin, AbstractIcReg, b_1bf_bool, p_1bf_bool

logger = logging.getLogger(__name__)


@dataclass
class RfSwitchArrayReg(AbstractIcReg):
    path0: bool = field(default=False)
    path1: bool = field(default=False)
    path2: bool = field(default=False)
    path3: bool = field(default=False)
    monitor: bool = field(default=False)

    def _parse_switch_pair(self, r):
        if r == 0x00:
            return False
        elif r == 0x03:
            return True
        elif r == 0x01 or r == 0x02:
            logger.warning("invalid state of RF switch for loopback, considered as inside")
            return True
        else:
            raise AssertionError


class AbstractRfSwitchArrayMixin(AbstractIcMixin):
    def dump_regs(self) -> Dict[int, int]:
        """dumping all the available registers.

        :return: a mapping between an address and a value of the registers
        """
        regs = {0: self.read_reg(0)}
        return regs


class QubeRfSwitchArray0Reg(RfSwitchArrayReg):
    def parse(self, v: int) -> None:
        self.path0 = self._parse_switch_pair(v & 0b00000000000011)
        self.path1 = p_1bf_bool(v, 2)
        self.monitor = self._parse_switch_pair((v & 0b00000000011000) >> 3)
        self.path2 = p_1bf_bool(v, 5)
        self.path3 = p_1bf_bool(v, 6)

    def build(self) -> int:
        return (
            b_1bf_bool(self.path0, 0)
            | b_1bf_bool(self.path0, 1)
            | b_1bf_bool(self.path1, 2)
            | b_1bf_bool(self.monitor, 3)
            | b_1bf_bool(self.monitor, 4)
            | b_1bf_bool(self.path2, 5)
            | b_1bf_bool(self.path3, 6)
        )


class QubeRfSwitchArray1Reg(RfSwitchArrayReg):
    def parse(self, v: int) -> None:
        self.path0 = self._parse_switch_pair((v & 0b11000000000000) >> 12)
        self.path1 = p_1bf_bool(v, 11)
        self.monitor = self._parse_switch_pair((v & 0b00011000000000) >> 9)
        self.path2 = p_1bf_bool(v, 8)
        self.path3 = p_1bf_bool(v, 7)

    def build(self) -> int:
        return (
            b_1bf_bool(self.path0, 13)
            | b_1bf_bool(self.path0, 12)
            | b_1bf_bool(self.path1, 11)
            | b_1bf_bool(self.monitor, 10)
            | b_1bf_bool(self.monitor, 9)
            | b_1bf_bool(self.path2, 8)
            | b_1bf_bool(self.path3, 7)
        )


QubeRfSwitchRegs: Dict[int, type] = {
    0: QubeRfSwitchArray0Reg,
    1: QubeRfSwitchArray1Reg,
}

QubeSwitchRegNames: Dict[str, int] = {
    "Group0": 0,
    "Group1": 1,
}


class QubeRfSwitchArrayMixin(AbstractRfSwitchArrayMixin):
    Regs = QubeRfSwitchRegs
    RegNames = QubeSwitchRegNames

    def __init__(self, name):
        super().__init__(name)


class Quel1TypeARfSwitchArray0Reg(RfSwitchArrayReg):
    def parse(self, v: int) -> None:
        self.path0 = p_1bf_bool(v, 0)  # port0: read-in,  port1: read-out
        self.path1 = p_1bf_bool(v, 5)  # port3 for typeA, port2 for TypeB
        self.monitor = p_1bf_bool(v, 3)  # port5: monitor-in, port6: monitor-out
        self.path2 = p_1bf_bool(v, 2)  # port2 for typeA, port3 for TypeB
        self.path3 = p_1bf_bool(v, 6)  # port4

    def build(self) -> int:
        return (
            b_1bf_bool(self.path0, 0)
            | b_1bf_bool(self.path1, 5)
            | b_1bf_bool(self.monitor, 3)
            | b_1bf_bool(self.path2, 2)
            | b_1bf_bool(self.path3, 6)
        )


class Quel1TypeARfSwitchArray1Reg(RfSwitchArrayReg):
    def parse(self, v: int) -> None:
        self.path0 = p_1bf_bool(v, 12)  # port7: readin, port8: read-out
        self.path1 = p_1bf_bool(v, 8)  # port10 for typeA, port9 for typeB
        self.monitor = p_1bf_bool(v, 9)  # port12: monitor-in, port13: monitor-out
        self.path2 = p_1bf_bool(v, 7)  # port9 for typeA, port10 for typeB
        self.path3 = p_1bf_bool(v, 11)  # port11

    def build(self) -> int:
        return (
            b_1bf_bool(self.path0, 12)
            | b_1bf_bool(self.path1, 8)
            | b_1bf_bool(self.monitor, 9)
            | b_1bf_bool(self.path2, 7)
            | b_1bf_bool(self.path3, 11)
        )


class Quel1TypeBRfSwitchArray0Reg(RfSwitchArrayReg):
    def parse(self, v: int) -> None:
        self.path0 = p_1bf_bool(v, 0)  # port0: read-in,  port1: read-out
        self.path1 = p_1bf_bool(v, 2)  # port3 for typeA, port2 for TypeB
        self.monitor = p_1bf_bool(v, 3)  # port5: monitor-in, port6: monitor-out
        self.path2 = p_1bf_bool(v, 5)  # port2 for typeA, port3 for TypeB
        self.path3 = p_1bf_bool(v, 6)  # port4

    def build(self) -> int:
        return (
            b_1bf_bool(self.path0, 0)
            | b_1bf_bool(self.path1, 2)
            | b_1bf_bool(self.monitor, 3)
            | b_1bf_bool(self.path2, 5)
            | b_1bf_bool(self.path3, 6)
        )


class Quel1TypeBRfSwitchArray1Reg(RfSwitchArrayReg):
    def parse(self, v: int) -> None:
        self.path0 = p_1bf_bool(v, 12)  # port7: readin, port8: read-out
        self.path1 = p_1bf_bool(v, 11)  # port10 for typeA, port9 for typeB
        self.monitor = p_1bf_bool(v, 9)  # port12: monitor-in, port13: monitor-out
        self.path2 = p_1bf_bool(v, 7)  # port9 for typeA, port10 for typeB
        self.path3 = p_1bf_bool(v, 8)  # port11

    def build(self) -> int:
        return (
            b_1bf_bool(self.path0, 12)
            | b_1bf_bool(self.path1, 11)
            | b_1bf_bool(self.monitor, 9)
            | b_1bf_bool(self.path2, 7)
            | b_1bf_bool(self.path3, 8)
        )


Quel1TypeARfSwitchRegs: Dict[int, type] = {
    0: Quel1TypeARfSwitchArray0Reg,
    1: Quel1TypeARfSwitchArray1Reg,
}

Quel1TypeBRfSwitchRegs: Dict[int, type] = {
    0: Quel1TypeBRfSwitchArray0Reg,
    1: Quel1TypeBRfSwitchArray1Reg,
}

Quel1SwitchRegNames: Dict[str, int] = {
    "Group0": 0,
    "Group1": 1,
}


class Quel1TypeARfSwitchArrayMixin(AbstractRfSwitchArrayMixin):
    Regs = Quel1TypeARfSwitchRegs
    RegNames = Quel1SwitchRegNames

    def __init__(self, name):
        super().__init__(name)


class Quel1TypeBRfSwitchArrayMixin(AbstractRfSwitchArrayMixin):
    Regs = Quel1TypeBRfSwitchRegs
    RegNames = Quel1SwitchRegNames

    def __init__(self, name):
        super().__init__(name)


class RfSwitchArrayConfigHelper(AbstractIcConfigHelper):
    """Helper class for programming RF Switch Array of Qube and Quel-1."""

    # TODO: consider to introduce general mechanisms for supporting virtual registers.
    #       (paged registers may be managed with the mechanisms well.)
    def __init__(self, ic: AbstractRfSwitchArrayMixin):
        super().__init__(ic)

    def _read_reg(self, addr: int, refer_to_cache: bool = False) -> AbstractIcReg:
        data = self.ic.Regs[addr]()
        if addr in self.updated and refer_to_cache:
            data.parse(self.updated[addr])
        else:
            v = self.ic.read_reg(0)
            if addr == 0:
                data.parse(v & 0b00000001111111)
            elif addr == 1:
                data.parse(v & 0b11111110000000)
            else:
                raise ValueError(f"Invalid address: {addr}")

        return cast(AbstractIcReg, data)

    def flush(self, discard_after_flush=True):
        # notes: load all the virtual registers
        if 0 not in self.updated or 1 not in self.updated:
            if 0 not in self.updated:
                v0 = self._read_reg(0, True).build()
                self.updated[0] = v0 & 0b00000001111111
            if 1 not in self.updated:
                v1 = self._read_reg(1, True).build()
                self.updated[1] = v1 & 0b11111110000000

        self.ic.write_reg(0, self.updated[0] | self.updated[1])  # address is not used, actually.
        if discard_after_flush:
            self.discard()
