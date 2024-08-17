import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple, Union, cast

logger = logging.getLogger(__name__)


def p_1bf_bool(v: int, bidx: int) -> bool:
    return bool((v >> bidx) & 0x1)


def p_nbf(v: int, bidx0: int, bidx1: int) -> int:
    return (v >> bidx1) & ((1 << (bidx0 - bidx1 + 1)) - 1)


def b_1bf_bool(f: bool, bidx: int) -> int:
    return int(f) << bidx


def b_nbf(f: int, bidx0: int, bidx1: int) -> int:
    return (f & ((1 << (bidx0 - bidx1 + 1)) - 1)) << bidx1


class AbstractIcReg(metaclass=ABCMeta):
    @abstractmethod
    def parse(self, v: int) -> None:
        pass

    @abstractmethod
    def build(self) -> int:
        pass


class AbstractIcMixin(metaclass=ABCMeta):
    Regs: Dict[int, type] = {}
    RegNames: Dict[str, int] = {}

    def __init__(self, name: str):
        self.name = name

    def _read_and_parse_reg(self, regname: str) -> Tuple[int, AbstractIcReg]:
        addr = self.RegNames[regname]
        regcls = self.Regs[addr]
        reg = regcls()
        reg.parse(self.read_reg(addr))
        return addr, reg

    def _build_and_write_reg(self, addr: int, regobj: AbstractIcReg) -> None:
        self.write_reg(addr, regobj.build())

    @abstractmethod
    def dump_regs(self) -> Dict[int, int]:
        pass

    @abstractmethod
    def read_reg(self, addr: int) -> int:
        pass

    @abstractmethod
    def write_reg(self, addr: int, data: int) -> None:
        pass


class AbstractIcConfigHelper:
    def __init__(self, ic: AbstractIcMixin, no_read: bool = False):
        self.ic: AbstractIcMixin = ic
        self._no_read = no_read
        self.updated: Dict[int, int] = {}

    def dump_regs_parsed(self) -> Dict[int, AbstractIcReg]:
        parsed_dump: Dict[int, AbstractIcReg] = {}
        plain_dump = self.ic.dump_regs()
        for idx, regcls in self.ic.Regs.items():
            regobj = regcls()
            regobj.parse(plain_dump[idx])
            parsed_dump[idx] = regobj
        return parsed_dump

    def _parse_addr(self, addr: Union[str, int]) -> int:
        if isinstance(addr, str):
            return self.ic.RegNames[addr]
        elif isinstance(addr, int):
            return addr
        else:
            raise TypeError("illegal type of given address")

    def _zero_reg(self, addr: int, refer_to_cache: bool = False) -> AbstractIcReg:
        data = self.ic.Regs[addr]()
        if addr in self.updated and refer_to_cache:
            data.parse(self.updated[addr])
        else:
            data.parse(0)
        return cast(AbstractIcReg, data)

    def _read_reg(self, addr: int, refer_to_cache: bool = False) -> AbstractIcReg:
        data = self.ic.Regs[addr]()
        if addr in self.updated and refer_to_cache:
            data.parse(self.updated[addr])
        else:
            data.parse(self.ic.read_reg(addr))
        return cast(AbstractIcReg, data)

    def _write_reg(self, addr: int, data: Union[int, AbstractIcReg]) -> None:
        if isinstance(data, AbstractIcReg):
            if isinstance(data, self.ic.Regs[addr]):
                data = cast(AbstractIcReg, data).build()
            else:
                raise TypeError(f"mismatched register object for R{addr}")
        if not isinstance(data, int):
            raise TypeError("unexpected data is given")
        self.updated[addr] = data

    def read_reg(self, address: Union[int, str], refer_to_cache: bool = False) -> AbstractIcReg:
        """reading the value of a register at the address.

        :param address: the address of register to be read. you may use alias name instead.
        :param refer_to_cache: return the cached register value if True.
        :return: an instance of the corresponding register object
        """
        addr_ = self._parse_addr(address)
        return self._read_reg(addr_, refer_to_cache)

    def write_reg(self, address: Union[int, str], data: Union[int, AbstractIcReg]) -> None:
        """updating the value of a register into a cache. the cached values are written to the register when flush() is
        called.

        :param address: the address of a register to be updated.
        :param data: the value of the register.
        :return: None
        """
        addr_ = self._parse_addr(address)
        return self._write_reg(addr_, data)

    def write_field(self, addr: Union[int, str], **kwargs: Union[bool, int, IntEnum]):
        """a wrapper function of write_reg() providing a convenient way to update fields of a register.

        :param addr: the address of a register to be updated.
        :param kwargs: the values of fields of the register.
        :return: None
        """
        addr_ = self._parse_addr(addr)
        # TODO: consider the validity of this design carefully!
        if self._no_read:
            data: AbstractIcReg = self._zero_reg(addr_, True)
        else:
            data = self._read_reg(addr_, True)
        for k, v in kwargs.items():
            if hasattr(data, k):
                expected_type = type(data.__dict__[k])
                actual_type = type(v)
                if isinstance(v, expected_type):
                    data.__dict__[k] = v
                elif issubclass(expected_type, IntEnum):
                    data.__dict__[k] = expected_type(v)
                else:
                    raise TypeError(
                        f"unexpected value '{v}' for field '{k}', "
                        f"should be {expected_type} but type of the given value is {actual_type}"
                    )
            else:
                raise ValueError(f"invalid field name '{k}' for Reg:{addr}")
        self._write_reg(addr_, data)

    def discard(self):
        self.updated = {}

    @abstractmethod
    def flush(self, discard_after_flush=True):
        pass


@dataclass
class Gpio16(AbstractIcReg):
    b15: bool = field(default=False)
    b14: bool = field(default=False)
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
        self.b15 = p_1bf_bool(v, 15)
        self.b14 = p_1bf_bool(v, 14)
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
            b_1bf_bool(self.b15, 15)
            | b_1bf_bool(self.b14, 14)
            | b_1bf_bool(self.b13, 13)
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


@dataclass
class Gpio8(AbstractIcReg):
    b07: bool = field(default=False)
    b06: bool = field(default=False)
    b05: bool = field(default=False)
    b04: bool = field(default=False)
    b03: bool = field(default=False)
    b02: bool = field(default=False)
    b01: bool = field(default=False)
    b00: bool = field(default=False)

    def parse(self, v: int) -> None:
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
            b_1bf_bool(self.b07, 7)
            | b_1bf_bool(self.b06, 6)
            | b_1bf_bool(self.b05, 5)
            | b_1bf_bool(self.b04, 4)
            | b_1bf_bool(self.b03, 3)
            | b_1bf_bool(self.b02, 2)
            | b_1bf_bool(self.b01, 1)
            | b_1bf_bool(self.b00, 0)
        )


@dataclass
class Gpio6(AbstractIcReg):
    b05: bool = field(default=True)
    b04: bool = field(default=True)
    b03: bool = field(default=True)
    b02: bool = field(default=True)
    b01: bool = field(default=True)
    b00: bool = field(default=True)

    def parse(self, v: int) -> None:
        self.b05 = p_1bf_bool(v, 5)
        self.b04 = p_1bf_bool(v, 4)
        self.b03 = p_1bf_bool(v, 3)
        self.b02 = p_1bf_bool(v, 2)
        self.b01 = p_1bf_bool(v, 1)
        self.b00 = p_1bf_bool(v, 0)

    def build(self) -> int:
        return (
            b_1bf_bool(self.b05, 5)
            | b_1bf_bool(self.b04, 4)
            | b_1bf_bool(self.b03, 3)
            | b_1bf_bool(self.b02, 2)
            | b_1bf_bool(self.b01, 1)
            | b_1bf_bool(self.b00, 0)
        )


@dataclass
class Gpio4(AbstractIcReg):
    b03: bool = field(default=False)
    b02: bool = field(default=False)
    b01: bool = field(default=False)
    b00: bool = field(default=False)

    def parse(self, v: int) -> None:
        self.b03 = p_1bf_bool(v, 3)
        self.b02 = p_1bf_bool(v, 2)
        self.b01 = p_1bf_bool(v, 1)
        self.b00 = p_1bf_bool(v, 0)

    def build(self) -> int:
        return b_1bf_bool(self.b03, 3) | b_1bf_bool(self.b02, 2) | b_1bf_bool(self.b01, 1) | b_1bf_bool(self.b00, 0)


@dataclass
class Gpio2(AbstractIcReg):
    b01: bool = field(default=False)
    b00: bool = field(default=False)

    def parse(self, v: int) -> None:
        self.b01 = p_1bf_bool(v, 1)
        self.b00 = p_1bf_bool(v, 0)

    def build(self) -> int:
        return b_1bf_bool(self.b01, 1) | b_1bf_bool(self.b00, 0)


@dataclass
class Uint16(AbstractIcReg):
    v: int = field(default=0)

    def parse(self, v: int) -> None:
        self.v = p_nbf(v, 15, 0)

    def build(self) -> int:
        return b_nbf(self.v, 15, 0)
