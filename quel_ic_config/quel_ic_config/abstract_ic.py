import logging
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import Dict, Union, cast

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
    def __init__(self, ic: AbstractIcMixin):
        self.ic: AbstractIcMixin = ic
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
        """read the value of a register at the address.
        :param address: the address of register to be read. you may use alias name instead.
        :param refer_to_cache: return the cached register value if True.
        :return: an instance of the corresponding register object
        """
        addr_ = self._parse_addr(address)
        return self._read_reg(addr_, refer_to_cache)

    def write_reg(self, address: Union[int, str], data: Union[int, AbstractIcReg]) -> None:
        """update the value of a register into a cache. the cached values are written to the register when flush() is
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
        data: AbstractIcReg = self._read_reg(addr_, True)  # TODO: consider the validity of this design carefully!
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
