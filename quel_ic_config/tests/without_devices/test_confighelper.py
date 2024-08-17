from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, cast

import pytest

from quel_ic_config.abstract_ic import (
    AbstractIcConfigHelper,
    AbstractIcMixin,
    AbstractIcReg,
    b_1bf_bool,
    b_nbf,
    p_1bf_bool,
    p_nbf,
)


class ZigReg0E(IntEnum):
    UP = 0
    DOWN = 1


@dataclass
class ZigReg0(AbstractIcReg):
    a: int = field(default=0)  # [15:5]
    b: bool = field(default=False)  # [4]
    c: int = field(default=0)  # [3:2]
    d: int = field(default=0)  # [1:1]
    e: ZigReg0E = field(default=ZigReg0E.UP)  # [0:0]

    def parse(self, v: int) -> None:
        self.a = p_nbf(v, 15, 5)
        self.b = p_1bf_bool(v, 4)
        self.c = p_nbf(v, 3, 2)
        self.d = p_nbf(v, 1, 1)
        self.e = ZigReg0E(p_nbf(v, 0, 0))

    def build(self) -> int:
        return (
            b_nbf(self.a, 15, 5)
            | b_1bf_bool(self.b, 4)
            | b_nbf(self.c, 3, 2)
            | b_nbf(self.d, 1, 1)
            | int(b_nbf(self.e, 0, 0))
        )


class ZigMixin(AbstractIcMixin):
    Regs = {0: ZigReg0}
    RegNames = {"R0": 0}

    def __init__(self, name: str):
        super().__init__(name)
        self.reg0: int = self.init_reg0()

    def init_reg0(self) -> int:
        a = 1234
        b = True
        c = 2
        d = 1
        e = ZigReg0E.DOWN
        return (a << 5) | (int(b) << 4) | (c << 2) | (d << 1) | int(e)

    def dump_regs(self) -> Dict[int, int]:
        regs: Dict[int, int] = {}
        regs[0] = self.read_reg(0)
        return regs

    def read_reg(self, addr: int) -> int:
        if addr == 0:
            return self.reg0
        else:
            raise ValueError("invalid address")

    def write_reg(self, addr: int, data: int):
        if addr == 0:
            self.reg0 = data
        else:
            raise ValueError("invalid address")


class ZigConfigHelper(AbstractIcConfigHelper):
    def __init__(self, ic: ZigMixin):
        super().__init__(ic)

    def flush(self, discard_after_flush=True):
        if 0 in self.updated:
            self.ic.write_reg(0, self.updated[0])
        if discard_after_flush:
            self.discard()


def test_read():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    regobj = h.read_reg("R0")
    assert isinstance(regobj, ZigReg0)
    assert regobj.a == 1234
    assert isinstance(regobj.b, bool)
    assert regobj.b
    assert regobj.c == 2
    assert regobj.d == 1
    assert regobj.e == ZigReg0E.DOWN


def test_write():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    a = 1999
    b = False
    c = 1
    d = 0
    e = ZigReg0E.DOWN

    h.write_reg("R0", (a << 5) | (int(b) << 4) | (c << 2) | (d << 1) | int(e))
    h.flush()
    regobj = h.read_reg("R0")
    assert isinstance(regobj, ZigReg0)
    assert regobj.a == a
    assert isinstance(regobj.b, bool)
    assert not regobj.b
    assert regobj.c == c
    assert regobj.d == d
    assert regobj.e == ZigReg0E.DOWN


def test_discard():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    regobj_before = h.read_reg("R0")

    h.write_reg("R0", 0xEDBA)
    h.discard()
    h.flush()

    regobj_after = h.read_reg("R0")
    assert regobj_before.build() == regobj_after.build()


def test_field_1by1():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    a = 2001
    h.write_field(0, a=a)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).a == a

    b = False
    h.write_field(0, b=b)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).b == b

    c = 3
    h.write_field(0, c=c)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).c == c

    d = 0
    h.write_field(0, d=d)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).d == d

    e = ZigReg0E.UP
    h.write_field(0, e=e)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).e == e

    e2 = 0
    h.write_field(0, e=e2)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).e == e2


def test_field_all():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    a = 1357
    b = False
    c = 3
    d = 0
    e = ZigReg0E.UP
    h.write_field(0, a=a, b=b, c=c, d=d, e=e)
    h.flush()
    regobj = cast(ZigReg0, h.read_reg(0))
    assert regobj.a == a
    assert regobj.b == b
    assert regobj.c == c
    assert regobj.d == d
    assert regobj.e == e


def test_field_redo():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    a = 1999
    a_dummy = 1234
    b = False
    b_dummy = True
    c = 3
    c_dummy = 1
    d = 0
    d_dummy = 1
    h.write_field(0, a=a_dummy, b=b_dummy, c=c_dummy, d=d_dummy)
    h.write_field(0, a=a, b=b, c=c, d=d)
    h.flush()
    regobj = cast(ZigReg0, h.read_reg(0))
    assert regobj.a == a
    assert regobj.b == b
    assert regobj.c == c
    assert regobj.d == d


def test_cache():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    regobj0 = cast(ZigReg0, h.read_reg(0, False))
    regobj0.a += 1
    h.write_reg(0, regobj0)

    regobj1 = cast(ZigReg0, h.read_reg(0, True))
    assert regobj1.a == 1235
    regobj1.a += 1
    h.write_reg(0, regobj1)

    h.flush()
    regobj2 = cast(ZigReg0, h.read_reg(0, True))
    assert regobj2.a == 1236


def test_field_cache():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    regobj0 = cast(ZigReg0, h.read_reg(0, False))
    a1 = regobj0.a + 1
    assert a1 == 1235
    h.write_field(0, a=a1)

    regobj1 = cast(ZigReg0, h.read_reg(0, True))
    a2 = regobj1.a + 1
    assert a2 == 1236
    h.write_field(0, a=a2)

    regobj2 = cast(ZigReg0, h.read_reg(0, True))
    assert regobj2.a == 1236

    regobj3 = cast(ZigReg0, h.read_reg(0, False))
    assert regobj3.a == 1234

    h.discard()
    regobj4 = cast(ZigReg0, h.read_reg(0, True))
    assert regobj4.a == 1234


def test_invalid_field():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    with pytest.raises(ValueError):
        h.write_field(0, xxx=10)

    with pytest.raises(TypeError):
        h.write_field(0, b=123)


def test_invalid_value():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)

    with pytest.raises(ValueError):
        h.write_field(0, e=3)

    b = cast(ZigReg0, h.read_reg(0)).b
    h.write_field(0, c=7)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).c == (7 & 0b11)  # field 'b' is 2bit.
    assert cast(ZigReg0, h.read_reg(0)).b == b


def test_enum_and_int():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)
    h.write_field(0, e=0)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).e == ZigReg0E.UP

    h.write_field(0, e=ZigReg0E.DOWN)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).e == ZigReg0E.DOWN


def test_bool_and_int():
    ic = ZigMixin("zig9000")
    h = ZigConfigHelper(ic)
    h.write_field(0, c=0)
    h.flush()
    assert not cast(ZigReg0, h.read_reg(0)).c

    h.write_field(0, c=True)
    h.flush()
    assert cast(ZigReg0, h.read_reg(0)).c

    h.write_field(0, c=0)
    h.flush()
    assert not cast(ZigReg0, h.read_reg(0)).c
