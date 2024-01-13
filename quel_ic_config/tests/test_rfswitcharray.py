from typing import Tuple

import pytest

from quel_ic_config.rfswitcharray import (
    QubeRfSwitchArrayMixin,
    QubeRfSwitchRegs,
    Quel1TypeARfSwitchArrayMixin,
    Quel1TypeARfSwitchRegs,
    Quel1TypeBRfSwitchArrayMixin,
    Quel1TypeBRfSwitchRegs,
    RfSwitchArrayConfigHelper,
)


class JigQuel1TypeARfSwitchArray(Quel1TypeARfSwitchArrayMixin):
    def __init__(self):
        super().__init__("JigQuel1TypeARfSwitchArray")
        self.virtualregisterfile = [0]

    def read_reg(self, addr: int) -> int:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        return self.virtualregisterfile[addr]

    def write_reg(self, addr: int, data: int) -> None:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        self.virtualregisterfile[addr] = data


class JigQuel1TypeBRfSwitchArray(Quel1TypeBRfSwitchArrayMixin):
    def __init__(self):
        super().__init__("JigQuel1TypeBRfSwitchArray")
        self.virtualregisterfile = [0, 0]

    def read_reg(self, addr: int) -> int:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        return self.virtualregisterfile[addr]

    def write_reg(self, addr: int, data: int) -> None:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        self.virtualregisterfile[addr] = data


class JigQubeRfSwitchArray(QubeRfSwitchArrayMixin):
    def __init__(self):
        super().__init__("JigQubeRfSwitchArray")
        self.virtualregisterfile = [0, 0]

    def read_reg(self, addr: int) -> int:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        return self.virtualregisterfile[addr]

    def write_reg(self, addr: int, data: int) -> None:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        self.virtualregisterfile[addr] = data


bits_to_check_quel1 = {
    0: (6, 5, 3, 2, 0),
    1: (12, 11, 9, 8, 7),
}


bits_to_check_qube_normal = {
    0: (6, 5, 2),
    1: (11, 8, 7),
}


bits_to_check_qube_grouped = {
    0: (4, 3, 1, 0),
    1: (13, 12, 10, 9),
}


@pytest.mark.parametrize(
    "idx,",
    [reg for reg in Quel1TypeARfSwitchRegs],
)
def test_bitfield_quel1_a(idx: int):
    regobj = Quel1TypeARfSwitchRegs[idx]()
    regobj.parse(0)
    constant = regobj.build()
    for i in range(16):
        v = 1 << i
        if i in bits_to_check_quel1[idx]:
            assert v & constant == 0
            regobj.parse(v)
            assert v == regobj.build() & (~constant)
        else:
            regobj.parse(v)
            assert regobj.build() == constant


@pytest.mark.parametrize(
    "idx,",
    [reg for reg in Quel1TypeBRfSwitchRegs],
)
def test_bitfield_quel1_b(idx: int):
    regobj = Quel1TypeBRfSwitchRegs[idx]()
    regobj.parse(0)
    constant = regobj.build()
    for i in range(16):
        v = 1 << i
        if i in bits_to_check_quel1[idx]:
            assert v & constant == 0
            regobj.parse(v)
            assert v == regobj.build() & (~constant)
        else:
            regobj.parse(v)
            assert regobj.build() == constant


@pytest.mark.parametrize(
    "idx,",
    [reg for reg in QubeRfSwitchRegs],
)
def test_bitfield_qube(idx: int):
    regobj = QubeRfSwitchRegs[idx]()
    regobj.parse(0)
    constant = regobj.build()
    for i in range(16):
        v = 1 << i
        if i in bits_to_check_qube_normal[idx]:
            assert v & constant == 0
            regobj.parse(v)
            assert v == regobj.build() & (~constant)
        elif i in bits_to_check_qube_grouped[idx]:
            # these pairs of RF switches can be controlled independently, however, the library constrains to the pairs
            # of the switches to be identical, namely (inside, inside) or (outside, outside).
            # This is because the other combinations are meaningless.
            if v == 1 or v == 2:
                v = 3
            elif v == 8 or v == 16:
                v = 24
            elif v == 512 or v == 1024:
                v = 1536
            elif v == 4096 or v == 8192:
                v = 12288
            else:
                assert False
            assert v & constant == 0
            regobj.parse(v)
            assert v == regobj.build() & (~constant)
        else:
            regobj.parse(v)
            assert regobj.build() == constant


@pytest.mark.parametrize(
    "jsacls,answer",
    [
        [
            JigQuel1TypeARfSwitchArray,
            (1 << 0, 1 << 5, 1 << 2, 1 << 6, 1 << 3, 1 << 12, 1 << 8, 1 << 7, 1 << 11, 1 << 9),
        ],
        [
            JigQuel1TypeBRfSwitchArray,
            (1 << 0, 1 << 2, 1 << 5, 1 << 6, 1 << 3, 1 << 12, 1 << 11, 1 << 7, 1 << 8, 1 << 9),
        ],
        [
            JigQubeRfSwitchArray,
            (
                (1 << 0) | (1 << 1),
                1 << 2,
                1 << 5,
                1 << 6,
                (1 << 3) | (1 << 4),
                (1 << 12) | (1 << 13),
                1 << 11,
                1 << 8,
                1 << 7,
                (1 << 9) | (1 << 10),
            ),
        ],
    ],
)
def test_correspondences(jsacls: type, answer: Tuple[int, int, int, int, int, int, int, int, int, int]):
    ic = jsacls()
    helper = RfSwitchArrayConfigHelper(ic)
    helper.write_field(0, path0=False, path1=False, path2=False, path3=False, monitor=False)
    helper.write_field(1, path0=False, path1=False, path2=False, path3=False, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == 0

    helper.write_field(0, path0=True, path1=False, path2=False, path3=False, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == answer[0]

    helper.write_field(0, path0=False, path1=True, path2=False, path3=False, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == answer[1]

    helper.write_field(0, path0=False, path1=False, path2=True, path3=False, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == answer[2]

    helper.write_field(0, path0=False, path1=False, path2=False, path3=True, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == answer[3]

    helper.write_field(0, path0=False, path1=False, path2=False, path3=False, monitor=True)
    helper.flush()
    assert ic.dump_regs()[0] == answer[4]

    helper.write_field(0, path0=False, path1=False, path2=False, path3=False, monitor=False)
    helper.write_field(1, path0=True, path1=False, path2=False, path3=False, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == answer[5]

    helper.write_field(1, path0=False, path1=True, path2=False, path3=False, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == answer[6]

    helper.write_field(1, path0=False, path1=False, path2=True, path3=False, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == answer[7]

    helper.write_field(1, path0=False, path1=False, path2=False, path3=True, monitor=False)
    helper.flush()
    assert ic.dump_regs()[0] == answer[8]

    helper.write_field(1, path0=False, path1=False, path2=False, path3=False, monitor=True)
    helper.flush()
    assert ic.dump_regs()[0] == answer[9]
