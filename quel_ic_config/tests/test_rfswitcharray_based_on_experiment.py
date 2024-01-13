import pytest

from quel_ic_config.quel1_config_subsystem_common import Quel1BoxType
from quel_ic_config.rfswitcharray import (
    AbstractRfSwitchArrayMixin,
    QubeRfSwitchArrayMixin,
    Quel1TypeARfSwitchArrayMixin,
    Quel1TypeBRfSwitchArrayMixin,
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


def dac2bit(unittype: Quel1BoxType, group: int, line: int) -> int:
    """this code is extracted from the conventional test code of rfswitch control,
    which is confirmed working correctly.
    :param unittype: a type of unit
    :param group: index of MxFE.
    :param line: index of DAC.
    :return: index of bit in the register.
    """
    if group == 0:
        if unittype == Quel1BoxType.QuEL1_TypeA:
            sw_to_open: int = [0, 5, 2, 6][line]
        elif unittype in {
            Quel1BoxType.QuEL1_TypeB,
            Quel1BoxType.QuBE_RIKEN_TypeA,
            Quel1BoxType.QuBE_RIKEN_TypeB,
        }:
            sw_to_open = [0, 2, 5, 6][line]
        else:
            raise AssertionError
    elif group == 1:
        if unittype == Quel1BoxType.QuEL1_TypeA:
            sw_to_open = [12, 8, 7, 11][line]
        elif unittype == Quel1BoxType.QuEL1_TypeB:
            sw_to_open = [12, 11, 7, 8][line]
        elif unittype in {
            Quel1BoxType.QuBE_RIKEN_TypeA,
            Quel1BoxType.QuBE_RIKEN_TypeB,
        }:
            sw_to_open = [12, 11, 8, 7][line]
        else:
            raise AssertionError
    else:
        raise AssertionError

    return sw_to_open


@pytest.mark.parametrize(
    ("boxtype", "group", "line"),
    [
        (Quel1BoxType.QuEL1_TypeA, 0, 0),
        (Quel1BoxType.QuEL1_TypeA, 0, 1),
        (Quel1BoxType.QuEL1_TypeA, 0, 2),
        (Quel1BoxType.QuEL1_TypeA, 0, 3),
        (Quel1BoxType.QuEL1_TypeA, 1, 0),
        (Quel1BoxType.QuEL1_TypeA, 1, 1),
        (Quel1BoxType.QuEL1_TypeA, 1, 2),
        (Quel1BoxType.QuEL1_TypeA, 1, 3),
        (Quel1BoxType.QuEL1_TypeB, 0, 0),
        (Quel1BoxType.QuEL1_TypeB, 0, 1),
        (Quel1BoxType.QuEL1_TypeB, 0, 2),
        (Quel1BoxType.QuEL1_TypeB, 0, 3),
        (Quel1BoxType.QuEL1_TypeB, 1, 0),
        (Quel1BoxType.QuEL1_TypeB, 1, 1),
        (Quel1BoxType.QuEL1_TypeB, 1, 2),
        (Quel1BoxType.QuEL1_TypeB, 1, 3),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 0, 0),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 0, 1),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 0, 2),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 0, 3),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 1, 0),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 1, 1),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 1, 2),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 1, 3),
    ],
)
def test_quel1(boxtype, group, line):
    if boxtype == Quel1BoxType.QuEL1_TypeA:
        obj: AbstractRfSwitchArrayMixin = JigQuel1TypeARfSwitchArray()
    elif boxtype == Quel1BoxType.QuEL1_TypeB:
        obj = JigQuel1TypeBRfSwitchArray()
    elif boxtype == Quel1BoxType.QuBE_RIKEN_TypeA:
        obj = JigQubeRfSwitchArray()
    else:
        raise AssertionError

    obj.write_reg(0, 0x0000)  # Ic Object have a raw bitfield.

    helper = RfSwitchArrayConfigHelper(obj)
    regname = f"Group{group}"
    fields = {
        f"path{line}": True,
    }
    helper.write_field(regname, **fields)
    helper.flush()

    if boxtype in {Quel1BoxType.QuBE_OU_TypeA, Quel1BoxType.QuBE_RIKEN_TypeA} and line == 0:
        v = 1 << dac2bit(boxtype, group, line)
        assert obj.dump_regs()[0] == (v | (v << 1))
    else:
        assert obj.dump_regs()[0] == (1 << dac2bit(boxtype, group, line))


def monitor_loop(group: int) -> int:
    if group == 0:
        sw = 0b00000000011000
    elif group == 1:
        sw = 0b00011000000000
    else:
        raise AssertionError
    return sw


def read_loop(group: int) -> int:
    if group == 0:
        sw = 0b00000000000011
    elif group == 1:
        sw = 0b11000000000000
    else:
        raise AssertionError
    return sw


@pytest.mark.parametrize(
    ("boxtype", "group", "adc_line"),
    [
        (Quel1BoxType.QuEL1_TypeA, 0, 0),
        (Quel1BoxType.QuEL1_TypeA, 1, 0),
        (Quel1BoxType.QuEL1_TypeA, 0, 1),
        (Quel1BoxType.QuEL1_TypeA, 1, 1),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 0, 0),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 1, 0),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 0, 1),
        (Quel1BoxType.QuBE_RIKEN_TypeA, 1, 1),
    ],
)
def test_loopback_switch(boxtype: Quel1BoxType, group: int, adc_line: int):
    if boxtype == Quel1BoxType.QuEL1_TypeA:
        obj: AbstractRfSwitchArrayMixin = JigQuel1TypeARfSwitchArray()
    elif boxtype == Quel1BoxType.QuBE_RIKEN_TypeA:
        obj = JigQubeRfSwitchArray()
    else:
        raise AssertionError
    obj.write_reg(0, 0x0000)  # Ic Object have a raw bitfield.

    helper = RfSwitchArrayConfigHelper(obj)
    regname = f"Group{group}"
    if adc_line == 0:
        field_name: str = "path0"
        ground_truth: int = read_loop(group)
    elif adc_line == 1:
        field_name = "monitor"
        ground_truth = monitor_loop(group)
    else:
        assert False

    fields = {
        field_name: True,
    }
    helper.write_field(regname, **fields)
    helper.flush()

    if boxtype in {Quel1BoxType.QuBE_OU_TypeA, Quel1BoxType.QuBE_RIKEN_TypeA}:
        assert obj.dump_regs()[0] == ground_truth
    elif boxtype == Quel1BoxType.QuEL1_TypeA:
        v = obj.dump_regs()[0]
        assert (v | (v << 1)) == ground_truth
    else:
        assert False
