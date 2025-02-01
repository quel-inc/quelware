import logging
import time
from typing import Dict

import adi_ad9082_v170 as adi_ad9082

logger = logging.getLogger()


class AD9082Dummy:
    def __init__(self, ic_idx):
        self.ic_idx: int = ic_idx
        self.dummy_regfile: Dict[int, int] = {}
        self.resetpin: int = 1
        self.was_reset: bool = False

    def set_dummy_regfile(self, address: int, data: int):
        self.dummy_regfile[address] = data

    def get_dummy_regfile(self, address: int):
        return self.dummy_regfile[address]

    def set_dummy_resetpin(self, level: int):
        if self.resetpin == 0 and level != 0:
            self.was_reset = True
        self.resetpin = level
        if self.resetpin == 0:
            self.dummy_regfile.clear()

    def read_reg(self, address: int):
        logger.info(f"read reg[{address:04x}] of ad9082[{self.ic_idx:d}]")
        return True, self.dummy_regfile[address]

    def write_reg(self, address: int, value: int):
        logger.info(f"write f{value:02x} into reg[{address:04x}] of ad9082[{self.ic_idx:d}")
        self.set_dummy_regfile(address & 0xFFFF, value & 0xFF)
        return (True,)

    def delay_us(self, us: int):
        logger.info(f"delay {us}us")
        time.sleep(us * 1e-6)
        return (True,)

    def log_write(self, loglevel: int, msg: str):
        print(f"level = {loglevel}, msg = '{msg}'")
        return (True,)

    def reset_pin_ctrl(self, level: int):
        logger.info(f"reset is {'asserted' if level == 0 else 'negated'}")
        self.set_dummy_resetpin(level)
        return (True,)


def test_readreg():
    wrapper0 = AD9082Dummy(0)
    wrapper0.set_dummy_regfile(0x1234, 0xCC)
    dev0 = adi_ad9082.Device()
    dev0.callback_set(
        wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write, wrapper0.reset_pin_ctrl
    )

    wrapper1 = AD9082Dummy(1)
    wrapper1.set_dummy_regfile(0x1234, 0x55)
    dev1 = adi_ad9082.Device()
    dev1.callback_set(
        wrapper1.read_reg, wrapper1.write_reg, wrapper1.delay_us, wrapper1.log_write, wrapper0.reset_pin_ctrl
    )

    reg1234 = adi_ad9082.RegData(0x1234)

    rc = adi_ad9082.hal_reg_get(dev0, reg1234)
    assert rc == 0
    assert reg1234.data == 0xCC

    rc = adi_ad9082.hal_reg_get(dev1, reg1234)
    assert rc == 0
    assert reg1234.data == 0x55


def test_writereg():
    wrapper0 = AD9082Dummy(0)
    wrapper0.set_dummy_regfile(0x1235, 0xFF)
    dev0 = adi_ad9082.Device()
    dev0.callback_set(
        wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write, wrapper0.reset_pin_ctrl
    )

    wrapper1 = AD9082Dummy(1)
    wrapper1.set_dummy_regfile(0x1235, 0xFF)
    dev1 = adi_ad9082.Device()
    dev1.callback_set(
        wrapper1.read_reg, wrapper1.write_reg, wrapper1.delay_us, wrapper1.log_write, wrapper0.reset_pin_ctrl
    )

    reg1235 = adi_ad9082.RegData(0x1235, 0xAA)
    rc = adi_ad9082.hal_reg_set(dev0, reg1235)
    assert rc == 0
    assert wrapper0.get_dummy_regfile(0x1235) == 0xAA

    reg1235.data = 0x33
    rc = adi_ad9082.hal_reg_set(dev1, reg1235)
    assert rc == 0
    assert wrapper1.get_dummy_regfile(0x1235) == 0x33

    rc = adi_ad9082.hal_reg_get(dev0, reg1235)
    assert rc == 0
    assert reg1235.data == 0xAA

    rc = adi_ad9082.hal_reg_get(dev1, reg1235)
    assert rc == 0
    assert reg1235.data == 0x33


def test_delay():
    wrapper0 = AD9082Dummy(0)
    dev0 = adi_ad9082.Device()
    dev0.callback_set(
        wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write, wrapper0.reset_pin_ctrl
    )

    t0 = time.perf_counter()
    adi_ad9082.hal_delay_us(dev0, 10000)
    t1 = time.perf_counter()
    assert (t1 - t0) >= 0.01

    t0 = time.perf_counter()
    adi_ad9082.hal_delay_us(dev0, 100000)
    t1 = time.perf_counter()
    assert (t1 - t0) >= 0.1


def test_softreset_and_logger(capfd):
    wrapper0 = AD9082Dummy(0)
    dev0 = adi_ad9082.Device()
    dev0.callback_set(
        wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write, wrapper0.reset_pin_ctrl
    )
    adi_ad9082.device_reset(dev0, adi_ad9082.SOFT_RESET)
    out, err = capfd.readouterr()
    assert (
        out == "level = 32, msg = 'adi_ad9082_device_reset(...)'\n"
        "level = 16, msg = 'ad9082: w@0000 = 81'\n"
        "level = 16, msg = 'ad9082: w@0000 = 00'\n"
    )
    assert err == ""


def test_hardreset(capfd):
    wrapper0 = AD9082Dummy(0)
    dev0 = adi_ad9082.Device()
    dev0.callback_set(
        wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write, wrapper0.reset_pin_ctrl
    )
    adi_ad9082.device_reset(dev0, adi_ad9082.HARD_RESET)
    assert wrapper0.was_reset
    assert wrapper0.resetpin != 0


def test_uninitialized(capfd):
    dev0 = adi_ad9082.Device()
    reg1236 = adi_ad9082.RegData(0x1236)
    assert adi_ad9082.hal_reg_get(dev0, reg1236) == -2


def test_unset():
    wrapper0 = AD9082Dummy(0)
    wrapper0.set_dummy_regfile(0x1236, 0xFF)
    dev0 = adi_ad9082.Device()
    dev0.callback_set(
        wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write, wrapper0.reset_pin_ctrl
    )
    dev0.callback_unset()
    reg1236 = adi_ad9082.RegData(0x1236)
    assert adi_ad9082.hal_reg_get(dev0, reg1236) == -2
