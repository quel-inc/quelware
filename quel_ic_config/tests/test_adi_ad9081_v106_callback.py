import logging
import time
from typing import Dict

import adi_ad9081_v106 as ad9081

logger = logging.getLogger()


class AD9082Dummy:
    def __init__(self, ic_idx):
        self.ic_idx = ic_idx
        self.dummy_regfile: Dict[int, int] = {}

    def set_dummy_regfile(self, address: int, data: int):
        self.dummy_regfile[address] = data

    def get_dummy_regfile(self, address: int):
        return self.dummy_regfile[address]

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

    def log_write(self, level: int, msg: str):
        print(f"level = {level}, msg = '{msg}'")
        return (True,)


def test_readreg():
    wrapper0 = AD9082Dummy(0)
    wrapper0.set_dummy_regfile(0x1234, 0xCC)
    dev0 = ad9081.Device()
    dev0.callback_set(wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write)

    wrapper1 = AD9082Dummy(1)
    wrapper1.set_dummy_regfile(0x1234, 0x55)
    dev1 = ad9081.Device()
    dev1.callback_set(wrapper1.read_reg, wrapper1.write_reg, wrapper1.delay_us, wrapper1.log_write)

    reg1234 = ad9081.AddrData(0x1234)

    rc = ad9081.hal_reg_get(dev0, reg1234)
    assert rc == 0
    assert reg1234.data == 0xCC

    rc = ad9081.hal_reg_get(dev1, reg1234)
    assert rc == 0
    assert reg1234.data == 0x55


def test_writereg():
    wrapper0 = AD9082Dummy(0)
    wrapper0.set_dummy_regfile(0x1235, 0xFF)
    dev0 = ad9081.Device()
    dev0.callback_set(wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write)

    wrapper1 = AD9082Dummy(1)
    wrapper1.set_dummy_regfile(0x1235, 0xFF)
    dev1 = ad9081.Device()
    dev1.callback_set(wrapper1.read_reg, wrapper1.write_reg, wrapper1.delay_us, wrapper1.log_write)

    reg1235 = ad9081.AddrData(0x1235, 0xAA)
    rc = ad9081.hal_reg_set(dev0, reg1235)
    assert rc == 0
    assert wrapper0.get_dummy_regfile(0x1235) == 0xAA

    reg1235.data = 0x33
    rc = ad9081.hal_reg_set(dev1, reg1235)
    assert rc == 0
    assert wrapper1.get_dummy_regfile(0x1235) == 0x33

    rc = ad9081.hal_reg_get(dev0, reg1235)
    assert rc == 0
    assert reg1235.data == 0xAA

    rc = ad9081.hal_reg_get(dev1, reg1235)
    assert rc == 0
    assert reg1235.data == 0x33


def test_delay():
    wrapper0 = AD9082Dummy(0)
    dev0 = ad9081.Device()
    dev0.callback_set(wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write)

    t0 = time.perf_counter()
    ad9081.hal_delay_us(dev0, 10000)
    t1 = time.perf_counter()
    assert (t1 - t0) >= 0.01

    t0 = time.perf_counter()
    ad9081.hal_delay_us(dev0, 100000)
    t1 = time.perf_counter()
    assert (t1 - t0) >= 0.1


def test_log(capfd):
    wrapper0 = AD9082Dummy(0)
    dev0 = ad9081.Device()
    dev0.callback_set(wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write)
    ad9081.device_reset(dev0, ad9081.SOFT_RESET)
    out, err = capfd.readouterr()
    assert out == "level = 32, msg = 'adi_ad9081_device_reset(...)'\n"
    assert err == ""


def test_uninitialized(capfd):
    dev0 = ad9081.Device()
    reg1236 = ad9081.AddrData(0x1236)
    assert ad9081.hal_reg_get(dev0, reg1236) == -2


def test_unset():
    wrapper0 = AD9082Dummy(0)
    wrapper0.set_dummy_regfile(0x1236, 0xFF)
    dev0 = ad9081.Device()
    dev0.callback_set(wrapper0.read_reg, wrapper0.write_reg, wrapper0.delay_us, wrapper0.log_write)
    dev0.callback_unset()
    reg1236 = ad9081.AddrData(0x1236)
    assert ad9081.hal_reg_get(dev0, reg1236) == -2
