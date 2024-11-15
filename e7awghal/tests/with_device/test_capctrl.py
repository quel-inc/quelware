import logging
import time
from concurrent.futures import CancelledError

import numpy as np
import pytest

from e7awghal.abstract_cap import AbstractCapUnit
from e7awghal.capparam import CapParam, CapSection
from e7awghal.capunit import CapUnitSimplified
from e7awghal.common_defs import E7awgMemoryError
from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal
from testlibs.capunit_with_hlapi import CapUnitHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.50",
        },
    },
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def proxy(request) -> AbstractQuel1Au50Hal:
    proxy = create_quel1au50hal_for_test(
        ipaddr_wss=request.param["box_config"]["ipaddr_wss"], auth_callback=lambda: True
    )
    proxy.initialize()
    return proxy


def test_version(proxy):
    assert proxy.capctrl.version == "a:2024/01/25-2"


@pytest.mark.parametrize(
    ("addr0", "size", "val0"),
    [
        (0x0000, 0x1000, 0),
        (0x0000, 0x1000, 0x12345),
        (0x0020, 0x0001, 0xFFFF_AAAA),
        (0x0044, 0x0008, 0xFFFF_AAAB),
        (0x012C, 0x0014, 0xC000_0000),
        (0x0004, 0x0100, 0x12345),
        (0x0010, 0x0200, 0x12345),
        (0x0080, 0x0401, 0x12345),
    ],
)
def test_read_write(addr0, size, val0, proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)

    assert isinstance(cu0, CapUnitSimplified)

    # clean up
    zw = np.zeros(0x1000, dtype=np.uint32)
    cu0._write_param_regs(0xB000, zw)
    zr = cu0._read_param_regs(0xB000, 0x1000)
    assert (zw == zr).all()

    # test
    zw0 = np.arange(size)
    zw0 += val0
    cu0._write_param_regs(0xB000 + addr0, zw0)
    zr0 = cu0._read_param_regs(0xB000, 0x1000)

    idx0 = addr0 // 4
    if idx0 > 0:
        assert (zr0[:idx0] == 0x0000_0000).all()
    assert (zr0[idx0 : idx0 + size] == zw0).all()
    if 0x1000 - (idx0 + size) > 0:
        assert (zr0[idx0 + size :] == 0).all()


def test_capture_1_1(proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()

    cp0 = CapParam()
    cp0.sections.append(CapSection(name="s0", num_capture_word=256))
    cu0.load_parameter(cp0)
    fut = cu0.start_now()
    cr = fut.result()

    raw = cr.rawwave()
    assert len(raw) == 256 * 4
    dct = cr.as_wave_dict()
    assert "s0" in dct
    assert dct["s0"].shape == (1, 256 * 4)
    assert (dct["s0"][0] == raw).all()
    lst = cr.as_wave_list()
    assert len(lst) == 1
    assert lst[0].shape == (1, 256 * 4)
    assert (lst[0] == raw).all()


def test_capture_1_1_verylong(proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()

    cp0 = CapParam()
    cp0.sections.append(CapSection(name="s0", num_capture_word=8388608))
    cu0.load_parameter(cp0)
    fut = cu0.start_now()
    cr = fut.result()

    t0 = time.perf_counter()
    raw = cr.rawwave()
    t1 = time.perf_counter()
    logger.info(f"it takes {t1-t0} second to retrieve 32M samples")
    assert len(raw) == 8388608 * 4

    dct = cr.as_wave_dict()
    assert "s0" in dct
    assert dct["s0"].shape == (1, 8388608 * 4)
    assert (dct["s0"][0] == raw).all()
    lst = cr.as_wave_list()
    assert len(lst) == 1
    assert lst[0].shape == (1, 8388608 * 4)
    assert (lst[0] == raw).all()


def test_capture_1_1_toolong(proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()

    cp0 = CapParam()
    cp0.sections.append(CapSection(name="s0", num_capture_word=8388609))
    with pytest.raises(E7awgMemoryError, match=f"failed to acquire {8388609*32} byte in '{cu0._mm._name}'"):
        cu0.load_parameter(cp0)


def test_capture_1_3(proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()

    cp0 = CapParam()
    cp0.sections.append(CapSection(name="s0", num_capture_word=256))
    cp0.sections.append(CapSection(name="s1", num_capture_word=128))
    cp0.sections.append(CapSection(name="s2", num_capture_word=64))
    cu0.load_parameter(cp0)
    fut = cu0.start_now()
    cr = fut.result()

    raw = cr.rawwave()
    assert len(raw) == (256 + 128 + 64) * 4

    dct = cr.as_wave_dict()
    pos = 0
    for k, n in (("s0", 256), ("s1", 128), ("s2", 64)):
        assert k in dct
        assert dct[k].shape == (1, n * 4)
        assert (dct[k][0] == raw[pos * 4 : (pos + n) * 4]).all()
        pos += n

    lst = cr.as_wave_list()
    pos = 0
    assert len(lst) == 3
    for i, n in enumerate((256, 128, 64)):
        assert lst[i].shape == (1, n * 4)
        assert (lst[i] == raw[pos * 4 : (pos + n) * 4]).all()
        pos += n


def test_capture_10_1(proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()

    num_repeat = 10
    cp0 = CapParam(num_repeat=num_repeat)
    cp0.sections.append(CapSection(name="s0", num_capture_word=256))
    cu0.load_parameter(cp0)
    fut = cu0.start_now()
    cr = fut.result()

    raw = cr.rawwave()
    assert len(raw) == 256 * 4 * num_repeat

    dct = cr.as_wave_dict()
    assert "s0" in dct
    assert dct["s0"].shape == (num_repeat, 256 * 4)
    for i in range(num_repeat):
        assert (dct["s0"][i] == raw[256 * 4 * i : 256 * 4 * (i + 1)]).all()

    lst = cr.as_wave_list()
    assert len(lst) == 1
    assert lst[0].shape == (num_repeat, 256 * 4)
    for i in range(num_repeat):
        assert (lst[0][i] == raw[256 * 4 * i : 256 * 4 * (i + 1)]).all()


def test_capture_15_3(proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()

    num_repeat = 15

    cp0 = CapParam(num_repeat=num_repeat)
    cp0.sections.append(CapSection(name="s0", num_capture_word=256))
    cp0.sections.append(CapSection(name="s1", num_capture_word=128))
    cp0.sections.append(CapSection(name="s2", num_capture_word=64))
    cu0.load_parameter(cp0)
    fut = cu0.start_now()
    cr = fut.result()

    raw = cr.rawwave()
    assert len(raw) == (256 + 128 + 64) * 4 * num_repeat

    dct = cr.as_wave_dict()
    for k, n in (("s0", 256), ("s1", 128), ("s2", 64)):
        assert k in dct
        assert dct[k].shape == (num_repeat, n * 4)

    pos = 0
    for j in range(num_repeat):
        for k, n in (("s0", 256), ("s1", 128), ("s2", 64)):
            assert (dct[k][j] == raw[pos * 4 : (pos + n) * 4]).all()
            pos += n

    lst = cr.as_wave_list()
    assert len(lst) == 3
    for i, n in enumerate((256, 128, 64)):
        assert lst[i].shape == (num_repeat, n * 4)

    pos = 0
    for j in range(num_repeat):
        for i, n in enumerate((256, 128, 64)):
            assert (lst[i][j] == raw[pos * 4 : (pos + n) * 4]).all()
            pos += n


def test_capture_no_configure(proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()

    with pytest.raises(RuntimeError, match=f"cap_unit-#{cu0.unit_index:02d} is not configured yet"):
        cu0.start_now()


@pytest.mark.parametrize(
    ("waiting_time",),
    [
        (0,),
        (0.25,),
    ],
)
def test_capture_cancel(waiting_time, proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()

    cp0 = CapParam()
    cp0.sections.append(CapSection(name="s0", num_capture_word=256))
    cu0.load_parameter(cp0)
    fut0 = cu0.wait_for_triggered_capture()
    time.sleep(waiting_time)
    cu0.cancel()
    with pytest.raises(CancelledError):
        fut0.result()


@pytest.mark.parametrize(
    ("waiting_time",),
    [
        (0,),
        (0.001,),
        (0.002,),
        (0.004,),
        (0.008,),
    ],
)
def test_capture_cancel_requires_terminate(waiting_time, proxy):
    cu0: AbstractCapUnit = proxy.capunit(0)
    assert isinstance(cu0, CapUnitHL)
    cu0.initialize()
    assert not cu0.is_busy()
    assert isinstance(cu0, CapUnitSimplified)
    assert cu0._current_reader is None

    cp0 = CapParam()
    cp0.sections.append(
        CapSection(name="s0", num_capture_word=8388608)
    )  # Notes: it takes 0.01677 sec to complete capture.
    cu0.load_parameter(cp0)
    fut0 = cu0.start_now()
    time.sleep(waiting_time)
    cu0.cancel()
    with pytest.raises(CancelledError):
        fut0.result()
