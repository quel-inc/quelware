import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic_core import ValidationError

from e7awghal.awgunit import AwgUnit
from e7awghal.capctrl import CapCtrlStandard
from e7awghal.capparam import CapParam, CapSection
from e7awghal.fwtype import E7FwType
from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal
from e7awghal.wavedata import AwgParam, WaveChunk
from testlibs.awgctrl_with_hlapi import AwgUnitHL
from testlibs.capunit_with_hlapi import CapUnitHL, CapUnitSimplifiedHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger(__name__)


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.74",
            "auidx": 2,
            "cmidx": 0,
        },
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.74",
            "auidx": 4,
            "cmidx": 3,
        },
    },
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def proxy_au_cm_w(request) -> tuple[AbstractQuel1Au50Hal, int, int, npt.NDArray[np.complex64]]:

    proxy = create_quel1au50hal_for_test(
        ipaddr_wss=request.param["box_config"]["ipaddr_wss"], auth_callback=lambda: True
    )
    proxy.initialize()
    assert proxy.fw_type() == E7FwType.SIMPLEMULTI_STANDARD

    au: AwgUnit = proxy.awgunit(request.param["box_config"]["auidx"])
    cmidx = request.param["box_config"]["cmidx"]

    w1 = np.arange(16384, dtype=np.complex64) - np.arange(16383, -1, -1, dtype=np.complex64) * 1j
    w2 = np.arange(16383, -1, -1, dtype=np.complex64) - np.arange(16384, dtype=np.complex64) * 1j
    w = np.hstack((w1, w2, w2, w1, w1, w2, w2, w1))
    w = np.hstack((w, w, w, w, w, w, w, w))
    assert w.shape == (1048576,)
    au.register_wavedata_from_complex64vector("w", w)
    param_w = AwgParam(num_wait_word=0, num_repeat=1)
    param_w.chunks.append(WaveChunk(name_of_wavedata="w", num_blank_word=0, num_repeat=1))
    au.load_parameter(param_w)

    return proxy, au.unit_index, cmidx, w


def au50loopback(proxy: AbstractQuel1Au50Hal, auidx: int, cmidx: int, cp: CapParam) -> list[npt.NDArray[np.complex64]]:
    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)

    cu = proxy.capunit(cc.units_of_module(cmidx)[0])
    if cmidx in {0, 1}:
        assert isinstance(cu, CapUnitHL)
    elif cmidx in {2, 3}:
        assert isinstance(cu, CapUnitSimplifiedHL)
    else:
        assert False
    cc.set_triggering_awgunit_idx(capmod_idx=cmidx, awgunit_idx=auidx)
    cc.add_triggerable_unit(cu.unit_index)

    cu.load_parameter(cp)
    fut = cu.wait_for_triggered_capture()

    au = proxy.awgunit(auidx)
    assert isinstance(au, AwgUnitHL)
    au.start_now().result()
    au.wait_done().result()

    rdr = fut.result()
    return rdr.as_wave_list()


def test_fixture(proxy_au_cm_w):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    au: AwgUnit = proxy.awgunit(auidx)
    ptr = au._lib.get_pointer_to_wavedata("w")
    data = proxy.hbmctrl.read_iq32(ptr, 16384)
    assert (data[:, 0] == w[0:16384].real).all()
    assert (data[:, 1] == w[0:16384].imag).all()


def test_basic(proxy_au_cm_w):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1)
    cp00.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    assert d00.shape == (1, 64)
    assert (d00[0] == w[64:128]).all()


@pytest.mark.parametrize(
    ["num_wait_word"],
    [
        (0,),
        (16,),
        (1024,),
    ],
)
def test_wait(proxy_au_cm_w, num_wait_word):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=num_wait_word, num_repeat=1)
    cp00.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    assert d00.shape == (1, 64)
    assert (d00[0] == w[64 + num_wait_word * 4 : 128 + num_wait_word * 4]).all()


@pytest.mark.parametrize(
    ["num_wait_word", "num_repeat"],
    [
        (-1, 1),
        (15, 1),
        (0x1_0000_0000, 1),
        (0xFFFF_FFFF_FFFF, 1),
        (0, 0),
        (0, -1),
        (0, 0x1_0000_0000),
        (0, 0xFFFF_FFFF_FFFF),
    ],
)
def test_wait_abnormal(proxy_au_cm_w, num_wait_word, num_repeat):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    with pytest.raises(ValidationError):
        cp00 = CapParam(num_wait_word=num_wait_word, num_repeat=num_repeat)
        _: Any = cp00


@pytest.mark.parametrize(
    ["num_repeat", "num_blank_word"],
    [
        (5, 1),
        (100, 1),
        (10000, 4),
    ],
)
def test_repeat(proxy_au_cm_w, num_repeat: int, num_blank_word: int):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=num_repeat)
    cp00.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=num_blank_word))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    assert d00.shape == (num_repeat, 64)
    for i in range(num_repeat):
        pos = (64 + num_blank_word * 4) * i
        assert (d00[i] == w[64 + pos : 128 + pos]).all()


@pytest.mark.parametrize(
    ["num_section", "num_blank_word"],
    [
        (5, 1),
        (100, 1),
        (4096, 3),
    ],
)
def test_sections(proxy_au_cm_w, num_section: int, num_blank_word: int):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1)
    for i in range(num_section):
        cp00.sections.append(CapSection(name=f"s{i:05d}", num_capture_word=16, num_blank_word=num_blank_word))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)

    assert len(d00) == num_section
    pos = 64
    for i in range(num_section):
        assert d00[i].shape == (1, 64)
        assert (d00[i] == w[pos : pos + 64]).all()
        pos += 64 + num_blank_word * 4


@pytest.mark.parametrize(
    ["num_section", "num_blank_word"],
    [
        (0, 1),
        (4097, 1),
    ],
)
def test_sections_abnormal_num_section(proxy_au_cm_w, num_section: int, num_blank_word: int):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1)
    for i in range(num_section):
        cp00.sections.append(CapSection(name=f"s{i:05d}", num_capture_word=16, num_blank_word=num_blank_word))
    with pytest.raises(ValidationError):
        _: Any = au50loopback(proxy, auidx, cmidx, cp00)


@pytest.mark.parametrize(
    ["num_section", "num_blank_word"],
    [
        (1, -1),
        (1, 0),
        (1, 0x1_0000_0000),
    ],
)
def test_sections_abnormal_num_blank_word(proxy_au_cm_w, num_section: int, num_blank_word: int):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1)
    with pytest.raises(ValidationError):
        for i in range(num_section):
            cp00.sections.append(CapSection(name=f"s{i:05d}", num_capture_word=16, num_blank_word=num_blank_word))


@pytest.mark.parametrize(
    [
        "sections",
    ],
    [
        (((16, 1), (32, 1), (64, 1), (128, 1), (256, 1), (512, 1), (1024, 1), (2048, 1), (4096, 1)),),
        ([(16 + i, 1 + i) for i in range(500)],),
    ],
)
def test_sections_variable(proxy_au_cm_w, sections: Sequence[tuple[int, int]]):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1)
    for i, (secword, blnword) in enumerate(sections):
        cp00.sections.append(CapSection(name=f"s{i:05d}", num_capture_word=secword, num_blank_word=blnword))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)

    assert len(d00) == len(sections)
    pos = 64
    for i, (secword, blnword) in enumerate(sections):
        assert d00[i].shape == (1, secword * 4)
        assert (d00[i] == w[pos : pos + secword * 4]).all()
        pos += (secword + blnword) * 4
