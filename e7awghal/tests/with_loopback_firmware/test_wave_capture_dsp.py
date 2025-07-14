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
from e7awghal.fwtype import E7FwAuxAttr, E7FwType
from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal
from e7awghal.wavedata import AwgParam, WaveChunk
from testlibs.awgctrl_with_hlapi import AwgUnitHL
from testlibs.capunit_with_hlapi import CapUnitHL
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

    w = np.arange(16384, dtype=np.complex64) - np.arange(16383, -1, -1, dtype=np.complex64) * 1j
    assert w.shape == (16384,)
    au.register_wavedata_from_complex64vector("w", w)
    param_w = AwgParam(num_wait_word=16, num_repeat=1)  # Notes: loopback firmware has delay of 64 samples
    param_w.chunks.append(WaveChunk(name_of_wavedata="w", num_blank_word=0, num_repeat=1))
    au.load_parameter(param_w)

    return proxy, au.unit_index, cmidx, w


def au50loopback(proxy: AbstractQuel1Au50Hal, auidx: int, cmidx: int, cp: CapParam) -> list[npt.NDArray[np.complex64]]:
    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)

    cu = proxy.capunit(cc.units_of_module(cmidx)[0])
    assert isinstance(cu, CapUnitHL)
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


@pytest.mark.parametrize(
    ["num_repeat"],
    [
        (2,),
        (100,),
    ],
)
def test_integration(proxy_au_cm_w, num_repeat):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=num_repeat, integration_enable=True)
    cp00.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=16))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    e00 = np.sum([w[i * 128 : i * 128 + 64] for i in range(num_repeat)], axis=0)  # 128 = (16+16)*4
    assert d00.shape == (1, 64)
    assert (d00[0] == e00).all()


@pytest.mark.parametrize(
    ["chunks"],
    [
        (((16, 1), (16, 1)),),
        (((16, 16), (32, 32), (64, 64), (128, 128), (256, 256)),),
    ],
)
def test_sum(proxy_au_cm_w, chunks: Sequence[tuple[int, int]]):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1, sum_enable=True)
    for i, (ncw, nbw) in enumerate(chunks):
        cp00.sections.append(CapSection(name="s{i:04d}", num_capture_word=ncw, num_blank_word=nbw))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)

    p = 0
    for i, (ncw, nbw) in enumerate(chunks):
        assert d00[i].shape == (1, 1)
        e00 = np.sum(w[p : p + ncw * 4])
        assert d00[i][0][0] == e00
        p += (ncw + nbw) * 4


def test_decimation(proxy_au_cm_w):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1, decimation_enable=True)
    cp00.sections.append(CapSection(name="s0", num_capture_word=32, num_blank_word=1))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    assert d00.shape == (1, 32)  # 32 = 32*4/4
    assert (d00[0] == w[0:128][::4]).all()


def test_sum_range(proxy_au_cm_w):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1, sum_enable=True, sum_range=(2, 17))
    cp00.sections.append(CapSection(name="s0", num_capture_word=32, num_blank_word=1))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    assert d00.shape == (1, 1)
    assert d00[0][0] == sum(w[8 : (17 + 1) * 4])


def test_sum_range_decimated(proxy_au_cm_w):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(num_wait_word=0, num_repeat=1, decimation_enable=True, sum_enable=True, sum_range=(2, 6))
    cp00.sections.append(CapSection(name="s0", num_capture_word=32, num_blank_word=1))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    assert d00.shape == (1, 1)
    assert d00[0][0] == sum(w[::4][2 * 4 : (6 + 1) * 4])  # Notes: I know w[::4] is inefficient.


@pytest.mark.parametrize(
    ["coeff", "exponent_offset"],
    [
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64), 14),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128), 14),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64), 14),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), 14),
        (np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64), 14),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex64), 14),
        (np.array([16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64), 0),
        (np.array([0, 0, 0, 0, 0, 0, 0, 8192, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64), 1),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4096], dtype=np.complex64), 2),
        (np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64), 15),
        (np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64), 15),
        (np.array([1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64), 14),
    ],
)
def test_cfir(proxy_au_cm_w, coeff, exponent_offset):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(
        num_wait_word=16,
        num_repeat=1,
        complexfir_enable=True,
        complexfir_coeff=coeff,
        complexfir_exponent_offset=exponent_offset,
    )
    cp00.sections.append(CapSection(name="s0", num_capture_word=32, num_blank_word=1))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    nonzero_idx = np.argmax(np.abs(coeff))
    nonzero_tap = coeff[nonzero_idx]
    assert d00.shape == (1, 128)
    assert (d00[0] == w[64 - nonzero_idx : 64 + 128 - nonzero_idx] * nonzero_tap).all()


@pytest.mark.parametrize(
    ["coeff", "exponent_offset"],
    [
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128), 15),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128), 16),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128), -1),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128), 12.5),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128), 14),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128), 14),
    ],
)
def test_cfir_abnormal(proxy_au_cm_w, coeff, exponent_offset):
    with pytest.raises(ValidationError):
        _: Any = CapParam(
            num_wait_word=16,
            num_repeat=1,
            complexfir_enable=True,
            complexfir_coeff=coeff,
            complexfir_exponent_offset=exponent_offset,
        )


@pytest.mark.parametrize(
    ["coeffs", "exponent_offset"],
    [
        (np.array([[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64), 14),
        (np.array([[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.float64), 14),
        (np.array([[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0]], dtype=np.float64), 14),
        (np.array([[16384, 0, 0, 0, 0, 0, 0, 0], [16384, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64), 0),
        (np.array([[-1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64), 15),
    ],
)
def test_rfirs(proxy_au_cm_w, coeffs, exponent_offset):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(
        num_wait_word=16,
        num_repeat=1,
        realfirs_enable=True,
        realfirs_real_coeff=coeffs[0],
        realfirs_imag_coeff=coeffs[1],
        realfirs_exponent_offset=exponent_offset,
    )
    cp00.sections.append(CapSection(name="s0", num_capture_word=32, num_blank_word=32))
    cp00.sections.append(CapSection(name="s1", num_capture_word=32, num_blank_word=32))
    if E7FwAuxAttr.DSP_v0b in proxy.auxattr:
        d00 = au50loopback(proxy, auidx, cmidx, cp00)[1]
        pos0 = (32 + 32) * 4
    elif E7FwAuxAttr.DSP_v1a in proxy.auxattr:
        d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]
        pos0 = 0
    else:
        assert False

    assert d00.shape == (1, 128)

    nonzero_idx0 = np.argmax(np.abs(coeffs[0]))
    nonzero_tap0 = coeffs[0][nonzero_idx0]
    assert (d00[0].real == w[pos0 + 64 - nonzero_idx0 : pos0 + 64 + 128 - nonzero_idx0].real * nonzero_tap0).all()

    nonzero_idx1 = np.argmax(np.abs(coeffs[1]))
    nonzero_tap1 = coeffs[1][nonzero_idx1]
    assert (d00[0].imag == w[pos0 + 64 - nonzero_idx1 : pos0 + 64 + 128 - nonzero_idx1].imag * nonzero_tap1).all()


@pytest.mark.parametrize(
    ["coeff_r", "coeff_i", "exponent_offset"],
    [
        (
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            np.array([0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            15,
        ),
        (
            np.array([0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            15,
        ),
        (
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            2.5,
        ),
        (
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            -1,
        ),
        (
            np.array([1j, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64),
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            14,
        ),
        (
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            np.array([1j, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64),
            14,
        ),
        (np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64), np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64), 14),
        (
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            14,
        ),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64), np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64), 14),
        (
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            14,
        ),
    ],
)
def test_rfirs_abnormal(proxy_au_cm_w, coeff_r, coeff_i, exponent_offset):
    with pytest.raises(ValidationError):
        _: Any = CapParam(
            num_wait_word=16,
            num_repeat=1,
            realfirs_enable=True,
            realfirs_real_coeff=coeff_r,
            realfirs_imag_coeff=coeff_i,
            realfirs_exponent_offset=exponent_offset,
        )


@pytest.mark.parametrize(
    ["windowfunc"],
    [
        (np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex128),),
        (np.array([0, 0, 0, -2, -1j, -2j, 0.5, 0.25, 0.125, -0.0625, 0, 0, 0, 0, -1, -1, -1], dtype=np.complex128),),
    ],
)
def test_windowfunc(proxy_au_cm_w, windowfunc):
    proxy, auidx, cmidx, w = proxy_au_cm_w

    cp00 = CapParam(
        num_wait_word=0,
        num_repeat=1,
        window_enable=True,
        window_coeff=windowfunc,
    )
    cp00.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    d00 = au50loopback(proxy, auidx, cmidx, cp00)[0]

    assert d00.shape == (1, 64)
    wf = np.zeros(64, dtype=windowfunc.dtype)
    wlen = min(len(windowfunc), 64)
    wf[:wlen] = windowfunc[:wlen]
    for i, v in enumerate(wf):
        assert d00[0][i] == w[i] * v


@pytest.mark.parametrize(
    ["windowfunc"],
    [
        (np.array([0, 0, 0, 2], dtype=np.complex128),),
        (np.array([], dtype=np.complex128),),
        (np.array([1, 1, 1, 1], dtype=np.complex64),),
        (np.zeros(2049, dtype=np.complex128),),
    ],
)
def test_windowfunc_abnormal(proxy_au_cm_w, windowfunc):
    with pytest.raises(ValidationError):
        _: Any = CapParam(
            num_wait_word=0,
            num_repeat=1,
            window_enable=True,
            window_coeff=windowfunc,
        )
