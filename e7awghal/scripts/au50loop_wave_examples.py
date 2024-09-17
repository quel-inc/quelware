import logging

import numpy as np
import numpy.typing as npt

from e7awghal import (
    AbstractCapParam,
    AbstractQuel1Au50Hal,
    AwgParam,
    AwgUnit,
    CapCtrlStandard,
    CapParam,
    CapSection,
    CapUnit,
    WaveChunk,
)
from e7awghal.capdata import CapIqDataReader
from testlibs.capunit_with_hlapi import CapUnitHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger()


def au50loopback(auidx: int, cmidx: int, cp0: AbstractCapParam, valid_data: dict[int, list[npt.NDArray[np.complex64]]]):
    au = proxy.awgunit(auidx)

    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)

    cus: list[CapUnit] = []
    cc.set_triggering_awgunit_idx(capmod_idx=cmidx, awgunit_idx=au.unit_index)
    cu = proxy.capunit(cc.units_of_module(cmidx)[0])
    cu.load_parameter(cp0)
    cc.add_triggerable_unit(cu.unit_index)
    cus.append(cu)

    futs = []
    for cu in cus:
        assert isinstance(cu, CapUnitHL)
        futs.append(cu.wait_for_triggered_capture())

    au.start_now().result()
    au.wait_done().result()
    for idx, fut in enumerate(futs):
        rdr = fut.result()
        valid_data[idx] = rdr.as_wave_list()


def au50loopback_rawdata(auidx: int, cmidx: int, cp0: AbstractCapParam) -> dict[int, CapIqDataReader]:
    au = proxy.awgunit(auidx)

    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)

    cus: list[CapUnit] = []
    cc.set_triggering_awgunit_idx(capmod_idx=cmidx, awgunit_idx=au.unit_index)
    cu = proxy.capunit(cc.units_of_module(cmidx)[0])
    cu.load_parameter(cp0)
    cc.add_triggerable_unit(cu.unit_index)
    cus.append(cu)

    futs = []
    for cu in cus:
        assert isinstance(cu, CapUnitHL)
        futs.append(cu.wait_for_triggered_capture())

    au.start_now().result()
    au.wait_done().result()
    rdrs: dict[int, CapIqDataReader] = {}
    for idx, fut in enumerate(futs):
        rdrs[idx] = fut.result()
    return rdrs


if __name__ == "__main__":
    """
    awg_unix-#2 --> cap_mod-#0
    awg_unix-#3 --> cap_mod-#2
    awg_unix-#4 --> cap_mod-#3
    awg_unix-#15 --> cap_mod-#1
    """
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    proxy: AbstractQuel1Au50Hal = create_quel1au50hal_for_test(ipaddr_wss="10.1.0.74")
    proxy.initialize()

    w = np.zeros(16384, dtype=np.complex64)
    w[:] = np.arange(16384) + np.arange(16383, -1, -1) * 1j
    circle = np.exp(1j * np.pi * np.arange(256) / 128)
    circle = np.round(circle * 32767).astype(np.complex64)

    auidx = 2
    cmidx = 0
    au: AwgUnit = proxy.awgunit(auidx)
    au.register_wavedata_from_complex64vector("w", w)
    au.register_wavedata_from_complex64vector("circle", circle)
    param_w16384 = AwgParam(num_wait_word=0, num_repeat=1)
    param_w16384.chunks.append(WaveChunk(name_of_wavedata="w", num_blank_word=0, num_repeat=1))
    param_circle = AwgParam(num_wait_word=16, num_repeat=1)
    param_circle.chunks.append(WaveChunk(name_of_wavedata="circle", num_blank_word=0, num_repeat=1))

    cu0 = proxy.capunit(0)

    au.load_parameter(param_w16384)

    # case-00
    data00: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp00: CapParam = CapParam(num_wait_word=0, num_repeat=5)
    cp00.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp00, data00)
    d00 = (data00[0])[0]

    assert d00.shape == (5, 64)
    assert d00[0][0] == w[64]
    assert d00[0][-1] == w[127]
    assert d00[1][0] == w[132]
    assert d00[1][-1] == w[195]
    assert d00[2][0] == w[200]
    assert d00[2][-1] == w[263]
    assert d00[3][0] == w[268]
    assert d00[3][-1] == w[331]
    assert d00[4][0] == w[336]
    assert d00[4][-1] == w[399]

    # case-01
    data01: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp01: CapParam = CapParam(num_wait_word=0, num_repeat=5, integration_enable=True)
    cp01.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp01, data01)
    d01 = (data01[0])[0]

    assert d01.shape == (1, 64)  # because integrated by DSP.
    assert d01[0][0] == sum(d00[:, 0])  # Notes: 1000 == 64 + 132 + 200 + 268 + 336
    assert d01[0][63] == sum(d00[:, 63])  # Notes: 1315 == 1000 + 5*63

    # case-02
    data02: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp02: CapParam = CapParam(num_wait_word=0, num_repeat=5, sum_enable=True)
    cp02.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp02, data02)
    d02 = (data02[0])[0]

    assert d02.shape == (5, 1)
    for i in range(d02.shape[0]):
        assert d02[i, 0] == sum(d00[i, :])

    # case-03
    data03: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp03: CapParam = CapParam(num_wait_word=0, num_repeat=5, sum_enable=True, sum_range=(2, 5))
    cp03.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp03, data03)
    d03 = (data03[0])[0]

    assert d03.shape == (5, 1)
    for i in range(d03.shape[0]):
        assert d03[i, 0] == sum(d00[i, 8:24])  # sum from 2*4 to 5*4+3

    # case-04
    data04: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp04: CapParam = CapParam(num_wait_word=0, num_repeat=5, decimation_enable=True)
    cp04.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp04, data04)
    d04 = (data04[0])[0]

    assert d04.shape == (5, 16)  # 16 == 64 // 4 because of decimation by 4
    for i in range(d04.shape[0]):
        assert all(d04[i] == d00[i][::4])

    # case-05
    data05: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp05: CapParam = CapParam(num_wait_word=0, num_repeat=5, decimation_enable=True, sum_enable=True)
    cp05.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp05, data05)
    d05 = (data05[0])[0]

    assert d05.shape == (5, 1)
    for i in range(d05.shape[0]):
        assert d05[i] == sum(d00[i][::4])

    # case-06
    data06: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp06: CapParam = CapParam(num_wait_word=0, num_repeat=5, decimation_enable=True, sum_enable=True, sum_range=(2, 2))
    cp06.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp06, data06)
    d06 = (data06[0])[0]

    assert d06.shape == (5, 1)
    for i in range(d06.shape[0]):
        assert d06[i] == sum(d04[i][8:12])  # from 2*4 to 2*4+3
        assert d06[i] == sum(d00[i][32:48][::4])  # from 2*16 to 2*16+15 by strides of 4

    # case-07
    data07: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp07: CapParam = CapParam(
        num_wait_word=0,
        num_repeat=5,
        window_enable=True,
    )
    cp07.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp07, data07)
    d07 = (data07[0])[0]

    assert d07.shape == (5, 64)
    for i in range(d07.shape[0]):
        assert d07[i][0] == d00[i][0]
        assert (d07[i][1:] == 0.0).all()

    # case-08
    data08: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp08: CapParam = CapParam(
        num_wait_word=0,
        num_repeat=5,
        window_enable=True,
        window_coeff=np.array([0, 0, 0, 0, 1], dtype=np.complex128),
    )
    cp08.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp08, data08)
    d08 = (data08[0])[0]

    assert d08.shape == (5, 64)
    for i in range(d08.shape[0]):
        assert (d08[i][:4] == 0.0).all()
        assert d08[i][4] == d00[i][4]
        assert (d08[i][5:] == 0.0).all()

    # case-09
    data09: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp09: CapParam = CapParam(
        num_wait_word=0,
        num_repeat=5,
        complexfir_enable=True,
    )
    cp09.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp09, data09)
    d09 = (data09[0])[0]

    assert d09.shape == (5, 64)
    for i in range(d09.shape[0]):
        assert (d09[i] == d00[i]).all()

    # case-10
    data10: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp10: CapParam = CapParam(
        num_wait_word=0,
        num_repeat=5,
        complexfir_enable=True,
        complexfir_coeff=np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex64
        ),
    )
    cp10.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp10, data10)
    d10 = (data10[0])[0]

    assert d10.shape == (5, 64)
    for i in range(d10.shape[0]):
        assert all((d10[i].real + 15.0 == d00[i].real))
        assert all((d10[i].imag - 15.0 == d00[i].imag))

    # case-11
    data11: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp11: CapParam = CapParam(
        num_wait_word=0,
        num_repeat=5,
        realfirs_enable=True,
    )
    cp11.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp11, data11)
    d11 = (data11[0])[0]

    assert d11.shape == (5, 64)
    for i in range(d11.shape[0]):
        assert (d11[i] == d00[i]).all()

    # case-12
    data12: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp12: CapParam = CapParam(
        num_wait_word=0,
        num_repeat=5,
        realfirs_enable=True,
        realfirs_real_coeff=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        realfirs_imag_coeff=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    cp12.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp12, data12)
    d12 = (data12[0])[0]

    assert d12.shape == (5, 64)
    for i in range(d12.shape[0]):
        if i == 0:
            # TODO: fix it!
            assert all((d12[7:i].real + 7.0 == d00[7:i].real))
            assert all((d12[7:i].imag - 7.0 == d00[7:i].imag))
        else:
            assert all((d12[i].real + 7.0 == d00[i].real))
            assert all((d12[i].imag - 7.0 == d00[i].imag))

    # case-13
    data13: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp13: CapParam = CapParam(
        num_wait_word=0,
        num_repeat=5,
        complexfir_enable=True,
        complexfir_coeff=np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex64
        ),
        realfirs_enable=True,
    )
    cp13.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp13, data13)
    d13 = (data13[0])[0]

    assert d13.shape == (5, 64)
    for i in range(d13.shape[0]):
        assert (d13[i].real == d10[i].real).all()
        assert (d13[i].imag == d10[i].imag).all()

    # case-14
    data14: dict[int, list[npt.NDArray[np.complex64]]] = {}
    cp14: CapParam = CapParam(
        num_wait_word=0,
        num_repeat=5,
        complexfir_enable=True,
        realfirs_enable=True,
        realfirs_real_coeff=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        realfirs_imag_coeff=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    cp14.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=1))
    au50loopback(auidx, cmidx, cp14, data14)
    d14 = (data14[0])[0]
    assert d14.shape == (5, 64)
    for i in range(d14.shape[0]):
        assert (d14[i].real == d12[i].real).all()
        assert (d14[i].imag == d12[i].imag).all()
