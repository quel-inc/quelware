import logging

import numpy as np
import numpy.typing as npt
import pytest

from e7awghal.awgunit import AwgUnit
from e7awghal.capctrl import CapCtrlStandard
from e7awghal.capparam import CapParam, CapSection
from e7awghal.common_defs import DECIMATION_RATE, SAMPLING_FREQ
from e7awghal.fwtype import E7FwType
from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal
from e7awghal.wavedata import AwgParam, WaveChunk
from e7awghal_utils.demodulation import table_for_demodulation
from testlibs.awgctrl_with_hlapi import AwgUnitHL
from testlibs.capunit_with_hlapi import CapUnitHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger(__name__)

TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.83",
            "auidx": 2,
            "cmidx": 0,
        },
    },
)

RO_FREQ: float = -150.0e6
RO_DURATION: float = 1.536e-6


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def proxy_au_cm(request):

    proxy = create_quel1au50hal_for_test(
        ipaddr_wss=request.param["box_config"]["ipaddr_wss"], auth_callback=lambda: True
    )
    assert proxy.fw_type() == E7FwType.SIMPLEMULTI_STANDARD

    proxy.initialize()

    auidx = request.param["box_config"]["auidx"]
    cmidx = request.param["box_config"]["cmidx"]

    au: AwgUnit = proxy.awgunit(auidx)
    t = np.linspace(0, RO_DURATION, int(RO_DURATION * SAMPLING_FREQ), endpoint=False, dtype=np.float32)
    w = 4096.0 * np.exp(1j * 2.0 * np.pi * RO_FREQ * t)
    au.register_wavedata_from_complex64vector("w", w.astype(np.complex64))
    param_w = AwgParam(num_wait_word=16)  # Notes: loopback firmware has delay of 64 samples
    param_w.chunks.append(WaveChunk(name_of_wavedata="w"))
    au.load_parameter(param_w)

    yield proxy, auidx, cmidx


def au50loopback(proxy: AbstractQuel1Au50Hal, auidx: int, cmidx: int, cp: CapParam) -> npt.NDArray[np.complex64]:
    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)
    cu = proxy.capunit(0)
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
    return (rdr.as_wave_list())[0][0]


@pytest.mark.parametrize(
    ["phase_offset"],
    [
        (0.0,),
        (np.pi / 4.0,),
        (-3.0 * np.pi / 4.0,),
    ],
)
def test_demodulation(proxy_au_cm, phase_offset):
    proxy, auidx, cmidx = proxy_au_cm

    for decimated_input in [False, True]:  # test both decimated and non-decimated input
        cp = CapParam(
            num_wait_word=0,
            num_repeat=1,
            window_enable=True,
            window_coeff=table_for_demodulation(
                ro_freq=RO_FREQ, ro_duration=RO_DURATION, decimated_input=decimated_input, phase_offset=phase_offset
            ),
        )
        cp.sections.append(
            CapSection(
                name="s0",
                num_capture_word=int(RO_DURATION * SAMPLING_FREQ / 4 / (DECIMATION_RATE if decimated_input else 1)),
            )
        )
        data = au50loopback(proxy, auidx, cmidx, cp)
        assert len(data) == int(RO_DURATION * SAMPLING_FREQ / (DECIMATION_RATE if decimated_input else 1))
        assert abs(np.angle(data[0], deg=False) + phase_offset) < 1e-6
