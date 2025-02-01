import logging

import numpy as np
import numpy.typing as npt
import pytest

from e7awghal.awgunit import AwgUnit
from e7awghal.capctrl import CapCtrlStandard
from e7awghal.capparam import CapParam, CapSection
from e7awghal.fwtype import E7FwType
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
def proxy_au_cm_data(request) -> tuple[AbstractQuel1Au50Hal, int, int, dict[str, npt.NDArray[np.complex64]]]:

    proxy = create_quel1au50hal_for_test(
        ipaddr_wss=request.param["box_config"]["ipaddr_wss"], auth_callback=lambda: True
    )
    proxy.initialize()
    assert proxy.fw_type() == E7FwType.SIMPLEMULTI_STANDARD

    au: AwgUnit = proxy.awgunit(request.param["box_config"]["auidx"])
    cmidx = request.param["box_config"]["cmidx"]

    circle = np.exp(1j * np.pi * np.arange(256) / 128)
    circle = np.round(circle * 32767).astype(np.complex64)
    au.register_wavedata_from_complex64vector("circle", circle)

    shifted_circle = np.exp(1j * np.pi * np.arange(256) / 128)
    shifted_circle = np.round(shifted_circle * 16383).astype(np.complex64)
    shifted_circle += 6319 - 8211j
    au.register_wavedata_from_complex64vector("shifted_circle", shifted_circle)
    return proxy, au.unit_index, cmidx, {"circle": circle, "shifted_circle": shifted_circle}


def au50loopback(
    proxy: AbstractQuel1Au50Hal, auidx: int, dataname: str, cmidx: int, cp: CapParam
) -> list[npt.NDArray[np.uint8]]:
    au = proxy.awgunit(auidx)
    assert isinstance(au, AwgUnitHL)
    param = AwgParam(num_wait_word=16, num_repeat=1)  # Notes: loopback firmware has delay of 64 samples
    param.chunks.append(WaveChunk(name_of_wavedata=dataname, num_blank_word=0, num_repeat=1))
    au.load_parameter(param)

    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)

    cu = proxy.capunit(cc.units_of_module(cmidx)[0])
    assert isinstance(cu, CapUnitHL)
    cc.set_triggering_awgunit_idx(capmod_idx=cmidx, awgunit_idx=auidx)
    cc.add_triggerable_unit(cu.unit_index)

    cu.load_parameter(cp)
    fut = cu.wait_for_triggered_capture()

    au.start_now().result()
    au.wait_done().result()

    rdr = fut.result()
    return rdr.as_class_list()


def _calc_class_single(u: npt.NDArray[np.float64], vs: npt.NDArray[np.float64]):
    dets = u[0] * vs[:, 1] - u[1] * vs[:, 0]
    n = len(dets)
    for i in range(n):
        if dets[i] <= 0 and dets[(i + 1) % n] >= 0:
            return (0, 1, 3, 2)[i]
    else:
        raise AssertionError(f"something wrong: {dets}, {dets <= 0}")


def calc_class(us: npt.NDArray[np.complex64], deg_main: float, deg_sub: float):
    e_main = np.array((np.cos(np.deg2rad(deg_main)), np.sin(np.deg2rad(deg_main))))
    e_sub = np.array((np.cos(np.deg2rad(deg_sub)), np.sin(np.deg2rad(deg_sub))))
    #              2e=0s   0e=1s   1e=3s    3e=2s
    vs = np.array([e_main, e_sub, -e_main, -e_sub])
    return np.array([_calc_class_single(np.array([u.real, u.imag]), vs) for u in us], dtype=np.uint8)


def verify_class(
    testname: str,
    target: npt.NDArray[np.uint8],
    us: npt.NDArray[np.complex64],
    deg_main: float,
    deg_sub: float,
    pivot_x: float = 0.0,
    pivot_y: float = 0.0,
):
    us1 = us - (pivot_x + pivot_y * 1j)  # TODO: take scale into account
    answer = calc_class(us1, deg_main, deg_sub)
    m = target == answer
    n = len(m)
    for i in range(n):
        if not m[i]:
            if not (
                answer[(i - 1) % n] == target[(i - 1) % n]
                and answer[(i + 1) % n] == target[(i + 1) % n]
                and (target[i] == answer[(i - 1) % n] or target[i] == answer[(i + 1) % n])
            ):
                return False
            else:
                logger.info(
                    f"mismatch at the edge: {i}-th sample @ deg_main={deg_main}, deg_sub={deg_sub} ({testname})"
                )
    return True


@pytest.mark.parametrize(
    ["dataname", "angle0", "angle1", "shift_i", "shift_q"],
    [
        ("circle", 0, 90, 0, 0),
        ("circle", 23, 161, 0, 0),
        ("circle", 23, -133, 0, 0),
        ("shifted_circle", 0, 90, 6319, -8211),
        ("shifted_circle", -50, 85, 6319, -8211),
        ("shifted_circle", 0, 90, 0, 0),
        ("shifted_circle", -120, -15, 0, 0),
    ],
)
def test_classification(proxy_au_cm_data, dataname: str, angle0: float, angle1: float, shift_i: float, shift_q: float):
    proxy, auidx, cmidx, data = proxy_au_cm_data

    cp00 = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True)
    cp00.classification_param.angle_main = angle0
    cp00.classification_param.angle_sub = angle1
    cp00.classification_param.pivot_x = shift_i
    cp00.classification_param.pivot_y = shift_q
    cp00.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=16))
    cls00 = au50loopback(proxy, auidx, dataname, cmidx, cp00)[0][0]
    assert verify_class("test_classification", cls00, data[dataname], angle0, angle1, shift_i, shift_q)


@pytest.mark.parametrize(
    ["dataname", "angle0", "angle1", "shift_i", "shift_q"],
    [
        ("circle", 23, 161, 0, 0),
        ("shifted_circle", 0, 90, 0, 0),
        ("shifted_circle", 23, -133, 0, 0),
        ("shifted_circle", -120, -15, 0, 0),
    ],
)
def test_classification_with_cfir(
    proxy_au_cm_data, dataname: str, angle0: float, angle1: float, shift_i: float, shift_q: float
):
    proxy, auidx, cmidx, data = proxy_au_cm_data

    cp00 = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True, complexfir_enable=True)
    cp00.classification_param.angle_main = angle0
    cp00.classification_param.angle_sub = angle1
    cp00.classification_param.pivot_x = shift_i
    cp00.classification_param.pivot_y = shift_q
    cp00.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=16))
    cls00 = au50loopback(proxy, auidx, dataname, cmidx, cp00)[0][0]
    assert verify_class("test_classification", cls00, data[dataname], angle0, angle1, shift_i, shift_q)


@pytest.mark.parametrize(
    ["dataname", "angle0", "angle1", "shift_i", "shift_q"],
    [
        ("circle", 23, 161, 0, 0),
        ("shifted_circle", 0, 90, 0, 0),
        ("shifted_circle", 23, -133, 0, 0),
        ("shifted_circle", -120, -15, 0, 0),
    ],
)
def test_classification_with_rfirs(
    proxy_au_cm_data, dataname: str, angle0: float, angle1: float, shift_i: float, shift_q: float
):
    proxy, auidx, cmidx, data = proxy_au_cm_data

    cp00 = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True, realfirs_enable=True)
    cp00.classification_param.angle_main = angle0
    cp00.classification_param.angle_sub = angle1
    cp00.classification_param.pivot_x = shift_i
    cp00.classification_param.pivot_y = shift_q
    cp00.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=16))
    cls00 = au50loopback(proxy, auidx, dataname, cmidx, cp00)[0][0]
    assert verify_class("test_classification", cls00, data[dataname], angle0, angle1, shift_i, shift_q)
