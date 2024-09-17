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


def au50loopback_reader(auidx: int, cmidx: int, cp: AbstractCapParam) -> CapIqDataReader:
    au = proxy.awgunit(auidx)

    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)
    cc.set_triggering_awgunit_idx(capmod_idx=cmidx, awgunit_idx=au.unit_index)

    cu = proxy.capunit(cc.units_of_module(cmidx)[0])
    assert isinstance(cu, CapUnitHL)
    cu.load_parameter(cp)
    cc.add_triggerable_unit(cu.unit_index)

    fut = cu.wait_for_triggered_capture()

    au.start_now().result()
    au.wait_done().result()

    rdr = fut.result()
    return rdr


def au50loopback_wave(auidx: int, cmidx: int, cp0: AbstractCapParam) -> list[npt.NDArray[np.complex64]]:
    rdr = au50loopback_reader(auidx, cmidx, cp0)
    return rdr.as_wave_list()


def au50loopback_class(auidx: int, cmidx: int, cp0: AbstractCapParam) -> list[npt.NDArray[np.uint8]]:
    rdr = au50loopback_reader(auidx, cmidx, cp0)
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
    vs = np.array([e_main, -e_sub, -e_main, e_sub])
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

    circle = np.exp(1j * np.pi * np.arange(256) / 128)
    circle = np.round(circle * 32767).astype(np.complex64)

    auidx = 2
    cmidx = 0
    au: AwgUnit = proxy.awgunit(auidx)
    au.register_wavedata_from_complex64vector("circle", circle)
    param_circle = AwgParam(num_wait_word=16, num_repeat=1)
    param_circle.chunks.append(WaveChunk(name_of_wavedata="circle", num_blank_word=0, num_repeat=1))

    cu0: CapUnit = proxy.capunit(0)

    au.load_parameter(param_circle)

    # case-20
    cp20: CapParam = CapParam(num_wait_word=0, num_repeat=1)
    cp20.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    d20 = au50loopback_wave(auidx, cmidx, cp20)[0][0]
    assert (d20 == circle).all()

    # case-21
    cp21: CapParam = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True)
    cp21.classification_param.angle_main = 0
    cp21.classification_param.angle_sub = -90
    cp21.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    rdr21 = au50loopback_reader(auidx, cmidx, cp21)
    cls21 = rdr21.as_class_list()[0][0]
    cls21_0 = np.rad2deg(np.angle(d20[cls21 == 0]))
    cls21_1 = np.rad2deg(np.angle(d20[cls21 == 1]))
    cls21_2 = np.rad2deg(np.angle(d20[cls21 == 2]))
    cls21_3 = np.rad2deg(np.angle(d20[cls21 == 3]))
    assert (cls21_0 >= 0.0).all() and (cls21_0 <= 90.0).all()
    assert (cls21_1 >= 90.0).all() and (cls21_1 <= 180.0).all()
    assert (cls21_3 >= -180.0).all() and (cls21_3 <= -90.0).all()
    assert (cls21_2 >= -90.0).all() and (cls21_2 <= 0.0).all()
    assert verify_class("case-21", cls21, circle, 0, -90)
    clsd21 = rdr21.as_class_dict()["s0"][0]
    assert (cls21 == clsd21).all()

    # case-22
    cp22: CapParam = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True)
    cp22.classification_param.angle_main = 23
    cp22.classification_param.angle_sub = -19
    cp22.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    cls22 = au50loopback_class(auidx, cmidx, cp22)[0][0]
    assert verify_class("case-22", cls22, circle, 23, -19)

    # case-xx
    for mdeg in (0, 25, 62, 99, 112, 142, -179, -159, -126, -89, -50, -30):
        for sdeg in (15, 45.1, 75, 105, 134, 165, -165, -134, -105, -75, -46, -15):
            # Notes: mismatch at edge happens at deg is either of 45, 135, -45, or -135.
            cpxx: CapParam = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True)
            cpxx.classification_param.angle_main = mdeg
            cpxx.classification_param.angle_sub = sdeg
            cpxx.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
            clsxx = au50loopback_class(auidx, cmidx, cpxx)[0][0]
            assert verify_class("case-xx:", clsxx, d20, mdeg, sdeg)

    # case-23 (reference)
    cp23: CapParam = CapParam(num_wait_word=0, num_repeat=1)
    cp23.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=16))
    cp23.sections.append(CapSection(name="s1", num_capture_word=16, num_blank_word=16))
    d23 = au50loopback_wave(auidx, cmidx, cp23)
    assert (d23[0][0] == d20[0:64]).all()
    assert (d23[1][0] == d20[128:192]).all()

    # case-24
    cp24: CapParam = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True)
    cp24.classification_param.angle_main = 23
    cp24.classification_param.angle_sub = -19
    cp24.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=16))
    cp24.sections.append(CapSection(name="s1", num_capture_word=16, num_blank_word=16))
    rdr24 = au50loopback_reader(auidx, cmidx, cp24)
    cls24 = rdr24.as_class_list()
    assert (cls24[0][0] == cls22[0:64]).all()
    assert (cls24[1][0] == cls22[128:192]).all()
    clsd24 = rdr24.as_class_dict()
    assert (cls24[0][0] == clsd24["s0"][0]).all()
    assert (cls24[1][0] == clsd24["s1"][0]).all()

    # case-25
    cp25: CapParam = CapParam(num_wait_word=0, num_repeat=1)
    cp25.sections.append(CapSection(name="s0", num_capture_word=9, num_blank_word=2))
    cp25.sections.append(CapSection(name="s1", num_capture_word=14, num_blank_word=39))
    d25 = au50loopback_wave(auidx, cmidx, cp25)
    assert (d25[0][0] == d20[0:36]).all()
    assert (d25[1][0] == d20[44:100]).all()

    # case-26
    cp26: CapParam = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True)
    cp26.classification_param.angle_main = 23
    cp26.classification_param.angle_sub = -19
    cp26.sections.append(CapSection(name="s0", num_capture_word=9, num_blank_word=2))
    cp26.sections.append(CapSection(name="s1", num_capture_word=14, num_blank_word=39))
    rdr26 = au50loopback_reader(auidx, cmidx, cp26)
    cls26 = rdr26.as_class_list()
    assert (cls26[0][0] == cls22[0:36]).all()
    assert (cls26[1][0] == cls22[44:100]).all()
    clsd26 = rdr26.as_class_dict()
    assert (cls26[0][0] == clsd26["s0"][0]).all()
    assert (cls26[1][0] == clsd26["s1"][0]).all()

    # case-27 (reference)
    cp27: CapParam = CapParam(num_wait_word=0, num_repeat=2)
    cp27.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=16))
    d27 = au50loopback_wave(auidx, cmidx, cp27)
    assert (d27[0][0] == d20[0:64]).all()
    assert (d27[0][1] == d20[128:192]).all()

    # case-28
    cp28: CapParam = CapParam(num_wait_word=0, num_repeat=2, classification_enable=True)
    cp28.classification_param.angle_main = 23
    cp28.classification_param.angle_sub = -19
    cp28.sections.append(CapSection(name="s0", num_capture_word=16, num_blank_word=16))
    rdr28 = au50loopback_reader(auidx, cmidx, cp28)
    cls28 = rdr28.as_class_list()
    assert (cls28[0][0] == cls22[0:64]).all()
    assert (cls28[0][1] == cls22[128:192]).all()
    clsd28 = rdr28.as_class_dict()
    assert (cls28[0][0] == clsd28["s0"][0]).all()
    assert (cls28[0][1] == clsd28["s0"][1]).all()

    # case-29 (reference)
    cp29: CapParam = CapParam(num_wait_word=0, num_repeat=2)
    cp29.sections.append(CapSection(name="s0", num_capture_word=17, num_blank_word=15))
    d29 = au50loopback_wave(auidx, cmidx, cp29)
    assert (d29[0][0] == d20[0:68]).all()
    assert (d29[0][1] == d20[128:196]).all()

    # case-30
    cp30: CapParam = CapParam(num_wait_word=0, num_repeat=2, classification_enable=True)
    cp30.classification_param.angle_main = 23
    cp30.classification_param.angle_sub = -19
    cp30.sections.append(CapSection(name="s0", num_capture_word=17, num_blank_word=15))
    rdr30 = au50loopback_reader(auidx, cmidx, cp30)
    cls30 = rdr30.as_class_list()
    assert (cls30[0][0] == cls22[0:68]).all()
    assert (cls30[0][1] == cls22[128:196]).all()
    clsd30 = rdr30.as_class_dict()
    assert (cls30[0][0] == clsd30["s0"][0]).all()
    assert (cls30[0][1] == clsd30["s0"][1]).all()

    # case-31 (reference)
    scircle = np.exp(1j * np.pi * np.arange(256) / 128)
    scircle31 = np.round(scircle * 16383 + 6789 - 9876j).astype(np.complex64)
    au.register_wavedata_from_complex64vector("scircle31", scircle31)
    param_scircle31 = AwgParam(num_wait_word=16, num_repeat=1)
    param_scircle31.chunks.append(WaveChunk(name_of_wavedata="scircle31", num_blank_word=0, num_repeat=1))
    au.load_parameter(param_scircle31)

    cp31: CapParam = CapParam(num_wait_word=0, num_repeat=1)
    cp31.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    d31 = au50loopback_wave(auidx, cmidx, cp31)[0][0]
    assert (d31 == scircle31).all()

    # case-32
    cp32: CapParam = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True)
    cp32.classification_param.angle_main = 0
    cp32.classification_param.angle_sub = -90
    cp32.classification_param.pivot_x = 6789
    cp32.classification_param.pivot_y = -9876
    cp32.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    cls32 = au50loopback_class(auidx, cmidx, cp32)[0][0]
    assert verify_class("case-32", cls32, d31, 0, -90, 6789, -9876)

    # case-yy
    for mdeg in (9, 132, -159, -25):
        for sdeg in (1, 93, -160, -46):
            for px in (-12000.1, -2609, 4305, 13033):
                for py in (-20013.2, -8222.2, 613, 15201):
                    cpyy: CapParam = CapParam(num_wait_word=0, num_repeat=1, classification_enable=True)
                    cpyy.classification_param.angle_main = mdeg
                    cpyy.classification_param.angle_sub = sdeg
                    cpyy.classification_param.pivot_x = px
                    cpyy.classification_param.pivot_y = py
                    cpyy.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
                    clsyy = au50loopback_class(auidx, cmidx, cpyy)[0][0]
                    assert verify_class("case-yy", clsyy, d31, mdeg, sdeg, px, py)

    # case-33
    cp33: CapParam = CapParam(num_wait_word=0, num_repeat=1, complexfir_enable=True, classification_enable=True)
    cp33.classification_param.angle_main = 0
    cp33.classification_param.angle_sub = -90
    cp33.classification_param.pivot_x = 6789
    cp33.classification_param.pivot_y = -9876
    cp33.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    cls33 = au50loopback_class(auidx, cmidx, cp33)[0][0]
    assert verify_class("case-33", cls33, d31, 0, -90, 6789, -9876)

    # case-34
    cp34: CapParam = CapParam(num_wait_word=0, num_repeat=1, realfirs_enable=True, classification_enable=True)
    cp34.classification_param.angle_main = 0
    cp34.classification_param.angle_sub = -90
    cp34.classification_param.pivot_x = 6789
    cp34.classification_param.pivot_y = -9876
    cp34.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    cls34 = au50loopback_class(auidx, cmidx, cp34)[0][0]
    assert verify_class("case-34", cls34, d31, 0, -90, 6789, -9876)

    # case-35
    cp35: CapParam = CapParam(
        num_wait_word=0, num_repeat=1, complexfir_enable=True, realfirs_enable=True, classification_enable=True
    )
    cp35.classification_param.angle_main = 0
    cp35.classification_param.angle_sub = -90
    cp35.classification_param.pivot_x = 6789
    cp35.classification_param.pivot_y = -9876
    cp35.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    cls35 = au50loopback_class(auidx, cmidx, cp35)[0][0]
    assert verify_class("case-35", cls35, d31, 0, -90, 6789, -9876)

    # case-36 (reference)
    cp36: CapParam = CapParam(num_wait_word=0, num_repeat=1, decimation_enable=True)
    cp36.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    d36 = au50loopback_wave(auidx, cmidx, cp36)[0][0]
    assert (d36 == scircle31[::4]).all()

    # case-37
    cp37: CapParam = CapParam(num_wait_word=0, num_repeat=1, decimation_enable=True, classification_enable=True)
    cp37.classification_param.angle_main = 0
    cp37.classification_param.angle_sub = -90
    cp37.classification_param.pivot_x = 6789
    cp37.classification_param.pivot_y = -9876
    cp37.sections.append(CapSection(name="s0", num_capture_word=64, num_blank_word=1))
    cls37 = au50loopback_class(auidx, cmidx, cp37)[0][0]
    assert verify_class("case-37", cls37, d36, 0, -90, 6789, -9876)
