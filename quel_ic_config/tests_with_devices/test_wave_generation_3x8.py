import logging
import os
import shutil
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak

from testlibs.basic_scan_common import init_box, init_e4405b, measure_floor_noise

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = {
    "ipaddr_wss": "10.1.0.42",
    "ipaddr_sss": "10.2.0.42",
    "ipaddr_css": "10.5.0.42",
    "boxtype": "quel1-a",
    "config_root": "settings",
    "mxfe": "both",
    "port_availability": {
        "unavailable": [(0, 1), (1, 1), (1, 2)],
        "via_monitor_out": [],
    },
}


OUTPUT_SETTINGS = {
    "spectrum_image_path": "./artifacts/spectrum",
}

MAX_BACKGROUND_NOISE = -62.0  # dBm


@pytest.fixture(scope="session")
def e4405b():
    e4405b = init_e4405b()
    return e4405b


@pytest.fixture(scope="session", params=(DEVICE_SETTINGS,))
def css_p2(request, e4405b):
    param0 = request.param

    linkup0, linkup1, css_p2_g0, css_p2_g1, _, _ = init_box(
        ipaddr_wss=param0["ipaddr_wss"],
        ipaddr_sss=param0["ipaddr_sss"],
        ipaddr_css=param0["ipaddr_css"],
        mxfe=param0["mxfe"],
        boxtype=param0["boxtype"],
        config_root=param0["config_root"],
    )
    assert linkup0
    assert linkup1

    max_noise = measure_floor_noise(e4405b)
    assert max_noise < MAX_BACKGROUND_NOISE
    yield css_p2_g0, css_p2_g1

    css_p2_g0.stop()
    css_p2_g1.stop()


@pytest.fixture(scope="session", params=(DEVICE_SETTINGS,))
def port_availability(request):
    # Notes: configration of CI devices will be managed by using DB or whatever in near future.
    return request.param["port_availability"]


@pytest.fixture(scope="session", params=(OUTPUT_SETTINGS,))
def outdir(request):
    mpl.use("Qt5Agg")  # TODO: reconsider where to execute.

    dirname = request.param["spectrum_image_path"]
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    dpath = Path(dirname)
    os.makedirs(dpath)
    os.makedirs(dpath / "awg")
    os.makedirs(dpath / "vatt")
    os.makedirs(dpath / "sideband")
    return dpath


@pytest.mark.parametrize(
    ("mxfe", "line", "awg_idx", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 0, 12000, 1000, 0),
        (0, 0, 0, 12000, 1500, 0),
        (0, 0, 0, 12000, 1500, 500),
        (0, 0, 0, 12000, 1500, 600),
        (0, 0, 0, 12000, 2000, -600),  # 12000 - (1500 - 600) > 11000 (!)
        (0, 1, 0, 12000, 1000, 100),
        (0, 1, 0, 12000, 1500, 100),
        (0, 2, 0, 11000, 2000, 25),
        (0, 2, 0, 11000, 2000, 525),
        (0, 2, 1, 11000, 2000, 50),
        (0, 2, 1, 11000, 2000, 550),
        (0, 2, 2, 11000, 2000, 75),
        (0, 2, 2, 11000, 2000, 575),
        (0, 3, 0, 10500, 2000, 25),
        (0, 3, 0, 10500, 2000, 525),
        (0, 3, 1, 10500, 2000, 50),
        (0, 3, 1, 10500, 2000, 550),
        (0, 3, 2, 10500, 2000, 75),
        (0, 3, 2, 10500, 2000, 575),
        (1, 0, 0, 12000, 1250, 0),
        (1, 0, 0, 12000, 1750, 0),
        (1, 0, 0, 12000, 1750, 500),
        (1, 0, 0, 12000, 1750, 600),
        (1, 0, 0, 12000, 1750, -600),
        (1, 1, 0, 12000, 1250, 100),
        (1, 1, 0, 12000, 1750, 100),
        (1, 2, 0, 11000, 2000, -25),
        (1, 2, 0, 11000, 2000, -525),
        (1, 2, 1, 11000, 2000, -50),
        (1, 2, 1, 11000, 2000, -550),
        (1, 2, 2, 11000, 2000, -75),
        (1, 2, 2, 11000, 2000, -575),
        (1, 3, 0, 10500, 2000, -25),
        (1, 3, 0, 10500, 2000, -525),
        (1, 3, 1, 10500, 2000, -50),
        (1, 3, 1, 10500, 2000, -550),
        (1, 3, 2, 10500, 2000, -75),
        (1, 3, 2, 10500, 2000, -575),
    ],
)
def test_all_single_awgs(
    mxfe: int,
    line: int,
    awg_idx: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    css_p2,
    port_availability,
    e4405b,
    outdir,
):
    mxfe_g = css_p2[mxfe]
    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    mxfe_g.run(line, awg_idx, lo_mhz=lo_mhz, cnco_mhz=cnco_mhz, fnco_mhz=fnco_mhz)
    if via_monitor:
        mxfe_g.open_monitor_out()
    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(e4405b, -50.0)
    mxfe_g.stop()
    if via_monitor:
        mxfe_g.close_monitor_out()

    expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    e0 = ExpectedSpectrumPeaks([(expected_freq, -20)])
    e0.validate_with_measurement_condition(e4405b.max_freq_error_get())
    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / "awg" / f"mxfe{mxfe:d}-line{line:d}-ch{awg_idx:d}-{int(expected_freq)//1000000:d}MHz.png")

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)


@pytest.mark.parametrize(
    ("mxfe", "line", "awg_idx", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 0, 12000, 1500, 0),
        (0, 1, 0, 12000, 1500, 100),
        (0, 2, 0, 11000, 2000, 525),
        (0, 3, 0, 10500, 2000, 525),
        (1, 0, 0, 12000, 1500, 0),
        (1, 1, 0, 12000, 1500, 100),
        (1, 2, 0, 11000, 2000, 525),
        (1, 3, 0, 10500, 2000, 525),
    ],
)
def test_vatt(
    mxfe: int,
    line: int,
    awg_idx: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    css_p2,
    port_availability,
    e4405b,
    outdir,
):
    mxfe_g = css_p2[mxfe]
    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    pwr: Dict[int, float] = {}
    for vatt in (0x380, 0x500, 0x680, 0x800, 0x980, 0xB00):
        """
        V_ref of AD5328 == 3.3V
        The expected output voltages are 0.72V, 1.03V, 1.34V, 1.65V, 1.96V, and 2.27V, respectively.
        Their corresponding gains at 10GHz are approximately -10dB, -7dB, -2dB, 3dB, 7dB, and 12dB, respectively.
        """
        mxfe_g.run(line, awg_idx, lo_mhz=lo_mhz, cnco_mhz=cnco_mhz, fnco_mhz=fnco_mhz, vatt=vatt)
        if via_monitor:
            mxfe_g.open_monitor_out()
        # notes: -60.0dBm fails due to spurious below 7GHz.
        m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(e4405b, -50.0)
        mxfe_g.stop()
        if via_monitor:
            mxfe_g.close_monitor_out()

        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
        e0 = ExpectedSpectrumPeaks([(expected_freq, -40)])
        e0.validate_with_measurement_condition(e4405b.max_freq_error_get())
        d0 = e0.extract_matched(m0)
        assert len(d0) == 1

        d00 = d0.pop()
        pwr[vatt] = d00.power

        plt.cla()
        plt.plot(t0[:, 0], t0[:, 1])
        plt.savefig(outdir / "vatt" / f"mxfe{mxfe:d}-line{line:d}-ch{awg_idx:d}-vatt{vatt:04x}.png")

    logger.info(f"vatt vs power@{(mxfe, line)}: {pwr}")
    pwrl = list(pwr.values())
    for i in range(1, len(pwrl)):
        if i == 1:
            assert 2.25 <= pwrl[i] - pwrl[i - 1] <= 4.0
        else:
            assert 3.0 <= pwrl[i] - pwrl[i - 1] <= 5.0


@pytest.mark.parametrize(
    ("mxfe", "line", "awg_idx", "lo_mhz", "cnco_mhz", "fnco_mhz", "sideband"),
    [
        (0, 0, 0, 9000, 2000, 0, "U"),
        (0, 0, 0, 12000, 2000, 0, "L"),
        (0, 1, 0, 9000, 1900, 0, "U"),
        (0, 1, 0, 12000, 1900, 0, "L"),
        (0, 2, 0, 8000, 1500, 0, "U"),
        (0, 2, 0, 12000, 2500, 0, "L"),
        (0, 3, 0, 8000, 2000, 0, "U"),
        (0, 3, 0, 12000, 3000, 0, "L"),
        (1, 0, 0, 9000, 1900, 0, "U"),
        (1, 0, 0, 12000, 1900, 0, "L"),
        (1, 1, 0, 9000, 1800, 0, "U"),
        (1, 1, 0, 12000, 1800, 0, "L"),
        (1, 2, 0, 8000, 1600, 0, "U"),
        (1, 2, 0, 12000, 2600, 0, "L"),
        (1, 3, 0, 8000, 1900, 0, "U"),
        (1, 3, 0, 12000, 2900, 0, "L"),
    ],
)
def test_sideband(
    mxfe: int,
    line: int,
    awg_idx: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    sideband: str,
    css_p2,
    port_availability,
    e4405b,
    outdir,
):
    mxfe_g = css_p2[mxfe]
    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    mxfe_g.run(line, awg_idx, lo_mhz=lo_mhz, cnco_mhz=cnco_mhz, fnco_mhz=fnco_mhz, sideband=sideband)
    if via_monitor:
        mxfe_g.open_monitor_out()
    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(e4405b, -50.0)
    mxfe_g.stop()
    if via_monitor:
        mxfe_g.close_monitor_out()

    if sideband == "L":
        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
    elif sideband == "U":
        expected_freq = (lo_mhz + (cnco_mhz + fnco_mhz)) * 1e6
    else:
        raise AssertionError

    e0 = ExpectedSpectrumPeaks([(expected_freq, -20)])
    e0.validate_with_measurement_condition(e4405b.max_freq_error_get())
    j0, s0, w0 = e0.match(m0)

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / "sideband" / f"mxfe{mxfe:d}-line{line:d}-ch{awg_idx:d}-sideband{sideband}.png")
