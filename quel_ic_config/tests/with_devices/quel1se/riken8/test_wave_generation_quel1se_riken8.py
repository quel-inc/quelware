import logging
import os
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from quel_ic_config.quel1_box_intrinsic import Quel1BoxIntrinsic, Quel1BoxType
from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak
from testlibs.gen_cw import boxi_gen_cw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def make_outdir(dirpath: Path):
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    os.makedirs(dirpath, exist_ok=True)
    return dirpath


# Notes: 5.8 -- 8.0GHz
@pytest.mark.parametrize(
    ("idx", "mxfe", "line", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 0, 0, 8500, 2700, 0),
        (1, 0, 0, 0, 9500, 1500, 0),
        (2, 0, 0, 0, 8500, 2700, -200),
        (3, 0, 0, 0, 9500, 1500, 200),
        (4, 0, 2, 0, 9000, 3200, 0),
        (5, 0, 2, 0, 10000, 2000, 0),
        (6, 0, 2, 0, 9000, 3000, 200),
        (7, 0, 2, 0, 10000, 2200, -200),
    ],
)
def test_line_mixer(
    idx: int,
    mxfe: int,
    line: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures8,
    fixture_ms2720t1,
):
    box, params, topdirpath = fixtures8
    if params["label"] not in {"staging-094"}:
        pytest.skip()

    boxi: Quel1BoxIntrinsic = box._dev
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "line_with_mixer")

    spa = fixture_ms2720t1

    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    task = boxi_gen_cw(
        boxi,
        mxfe,
        line,
        channel,
        fnco_freq=fnco_mhz * 1e6,
        cnco_freq=cnco_mhz * 1e6,
        fullscale_current=40527,
        lo_freq=lo_mhz * 1e6,
        sideband="L",
        vatt=0xA00,
        via_monitor=via_monitor,
    )

    expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    max_sprious_peek = -50.0
    if line == 1 and boxi.css.boxtype in {
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
    }:
        expected_freq *= 2
        max_sprious_peek = -42.0
    e0 = ExpectedSpectrumPeaks([(expected_freq, -20 - relative_loss)])
    e0.validate_with_measurement_condition(spa.max_freq_error_get())

    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(spa, max_sprious_peek)
    # notes: stop all the awgs of the line
    task.cancel()

    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / f"{idx:02d}_mxfe{mxfe:d}-line{line:d}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png")

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)


# Notes: 2 -- 5.8GHz
@pytest.mark.parametrize(
    ("idx", "mxfe", "line", "channel", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 1, 0, 3000, 0),
        (1, 0, 3, 0, 2000, 0),
        (2, 0, 3, 1, 5800, 0),
        (3, 0, 3, 2, 5800, -600),
        (4, 1, 0, 0, 2100, 0),
        (5, 1, 1, 0, 5000, 700),
        (6, 1, 1, 1, 5650, 0),
        (7, 1, 1, 2, 2200, 0),
        (8, 1, 2, 0, 2300, 0),
        (9, 1, 2, 1, 5600, 0),
        (10, 1, 2, 2, 4900, 800),
        (11, 1, 3, 0, 2400, 0),
    ],
)
def test_line_without_mixer(
    idx: int,
    mxfe: int,
    line: int,
    channel: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures8,
    fixture_ms2720t1,
):

    box, params, topdirpath = fixtures8
    if params["label"] not in {"staging-094"}:
        pytest.skip()

    boxi: Quel1BoxIntrinsic = box._dev
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "line_without_mixer")

    spa = fixture_ms2720t1

    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    task = boxi_gen_cw(
        boxi,
        mxfe,
        line,
        channel,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=15000,
        via_monitor=via_monitor,
    )

    expected_freq = (cnco_mhz + fnco_mhz) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    max_sprious_peek = -50.0
    if line == 1 and boxi.css.boxtype in {
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
    }:
        expected_freq *= 2
        max_sprious_peek = -42.0

    # allowing harmonics for 2-5.8GHz port
    e0 = ExpectedSpectrumPeaks(
        [
            (expected_freq * i, -20 - relative_loss)
            for i in range(1, 4)
            if expected_freq * i < spa.freq_center + spa.freq_span / 2
        ]
    )
    e0.validate_with_measurement_condition(spa.max_freq_error_get())

    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(spa, max_sprious_peek)
    logger.info(f"m0 = {m0}")
    # notes: stop all the awgs of the line
    task.cancel()

    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / f"{idx:02d}_mxfe{mxfe:d}-line{line:d}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png")

    logger.info(f"w0 = {w0}")
    logger.info(f"j0 = {j0}")
    assert len(s0) == 0
    assert j0[0]


@pytest.mark.parametrize(
    ("idx", "mxfe", "line", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 0, 0, 9000, 2000, 0),
        (1, 0, 2, 0, 9000, 2100, 100),
    ],
)
def test_vatt(
    idx: int,
    mxfe: int,
    line: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures8,
    fixture_ms2720t1,
):
    box, params, topdirpath = fixtures8
    if params["label"] not in {"staging-094"}:
        pytest.skip()

    boxi: Quel1BoxIntrinsic = box._dev
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "vatt")

    spa = fixture_ms2720t1

    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    pwr: Dict[int, float] = {}
    is_pump = line == 1 and boxi.css.boxtype in {
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
    }
    if is_pump:
        vatt_list = (0x780, 0x800, 0x880, 0x900, 0x980, 0xA00)
    else:
        vatt_list = (0x380, 0x500, 0x680, 0x800, 0x980, 0xB00)

    for vatt in vatt_list:
        """
        V_ref of AD5328 == 3.3V
        The expected output voltages are 0.72V, 1.03V, 1.34V, 1.65V, 1.96V, and 2.27V, respectively.
        Their corresponding gains at 10GHz are approximately -10dB, -7dB, -2dB, 3dB, 7dB, and 12dB, respectively.
        """
        task = boxi_gen_cw(
            boxi,
            mxfe,
            line,
            channel,
            lo_freq=lo_mhz * 1e6,
            cnco_freq=cnco_mhz * 1e6,
            fnco_freq=fnco_mhz * 1e6,
            fullscale_current=40527,
            sideband="L",
            vatt=vatt,
            via_monitor=via_monitor,
        )

        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
        if is_pump:
            expected_freq *= 2
            max_sprious_peek = -42.0
        else:
            max_sprious_peek = -50.0
        e0 = ExpectedSpectrumPeaks([(expected_freq, -40 - relative_loss)])
        e0.validate_with_measurement_condition(spa.max_freq_error_get())

        # notes: -60.0dBm fails due to spurious below 7GHz.
        m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(spa, max_sprious_peek)
        task.cancel()

        plt.cla()
        plt.plot(t0[:, 0], t0[:, 1])
        plt.savefig(outdir / f"{idx:02d}-mxfe{mxfe:d}-line{line:d}-ch{channel:d}-vatt{vatt:04x}.png")

        d0 = e0.extract_matched(m0)
        assert len(d0) == 1

        d00 = d0.pop()
        pwr[vatt] = d00.power

    logger.info(f"vatt vs power@{(mxfe, line)}: {pwr}")
    pwrl = list(pwr.values())
    for i in range(1, len(pwrl)):
        if is_pump:
            assert 2.0 <= pwrl[i] - pwrl[i - 1] <= 6.5
        else:
            assert 2.2 <= pwrl[i] - pwrl[i - 1] <= 5.0


# TODO: enable it after implementing the control of output divider of ADRF6780
@pytest.mark.parametrize(
    ("idx", "mxfe", "line", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz", "sideband"),
    [
        (0, 0, 0, 0, 5000, 2500, 0, "U"),
        (1, 0, 0, 0, 10000, 2000, 0, "L"),
        (2, 0, 2, 0, 4900, 2200, 0, "U"),
        (3, 0, 2, 0, 10000, 2500, 0, "L"),
    ],
)
def test_sideband(
    idx: int,
    mxfe: int,
    line: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    sideband: str,
    fixtures8,
    fixture_ms2720t1,
):
    box, params, topdirpath = fixtures8
    if params["label"] not in {"staging-094"}:
        pytest.skip()

    boxi: Quel1BoxIntrinsic = box._dev
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "sideband")

    spa = fixture_ms2720t1

    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    task = boxi_gen_cw(
        boxi,
        mxfe,
        line,
        channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband=sideband,
        via_monitor=via_monitor,
    )

    if sideband == "L":
        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
    elif sideband == "U":
        expected_freq = (lo_mhz + (cnco_mhz + fnco_mhz)) * 1e6
    else:
        raise AssertionError
    if line == 1 and boxi.css.boxtype in {
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
    }:
        expected_freq *= 2
        max_sprious_peek = -42.0
    else:
        max_sprious_peek = -50.0
    e0 = ExpectedSpectrumPeaks([(expected_freq, -20 - relative_loss)])
    e0.validate_with_measurement_condition(spa.max_freq_error_get())

    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(spa, max_sprious_peek)
    task.cancel()

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / f"{idx:02d}-mxfe{mxfe:d}-line{line:d}-ch{channel:d}-sideband{sideband}.png")

    j0, s0, w0 = e0.match(m0)

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)
