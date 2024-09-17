import logging
import os
from pathlib import Path
from typing import Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from quel_ic_config.quel1_box import Quel1BoxType
from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak
from testlibs.gen_cw import box_gen_cw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def make_outdir(dirpath: Path):
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    os.makedirs(dirpath, exist_ok=True)
    return dirpath


def decode_port_subport(port_subport: Union[int, Tuple[int, int]]) -> Tuple[int, int, str]:
    if isinstance(port_subport, int):
        port: int = port_subport
        subport: int = 0
        portname: str = f"{port:02d}"
    else:
        port, subport = port_subport
        portname = f"{port:02d}-{subport:02d}"

    return port, subport, portname


def has_doubler(boxtype: Quel1BoxType, port_subport: Union[int, Tuple[int, int]]) -> bool:
    if boxtype in {
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
    }:
        return port_subport in {2, 11}
    elif boxtype == Quel1BoxType.QuEL1_TypeA:
        return port_subport in {3, 10}
    else:
        return False


# Notes: 5.8 -- 8.0GHz
@pytest.mark.parametrize(
    ("idx", "port_subport", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 1, 0, 8500, 2700, 0),
        (1, 1, 0, 9500, 1500, 0),
        (2, 1, 0, 8500, 2700, -200),
        (3, 1, 0, 9500, 1500, 200),
        (4, 2, 0, 9000, 3200, 0),
        (5, 2, 0, 10000, 2000, 0),
        (6, 2, 0, 9000, 3000, 200),
        (7, 2, 0, 10000, 2200, -200),
    ],
)
def test_port_with_mixer(
    idx: int,
    port_subport: Union[int, Tuple[int, int]],
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures8,
    fixture_ms2720t1,
):
    box, params, topdirpath = fixtures8
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "port_with_mixer")

    spa = fixture_ms2720t1

    port, subport, portname = decode_port_subport(port_subport)

    via_monitor = False
    if port_subport in port_availability["unavailable"]:
        pytest.skip(f"{portname} is unavailable.")
    elif port in port_availability["via_monitor_out"]:
        via_monitor = True

    task = box_gen_cw(
        box,
        port_subport,
        channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband="L",
        via_monitor=via_monitor,
    )

    expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    max_sprious_peek = -50.0
    if has_doubler(box.css.boxtype, port_subport):
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
    plt.savefig(outdir / f"{idx:02d}_{portname}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png")

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)


# Notes: 2 -- 5.8GHz
@pytest.mark.parametrize(
    ("idx", "port_subport", "channel", "cnco_mhz", "fnco_mhz"),
    [
        (0, (1, 1), 0, 3000, 0),
        (1, 3, 0, 2000, 0),
        (2, 3, 1, 5800, 0),
        (3, 3, 2, 5800, -600),
        (4, 6, 0, 2100, 0),
        (5, 6, 1, 5700, 0),
        (6, 6, 2, 5000, 800),
        (7, 7, 0, 2200, 0),
        (8, 7, 1, 5600, 0),
        (9, 7, 2, 4900, 800),
        (10, 8, 0, 2300, 0),
        (11, 9, 0, 2400, 0),
    ],
)
def test_port_without_mixer(
    idx: int,
    port_subport: Union[int, Tuple[int, int]],
    channel: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures8,
    fixture_ms2720t1,
):
    box, params, topdirpath = fixtures8
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "port_without_mixer")

    spa = fixture_ms2720t1

    port, subport, portname = decode_port_subport(port_subport)

    via_monitor = False
    if port_subport in port_availability["unavailable"]:
        pytest.skip(f"({portname} is unavailable.")
    elif port_subport in port_availability["via_monitor_out"]:
        via_monitor = True

    task = box_gen_cw(
        box,
        port_subport,
        channel,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=20000,
        via_monitor=via_monitor,
    )
    expected_freq = (cnco_mhz + fnco_mhz) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    max_sprious_peek = -50.0
    if has_doubler(box.css.boxtype, port_subport):
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
    plt.savefig(outdir / f"{idx:02d}_{portname}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png")

    logger.info(f"w0 = {w0}")
    logger.info(f"j0 = {j0}")
    assert len(s0) == 0
    assert j0[0]
