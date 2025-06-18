import logging
import os
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from e7awghal import AwgParam, WaveChunk
from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak, SpectrumAnalyzer

from quel_ic_config.quel1_box import Quel1BoxIntrinsic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def make_outdir(dirpath: Path):
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    os.makedirs(dirpath, exist_ok=True)
    return dirpath


def e4405b_measure(
    e4405b: SpectrumAnalyzer,
    boxi: Quel1BoxIntrinsic,
    group: int,
    line: int,
    channel: int,
    fnco_mhz: float,
    cnco_mhz: float,
    fsc: Optional[int],
    lo_mhz: float,
    sideband: Optional[str],
    vatt: Optional[int],
    standard_power: float,
    relative_loss: float,
) -> tuple[float, ExpectedSpectrumPeaks, list[MeasuredSpectrumPeak], npt.NDArray[np.float64]]:
    ap: AwgParam = AwgParam(num_repeat=0xFFFF_FFFF)
    ap.chunks.append(WaveChunk(name_of_wavedata="test_wave_generation:cw32767", num_repeat=0xFFFF_FFFF))
    boxi.config_channel(group, line, channel, fnco_freq=fnco_mhz * 1e6, awg_param=ap)
    boxi.config_line(
        group,
        line,
        cnco_freq=cnco_mhz * 1e6,
        fullscale_current=fsc,
        lo_freq=lo_mhz * 1e6,
        sideband=sideband,
        vatt=vatt,
        rfswitch="pass",
    )
    task = boxi.start_wavegen({(group, line, channel)})

    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(e4405b, -50.0)

    task.cancel()
    with pytest.raises(CancelledError):
        task.result()
    del task

    if sideband == "L":
        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
    elif sideband == "U":
        expected_freq = (lo_mhz + (cnco_mhz + fnco_mhz)) * 1e6
    elif sideband is None:
        expected_freq = (cnco_mhz + fnco_mhz) * 1e6
        assert lo_mhz is None
        assert vatt is None
    else:
        raise AssertionError

    e0 = ExpectedSpectrumPeaks([(expected_freq, standard_power - relative_loss)])
    e0.validate_with_measurement_condition(e4405b.max_freq_error_get())
    return expected_freq, e0, m0, t0


@pytest.mark.parametrize(
    ("case_idx", "group", "line", "channel", "fnco_mhz", "cnco_mhz", "fsc", "lo_mhz", "sideband", "vatt"),
    [
        (0, 0, 0, 0, 500, 1400, 40527, 12000, "L", 0xA00),
        (1, 0, 0, 0, -600, 2800, 40527, 12000, "L", 0xA00),
        (2, 0, 1, 0, 100, 2000, 40527, 12000, "L", 0xA00),
        (3, 0, 1, 0, 100, 2500, 40527, 12000, "L", 0xA00),
        (4, 0, 2, 0, 25, 2000, 40527, 11000, "L", 0xA00),
        (5, 0, 2, 0, 525, 2000, 40527, 11000, "L", 0xA00),
        (6, 0, 2, 1, 50, 2000, 40527, 11000, "L", 0xA00),
        (7, 0, 2, 1, 550, 2000, 40527, 11000, "L", 0xA00),
        (8, 0, 2, 2, 75, 2000, 40527, 11000, "L", 0xA00),
        (9, 0, 2, 2, 575, 2000, 40527, 11000, "L", 0xA00),
        (10, 0, 3, 0, 25, 2000, 40527, 10500, "L", 0xA00),
        (11, 0, 3, 0, 525, 2000, 40527, 10500, "L", 0xA00),
        (12, 0, 3, 1, 50, 2000, 40527, 10500, "L", 0xA00),
        (13, 0, 3, 1, 550, 2000, 40527, 10500, "L", 0xA00),
        (14, 0, 3, 2, 75, 2000, 40527, 10500, "L", 0xA00),
        (15, 0, 3, 2, 575, 2000, 40527, 10500, "L", 0xA00),
        (16, 1, 0, 0, 600, 1750, 40527, 12000, "L", 0xA00),
        (17, 1, 0, 0, -600, 3000, 40527, 12000, "L", 0xA00),
        (18, 1, 1, 0, -100, 2600, 40527, 12000, "L", 0xA00),
        (19, 1, 1, 0, -800, 2500, 40527, 10500, "L", 0xA00),
        (20, 1, 2, 0, -25, 2000, 40527, 11000, "L", 0xA00),
        (21, 1, 2, 0, -525, 2000, 40527, 11000, "L", 0xA00),
        (22, 1, 2, 1, -50, 2000, 40527, 11000, "L", 0xA00),
        (23, 1, 2, 1, -550, 2000, 40527, 11000, "L", 0xA00),
        (24, 1, 2, 2, -75, 2000, 40527, 11000, "L", 0xA00),
        (25, 1, 2, 2, -575, 2000, 40527, 11000, "L", 0xA00),
        (26, 1, 3, 0, -25, 2000, 40527, 10500, "L", 0xA00),
        (27, 1, 3, 0, 800, 2000, 40527, 10500, "L", 0xA00),
        (28, 1, 3, 1, -50, 2000, 40527, 10500, "L", 0xA00),
        (29, 1, 3, 1, -550, 2000, 40527, 10500, "L", 0xA00),
        (30, 1, 3, 2, -75, 2000, 40527, 10500, "L", 0xA00),
        (31, 1, 3, 2, -575, 2000, 40527, 10500, "L", 0xA00),
    ],
)
def test_awgs(
    case_idx: int,
    group: int,
    line: int,
    channel: int,
    fnco_mhz: float,
    cnco_mhz: float,
    fsc: int,
    lo_mhz: float,
    sideband: str,
    vatt: int,
    fixtures1,
    fixture_e4405b,
):
    box, params, topdirpath = fixtures1
    if params["label"] not in {"staging-060"}:
        pytest.skip()
    boxi = box._dev
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "awgs")
    e4405b = fixture_e4405b

    if (group, line) in port_availability["unavailable"]:
        pytest.skip(f"({group}, {line}) is unavailable.")

    expected_freq, e0, m0, t0 = e4405b_measure(
        e4405b, boxi, group, line, channel, fnco_mhz, cnco_mhz, fsc, lo_mhz, sideband, vatt, -20.0, relative_loss
    )
    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(
        outdir / f"{case_idx:02d}_group{group:d}-line{line:d}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png"
    )

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)


@pytest.mark.parametrize(
    ("case_idx", "group", "line", "channel", "fnco_mhz", "cnco_mhz", "fsc", "lo_mhz", "sideband", "vatt_"),
    [
        (0, 0, 0, 0, 0, 2500, 40527, 12000, "L", None),
        (1, 0, 1, 0, 100, 2500, 40527, 12000, "L", None),
        (2, 0, 2, 0, 525, 2000, 40527, 11000, "L", None),
        (3, 0, 3, 0, 525, 2000, 40527, 10500, "L", None),
        (4, 1, 0, 0, 0, 2000, 40527, 12000, "L", None),
        (5, 1, 1, 0, 100, 2500, 40527, 12000, "L", None),
        (6, 1, 2, 0, 525, 2000, 40527, 11000, "L", None),
        (7, 1, 3, 0, 525, 2000, 40527, 10500, "L", None),
    ],
)
def test_vatt(
    case_idx: int,
    group: int,
    line: int,
    channel: int,
    fnco_mhz: float,
    cnco_mhz: float,
    fsc: int,
    lo_mhz: float,
    sideband: str,
    vatt_: Optional[int],  # NOT USED
    fixtures1,
    fixture_e4405b,
):
    box, params, topdirpath = fixtures1
    if params["label"] not in {"staging-060"}:
        pytest.skip()
    boxi = box._dev
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "vatt")
    e4405b = fixture_e4405b

    if (group, line) in port_availability["unavailable"]:
        pytest.skip(f"({group}, {line}) is unavailable.")

    pwr: dict[int, float] = {}
    for vatt in (0x380, 0x500, 0x680, 0x800, 0x980, 0xB00):
        """
        V_ref of AD5328 == 3.3V
        The expected output voltages are 0.72V, 1.03V, 1.34V, 1.65V, 1.96V, and 2.27V, respectively.
        Their corresponding gains at 10GHz are approximately -10dB, -7dB, -2dB, 3dB, 7dB, and 12dB, respectively.
        """
        expected_freq, e0, m0, t0 = e4405b_measure(
            e4405b, boxi, group, line, channel, fnco_mhz, cnco_mhz, fsc, lo_mhz, sideband, vatt, -40.0, relative_loss
        )

        plt.cla()
        plt.plot(t0[:, 0], t0[:, 1])
        plt.savefig(outdir / f"{case_idx:02d}-group{group:d}-line{line:d}-ch{channel:d}-vatt{vatt:04x}.png")

        d0 = e0.extract_matched(m0)
        assert len(d0) == 1

        d00 = d0.pop()
        pwr[vatt] = d00.power

    logger.info(f"vatt vs power@{(group, line)}: {pwr}")
    pwrl = list(pwr.values())
    for i in range(1, len(pwrl)):
        if i == 1:
            assert 2.25 <= pwrl[i] - pwrl[i - 1] <= 4.25
        else:
            assert 2.4 <= pwrl[i] - pwrl[i - 1] <= 5.0


@pytest.mark.parametrize(
    ("case_idx", "group", "line", "channel", "fnco_mhz", "cnco_mhz", "fsc", "lo_mhz", "sideband", "vatt"),
    [
        (0, 0, 0, 0, 0, 1900, 40527, 8000, "U", 0xA00),
        (1, 0, 0, 0, 0, 2000, 40527, 12000, "L", 0xA00),
        (2, 0, 1, 0, 0, 1800, 40527, 8000, "U", 0xA00),
        (3, 0, 1, 0, 0, 2500, 40527, 12000, "L", 0xA00),
        (4, 0, 2, 0, 0, 1500, 40527, 8000, "U", 0xA00),
        (5, 0, 2, 0, 0, 2500, 40527, 12000, "L", 0xA00),
        (6, 0, 3, 0, 0, 2000, 40527, 8000, "U", 0xA00),
        (7, 0, 3, 0, 0, 3000, 40527, 12000, "L", 0xA00),
        (8, 1, 0, 0, 0, 1800, 40527, 8000, "U", 0xA00),
        (9, 1, 0, 0, 0, 2200, 40527, 12000, "L", 0xA00),
        (10, 1, 1, 0, 0, 1800, 40527, 8000, "U", 0xA00),
        (11, 1, 1, 0, 0, 1800, 40527, 11500, "L", 0xA00),
        (12, 1, 2, 0, 0, 1600, 40527, 8000, "U", 0xA00),
        (13, 1, 2, 0, 0, 2600, 40527, 12000, "L", 0xA00),
        (14, 1, 3, 0, 0, 1900, 40527, 8000, "U", 0xA00),
        (15, 1, 3, 0, 0, 2900, 40527, 12000, "L", 0xA00),
    ],
)
def test_sideband(
    case_idx: int,
    group: int,
    line: int,
    channel: int,
    fnco_mhz: float,
    cnco_mhz: float,
    fsc: int,
    lo_mhz: float,
    sideband: str,
    vatt: int,
    fixtures1,
    fixture_e4405b,
):
    box, params, topdirpath = fixtures1
    if params["label"] not in {"staging-060"}:
        pytest.skip()
    boxi = box._dev
    port_availability = params["port_availability"]
    relative_loss = params["relative_loss"]
    outdir = make_outdir(topdirpath / "sideband")
    e4405b = fixture_e4405b

    if (group, line) in port_availability["unavailable"]:
        pytest.skip(f"({group}, {line}) is unavailable.")

    expected_freq, e0, m0, t0 = e4405b_measure(
        e4405b, boxi, group, line, channel, fnco_mhz, cnco_mhz, fsc, lo_mhz, sideband, vatt, -20.0, relative_loss
    )
    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / f"{case_idx:02d}-group{group:d}-line{line:d}-ch{channel:d}-sideband{sideband}.png")

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)
