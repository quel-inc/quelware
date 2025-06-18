import logging
import os
import shutil
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak, SpectrumAnalyzer

from quel_ic_config.quel1_box import Quel1Box
from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption
from testlibs.gen_cw import box_gen_cw
from testlibs.register_cw import register_cw_to_all_ports
from testlibs.spa_helper import init_ms2xxxx, measure_floor_noise

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.80",
            "ipaddr_sss": "10.2.0.80",
            "ipaddr_css": "10.5.0.80",
            "boxtype": Quel1BoxType.fromstr("quel1-nec"),
            "config_root": None,
            "config_options": [
                Quel1ConfigOption.USE_READ_IN_MXFE0,
                Quel1ConfigOption.USE_READ_IN_MXFE1,
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE0,
                Quel1ConfigOption.DAC_CNCO_1500MHz_MXFE0,
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE1,
                Quel1ConfigOption.DAC_CNCO_2000MHz_MXFE1,
            ],
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": True,
        },
        "port_availability": {
            "unavailable": [1, 4, 7, 10],
            "via_monitor_out": [],
        },
        "spa_type": "ms2090a-1",
        "spectrum_image_path": "./artifacts/spectrum-078",
        "relative_loss": 0,
    },
)


MAX_BACKGROUND_NOISE = -62.0  # dBm


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def fixtures(request):
    param0 = request.param

    box = Quel1Box.create(**param0["box_config"])
    linkstatus = box.relinkup(**param0["linkup_config"])
    assert linkstatus[0]
    assert linkstatus[1]

    register_cw_to_all_ports(box)

    ms2xxxx: SpectrumAnalyzer = init_ms2xxxx(request.param["spa_type"])
    ms2xxxx.input_attenuation = 10.0
    max_noise = measure_floor_noise(ms2xxxx)
    assert max_noise < MAX_BACKGROUND_NOISE
    yield box, ms2xxxx, make_outdir(request.param), request.param["port_availability"], request.param["relative_loss"]

    box.initialize_all_awgunits()
    box.activate_monitor_loop(0)
    box.activate_monitor_loop(1)


def make_outdir(param):
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    dirname = param["spectrum_image_path"]
    if os.path.exists(dirname):
        logger.info(f"deleting the existing directory: '{dirname}'")
        shutil.rmtree(dirname)

    dpath = Path(dirname)
    os.makedirs(dpath)
    os.makedirs(dpath / "awg")
    os.makedirs(dpath / "vatt")
    os.makedirs(dpath / "sideband")
    return dpath


@pytest.mark.parametrize(
    ("idx", "port", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 0, 12000, 2000, 0),
        (1, 0, 0, 12000, 2500, 0),
        (2, 0, 0, 12000, 1400, 500),
        (3, 0, 0, 12000, 1800, 600),
        (4, 0, 0, 12000, 2800, -600),
        (5, 3, 0, 12000, 2000, 100),
        (6, 3, 0, 12000, 2500, 100),
        (7, 1, 0, 11000, 2000, 25),
        (8, 1, 0, 11000, 2000, 525),
        (9, 1, 1, 11000, 2000, 50),
        (10, 1, 1, 11000, 2000, 550),
        (11, 1, 2, 11000, 2000, 75),
        (12, 1, 2, 11000, 2000, 575),
        (13, 4, 0, 10500, 2000, 25),
        (14, 4, 0, 10500, 2000, 525),
        (15, 4, 1, 10500, 2000, 50),
        (16, 4, 1, 10500, 2000, 550),
        (17, 4, 2, 10500, 2000, 75),
        (18, 4, 2, 10500, 2000, 575),
        (19, 6, 0, 12000, 2050, 0),
        (20, 6, 0, 12000, 2450, 0),
        (21, 6, 0, 12000, 2150, 200),
        (22, 6, 0, 12000, 1750, 600),
        (23, 6, 0, 12000, 3000, -600),
        (24, 9, 0, 12000, 1900, 100),
        (25, 9, 0, 12000, 2600, -100),
        (26, 7, 0, 11000, 2000, -25),
        (27, 7, 0, 11000, 2000, -525),
        (28, 7, 1, 11000, 2000, -50),
        (29, 7, 1, 11000, 2000, -550),
        (30, 7, 2, 11000, 2000, -75),
        (31, 7, 2, 11000, 2000, -575),
        (32, 10, 0, 10500, 2000, -25),
        (33, 10, 0, 10500, 2000, -525),
        (34, 10, 0, 10500, 2500, -800),  # for 2000Msps
        (35, 10, 0, 10500, 2000, 800),  # for 2000Msps
        (36, 10, 1, 10500, 2000, -50),
        (37, 10, 1, 10500, 2000, -550),
        (38, 10, 2, 10500, 2000, -75),
        (39, 10, 2, 10500, 2000, -575),
    ],
)
def test_all_single_awgs(
    idx: int,
    port: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures,
):
    box, ms2xxxx, outdir, port_availability, relative_loss = fixtures
    assert isinstance(box, Quel1Box)

    via_monitor = False
    if port in port_availability["unavailable"]:
        pytest.skip(f"{port} is unavailable.")
    elif port in port_availability["via_monitor_out"]:
        via_monitor = True

    # TODO: fix
    task = box_gen_cw(
        box,
        port,
        channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband="L",
        via_monitor=via_monitor,
    )
    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(ms2xxxx, -50.0)
    # notes: stop all the awgs of the line
    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    e0 = ExpectedSpectrumPeaks([(expected_freq, -20 - relative_loss)])
    e0.validate_with_measurement_condition(ms2xxxx.max_freq_error_get())
    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / "awg" / f"{idx:02d}_port{port:d}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png")

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)


@pytest.mark.parametrize(
    ("idx", "port", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 0, 12000, 2500, 0),
        (1, 1, 0, 12000, 2500, 100),
        (2, 3, 0, 11000, 2000, 525),
        (3, 4, 0, 10500, 2000, 525),
        (4, 6, 0, 12000, 2500, 0),
        (5, 7, 0, 12000, 2500, 100),
        (6, 9, 0, 11000, 2000, 525),
        (7, 9, 0, 10500, 2000, 525),
    ],
)
def test_vatt(
    idx: int,
    port: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures,
):
    box, ms2xxxx, outdir, port_availability, relative_loss = fixtures
    assert isinstance(box, Quel1Box)
    via_monitor = False
    if port in port_availability["unavailable"]:
        pytest.skip(f"{port} is unavailable.")
    elif port in port_availability["via_monitor_out"]:
        via_monitor = True

    pwr: Dict[int, float] = {}
    for vatt in (0x380, 0x500, 0x680, 0x800, 0x980, 0xB00):
        """
        V_ref of AD5328 == 3.3V
        The expected output voltages are 0.72V, 1.03V, 1.34V, 1.65V, 1.96V, and 2.27V, respectively.
        Their corresponding gains at 10GHz are approximately -10dB, -7dB, -2dB, 3dB, 7dB, and 12dB, respectively.
        """
        task = box_gen_cw(
            box,
            port,
            channel,
            lo_freq=lo_mhz * 1e6,
            cnco_freq=cnco_mhz * 1e6,
            fnco_freq=fnco_mhz * 1e6,
            fullscale_current=40527,
            sideband="L",
            vatt=vatt,
            via_monitor=via_monitor,
        )
        # notes: -60.0dBm fails due to spurious below 7GHz.
        m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(ms2xxxx, -50.0)
        task.cancel()
        with pytest.raises(CancelledError):
            task.result()

        plt.cla()
        plt.plot(t0[:, 0], t0[:, 1])
        plt.savefig(outdir / "vatt" / f"{idx:02d}-port{port:d}-ch{channel:d}-vatt{vatt:04x}.png")

        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
        e0 = ExpectedSpectrumPeaks([(expected_freq, -40 - relative_loss)])
        e0.validate_with_measurement_condition(ms2xxxx.max_freq_error_get())
        d0 = e0.extract_matched(m0)
        assert len(d0) == 1

        d00 = d0.pop()
        pwr[vatt] = d00.power

    logger.info(f"vatt vs power@{(port)}: {pwr}")
    pwrl = list(pwr.values())
    for i in range(1, len(pwrl)):
        if i == 1:
            assert 2.25 <= pwrl[i] - pwrl[i - 1] <= 4.25
        else:
            assert 2.4 <= pwrl[i] - pwrl[i - 1] <= 5.0


@pytest.mark.parametrize(
    ("idx", "port", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz", "sideband"),
    [
        (0, 0, 0, 8000, 1900, 0, "U"),
        (1, 0, 0, 12000, 2000, 0, "L"),
        (2, 1, 0, 8000, 1800, 0, "U"),
        (3, 1, 0, 12000, 2500, 0, "L"),
        (4, 3, 0, 8000, 1500, 0, "U"),
        (5, 3, 0, 12000, 2500, 0, "L"),
        (6, 4, 0, 8000, 2000, 0, "U"),
        (7, 4, 0, 12000, 3000, 0, "L"),
        (8, 6, 0, 8000, 1800, 0, "U"),
        (9, 6, 0, 12000, 2200, 0, "L"),
        (10, 7, 0, 8000, 1800, 0, "U"),
        (11, 7, 0, 11500, 1800, 0, "L"),
        (12, 9, 0, 8000, 1600, 0, "U"),
        (13, 9, 0, 12000, 2600, 0, "L"),
        (14, 10, 0, 8000, 1900, 0, "U"),
        (15, 10, 0, 12000, 2900, 0, "L"),
    ],
)
def test_sideband(
    idx: int,
    port: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    sideband: str,
    fixtures,
):
    box, ms2xxxx, outdir, port_availability, relative_loss = fixtures
    via_monitor = False
    if port in port_availability["unavailable"]:
        pytest.skip(f"{port} is unavailable.")
    elif port in port_availability["via_monitor_out"]:
        via_monitor = True

    task = box_gen_cw(
        box,
        port,
        channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband=sideband,
        via_monitor=via_monitor,
    )
    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(ms2xxxx, -50.0)
    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / "sideband" / f"{idx:02d}-port{port:d}-ch{channel:d}-sideband{sideband}.png")

    if sideband == "L":
        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
    elif sideband == "U":
        expected_freq = (lo_mhz + (cnco_mhz + fnco_mhz)) * 1e6
    else:
        raise AssertionError

    e0 = ExpectedSpectrumPeaks([(expected_freq, -20 - relative_loss)])
    e0.validate_with_measurement_condition(ms2xxxx.max_freq_error_get())
    j0, s0, w0 = e0.match(m0)

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)
