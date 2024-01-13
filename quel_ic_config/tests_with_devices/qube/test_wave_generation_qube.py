import logging
import os
import shutil
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from quel_ic_config_utils.simple_box import Quel1BoxType, Quel1ConfigOption, SimpleBoxIntrinsic, init_box_with_linkup
from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak, SpectrumAnalyzer
from testlibs.spa_helper import init_ms2xxxx, measure_floor_noise

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.17",
            "ipaddr_sss": "10.2.0.17",
            "ipaddr_css": "10.5.0.17",
            "boxtype": Quel1BoxType.fromstr("qube-riken-a"),
            "config_root": None,
            "config_options": [
                Quel1ConfigOption.USE_READ_IN_MXFE0,
                Quel1ConfigOption.USE_READ_IN_MXFE1,
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE0,
                Quel1ConfigOption.DAC_CNCO_1500MHz_MXFE0,
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE1,
                Quel1ConfigOption.DAC_CNCO_2000MHz_MXFE1,
            ],
            "mxfes_to_linkup": (0, 1),
            "use_204b": True,
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "spa_type": "MS2XXXX",
        "spa_name": "ms2720t-1",
        "spa_parameters": {
            "resolution_bandwidth": 1e4,
        },
        "max_background_noise": -54.0,
        # "spa_name": "ms2090-1",
        # "spa_parameters": {},
        # "max_background_noise": -65.0,
        "spectrum_image_path": "./artifacts/spectrum-017",
        "relative_loss": 0,
    },
)


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def fixtures(request):
    param0 = request.param

    linkstatus, _, _, _, _, box = init_box_with_linkup(**param0["box_config"], refer_by_port=False)
    assert linkstatus[0]
    assert linkstatus[1]
    assert isinstance(box, SimpleBoxIntrinsic)

    if request.param["spa_type"] == "MS2XXXX":
        spa: SpectrumAnalyzer = init_ms2xxxx(request.param["spa_name"], **request.param["spa_parameters"])
    else:
        # Notes: to be added by need.
        assert False
    max_noise = measure_floor_noise(spa)
    assert max_noise < request.param["max_background_noise"]
    yield box, spa, make_outdir(request.param), request.param["port_availability"], request.param["relative_loss"]

    box.easy_stop_all()
    box.activate_monitor_loop(0)
    box.activate_monitor_loop(1)


def make_outdir(param):
    mpl.use("Qt5Agg")  # TODO: reconsider where to execute.

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
    ("idx", "mxfe", "line", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 0, 0, 12000, 2000, 0),
        (1, 0, 0, 0, 12000, 2500, 0),
        (2, 0, 0, 0, 12000, 1400, 500),
        (3, 0, 0, 0, 12000, 1800, 600),
        (4, 0, 0, 0, 12000, 2800, -600),
        (5, 0, 1, 0, 12000, 2000, 100),
        (6, 0, 1, 0, 12000, 2500, 100),
        (7, 0, 2, 0, 11000, 2000, 25),
        (8, 0, 2, 0, 11000, 2000, 525),
        (9, 0, 2, 1, 11000, 2000, 50),
        (10, 0, 2, 1, 11000, 2000, 550),
        (11, 0, 2, 2, 11000, 2000, 75),
        (12, 0, 2, 2, 11000, 2000, 575),
        (13, 0, 3, 0, 10500, 2000, 25),
        (14, 0, 3, 0, 10500, 2000, 525),
        (15, 0, 3, 1, 10500, 2000, 50),
        (16, 0, 3, 1, 10500, 2000, 550),
        (17, 0, 3, 2, 10500, 2000, 75),
        (18, 0, 3, 2, 10500, 2000, 575),
        (19, 1, 0, 0, 12000, 2050, 0),
        (20, 1, 0, 0, 12000, 2450, 0),
        (21, 1, 0, 0, 12000, 2150, 200),
        (22, 1, 0, 0, 12000, 1750, 600),
        (23, 1, 0, 0, 12000, 3000, -600),
        (24, 1, 1, 0, 12000, 1900, 100),
        (25, 1, 1, 0, 12000, 2600, -100),
        (26, 1, 2, 0, 11000, 2000, -25),
        (27, 1, 2, 0, 11000, 2000, -525),
        (28, 1, 2, 1, 11000, 2000, -50),
        (29, 1, 2, 1, 11000, 2000, -550),
        (30, 1, 2, 2, 11000, 2000, -75),
        (31, 1, 2, 2, 11000, 2000, -575),
        (32, 1, 3, 0, 10500, 2000, -25),
        (33, 1, 3, 0, 10500, 2000, -525),
        (34, 1, 3, 0, 10500, 2500, -800),  # for 2000Msps
        (35, 1, 3, 0, 10500, 2000, 800),  # for 2000Msps
        (36, 1, 3, 1, 10500, 2000, -50),
        (37, 1, 3, 1, 10500, 2000, -550),
        (38, 1, 3, 2, 10500, 2000, -75),
        (39, 1, 3, 2, 10500, 2000, -575),
    ],
)
def test_all_single_awgs(
    idx: int,
    mxfe: int,
    line: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures,
):
    box, spa, outdir, port_availability, relative_loss = fixtures
    assert isinstance(box, SimpleBoxIntrinsic)

    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    # TODO: fix
    box.easy_start_cw(
        mxfe,
        line,
        channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        vatt=0xA00,
        sideband="L",
        amplitude=32767.0,
        control_port_rfswitch=not via_monitor,
        control_monitor_rfswitch=via_monitor,
    )
    expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    max_sprious_peek = -50.0
    if line == 1 and box.css._boxtype in {
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
    box.easy_stop(mxfe, line, control_port_rfswitch=not via_monitor, control_monitor_rfswitch=via_monitor)

    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(
        outdir / "awg" / f"{idx:02d}_mxfe{mxfe:d}-line{line:d}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png"
    )

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)


@pytest.mark.parametrize(
    ("idx", "mxfe", "line", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 0, 0, 12000, 2500, 0),
        (1, 0, 1, 0, 12000, 2500, 100),
        (2, 0, 2, 0, 11000, 2000, 525),
        (3, 0, 3, 0, 10500, 2000, 525),
        (4, 1, 0, 0, 12000, 2500, 0),
        (5, 1, 1, 0, 12000, 2500, 100),
        (6, 1, 2, 0, 11000, 2000, 525),
        (7, 1, 3, 0, 10500, 2000, 525),
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
    fixtures,
):
    box, e4405b, outdir, port_availability, relative_loss = fixtures
    assert isinstance(box, SimpleBoxIntrinsic)
    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    pwr: Dict[int, float] = {}
    is_pump = line == 1 and box.css._boxtype in {
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
        box.easy_start_cw(
            mxfe,
            line,
            channel,
            lo_freq=lo_mhz * 1e6,
            cnco_freq=cnco_mhz * 1e6,
            fnco_freq=fnco_mhz * 1e6,
            sideband="L",
            vatt=vatt,
            amplitude=32767.0,
            control_port_rfswitch=not via_monitor,
            control_monitor_rfswitch=via_monitor,
        )

        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
        if is_pump:
            expected_freq *= 2
            max_sprious_peek = -42.0
        else:
            max_sprious_peek = -50.0
        e0 = ExpectedSpectrumPeaks([(expected_freq, -40 - relative_loss)])
        e0.validate_with_measurement_condition(e4405b.max_freq_error_get())

        # notes: -60.0dBm fails due to spurious below 7GHz.
        m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(e4405b, max_sprious_peek)
        box.easy_stop(mxfe, line, control_port_rfswitch=not via_monitor, control_monitor_rfswitch=via_monitor)

        plt.cla()
        plt.plot(t0[:, 0], t0[:, 1])
        plt.savefig(outdir / "vatt" / f"{idx:02d}-mxfe{mxfe:d}-line{line:d}-ch{channel:d}-vatt{vatt:04x}.png")

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


@pytest.mark.parametrize(
    ("idx", "mxfe", "line", "channel", "lo_mhz", "cnco_mhz", "fnco_mhz", "sideband"),
    [
        (0, 0, 0, 0, 8000, 1900, 0, "U"),
        (1, 0, 0, 0, 12000, 2000, 0, "L"),
        (2, 0, 1, 0, 8000, 1800, 0, "U"),
        (3, 0, 1, 0, 12000, 2500, 0, "L"),
        (4, 0, 2, 0, 8000, 1500, 0, "U"),
        (5, 0, 2, 0, 12000, 2500, 0, "L"),
        (6, 0, 3, 0, 8000, 2000, 0, "U"),
        (7, 0, 3, 0, 12000, 3000, 0, "L"),
        (8, 1, 0, 0, 8000, 1800, 0, "U"),
        (9, 1, 0, 0, 12000, 2200, 0, "L"),
        (10, 1, 1, 0, 8000, 1800, 0, "U"),
        (11, 1, 1, 0, 11500, 1800, 0, "L"),
        (12, 1, 2, 0, 8000, 1600, 0, "U"),
        (13, 1, 2, 0, 12000, 2600, 0, "L"),
        (14, 1, 3, 0, 8000, 1900, 0, "U"),
        (15, 1, 3, 0, 12000, 2900, 0, "L"),
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
    fixtures,
):
    box, e4405b, outdir, port_availability, relative_loss = fixtures
    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    box.easy_start_cw(
        mxfe,
        line,
        channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        vatt=0xA00,
        sideband=sideband,
        amplitude=32767.0,
        control_port_rfswitch=not via_monitor,
        control_monitor_rfswitch=via_monitor,
    )

    if sideband == "L":
        expected_freq = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e6
    elif sideband == "U":
        expected_freq = (lo_mhz + (cnco_mhz + fnco_mhz)) * 1e6
    else:
        raise AssertionError
    if line == 1 and box.css._boxtype in {
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
    }:
        expected_freq *= 2
        max_sprious_peek = -42.0
    else:
        max_sprious_peek = -50.0
    e0 = ExpectedSpectrumPeaks([(expected_freq, -20 - relative_loss)])
    e0.validate_with_measurement_condition(e4405b.max_freq_error_get())

    # notes: -60.0dBm fails due to spurious below 7GHz.
    m0, t0 = MeasuredSpectrumPeak.from_spectrumanalyzer_with_trace(e4405b, max_sprious_peek)
    box.easy_stop(mxfe, line, control_port_rfswitch=not via_monitor, control_monitor_rfswitch=via_monitor)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / "sideband" / f"{idx:02d}-mxfe{mxfe:d}-line{line:d}-ch{channel:d}-sideband{sideband}.png")

    j0, s0, w0 = e0.match(m0)

    assert len(s0) == 0
    assert len(w0) == 0
    assert all(j0)
