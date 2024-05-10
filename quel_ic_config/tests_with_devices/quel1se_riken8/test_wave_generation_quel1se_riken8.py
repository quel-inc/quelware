import logging
import os
import shutil
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from quel_ic_config.quel1_box import Quel1BoxIntrinsic, Quel1BoxType
from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak, SpectrumAnalyzer
from testlibs.spa_helper import init_ms2xxxx, measure_floor_noise

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.132",
            "ipaddr_sss": "10.2.0.132",
            "ipaddr_css": "10.5.0.132",
            "boxtype": Quel1BoxType.fromstr("quel1se-riken8"),
            "config_root": None,
            "config_options": [],
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "spa_type": "MS2XXXX",
        "spa_name": "ms2720t-1",
        "spa_parameters": {
            "freq_center": 5e9,
            "freq_span": 8e9,
            "resolution_bandwidth": 1e4,
        },
        "max_background_noise": -50.0,
        # "spa_name": "ms2090-1",
        # "spa_parameters": {},
        # "max_background_noise": -65.0,
        "spectrum_image_path": "./artifacts/spectrum-132",
        "relative_loss": 0,
        "linkup": False,
    },
)


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def fixtures(request):
    param0 = request.param

    box = Quel1BoxIntrinsic.create(**param0["box_config"])
    if request.param["linkup"]:
        linkstatus = box.relinkup(**param0["linkup_config"])
    else:
        linkstatus = box.reconnect()
    assert linkstatus[0]
    assert linkstatus[1]

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
    box.css.terminate()


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
def test_all_single_awgs_with_mixer(
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
    assert isinstance(box, Quel1BoxIntrinsic)

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
        fullscale_current=40527,
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


# Notes: 2 -- 5.8GHz
@pytest.mark.parametrize(
    ("idx", "mxfe", "line", "channel", "cnco_mhz", "fnco_mhz"),
    [
        (0, 0, 1, 0, 3000, 0),
        (1, 0, 3, 0, 2000, 0),
        (2, 0, 3, 1, 5800, 0),
        (3, 0, 3, 2, 5800, -600),
        (4, 1, 0, 0, 2100, 0),
        (5, 1, 0, 1, 5700, 0),
        (6, 1, 0, 2, 5000, 800),
        (7, 1, 1, 0, 2200, 0),
        (8, 1, 1, 1, 5600, 0),
        (9, 1, 1, 2, 4900, 800),
        (10, 1, 2, 0, 2300, 0),
        (11, 1, 3, 0, 2400, 0),
    ],
)
def test_all_single_awgs_without_mixer(
    idx: int,
    mxfe: int,
    line: int,
    channel: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures,
):
    box, spa, outdir, port_availability, relative_loss = fixtures
    assert isinstance(box, Quel1BoxIntrinsic)

    via_monitor = False
    if (mxfe, line) in port_availability["unavailable"]:
        pytest.skip(f"({mxfe}, {line}) is unavailable.")
    elif (mxfe, line) in port_availability["via_monitor_out"]:
        via_monitor = True

    box.easy_start_cw(
        mxfe,
        line,
        channel,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=15000,
        amplitude=32767.0,
        control_port_rfswitch=not via_monitor,
        control_monitor_rfswitch=via_monitor,
    )
    expected_freq = (cnco_mhz + fnco_mhz) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    max_sprious_peek = -50.0
    if line == 1 and box.css._boxtype in {
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
    box.easy_stop(mxfe, line, control_port_rfswitch=not via_monitor, control_monitor_rfswitch=via_monitor)

    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(
        outdir / "awg" / f"{idx:02d}_mxfe{mxfe:d}-line{line:d}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png"
    )

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
    fixtures,
):
    box, e4405b, outdir, port_availability, relative_loss = fixtures
    assert isinstance(box, Quel1BoxIntrinsic)
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
