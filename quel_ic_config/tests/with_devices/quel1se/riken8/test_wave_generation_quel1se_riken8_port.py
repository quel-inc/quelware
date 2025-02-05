import logging
import os
import shutil
from pathlib import Path
from typing import Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from quel_ic_config.quel1_box import Quel1Box, Quel1BoxType
from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak, SpectrumAnalyzer
from testlibs.spa_helper import init_ms2xxxx, measure_floor_noise

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.94",
            "ipaddr_sss": "10.2.0.94",
            "ipaddr_css": "10.5.0.94",
            "boxtype": Quel1BoxType.fromstr("quel1se-riken8"),
        },
        "linkup_config": {
            "config_root": None,
            "config_options": [],
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
        "spectrum_image_path": "./artifacts/spectrum-094-port",
        "relative_loss": 9,
        "linkup": False,
    },
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def fixtures(request):
    param0 = request.param

    box = Quel1Box.create(**param0["box_config"])
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
    del box


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
def test_all_single_awgs_with_mixer(
    idx: int,
    port_subport: Union[int, Tuple[int, int]],
    channel: int,
    lo_mhz: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures,
):
    box, spa, outdir, port_availability, relative_loss = fixtures
    assert isinstance(box, Quel1Box)

    port, subport, portname = decode_port_subport(port_subport)

    via_monitor = False
    if port_subport in port_availability["unavailable"]:
        pytest.skip(f"{portname} is unavailable.")
    elif port in port_availability["via_monitor_out"]:
        via_monitor = True

    # TODO: fix
    box.easy_start_cw(
        port=port,
        subport=subport,
        channel=channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband="L",
        amplitude=32767.0,
        control_port_rfswitch=not via_monitor,
        control_monitor_rfswitch=via_monitor,
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
    box.easy_stop(
        port=port, subport=subport, control_port_rfswitch=not via_monitor, control_monitor_rfswitch=via_monitor
    )

    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / "awg" / f"{idx:02d}_{portname}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png")

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
        (5, 7, 0, 2200, 0),
        (6, 7, 1, 5700, 0),
        (7, 7, 2, 5000, 800),
        (8, 8, 0, 2300, 0),
        (9, 8, 1, 5600, 0),
        (10, 8, 2, 4900, 800),
        (11, 9, 0, 2400, 0),
    ],
)
def test_all_single_awgs_without_mixer(
    idx: int,
    port_subport: Union[int, Tuple[int, int]],
    channel: int,
    cnco_mhz: int,
    fnco_mhz: int,
    fixtures,
):
    box, spa, outdir, port_availability, relative_loss = fixtures
    assert isinstance(box, Quel1Box)

    port, subport, portname = decode_port_subport(port_subport)

    via_monitor = False
    if port_subport in port_availability["unavailable"]:
        pytest.skip(f"({portname} is unavailable.")
    elif port_subport in port_availability["via_monitor_out"]:
        via_monitor = True

    box.easy_start_cw(
        port=port,
        subport=subport,
        channel=channel,
        cnco_freq=cnco_mhz * 1e6,
        fnco_freq=fnco_mhz * 1e6,
        fullscale_current=20000,
        amplitude=32767.0,
        control_port_rfswitch=not via_monitor,
        control_monitor_rfswitch=via_monitor,
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
    box.easy_stop(
        port=port, subport=subport, control_port_rfswitch=not via_monitor, control_monitor_rfswitch=via_monitor
    )

    j0, s0, w0 = e0.match(m0)

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.savefig(outdir / "awg" / f"{idx:02d}_{portname}-ch{channel:d}-{int(expected_freq) // 1000000:d}MHz.png")

    logger.info(f"w0 = {w0}")
    logger.info(f"j0 = {j0}")
    assert len(s0) == 0
    assert j0[0]
