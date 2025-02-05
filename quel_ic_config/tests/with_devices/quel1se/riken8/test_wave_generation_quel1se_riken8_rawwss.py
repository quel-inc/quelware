import logging
import os
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from e7awgsw import WaveSequence

from quel_ic_config.quel1_box import Quel1BoxType
from quel_ic_config.quel1_box_with_raw_wss import Quel1BoxWithRawWss
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
        "spectrum_image_path": "./artifacts/spectrum-094-rawwss",
        "relative_loss": 9,
        "linkup": False,
    },
)


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def fixtures(request):
    param0 = request.param

    box = Quel1BoxWithRawWss.create(**param0["box_config"])
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


def make_pulses_wave_param(
    num_delay_sample: int,
    num_global_repeat: int,
    num_wave_samples: Sequence[int],
    num_blank_samples: Sequence[int],
    num_repeats: Sequence[int],
    amplitudes: Sequence[complex],
) -> WaveSequence:
    if num_delay_sample % 64 != 0:
        raise ValueError(f"num_delay_sample (= {num_delay_sample}) is not multiple of 64")
    wave = WaveSequence(num_wait_words=num_delay_sample // 4, num_repeats=num_global_repeat)

    for idx in range(len(num_wave_samples)):
        if num_wave_samples[idx] % 64 != 0:
            raise ValueError(f"num_wave_samples[{idx}] (= {num_wave_samples[idx]}) is not multiple of 64")
        if num_blank_samples[idx] % 4 != 0:
            raise ValueError(f"num_blank_samples[{idx}] (= {num_blank_samples[idx]}) is not multiple of 4")
        iq = np.zeros(num_wave_samples[idx], dtype=np.complex64)
        iq[:] = (1 + 0j) * amplitudes[idx]
        block_assq: List[Tuple[int, int]] = list(zip(iq.real.astype(int), iq.imag.astype(int)))
        wave.add_chunk(iq_samples=block_assq, num_blank_words=num_blank_samples[idx] // 4, num_repeats=num_repeats[idx])
    return wave


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
        (6, 7, 1, 5650, 0),
        (7, 7, 2, 5000, 700),
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
    assert isinstance(box, Quel1BoxWithRawWss)

    port, subport, portname = decode_port_subport(port_subport)

    via_monitor = False
    if port_subport in port_availability["unavailable"]:
        pytest.skip(f"({portname} is unavailable.")
    elif port_subport in port_availability["via_monitor_out"]:
        via_monitor = True

    wave_param = make_pulses_wave_param(
        num_delay_sample=0,
        num_global_repeat=1,
        num_wave_samples=(64,),
        num_blank_samples=(0,),
        num_repeats=(0xFFFFFFFF,),
        amplitudes=(32767.0,),
    )

    box.config_port(port=port, subport=subport, cnco_freq=cnco_mhz * 1e6, fullscale_current=20000)
    box.config_channel(port=port, subport=subport, channel=channel, fnco_freq=fnco_mhz * 1e6, wave_param=wave_param)
    if via_monitor:
        group, _ = box._convert_output_port_decoded(port, subport)
        box.deactivate_monitor_loop(group)
    else:
        box.config_rfswitch(port=port, rfswitch="pass")
    box.start_emission(channels=((port_subport, channel),))

    expected_freq = (cnco_mhz + fnco_mhz) * 1e6  # Note that LSB mode (= default sideband mode) is assumed.
    max_sprious_peek = -50.0

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
