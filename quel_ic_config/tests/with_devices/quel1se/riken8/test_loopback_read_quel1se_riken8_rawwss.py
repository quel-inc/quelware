import logging
import os
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from e7awgsw import CaptureParam, WaveSequence

from quel_ic_config.quel1_box import CaptureReturnCode
from quel_ic_config.quel1_box_with_raw_wss import Quel1BoxWithRawWss
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = (
    {
        "label": "staging-094-rawwss",
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
        "linkup": False,
    },
)

OUTPUT_SETTING = {
    "spectrum_image_path": Path("./artifacts/loopback_readin"),
}


@pytest.fixture(scope="session", params=DEVICE_SETTINGS)
def fixtures(request):
    param0 = request.param

    box = Quel1BoxWithRawWss.create(**param0["box_config"])
    if request.param["linkup"]:
        linkstatus = box.relinkup(**param0["linkup_config"])
    else:
        linkstatus = box.reconnect()
    assert linkstatus[0]
    assert linkstatus[1]

    yield make_outdir(param0["label"]), box

    box.easy_stop_all()


def make_outdir(label: str):
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    dirpath = OUTPUT_SETTING["spectrum_image_path"] / label
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath / label)
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


def make_capture_param(
    num_delay_sample: int,
    num_integration_section: int,
    num_capture_samples: Sequence[int],
    num_blank_samples: Sequence[int],
):
    capprm = CaptureParam()

    if num_delay_sample % 4 != 0:
        raise ValueError(f"num_delay_sample (= {num_delay_sample} is not multiple of 4.")
    capprm.capture_delay = num_delay_sample // 4

    capprm.num_integ_sections = num_integration_section
    for idx in range(len(num_capture_samples)):
        if num_capture_samples[idx] % 4 != 0:
            raise ValueError(f"num_capture_samples[{idx}] (= {num_capture_samples[idx]}) is not multiple of 4")
        if num_blank_samples[idx] % 4 != 0:
            raise ValueError(f"num_blank_samples[{idx}] (= {num_blank_samples[idx]}) is not multiple of 4")
        capprm.add_sum_section(
            num_words=num_capture_samples[idx] // 4, num_post_blank_words=num_blank_samples[idx] // 4
        )

    return capprm


@pytest.mark.parametrize(
    ("port_subport", "channel", "lo_mhz", "cnco_mhz_tx", "cnco_mhz_rx", "fnco_mhz_tx", "fnco_mhz_rx", "sideband"),
    [
        (1, 0, 8500, 1500, 1500, 0, 0, "L"),
        (1, 0, 8500, 1500, 1500, -150, 0, "L"),
    ],
)
def test_read_loopback(
    port_subport: Union[int, Tuple[int, int]],
    channel: int,
    lo_mhz: int,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    sideband: str,
    fixtures,
):
    outdir, box = fixtures

    port, subport, port_name = decode_port_subport(port_subport)
    num_samples = 65536 * 16 * 4

    wave_param = make_pulses_wave_param(
        num_delay_sample=0,
        num_global_repeat=1,
        num_wave_samples=(64,),
        num_blank_samples=(0,),
        num_repeats=(0xFFFFFFFF,),
        amplitudes=(32767.0,),
    )
    capture_param = make_capture_param(
        num_delay_sample=0,
        num_integration_section=1,
        num_capture_samples=(num_samples,),
        num_blank_samples=(4,),
    )

    box.config_port(
        port=port,
        subport=0,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz_tx * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband=sideband,
    )
    box.config_channel(
        port=port,
        subport=0,
        channel=channel,
        fnco_freq=fnco_mhz_tx * 1e6,
        wave_param=wave_param,
    )

    box.config_port(port=0, cnco_freq=cnco_mhz_rx * 1e6, rfswitch="loop")
    box.config_runit(port=0, runit=0, fnco_freq=fnco_mhz_rx * 1e6, capture_param=capture_param)
    fut = box.capture_start(port=0, runits=(0,), triggering_channel=(port_subport, channel))
    box.start_emission(channels=((port_subport, channel),))
    retcode, capdata = fut.result()
    box.stop_emission(channels=((port_subport, channel),))

    assert retcode == CaptureReturnCode.SUCCESS
    assert 0 in capdata
    x = capdata[0]
    assert len(x) == num_samples

    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq = ((cnco_mhz_tx - cnco_mhz_rx) + (fnco_mhz_tx - fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_samples}")

    plt.cla()
    plt.plot(f, p / num_samples)
    plt.savefig(
        outdir / f"{port_name}-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / len(x) >= 2000.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 1000.0
