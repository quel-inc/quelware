import logging
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from quel_ic_config_utils.simple_box import Quel1BoxType, Quel1ConfigOption, SimpleBoxIntrinsic, init_box_with_linkup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = (
    {
        "label": "staging-017",
        "config": {
            "ipaddr_wss": "10.1.0.17",
            "ipaddr_sss": "10.2.0.17",
            "ipaddr_css": "10.5.0.17",
            "boxtype": Quel1BoxType.fromstr("qube-riken-a"),
            "config_root": None,
            "config_options": [Quel1ConfigOption.USE_MONITOR_IN_MXFE0, Quel1ConfigOption.USE_MONITOR_IN_MXFE1],
            "mxfes_to_linkup": (0, 1),
            "use_204b": True,
        },
    },
    {
        "label": "staging-018",
        "config": {
            "ipaddr_wss": "10.1.0.18",
            "ipaddr_sss": "10.2.0.18",
            "ipaddr_css": "10.5.0.18",
            "boxtype": Quel1BoxType.fromstr("qube-riken-a"),
            "background_noise_threshold": 400.0,
            "config_root": None,
            "config_options": [Quel1ConfigOption.USE_MONITOR_IN_MXFE0, Quel1ConfigOption.USE_MONITOR_IN_MXFE1],
            "mxfes_to_linkup": (0, 1),
            "use_204b": True,
        },
    },
)


OUTPUT_SETTING = {
    "spectrum_image_path": Path("./artifacts/loopback_monitor"),
}


@pytest.fixture(scope="session", params=DEVICE_SETTINGS)
def fixtures(request):
    param0 = request.param

    linkstatus, _, _, _, _, box = init_box_with_linkup(**param0["config"], refer_by_port=False)
    assert linkstatus[0]
    assert linkstatus[1]
    assert isinstance(box, SimpleBoxIntrinsic)
    yield make_outdir(param0["label"]), box

    box.easy_stop_all()


def make_outdir(dirname: str) -> Path:
    mpl.use("Qt5Agg")  # TODO: reconsider where to execute.

    dirpath = OUTPUT_SETTING["spectrum_image_path"] / dirname
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    return dirpath


@pytest.mark.parametrize(
    ("mxfe", "line", "channel", "lo_mhz", "cnco_mhz_tx", "cnco_mhz_rx", "fnco_mhz_tx", "fnco_mhz_rx", "sideband"),
    [
        (0, 0, 0, 11500, 1500, 1500, 0, 0, "L"),
        (0, 0, 0, 11500, 1500, 1500, 10, 0, "L"),
        (0, 0, 0, 11500, 1500, 1500, -150, 0, "L"),
        (0, 0, 0, 11500, 1500, 1600, 0, 0, "L"),
        (0, 0, 0, 11500, 1500, 1500, 0, 50, "L"),
        (0, 1, 0, 11500, 1500, 1500, 0, 0, "L"),
        (0, 1, 0, 11500, 1500, 1500, 20, 0, "L"),
        (0, 1, 0, 11500, 1500, 1500, -160, 0, "L"),
        (0, 1, 0, 11500, 1500, 1400, 40, 0, "L"),
        (0, 2, 0, 11500, 2000, 2000, 0, 0, "L"),
        (0, 2, 1, 11500, 2000, 2000, 30, 0, "L"),
        (0, 2, 2, 11500, 2000, 2000, -170, 0, "L"),
        (0, 2, 2, 11500, 2000, 2000, 0, -60, "L"),
        (0, 3, 0, 11500, 2000, 2000, 0, 0, "L"),
        (0, 3, 1, 11500, 2000, 2000, 40, 0, "L"),
        (0, 3, 1, 11500, 2000, 2000, 0, -80, "L"),
        (0, 3, 2, 11500, 2000, 2000, -180, 0, "L"),
        (1, 0, 0, 11500, 1500, 1500, 0, 0, "L"),
        (1, 0, 0, 11500, 1500, 1500, 100, 0, "L"),
        (1, 0, 0, 11500, 1500, 1500, -20, 0, "L"),
        (1, 0, 0, 11500, 1500, 1650, 0, 0, "L"),
        (1, 0, 0, 11500, 1500, 1500, 0, 120, "L"),
        (1, 1, 0, 11500, 1500, 1500, 0, 0, "L"),
        (1, 1, 0, 11500, 1500, 1500, 110, 0, "L"),
        (1, 1, 0, 11500, 1500, 1500, -30, 0, "L"),
        (1, 1, 0, 11500, 1500, 1400, -30, 0, "L"),
        (1, 2, 0, 11500, 2000, 2000, 0, 0, "L"),
        (1, 2, 1, 11500, 2000, 2000, 120, 0, "L"),
        (1, 2, 2, 11500, 2000, 2000, -40, 0, "L"),
        (1, 2, 2, 11500, 2000, 2000, 0, -180, "L"),
        (1, 3, 0, 11500, 2000, 2000, 0, 0, "L"),
        (1, 3, 1, 11500, 2000, 2000, 130, 0, "L"),
        (1, 3, 1, 11500, 2000, 2000, 0, -195, "L"),
        (1, 3, 2, 11500, 2000, 2000, -50, 0, "L"),
    ],
)
def test_monitor_loopback(
    mxfe: int,
    line: int,
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

    box.easy_start_cw(
        group=mxfe,
        line=line,
        channel=channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        vatt=0xC00,
        sideband=sideband,
        amplitude=32767.0,
    )

    num_samples = 65536 * 4

    x = box.easy_capture(
        group=mxfe,
        rline="m",
        runit=0,
        lo_freq=lo_mhz * 1e6 if line != 1 else None,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_samples=num_samples,
    )

    box.easy_stop(group=mxfe, line=line, channel=channel)

    assert (x is not None) and (len(x) == num_samples)
    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq = ((cnco_mhz_tx - cnco_mhz_rx) + (fnco_mhz_tx - fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_samples}")

    plt.cla()
    plt.plot(f, p / num_samples)
    plt.savefig(
        outdir / f"mxfe{mxfe:d}-line{line:d}-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / num_samples >= 1000.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / num_samples >= 500.0
