import logging
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from quel_ic_config.quel1_box import Quel1BoxIntrinsic
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = (
    {
        "label": "staging-094",
        "box_config": {
            "ipaddr_wss": "10.1.0.94",
            "ipaddr_sss": "10.2.0.94",
            "ipaddr_css": "10.5.0.94",
            "boxtype": Quel1BoxType.fromstr("quel1se-riken8"),
            "config_root": None,
            "config_options": [],
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "linkup": False,
    },
)

OUTPUT_SETTING = {
    "spectrum_image_path": Path("./artifacts/loopback_readin"),
}


@pytest.fixture(scope="module", params=DEVICE_SETTINGS)
def fixtures(request):
    param0 = request.param

    box = Quel1BoxIntrinsic.create(**param0["box_config"])
    if request.param["linkup"]:
        linkstatus = box.relinkup(**param0["linkup_config"])
    else:
        linkstatus = box.reconnect()
    assert linkstatus[0]
    assert linkstatus[1]

    yield make_outdir(param0["label"]), box

    box.easy_stop_all()
    del box


def make_outdir(label: str):
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    dirpath = OUTPUT_SETTING["spectrum_image_path"] / label
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    return dirpath


@pytest.mark.parametrize(
    ("mxfe", "line", "channel", "lo_mhz", "cnco_mhz_tx", "cnco_mhz_rx", "fnco_mhz_tx", "fnco_mhz_rx", "sideband"),
    [
        (0, 0, 0, 8500, 1500, 1500, 0, 0, "L"),
        (0, 0, 0, 8500, 1500, 1500, 10, 0, "L"),
        (0, 0, 0, 8500, 1500, 1500, -150, 0, "L"),
        (0, 0, 0, 8500, 1500, 1600, 0, 0, "L"),
        (0, 0, 0, 8500, 1500, 1500, 0, 50, "L"),
    ],
)
def test_read_loopback(
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
        fullscale_current=40527,
        vatt=0xA00,
        sideband=sideband,
        amplitude=32767.0,
    )

    num_samples = 65536 * 16 * 4
    x = box.easy_capture(
        group=mxfe,
        rline="r",
        runit=0,
        lo_freq=None,
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
        outdir / f"mxfe{mxfe:d}-line0-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / len(x) >= 2000.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 1000.0
