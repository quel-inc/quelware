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
            "config_options": set(),
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "linkup": False,
    },
)

OUTPUT_SETTING = {
    "spectrum_image_path": Path("./artifacts/loopback_monitorin"),
}


@pytest.fixture(scope="session", params=DEVICE_SETTINGS)
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


def make_outdir(label: str):
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    dirpath = OUTPUT_SETTING["spectrum_image_path"] / label
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath / label)
    return dirpath


@pytest.mark.parametrize(
    ("group", "line", "channel", "lo_mhz_rx", "d_ratio_rx", "cnco_mhz_tx", "cnco_mhz_rx", "fnco_mhz_tx", "fnco_mhz_rx"),
    [
        (0, 1, 0, 3000, 4, 4000, 1000, 0, 0),
        (0, 1, 0, 5000, 2, 4000, 1000, 0, 0),
        (0, 3, 0, 3000, 4, 4100, 1100, 0, 0),
        (0, 3, 0, 5000, 2, 4100, 1000, 0, -100),
        (1, 0, 0, 3000, 4, 4000, 900, 0, 100),
        (1, 0, 0, 5000, 2, 4000, 1000, 0, 0),
        (1, 1, 0, 3000, 4, 4100, 1100, 0, 0),
        (1, 1, 0, 5000, 2, 4100, 900, 0, 0),
        (1, 2, 0, 3000, 4, 4000, 1000, 0, 0),
        (1, 2, 0, 5000, 2, 4000, 1000, 0, 0),
        (1, 3, 0, 3000, 4, 4100, 1100, 0, 0),
        (1, 3, 0, 5000, 2, 4100, 900, 0, 0),
    ],
)
def test_monitor_loopback_cf(
    group: int,
    line: int,
    channel: int,
    lo_mhz_rx: int,
    d_ratio_rx: int,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    fixtures,
):
    outdir, box = fixtures

    box.easy_start_cw(
        group=group,
        line=line,
        channel=channel,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        fullscale_current=40527,
        amplitude=32767.0,
    )

    num_samples = 65536 * 16 * 4
    x = box.easy_capture(
        group=group,
        rline="m",
        runit=0,
        lo_freq=lo_mhz_rx * 1e6,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_samples=num_samples,
    )

    box.easy_stop(group=group, line=line, channel=channel)

    assert (x is not None) and (len(x) == num_samples)

    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq_in = abs(cnco_mhz_tx + fnco_mhz_tx - lo_mhz_rx)
    expected_freq = (expected_freq_in - (cnco_mhz_rx + fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_samples}")

    plt.cla()
    plt.plot(f, p / num_samples)
    plt.savefig(
        outdir / f"mxfe{group:d}-line0-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / len(x) >= 500.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 300.0


@pytest.mark.parametrize(
    (
        "group",
        "line",
        "channel",
        "lo_mhz_tx",
        "lo_mhz_rx",
        "d_ratio_rx",
        "cnco_mhz_tx",
        "cnco_mhz_rx",
        "fnco_mhz_tx",
        "fnco_mhz_rx",
    ),
    [
        (0, 0, 0, 8500, 8500, 1, 1500, 1500, 0, 0),
        (0, 0, 0, 8500, 6000, 2, 1500, 1000, 0, 0),
        (0, 0, 0, 8000, 8000, 1, 2000, 2000, 0, 0),
        (0, 0, 0, 8500, 5500, 2, 1500, 1500, 0, 0),
        (0, 2, 0, 8500, 8500, 1, 1500, 1500, 0, 0),
        (0, 2, 0, 8500, 6000, 2, 1500, 1000, 0, 0),
        (0, 2, 0, 8000, 8000, 1, 2000, 2000, 0, 0),
        (0, 2, 0, 8500, 5500, 2, 1500, 1500, 0, 0),
    ],
)
def test_monitor_loopback_rp(
    group: int,
    line: int,
    channel: int,
    lo_mhz_tx: int,
    lo_mhz_rx: int,
    d_ratio_rx: int,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    fixtures,
):
    outdir, box = fixtures

    box.easy_start_cw(
        group=group,
        line=line,
        channel=channel,
        lo_freq=lo_mhz_tx * 1e6,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        fullscale_current=40527,
        amplitude=32767.0,
        sideband="L",
        vatt=0xC00,
    )

    num_samples = 65536 * 16 * 4
    x = box.easy_capture(
        group=group,
        rline="m",
        runit=0,
        lo_freq=lo_mhz_rx * 1e6,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_samples=num_samples,
    )

    box.easy_stop(group=group, line=line, channel=channel)

    assert (x is not None) and (len(x) == num_samples)

    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq_in = abs((lo_mhz_tx - (cnco_mhz_tx + fnco_mhz_tx)) - lo_mhz_rx)
    expected_freq = (expected_freq_in - (cnco_mhz_rx + fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_samples}")

    plt.cla()
    plt.plot(f, p / num_samples)
    plt.savefig(
        outdir / f"mxfe{group:d}-line0-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / len(x) >= 230.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 150.0