import logging
import os
from concurrent.futures import CancelledError
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from testlibs.easy_capture import boxi_easy_capture
from testlibs.gen_cw import boxi_gen_cw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def make_outdir(dirpath: Path):

    os.makedirs(dirpath, exist_ok=True)
    return dirpath


def _test_monitor_loopback_rcp(
    group: int,
    line: int,
    channel: int,
    lo_mhz_tx: int,
    lo_mhz_rx: int,
    d_ratio_rx: int,
    sideband: str,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    fixtures,
):
    box, param, topdirpath = fixtures
    boxi = box._dev
    outdir = make_outdir(topdirpath / "monitor_loopack_rcp")

    task = boxi_gen_cw(
        boxi=boxi,
        group=group,
        line=line,
        channel=channel,
        lo_freq=lo_mhz_tx * 1e6,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        fullscale_current=40527,
        sideband=sideband,
        vatt=0xC00,
        via_monitor=False,
    )

    num_samples = 65536 * 16 * 4
    x = boxi_easy_capture(
        boxi=boxi,
        group=group,
        rline="m",
        runit=0,
        lo_freq=lo_mhz_rx * 1e6,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_capture_sample=num_samples,
    )

    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    assert (x is not None) and (len(x) == num_samples)

    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    if sideband == "L":
        expected_freq_in = abs((lo_mhz_tx - (cnco_mhz_tx + fnco_mhz_tx)) - lo_mhz_rx)
    elif sideband == "U":
        expected_freq_in = abs((lo_mhz_tx + (cnco_mhz_tx + fnco_mhz_tx)) - lo_mhz_rx)
    else:
        assert False

    expected_freq = (expected_freq_in - (cnco_mhz_rx + fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_samples}")

    plt.cla()
    plt.plot(f, p / num_samples)
    plt.savefig(
        outdir / f"mxfe{group:d}-line{line:d}-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        # loopback signal is weaker than quel1-a and quel1-b
        assert p[max_idx] / len(x) >= 1200.0
    else:
        # loopback signal is weaker than quel1-a and quel1-b
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 700.0


@pytest.mark.parametrize(
    (
        "group",
        "line",
        "channel",
        "lo_mhz_tx",
        "lo_mhz_rx",
        "d_ratio_rx",
        "sideband",
        "cnco_mhz_tx",
        "cnco_mhz_rx",
        "fnco_mhz_tx",
        "fnco_mhz_rx",
    ),
    [
        # port-#1
        (0, 0, 0, 8500, 8500, 1, "U", 1500, 1500, 0, 0),
        (0, 0, 0, 8000, 8000, 1, "U", 2300, 2300, 0, 0),
        (0, 0, 0, 8000, 8000, 1, "U", 2000, 2000, 0, 100),
        # port-#3
        (0, 1, 0, 8500, 8500, 1, "U", 1500, 1500, 0, 0),
        (0, 1, 0, 8000, 8000, 1, "U", 2400, 2400, 0, 0),
        (0, 1, 0, 8000, 8000, 1, "U", 2000, 2000, 100, 0),
        # port-#2
        (0, 2, 0, 11000, 8000, 1, "L", 1500, 1500, 0, 0),
        (0, 2, 0, 11500, 8000, 1, "L", 2200, 1300, 0, 0),
        (0, 2, 0, 11500, 8000, 1, "L", 2200, 1300, 0, -100),
        # port-#4
        (0, 3, 0, 11000, 8500, 1, "L", 1500, 1000, 0, 0),
        (0, 3, 0, 11500, 8000, 1, "L", 2200, 1300, 0, 0),
        (0, 3, 0, 11500, 8000, 1, "L", 2200, 1300, -100, 0),
        # port-#8
        (1, 0, 0, 8500, 8500, 1, "U", 1500, 1500, 0, 0),
        (1, 0, 0, 8000, 8000, 1, "U", 2300, 2300, 0, 0),
        (1, 0, 0, 8000, 8000, 1, "U", 2000, 2000, 0, 100),
        # port-#10
        (1, 1, 0, 8500, 8500, 1, "U", 1500, 1500, 0, 0),
        (1, 1, 0, 8000, 8000, 1, "U", 2400, 2400, 0, 0),
        (1, 1, 0, 8000, 8000, 1, "U", 2000, 2000, 100, 0),
        # port-#9
        (1, 2, 0, 11000, 8000, 1, "L", 1500, 1500, 0, 0),
        (1, 2, 0, 11500, 8000, 1, "L", 2200, 1300, 0, 0),
        (1, 2, 0, 11500, 8000, 1, "L", 2200, 1300, 0, -100),
        # port-#11
        (1, 3, 0, 11000, 8500, 1, "L", 1500, 1000, 0, 0),
        (1, 3, 0, 11500, 8000, 1, "L", 2200, 1300, 0, 0),
        (1, 3, 0, 11500, 8000, 1, "L", 2200, 1300, -100, 0),
    ],
)
def test_typea_monitor_loopback_rcp(
    group: int,
    line: int,
    channel: int,
    lo_mhz_tx: int,
    lo_mhz_rx: int,
    d_ratio_rx: int,
    sideband: str,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    fixtures11a,
):
    _test_monitor_loopback_rcp(
        group,
        line,
        channel,
        lo_mhz_tx,
        lo_mhz_rx,
        d_ratio_rx,
        sideband,
        cnco_mhz_tx,
        cnco_mhz_rx,
        fnco_mhz_tx,
        fnco_mhz_rx,
        fixtures11a,
    )


@pytest.mark.parametrize(
    (
        "group",
        "line",
        "channel",
        "lo_mhz_tx",
        "lo_mhz_rx",
        "d_ratio_rx",
        "sideband",
        "cnco_mhz_tx",
        "cnco_mhz_rx",
        "fnco_mhz_tx",
        "fnco_mhz_rx",
    ),
    [
        # port-#1
        (0, 0, 0, 11500, 8500, 1, "L", 1500, 1500, 0, 0),
        (0, 0, 0, 11600, 8000, 1, "L", 2300, 1300, 0, 0),
        (0, 0, 0, 11400, 8000, 1, "L", 2000, 1300, 0, 100),
        # port-#3 (be aware that DAC and ADC shares the same LO)
        (0, 1, 0, 11500, 11500, 1, "L", 1500, 1500, 0, 0),
        (0, 1, 0, 11000, 11000, 1, "L", 2300, 2300, 100, 100),
        (0, 1, 0, 11400, 11400, 1, "L", 2000, 2200, 100, -100),
        # port-#2
        (0, 2, 0, 11000, 8000, 1, "L", 1500, 1500, 0, 0),
        (0, 2, 0, 11500, 8000, 1, "L", 2200, 1300, 0, 0),
        (0, 2, 0, 11500, 8000, 1, "L", 2200, 1300, 0, -100),
        # port-#4
        (0, 3, 0, 11000, 8500, 1, "L", 1500, 1000, 0, 0),
        (0, 3, 0, 11500, 8000, 1, "L", 2200, 1300, 0, 0),
        (0, 3, 0, 11500, 8000, 1, "L", 2200, 1300, -100, 0),
        # port-#8
        (1, 0, 0, 11500, 8500, 1, "L", 1500, 1500, 0, 0),
        (1, 0, 0, 11600, 8000, 1, "L", 2300, 1300, 0, 0),
        (1, 0, 0, 11400, 8000, 1, "L", 2000, 1300, 0, 100),
        # port-#10 (be aware that DAC and ADC shares the same LO)
        (1, 1, 0, 11500, 11500, 1, "L", 1600, 1600, 0, 0),
        (1, 1, 0, 11000, 11000, 1, "L", 2000, 2000, -100, -100),
        (1, 1, 0, 11400, 11400, 1, "L", 2500, 2300, -100, 100),
        # port-#9
        (1, 2, 0, 11000, 8000, 1, "L", 1500, 1500, 0, 0),
        (1, 2, 0, 11500, 8000, 1, "L", 2200, 1300, 0, 0),
        (1, 2, 0, 11500, 8000, 1, "L", 2200, 1300, 0, -100),
        # port-#11
        (1, 3, 0, 11000, 8500, 1, "L", 1500, 1000, 0, 0),
        (1, 3, 0, 11500, 8000, 1, "L", 2200, 1300, 0, 0),
        (1, 3, 0, 11500, 8000, 1, "L", 2200, 1300, -100, 0),
    ],
)
def test_typeb_monitor_loopback_rcp(
    group: int,
    line: int,
    channel: int,
    lo_mhz_tx: int,
    lo_mhz_rx: int,
    d_ratio_rx: int,
    sideband: str,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    fixtures11b,
):
    _test_monitor_loopback_rcp(
        group,
        line,
        channel,
        lo_mhz_tx,
        lo_mhz_rx,
        d_ratio_rx,
        sideband,
        cnco_mhz_tx,
        cnco_mhz_rx,
        fnco_mhz_tx,
        fnco_mhz_rx,
        fixtures11b,
    )
