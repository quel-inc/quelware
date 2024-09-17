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
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    os.makedirs(dirpath, exist_ok=True)
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
    fixtures8,
):
    box, param, topdirpath = fixtures8
    boxi = box._dev
    outdir = make_outdir(topdirpath / "monitor_loopback_cf")

    task = boxi_gen_cw(
        boxi,
        group,
        line,
        channel,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        fullscale_current=40527,
        via_monitor=False,
    )

    num_sample = 65536 * 16 * 4

    x = boxi_easy_capture(
        boxi,
        group,
        "m",
        runit=0,
        lo_freq=lo_mhz_rx * 1e6,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_capture_sample=num_sample,
    )

    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    assert (x is not None) and (len(x) == num_sample)

    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq_in = abs(cnco_mhz_tx + fnco_mhz_tx - lo_mhz_rx)
    expected_freq = (expected_freq_in - (cnco_mhz_rx + fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_sample}")

    plt.cla()
    plt.plot(f, p / num_sample)
    plt.savefig(
        outdir / f"group{group:d}-line{line:d}-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
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
    fixtures8,
):
    box, param, topdirpath = fixtures8
    boxi = box._dev
    outdir = make_outdir(topdirpath / "monitor_loopback_rp")

    task = boxi_gen_cw(
        boxi,
        group,
        line,
        channel,
        lo_freq=lo_mhz_tx * 1e6,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        fullscale_current=40527,
        vatt=0xC00,
        sideband="L",
        via_monitor=False,
    )

    num_sample = 65536 * 16 * 4
    x = boxi_easy_capture(
        boxi,
        group,
        "m",
        runit=0,
        lo_freq=lo_mhz_rx * 1e6,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_capture_sample=num_sample,
    )

    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    assert (x is not None) and (len(x) == num_sample)

    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq_in = abs((lo_mhz_tx - (cnco_mhz_tx + fnco_mhz_tx)) - lo_mhz_rx)
    expected_freq = (expected_freq_in - (cnco_mhz_rx + fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_sample}")

    plt.cla()
    plt.plot(f, p / num_sample)
    plt.savefig(
        outdir / f"group{group:d}-line{line:d}-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / len(x) >= 230.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 150.0


@pytest.mark.parametrize(
    ("group", "line", "channel", "lo_mhz", "cnco_mhz_tx", "cnco_mhz_rx", "fnco_mhz_tx", "fnco_mhz_rx", "sideband"),
    [
        (0, 0, 0, 8500, 1500, 1500, 0, 0, "L"),
        (0, 0, 0, 8500, 1500, 1500, 10, 0, "L"),
        (0, 0, 0, 8500, 1500, 1500, -150, 0, "L"),
        (0, 0, 0, 8500, 1500, 1600, 0, 0, "L"),
        (0, 0, 0, 8500, 1500, 1500, 0, 50, "L"),
    ],
)
def test_read_loopback(
    group: int,
    line: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    sideband: str,
    fixtures8,
):
    box, param, topdirpath = fixtures8
    boxi = box._dev
    outdir = make_outdir(topdirpath / "read_loopback")

    task = boxi_gen_cw(
        boxi,
        group,
        line,
        channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband=sideband,
        via_monitor=False,
    )

    num_sample = 65536 * 16 * 4
    x = boxi_easy_capture(
        boxi,
        group=group,
        rline="r",
        runit=0,
        lo_freq=None,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_capture_sample=num_sample,
    )

    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    assert (x is not None) and (len(x) == num_sample)

    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq = ((cnco_mhz_tx - cnco_mhz_rx) + (fnco_mhz_tx - fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_sample}")

    plt.cla()
    plt.plot(f, p / num_sample)
    plt.savefig(
        outdir / f"group{group:d}-line0-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / len(x) >= 2000.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 1000.0
