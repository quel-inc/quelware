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
    ("group", "line", "channel", "lo_mhz", "cnco_mhz_tx", "cnco_mhz_rx", "fnco_mhz_tx", "fnco_mhz_rx", "sideband"),
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
    group: int,
    line: int,
    channel: int,
    lo_mhz: int,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    sideband: str,
    fixtures1,
):
    box, params, topdirpath = fixtures1
    if params["label"] not in {"staging-074"}:
        pytest.skip()
    boxi = box._dev
    outdir = make_outdir(topdirpath / "monitor_loopbck")

    task = boxi_gen_cw(
        boxi=boxi,
        group=group,
        line=line,
        channel=channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband=sideband,
        via_monitor=False,
    )

    num_capture_sample = 65536 * 4

    x = boxi_easy_capture(
        boxi=boxi,
        group=group,
        rline="m",
        runit=0,
        lo_freq=lo_mhz * 1e6 if line != 1 else None,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_capture_sample=num_capture_sample,
    )

    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    assert (x is not None) and (len(x) == num_capture_sample)
    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq = ((cnco_mhz_tx - cnco_mhz_rx) + (fnco_mhz_tx - fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_capture_sample}")

    plt.cla()
    plt.plot(f, p / num_capture_sample)
    plt.savefig(
        outdir / f"monitor-mxfe{group:d}-line{line:d}-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / num_capture_sample >= 2000.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / num_capture_sample >= 1000.0


@pytest.mark.parametrize(
    ("group", "line", "channel", "lo_mhz", "cnco_mhz_tx", "cnco_mhz_rx", "fnco_mhz_tx", "fnco_mhz_rx", "sideband"),
    [
        (0, 0, 0, 11500, 1500, 1500, 0, 0, "L"),
        (0, 0, 0, 11500, 1500, 1500, 10, 0, "L"),
        (0, 0, 0, 11500, 1500, 1500, -150, 0, "L"),
        (0, 0, 0, 11500, 1500, 1600, 0, 0, "L"),
        (0, 0, 0, 11500, 1500, 1500, 0, 50, "L"),
        (1, 0, 0, 11500, 1500, 1500, 0, 0, "L"),
        (1, 0, 0, 11500, 1500, 1500, 100, 0, "L"),
        (1, 0, 0, 11500, 1500, 1500, -20, 0, "L"),
        (1, 0, 0, 11500, 1500, 1700, 0, 0, "L"),
        (1, 0, 0, 11500, 1500, 1500, 0, -50, "L"),
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
    fixtures1,
):
    box, params, topdirpath = fixtures1
    if params["label"] not in {"staging-074"}:
        pytest.skip()
    boxi = box._dev
    outdir = make_outdir(topdirpath / "read_loopbck")

    task = boxi_gen_cw(
        boxi=boxi,
        group=group,
        line=line,
        channel=channel,
        lo_freq=lo_mhz * 1e6,
        cnco_freq=cnco_mhz_tx * 1e6,
        fnco_freq=fnco_mhz_tx * 1e6,
        fullscale_current=40527,
        vatt=0xA00,
        sideband=sideband,
        via_monitor=False,
    )

    num_capture_sample = 65536 * 16 * 4

    x = boxi_easy_capture(
        boxi=boxi,
        group=group,
        rline="r",
        runit=0,
        lo_freq=None,
        cnco_freq=cnco_mhz_rx * 1e6,
        fnco_freq=fnco_mhz_rx * 1e6,
        activate_internal_loop=True,
        num_capture_sample=num_capture_sample,
    )

    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    assert (x is not None) and (len(x) == num_capture_sample)

    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq = ((cnco_mhz_tx - cnco_mhz_rx) + (fnco_mhz_tx - fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/num_capture_sample}")

    plt.cla()
    plt.plot(f, p / num_capture_sample)
    plt.savefig(
        outdir / f"read-mxfe{group:d}-line0-ch{channel:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / len(x) >= 2000.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 1000.0
