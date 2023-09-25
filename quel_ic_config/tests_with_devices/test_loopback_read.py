import logging
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from testlibs.basic_scan_common import Quel1BoxType, Quel1ConfigOption, init_box

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = (
    {
        "ipaddr_wss": "10.1.0.42",
        "ipaddr_sss": "10.2.0.42",
        "ipaddr_css": "10.5.0.42",
        "boxtype": Quel1BoxType.fromstr("quel1-a"),
        "config_root": Path("settings"),
        "config_options": [Quel1ConfigOption.USE_READ_IN_MXFE0, Quel1ConfigOption.USE_READ_IN_MXFE1],
        "mxfe_combination": "both",
    },
)


OUTPUT_SETTING = {
    "spectrum_image_path": "./artifacts/loopback_readin",
}


@pytest.fixture(scope="session", params=DEVICE_SETTINGS)
def css_p2_p3(request):
    param0 = request.param

    linkup0, linkup1, _, wss, _, css_p2_g0, css_p2_g1, css_p3_g0, css_p3_g1, _ = init_box(**param0)
    assert linkup0
    assert linkup1
    assert css_p2_g0 is not None
    assert css_p2_g1 is not None
    assert css_p3_g0 is not None
    assert css_p3_g1 is not None

    # max_noise = measure_floor_noise(e4405b)
    # assert max_noise < MAX_BACKGROUND_NOISE
    yield (css_p2_g0, css_p3_g0), (css_p2_g1, css_p3_g1)

    css_p2_g0.stop()
    css_p2_g1.stop()
    if len(css_p3_g0.active_units) != 0:
        css_p3_g0.complete()
        assert False
    if len(css_p3_g1.active_units) != 0:
        css_p3_g1.complete()
        assert False


@pytest.fixture(scope="session")
def outdir(request):
    mpl.use("Qt5Agg")  # TODO: reconsider where to execute.

    dirname = OUTPUT_SETTING["spectrum_image_path"]
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    dpath = Path(dirname)
    os.makedirs(dpath)
    return dpath


@pytest.mark.parametrize(
    ("mxfe", "awg_idx", "lo_mhz", "cnco_mhz_tx", "cnco_mhz_rx", "fnco_mhz_tx", "fnco_mhz_rx", "sideband"),
    [
        (0, 0, 11500, 1500, 1500, 0, 0, "L"),
        (0, 0, 11500, 1500, 1500, 10, 0, "L"),
        (0, 0, 11500, 1500, 1500, -150, 0, "L"),
        (0, 0, 11500, 1500, 1600, 0, 0, "L"),
        (0, 0, 11500, 1500, 1500, 0, 50, "L"),
        (1, 0, 11500, 1500, 1500, 0, 0, "L"),
        (1, 0, 11500, 1500, 1500, 100, 0, "L"),
        (1, 0, 11500, 1500, 1500, -20, 0, "L"),
        (1, 0, 11500, 1500, 1700, 0, 0, "L"),
        (1, 0, 11500, 1500, 1500, 0, -50, "L"),
    ],
)
def test_read_loopback(
    mxfe: int,
    awg_idx: int,
    lo_mhz: int,
    cnco_mhz_tx: int,
    cnco_mhz_rx: int,
    fnco_mhz_tx: int,
    fnco_mhz_rx: int,
    sideband: str,
    css_p2_p3,
    outdir,
):
    mxfe_g, mxfe_c = css_p2_p3[mxfe]
    mxfe_g.run(0, awg_idx, lo_mhz=lo_mhz, cnco_mhz=cnco_mhz_tx, fnco_mhz=fnco_mhz_tx, sideband=sideband)
    mxfe_c.run(
        "r",
        num_words=65536 * 16,
        active_units={0},
        enable_internal_loop=True,
        lo_mhz=None,
        cnco_mhz=cnco_mhz_rx,
        fnco_mhz=fnco_mhz_rx,
    )
    captured = mxfe_c.complete()
    mxfe_g.stop()

    assert captured is not None
    x = list(captured.values())[0]
    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    expected_freq = ((cnco_mhz_tx - cnco_mhz_rx) + (fnco_mhz_tx - fnco_mhz_rx)) * 1e6

    logger.info(f"freq error = {f[max_idx] - expected_freq}Hz")
    logger.info(f"power = {p[max_idx]/len(x)}")

    plt.cla()
    plt.plot(f, p / len(x))
    plt.savefig(
        outdir / f"mxfe{mxfe:d}-line0-ch{awg_idx:d}-cnco{cnco_mhz_tx:d}_{cnco_mhz_rx:d}"
        f"-fnco{fnco_mhz_tx:d}_{fnco_mhz_rx:d}.png"
    )

    assert abs(f[max_idx] - expected_freq) < abs(f[1] - f[0])
    if fnco_mhz_rx == 0:
        assert p[max_idx] / len(x) >= 2000.0
    else:
        # TODO: investigate why the amplitude of the received signal is smaller when fnco_rx != 0.
        assert p[max_idx] / len(x) >= 1000.0
