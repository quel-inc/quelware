import logging
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from testlibs.basic_scan_common import init_box

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = {
    "ipaddr_wss": "10.1.0.42",
    "ipaddr_sss": "10.2.0.42",
    "ipaddr_css": "10.5.0.42",
    "boxtype": "quel1-a",
    "config_root": "settings",
    "config_options": ["use_monitor_in_mxfe0", "use_monitor_in_mxfe1"],
    "mxfe": "both",
}


OUTPUT_SETTINGS = {
    "spectrum_image_path": "./artifacts/loopback_monitor",
}


@pytest.fixture(scope="session", params=(DEVICE_SETTINGS,))
def css_p2_p3(request):
    param0 = request.param

    linkup0, linkup1, css_p2_g0, css_p2_g1, css_p3_g0, css_p3_g1 = init_box(
        ipaddr_wss=param0["ipaddr_wss"],
        ipaddr_sss=param0["ipaddr_sss"],
        ipaddr_css=param0["ipaddr_css"],
        mxfe=param0["mxfe"],
        boxtype=param0["boxtype"],
        config_root=param0["config_root"],
        config_options=param0["config_options"],
    )
    assert linkup0
    assert linkup1

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


@pytest.fixture(scope="session", params=(OUTPUT_SETTINGS,))
def outdir(request):
    mpl.use("Qt5Agg")  # TODO: reconsider where to execute.

    dirname = request.param["spectrum_image_path"]
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    dpath = Path(dirname)
    os.makedirs(dpath)
    return dpath


@pytest.mark.parametrize(
    ("mxfe", "line", "awg_idx", "lo_mhz", "cnco_mhz", "fnco_mhz", "sideband"),
    [
        (0, 0, 0, 11500, 1500, 0, "L"),
        (0, 0, 0, 11500, 1500, 10, "L"),
        (0, 0, 0, 11500, 1500, -150, "L"),
        (0, 1, 0, 11500, 1500, 0, "L"),
        (0, 1, 0, 11500, 1500, 20, "L"),
        (0, 1, 0, 11500, 1500, -160, "L"),
        (0, 2, 0, 11500, 2000, 0, "L"),
        (0, 2, 1, 11500, 2000, 30, "L"),
        (0, 2, 2, 11500, 2000, -170, "L"),
        (0, 3, 0, 11500, 2000, 0, "L"),
        (0, 3, 1, 11500, 2000, 40, "L"),
        (0, 3, 2, 11500, 2000, -180, "L"),
        (1, 0, 0, 11500, 1500, 0, "L"),
        (1, 0, 0, 11500, 1500, 100, "L"),
        (1, 0, 0, 11500, 1500, -20, "L"),
        (1, 1, 0, 11500, 1500, 0, "L"),
        (1, 1, 0, 11500, 1500, 110, "L"),
        (1, 1, 0, 11500, 1500, -30, "L"),
        (1, 2, 0, 11500, 2000, 0, "L"),
        (1, 2, 1, 11500, 2000, 120, "L"),
        (1, 2, 2, 11500, 2000, -40, "L"),
        (1, 3, 0, 11500, 2000, 0, "L"),
        (1, 3, 1, 11500, 2000, 130, "L"),
        (1, 3, 2, 11500, 2000, -50, "L"),
    ],
)
def test_readin_loopback(
    mxfe: int, line: int, awg_idx: int, lo_mhz: int, cnco_mhz: int, fnco_mhz: int, sideband: str, css_p2_p3, outdir
):
    mxfe_g, mxfe_c = css_p2_p3[mxfe]
    mxfe_g.run(line, awg_idx, lo_mhz=lo_mhz, cnco_mhz=cnco_mhz, fnco_mhz=fnco_mhz, sideband=sideband)
    mxfe_c.run(
        "m",
        num_words=65536 * 16,
        active_units={0},
        enable_internal_loop=True,
        lo_mhz=lo_mhz if line != 1 else None,
        cnco_mhz=cnco_mhz,
        fnco_mhz=0,
    )
    captured = mxfe_c.complete()
    mxfe_g.stop()

    assert captured is not None
    x = list(captured.values())[0]
    p = abs(np.fft.fft(x))
    f = np.fft.fftfreq(len(p), 1.0 / 500e6)
    max_idx = np.argmax(p)

    logger.info(f"freq error = {f[max_idx] - fnco_mhz * 1e6}Hz")
    logger.info(f"power = {p[max_idx]/len(x)}")

    plt.cla()
    plt.plot(f, p / len(x))
    plt.savefig(outdir / f"mxfe{mxfe:d}-line{line:d}-ch{awg_idx:d}-fnco{fnco_mhz:d}.png")

    assert p[max_idx] / len(x) >= 2000.0
    assert abs(f[max_idx] - fnco_mhz * 1e6) < abs(f[1] - f[0])
