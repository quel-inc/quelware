import logging
import time
from typing import Any, Final, Mapping

import matplotlib
import numpy as np

from testlibs.general_looptest_common import calc_angle, init_units, plot_iqs
from testlibs.updated_linkup_phase2 import Quel1WaveGen
from testlibs.updated_linkup_phase3 import Quel1WaveCap

logger = logging.getLogger(__name__)

DEVICE_SETTINGS: Mapping[str, Mapping[str, Any]] = {
    "BOX0": {
        "ipaddr_wss": "10.1.0.42",
        "ipaddr_sss": "10.2.0.42",
        "ipaddr_css": "10.5.0.42",
        "boxtype": "quel1-a",
        "mxfe": "both",
        "config_root": "settings",
        "config_options": ["use_read_in_mxfe0", "use_read_in_mxfe1"],
    },
    "BOX1": {
        "ipaddr_wss": "10.1.0.58",
        "ipaddr_sss": "10.2.0.58",
        "ipaddr_css": "10.5.0.58",
        "boxtype": "quel1-a",
        "mxfe": "both",
        "config_root": "settings",
    },
    "SENDER0": {
        "box": "BOX0",
        "mxfe": 1,
        "dac": 2,
        "vatt": 0x2C0,
    },
    "SENDER1": {
        "box": "BOX1",
        "mxfe": 1,
        "dac": 2,
        "vatt": 0x600,
    },
    "SENDER2": {
        "box": "BOX1",
        "mxfe": 0,
        "dac": 3,
        "vatt": 0x300,
    },
    "CAPTURER": {
        "box": "BOX0",
        "mxfe": 1,
        "input_port": "read_in",
    },
    "COMMON": {
        "lo_mhz": 12000,
        "cnco_mhz": 2000.0,
        "fnco_mhz": 0.0,
        "sideband": "L",
        "amplitude": 6000.0,
    },
}


def capture_cw(wg: Quel1WaveGen, dac_idx: int, amplitude: int, cp: Quel1WaveCap):
    wg.run(line=dac_idx, awg_idx=0, amplitude=amplitude, a=True, b=False, c=False)
    cp.run("r", enable_internal_loop=False, a=False, b=False, c=False)
    captured = cp.complete()
    wg.stop()
    if captured is None:
        # actually, c is not None when 'd' of cp.run() is True.
        raise RuntimeError("failed to capture data")
    return captured[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    matplotlib.use("Qt5Agg")

    N: Final[int] = 5
    common_ = DEVICE_SETTINGS["COMMON"]

    wgs_, (cp_, c_silent_) = init_units(["SENDER0", "SENDER1", "SENDER2"], DEVICE_SETTINGS)
    wg0_, dac_idx0_ = wgs_["SENDER0"]
    wg1_, dac_idx1_ = wgs_["SENDER1"]
    wg2_, dac_idx2_ = wgs_["SENDER2"]

    # loopback with CW start and stop.
    c0 = []
    c1 = []
    c2 = []
    for i in range(N):
        if i != 0:
            logger.info("wating 10sec")
            time.sleep(10)
        c0.append(capture_cw(wg0_, dac_idx0_, common_["amplitude"], cp_))
        c1.append(capture_cw(wg1_, dac_idx1_, common_["amplitude"], cp_))
        c2.append(capture_cw(wg2_, dac_idx2_, common_["amplitude"], cp_))

    logger.info(f"no_signal:power = {np.mean(abs(c_silent_)):.1f} +/- {np.sqrt(np.var(abs(c_silent_))):.1f}")
    logger.info(f"wg0:power = {np.mean(abs(c0[0])):.1f} +/- {np.sqrt(np.var(abs(c0[0]))):.1f}")
    logger.info(f"wg1:power = {np.mean(abs(c1[0])):.1f} +/- {np.sqrt(np.var(abs(c1[0]))):.1f}")
    logger.info(f"wg2:power = {np.mean(abs(c2[0])):.1f} +/- {np.sqrt(np.var(abs(c2[0]))):.1f}")

    for i, cl in enumerate((c0, c1, c2)):
        for j, c in enumerate(cl):
            wg_angle_avg, wg_angle_sd = calc_angle(c)
            logger.info(f"wg{i}:c{j}:angle = {wg_angle_avg:.2f}deg +/- {wg_angle_sd:.2f}deg")

    plot_iqs({"no signal": c_silent_, "s0": c0[0], "s1": c1[0], "s2": c2[0]})
