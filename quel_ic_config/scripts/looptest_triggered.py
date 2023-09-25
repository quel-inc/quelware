import logging
import time
from pathlib import Path
from typing import Any, Final, Mapping

import matplotlib
import numpy as np

from quel_ic_config import Quel1BoxType, Quel1ConfigOption
from testlibs.general_looptest_common import calc_angle, init_units, plot_iqs
from testlibs.updated_linkup_phase2 import Quel1WaveGen
from testlibs.updated_linkup_phase3 import Quel1WaveCap

logger = logging.getLogger(__name__)

DEVICE_SETTINGS: Mapping[str, Mapping[str, Any]] = {
    "BOX0": {
        "ipaddr_wss": "10.1.0.42",
        "ipaddr_sss": "10.2.0.42",
        "ipaddr_css": "10.5.0.42",
        "boxtype": Quel1BoxType.QuEL1_TypeA,
        "mxfe_combination": "both",
        "config_root": Path("settings"),
        "config_options": [Quel1ConfigOption.USE_READ_IN_MXFE0, Quel1ConfigOption.USE_READ_IN_MXFE1],
    },
    "SENDER0": {
        "box": "BOX0",
        "mxfe": 1,
        "dac": 2,
        "vatt": 0x2C0,
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


def capture_cw(wg: Quel1WaveGen, dac_idx: int, amplitude: int, cp: Quel1WaveCap, num_words: int):
    # Notes: the order of method invocation is not interchangeable due to the thread-unsafe design of e7awg_sw.
    # TODO: it should be re-considered ASAP because it requires too much effort to developers and users.
    cp.run(
        "r",
        enable_internal_loop=False,
        num_words=num_words,
        delay=0,
        triggering_awg_unit=(wg, dac_idx, 0),
        a=False,
        b=False,
        c=False,
    )
    wg.run(line=dac_idx, awg_idx=0, amplitude=amplitude, a=True, b=False, c=False)
    captured = cp.complete()
    wg.stop()
    if captured is None:
        raise RuntimeError("failed to capture data")
    return captured[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    matplotlib.use("Qt5Agg")

    N: Final[int] = 5
    NUM_WORDS_TO_CAPTURE: Final[int] = 512
    common_ = DEVICE_SETTINGS["COMMON"]

    wgs_, (cp_, c_silent_) = init_units(["SENDER0"], DEVICE_SETTINGS)
    wg0_, dac_idx0_ = wgs_["SENDER0"]

    c0 = []
    for i in range(N):
        if i != 0:
            logger.info("wating 10sec")
            time.sleep(10)
        c0.append(capture_cw(wg0_, dac_idx0_, common_["amplitude"], cp_, NUM_WORDS_TO_CAPTURE))

    logger.info(f"no_signal:power = {np.mean(abs(c_silent_)):.1f} +/- {np.sqrt(np.var(abs(c_silent_))):.1f}")
    logger.info(f"wg0:power = {np.mean(abs(c0[0])):.1f} +/- {np.sqrt(np.var(abs(c0[0]))):.1f}")

    for i, cl in enumerate((c0,)):
        for j, c in enumerate(cl):
            wg_angle_avg, wg_angle_sd = calc_angle(c[1024:])
            logger.info(f"wg{i}:c{j}:angle = {wg_angle_avg:.2f}deg +/- {wg_angle_sd:.2f}deg")

    to_plot = {"no signal": c_silent_}
    for i in range(N):
        to_plot[f"s0_{i}"] = c0[i]
    plot_iqs(to_plot)
