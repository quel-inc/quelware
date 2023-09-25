import logging
from pathlib import Path
from typing import Any, Collection, Dict, Final, Mapping, Tuple

import matplotlib
import numpy as np
from quel_clock_master import QuBEMasterClient, SequencerClient

from quel_ic_config import Quel1BoxType, Quel1ConfigOption
from testlibs.general_looptest_common import calc_angle, init_units, plot_iqs
from testlibs.updated_linkup_phase2 import Quel1WaveGen
from testlibs.updated_linkup_phase3 import Quel1WaveCap

logger = logging.getLogger(__name__)

DEVICE_SETTINGS: Mapping[str, Mapping[str, Any]] = {
    "CLOCK_MASTER": {
        "ipaddr": "10.3.0.13",
        "reset": True,
    },
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


# TODO: will be moved to general_looptest_common.py
def init_scheduler(
    senders: Collection[str], settings: Mapping[str, Mapping[str, Any]]
) -> Tuple[QuBEMasterClient, Dict[str, SequencerClient]]:
    cm = QuBEMasterClient(settings["CLOCK_MASTER"]["ipaddr"])
    cm.reset()  # TODO: confirm whether it is harmless or not.

    sqs: Dict[str, SequencerClient] = {}
    for sender in senders:
        box = settings[sender]["box"]
        if box not in sqs:
            bx_ipaddr = settings[box]["ipaddr_sss"]
            sqs[box] = SequencerClient(bx_ipaddr)
            # TODO: calling kick_softreset() here diminish the function. investigate the reason.

    cm.kick_clock_synch([sq.ipaddress for name, sq in sqs.items()])
    return cm, sqs


def check_clocks(cm: QuBEMasterClient, sqs: Dict[str, SequencerClient]) -> bool:
    valid_m, cntr_m = cm.read_clock()

    t = {}
    for name, sq in sqs.items():
        t[name] = sq.read_clock()

    flag = True
    if valid_m:
        logger.info(f"master: {cntr_m:d}")
    else:
        flag = False
        logger.info("master: not found")

    for name, (valid, cntr) in t.items():
        if valid:
            logger.info(f"{name:s}: {cntr:d}")
        else:
            flag = False
            logger.info(f"{name:s}: not found")

    return flag


def capture_cw(
    wg: Quel1WaveGen,
    dac_idx: int,
    amplitude: int,
    cp: Quel1WaveCap,
    num_words: int,
    sq: SequencerClient,
    num_iter: int,
):
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
    wg.run(
        line=dac_idx, awg_idx=0, amplitude=amplitude, a=True, b=False, c=False, num_repeats=(16, 1), use_schedule=True
    )
    valid_read, current_time = sq.read_clock()
    if valid_read:
        logger.info(f"current time is {current_time}")
    else:
        raise RuntimeError("failed to read current clock")

    # scheduling at once
    for i in range(num_iter):
        valid_sched = sq.add_sequencer(current_time + 125000000 * (i * 10 + 1))
        if not valid_sched:
            raise RuntimeError("failed to schedule AWG start")
    logger.info("scheduling completed")

    # capture
    captured_list = []
    for i in range(num_iter):
        captured = cp.complete(y=False, timeout=15.0)  # notes: keep reserving CPUNs.
        logger.info(f"capture completed ({i+1}/{num_iter})")
        wg.stop(y=False, z=False, terminate=False)  # notes: keep reserving AWGs and don't close RF switches
        if captured is None:
            raise RuntimeError("failed to capture data")
        captured_list.append(captured[0])
        # Note: it turns out that capture unit starts repeatedly if it is not initialized.
        #       this looks different from an example in qube_master repository although I haven't tested it yet.
        #       the order of wait_for_stop_awg() and wait_for_stop_captureunit() doesn't matter.
        # TODO: this looks tedious. elaborate APIs for the ease of use.
        if i != num_iter - 1:
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

    cp.complete(x=False, z=False)  # release active capture units
    wg.stop(x=False)  # release active AWGs and close RF switches
    return captured_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    matplotlib.use("Qt5Agg")

    N: Final[int] = 5
    NUM_WORDS_TO_CAPTURE: Final[int] = 512
    common_ = DEVICE_SETTINGS["COMMON"]

    wgs_, (cp_, c_silent_) = init_units(["SENDER0"], DEVICE_SETTINGS)
    wg0_, dac_idx0_ = wgs_["SENDER0"]

    cm_, sqs_ = init_scheduler(["SENDER0"], DEVICE_SETTINGS)
    check_clocks(cm_, sqs_)
    sqs0_ = sqs_["BOX0"]

    # loopback with CW start and stop.
    c0 = capture_cw(wg0_, dac_idx0_, common_["amplitude"], cp_, NUM_WORDS_TO_CAPTURE, sqs0_, N)

    logger.info(f"no_signal:power = {np.mean(abs(c_silent_)):.1f} +/- {np.sqrt(np.var(abs(c_silent_))):.1f}")
    logger.info(f"wg0:power = {np.mean(abs(c0[0][768:1280])):.1f} +/- {np.sqrt(np.var(abs(c0[0][768:1280]))):.1f}")

    for i, c0_i in enumerate(c0):
        wg_angle_avg, wg_angle_sd = calc_angle(c0_i[768:1280])
        logger.info(f"wg0:c{i}:angle = {wg_angle_avg:.2f}deg +/- {wg_angle_sd:.2f}deg")

    to_plot = {"no signal": c_silent_}
    for i, c0_i in enumerate(c0):
        to_plot[f"s0_{i}"] = c0_i
    plot_iqs(to_plot)
