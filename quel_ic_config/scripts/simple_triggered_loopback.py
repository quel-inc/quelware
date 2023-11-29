import logging
from typing import Any, Mapping

from quel_ic_config import Quel1BoxType, Quel1ConfigOption
from quel_ic_config_utils import CaptureReturnCode
from testlibs.general_looptest_common import BoxPool, find_chunks, init_pulsecap, init_pulsegen, plot_iqs

COMMON_SETTINGS: Mapping[str, Any] = {
    "lo_freq": 11500e6,
    "cnco_freq": 1500e6,
    "fnco_freq": 0,
    "sideband": "L",
    "amplitude": 6000.0,
}

DEVICE_SETTINGS: Mapping[str, Mapping[str, Any]] = {
    "CLOCK_MASTER": {
        "ipaddr": "10.3.0.13",
        "reset": True,
    },
    "BOX0": {
        "ipaddr_wss": "10.1.0.74",
        "ipaddr_sss": "10.2.0.74",
        "ipaddr_css": "10.5.0.74",
        "boxtype": Quel1BoxType.QuEL1_TypeA,
        "config_root": None,
        "config_options": [Quel1ConfigOption.USE_READ_IN_MXFE0],
    },
    "CAPTURER": {
        "box": "BOX0",
        "group": 0,
        "rline": "r",
        "background_noise_threshold": 250.0,
    },
    "SENDER0": {
        "box": "BOX0",
        "group": 1,
        "line": 2,
        "vatt": 0xA00,
        "wave_samples": 64,
        "num_repeats": (2, 1),
        "num_wait_words": (0, 0),
    },
}


# this delays the start of the whole wave
def simple_trigger_0(num_wait_word0: int = 0):
    global pgs, cp

    pg0 = pgs["SENDER0"]
    pg0.num_wait_words = (num_wait_word0, pg0.num_wait_words[1])
    pg0.init_wave()
    thunk = cp.capture_at_single_trigger_of(pg=pg0, num_samples=1024, delay=0)
    pg0.emit_now()
    s0, iq = thunk.result()
    iq0 = iq[0]
    assert s0 == CaptureReturnCode.SUCCESS
    chunks = find_chunks(iq0)
    return iq0, chunks


# this puts blanks between chunks
def simple_trigger_1(num_wait_word1: int = 0):
    global pgs, cp

    pg0 = pgs["SENDER0"]
    pg0.num_wait_words = (pg0.num_wait_words[0], num_wait_word1)
    pg0.init_wave()
    thunk = cp.capture_at_single_trigger_of(pg=pg0, num_samples=1024, delay=0)
    pg0.emit_now()
    s0, iq = thunk.result()
    iq0 = iq[0]
    assert s0 == CaptureReturnCode.SUCCESS
    chunks = find_chunks(iq0)
    return iq0, chunks


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Qt5agg")
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    boxpool = BoxPool(DEVICE_SETTINGS)
    boxpool.init(resync=True)
    pgs = init_pulsegen(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)
    cp = init_pulsecap(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)

    cp.check_noise(show_graph=False)

    iq0, chunks = simple_trigger_1(20)
    plot_iqs({"sender0": iq0})
