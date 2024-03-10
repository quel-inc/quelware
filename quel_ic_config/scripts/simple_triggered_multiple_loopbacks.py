import logging
from typing import Any, Collection, Mapping

from quel_ic_config import CaptureReturnCode, Quel1BoxType
from testlibs.general_looptest_common import (
    BoxPool,
    PulseCap,
    PulseGen,
    find_chunks,
    init_pulsecap,
    init_pulsegen,
    plot_iqs,
)

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
        "config_options": [],
    },
    "CAPTURER0": {
        "box": "BOX0",
        "group": 0,
        "rline": "r",
        "background_noise_threshold": 250.0,
        "gain": 1.0,
    },
    "CAPTURER1": {
        "box": "BOX0",
        "group": 0,
        "rline": "m",
        "background_noise_threshold": 250.0,
        "gain": 1.0,
    },
    "CAPTURER2": {
        "box": "BOX0",
        "group": 1,
        "rline": "r",
        "background_noise_threshold": 250.0,
        "gain": 1.0,
    },
    "CAPTURER3": {
        "box": "BOX0",
        "group": 1,
        "rline": "m",
        "background_noise_threshold": 250.0,
        "gain": 1.0,
    },
    "SENDER0": {
        "box": "BOX0",
        "group": 0,
        "line": 0,
        "vatt": 0xA00,
        "wave_samples": 64,
        "num_repeats": (2, 1),
        "num_wait_words": (0, 0),
    },
    "SENDER1": {
        "box": "BOX0",
        "group": 1,
        "line": 0,
        "vatt": 0xA00,
        "wave_samples": 64,
        "num_repeats": (2, 1),
        "num_wait_words": (0, 0),
    },
}


# this puts blanks between chunks
def simple_trigger_1(pg0: PulseGen, cps_: Collection[PulseCap], num_wait_word1: int = 0):
    pg0.num_wait_words = (pg0.num_wait_words[0], num_wait_word1)
    pg0.init_wave()
    thunks = {}
    gains = {}
    for cp in cps_:
        thunks[cp.name] = cp.capture_at_single_trigger_of(pg=pg0, num_samples=1024, delay=0)
        gains[cp.name] = cp.gain

    pg0.emit_now()
    results = {}
    for name, thunk in thunks.items():
        s0, iq = thunk.result()
        iq0 = iq[0] * gains[name]
        assert s0 == CaptureReturnCode.SUCCESS
        chunks = find_chunks(iq0, power_thr=200.0)
        results[name] = {
            "iq0": iq0,
            "chunks": chunks,
        }
    return results


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Qt5agg")
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    boxpool = BoxPool(DEVICE_SETTINGS)
    boxpool.init(resync=True)
    pgs = init_pulsegen(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)
    cps = init_pulsecap(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)

    _: Any
    _, box, _ = boxpool.get_box("BOX0")
    box.activate_read_loop(0)
    box.activate_monitor_loop(0)
    box.activate_read_loop(1)
    box.activate_monitor_loop(1)

    for name, cp in cps.items():
        cp.check_noise(show_graph=False)

    results = simple_trigger_1(pgs["SENDER1"], {cps["CAPTURER2"], cps["CAPTURER3"]}, 20)
    iqd = {name: result["iq0"] for name, result in results.items()}
    plot_iqs(iqd)

    results = simple_trigger_1(pgs["SENDER0"], {cps["CAPTURER0"], cps["CAPTURER1"]}, 20)
    iqd = {name: result["iq0"] for name, result in results.items()}
    plot_iqs(iqd)
