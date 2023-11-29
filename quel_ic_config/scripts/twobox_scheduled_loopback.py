import logging
from typing import Any, Mapping

from quel_ic_config import Quel1BoxType, Quel1ConfigOption
from quel_ic_config_utils import CaptureReturnCode
from testlibs.general_looptest_common import BoxPool, PulseGen, find_chunks, init_pulsecap, init_pulsegen, plot_iqs

COMMON_SETTINGS: Mapping[str, Any] = {
    "lo_freq": 11500e6,
    "cnco_freq": 1500e6,
    "fnco_freq": 0.0,
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
    "BOX1": {
        "ipaddr_wss": "10.1.0.58",
        "ipaddr_sss": "10.2.0.58",
        "ipaddr_css": "10.5.0.58",
        "boxtype": Quel1BoxType.QuEL1_TypeA,
        "config_root": None,
        "config_options": [],
    },
    "CAPTURER": {
        "box": "BOX0",
        "group": 0,
        "rline": "r",
        "background_noise_threshold": 256.0,
    },
    "SENDER0": {
        "box": "BOX0",
        "group": 1,
        "line": 2,
        "vatt": 0xA00,
        "wave_samples": 64,
        "num_repeats": (1, 1),
        "num_wait_words": (0, 0),
    },
    "SENDER1": {
        "box": "BOX1",
        "group": 1,
        "line": 2,
        "vatt": 0xA00,
        "wave_samples": 128,
        "num_repeats": (1, 1),
        "num_wait_words": (64, 0),
    },
}


def multiple_schedule(num_iters: int = 3):
    global boxpool, pgs, cp

    # TODO: can cause intractable wrong behavior if host-side software process fails to get in touch with the firmware
    #       process. need to design firmware more carefully.
    pg0: PulseGen = pgs["SENDER0"]
    pg1: PulseGen = pgs["SENDER1"]
    results = cp.capture_at_multiple_triggers_of(pg=pg0, num_iters=num_iters, num_samples=1024, delay=0)
    schedule = boxpool.emit_at(
        cp=cp,
        pgs={pg0, pg1},
        min_time_offset=125_000_000,
        time_counts=[i * 125_000_000 for i in range(num_iters)],
    )

    iq_list = []
    chunks_list = []
    for idx, (status, iqs) in enumerate(results):
        assert status == CaptureReturnCode.SUCCESS
        iq = iqs[0]
        iq_list.append(iq)
        chunks_list.append(find_chunks(iq))
    return iq_list, chunks_list, schedule


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Qt5agg")
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    boxpool = BoxPool(DEVICE_SETTINGS)
    boxpool.init(resync=True)
    pgs = init_pulsegen(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)
    cp = init_pulsecap(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)

    cp.check_noise(show_graph=False)
    boxpool.measure_timediff(cp)

    iq_list, chunks_list, schedule = multiple_schedule(3)
    plot_iqs({idx: iq for idx, iq in enumerate(iq_list)})
