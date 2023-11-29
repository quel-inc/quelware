import json
import logging
import time
from typing import Any, Dict, Mapping

import numpy as np
import numpy.typing as npt

from quel_ic_config import Quel1BoxType, Quel1ConfigOption
from quel_ic_config_utils import CaptureReturnCode
from testlibs.general_looptest_common import BoxPool, find_chunks, init_pulsecap, init_pulsegen

logger = logging.getLogger()


COMMON_SETTINGS: Mapping[str, Any] = {
    "lo_freq": 11500e6,
    "cnco_freq": 1500e6,
    "fnco_freq": 0.0,
    "sideband": "L",
    "amplitude": 16383.0,
}

DEVICE_SETTINGS: Dict[str, Mapping[str, Any]] = {
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
    "BOX2": {
        "ipaddr_wss": "10.1.0.60",
        "ipaddr_sss": "10.2.0.60",
        "ipaddr_css": "10.5.0.60",
        "boxtype": Quel1BoxType.QuEL1_TypeB,
        "config_root": None,
        "config_options": [],
    },
    "CAPTURER": {
        "box": "BOX0",
        "group": 0,
        "rline": "r",
        "background_noise_threshold": 256.0,
    },
}

wavelen = 51200
for k in range(2):
    for i in range(2):
        for j in range(2):
            m = k * 4 + i * 2 + j
            DEVICE_SETTINGS[f"SENDER{m}"] = {
                "box": f"BOX{k}",
                "group": i,
                "line": 2 + j,
                "vatt": 0xA00,
                "wave_samples": wavelen,
                "num_repeats": (1, 1),
                "num_wait_words": (wavelen // 2 * m, 0),
            }


for i in range(2):
    for j in range(4):
        m = 8 + i * 4 + j
        DEVICE_SETTINGS[f"SENDER{m}"] = {
            "box": "BOX2",
            "group": i,
            "line": j,
            "vatt": 0xA00,
            "wave_samples": wavelen,
            "num_repeats": (1, 1),
            "num_wait_words": (wavelen // 2 * m, 0),
        }


def multiple_schedule(num_iters: int = 1):
    global boxpool, pgs, cp

    # TODO: can cause intractable wrong behavior if host-side software process fails to get in touch with the firmware
    #       process. need to design firmware more carefully.
    results = cp.capture_at_multiple_triggers_of(pg=pgs["SENDER0"], num_iters=num_iters, num_samples=1600000, delay=0)
    schedule = boxpool.emit_at(
        cp=cp,
        pgs={pgs[f"SENDER{i}"] for i in range(16)},
        min_time_offset=1_250_000,
        time_counts=[i * 125_000_000 * 2 for i in range(num_iters)],
    )

    iq_list = []
    chunks_list = []
    for idx, (status, iqs) in enumerate(results):
        assert status == CaptureReturnCode.SUCCESS
        iq = iqs[0]
        iq_list.append(iq)
        chunks_list.append(find_chunks(iq))
    return iq_list, chunks_list, schedule


# notes 10kHz ~ 50000samples at 500MSps
def phase_stat(iq: npt.NDArray[np.complex64], num_samples=50000) -> Dict[str, float]:
    if len(iq) < num_samples:
        logger.info(f"processing {len(iq)} samples")
    else:
        iq = iq[:num_samples]

    pwr = np.abs(iq)
    angle: npt.NDArray[np.float64] = np.angle(iq)
    # Notes: angle changes from 3.14 --> -3.14 suddenly.
    if max(angle) >= 3.0 and min(angle) < -3.0:
        angle = (angle + 2 * np.pi) % (2 * np.pi)

    pwr_mean = np.mean(pwr)
    pwr_std = np.sqrt(np.var(pwr))
    agl_mean = float(np.mean(angle)) * 180.0 / np.pi
    agl_std = np.sqrt(float(np.var(angle))) * 180.0 / np.pi
    agl_deltamax = (np.max(angle) - np.min(angle)) * 180.0 / np.pi
    return {
        "pwr_mean": np.floor(pwr_mean * 1000 + 0.5) / 1000.0,
        "pwr_std": np.floor(pwr_std * 1000 + 0.5) / 1000.0,
        "agl_mean": np.floor(agl_mean * 1000 + 0.5) / 1000.0,
        "agl_std": np.floor(agl_std * 1000 + 0.5) / 1000.0,
        "agl_deltamax": np.floor(agl_deltamax * 1000 + 0.5) / 1000.0,
    }


if __name__ == "__main__":
    from argparse import ArgumentParser

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = ArgumentParser(description="a tool for observing the relative phase among three boxes")
    parser.add_argument("--duration", type=int, default=15, help="duration of measurement in second")
    args = parser.parse_args()

    boxpool = BoxPool(DEVICE_SETTINGS)
    boxpool.init(resync=False)
    pgs = init_pulsegen(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)
    cp = init_pulsecap(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)

    cp.check_noise(show_graph=False)
    boxpool.measure_timediff(cp)

    t_complete = time.perf_counter() + int(args.duration)
    while time.perf_counter() < t_complete:
        iq_list, chunks_list, schedule = multiple_schedule(1)
        iq = iq_list[0]
        chunk_list = chunks_list[0]

        for j, (s0, s1) in enumerate(chunk_list):
            stat: Dict[str, Any] = {"line": j, "time": schedule[f"{pgs[f'SENDER{j}'].boxname}"][0]}
            stat.update(phase_stat(iq[s0 + 1000 : s0 + 51000]))  # noqa: E203
            logger.info(json.dumps(stat))
