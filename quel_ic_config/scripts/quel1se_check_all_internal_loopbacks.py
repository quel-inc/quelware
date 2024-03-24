import argparse
import logging
from ipaddress import ip_address
from typing import Any, Dict, Mapping, Set

import matplotlib

from quel_ic_config import CaptureReturnCode, Quel1BoxType
from quel_ic_config_utils.common_arguments import complete_ipaddrs
from testlibs.general_looptest_common_updated import (
    BoxPool,
    PulseCap,
    PulseGen,
    create_pulsecap,
    create_pulsegen,
    find_chunks,
    plot_iqs,
)

logger = logging.getLogger()


def simple_trigger(cp1: PulseCap, pg1: PulseGen):
    thunk = cp1.capture_at_single_trigger_of(pg=pg1, num_samples=1024, delay_samples=0)
    pg1.emit_now()
    s0, iq = thunk.result()
    iq0 = iq[0]
    assert s0 == CaptureReturnCode.SUCCESS
    chunks = find_chunks(iq0, power_thr=200)
    return iq0, chunks


def single_schedule(cp: PulseCap, pg_trigger: PulseGen, pgs: Set[PulseGen], boxpool: BoxPool, power_thr: float):
    if pg_trigger not in pgs:
        raise ValueError("trigerring pulse generator is not included in activated pulse generators")
    thunk = cp.capture_at_single_trigger_of(pg=pg_trigger, num_samples=2048, delay_samples=0)
    boxpool.emit_at(cp=cp, pgs=pgs, min_time_offset=125_000_000, time_counts=(0,))

    s0, iqs = thunk.result()
    iq0 = iqs[0]
    assert s0 == CaptureReturnCode.SUCCESS
    chunks = find_chunks(iq0, power_thr=power_thr)
    return iq0, chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    matplotlib.use("Qt5agg")

    parser = argparse.ArgumentParser("observing the output of port 6, 7, 8, 9 via internal monitor loop")
    parser.add_argument(
        "--ipaddr_wss",
        type=ip_address,
        required=True,
        help="IP address of the wave generation/capture subsystem of the target box",
    )
    parser.add_argument(
        "--ipaddr_sss",
        type=ip_address,
        default=0,
        help="IP address of the synchronization subsystem of the target box",
    )
    parser.add_argument(
        "--ipaddr_css",
        type=ip_address,
        default=0,
        help="IP address of the configuration subsystem of the target box",
    )
    parser.add_argument(
        "--ipaddr_clk",
        type=ip_address,
        required=True,
        help="IP address of clock master",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)

    DEVICE_SETTINGS: Dict[str, Mapping[str, Any]] = {
        "CLOCK_MASTER": {
            "ipaddr": str(args.ipaddr_clk),
            "reset": True,
        },
        "BOX0": {
            "ipaddr_wss": str(args.ipaddr_wss),
            "ipaddr_sss": str(args.ipaddr_sss),
            "ipaddr_css": str(args.ipaddr_css),
            "boxtype": Quel1BoxType.QuEL1SE_RIKEN8,
            "config_root": None,
            "config_options": [],
            "ignore_crc_error_of_mxfe": {0, 1},
        },
    }

    VPORT_SETTINGS: Dict[str, Mapping[str, Mapping[str, Any]]] = {
        "CAPTURER_READ": {
            "create": {
                "boxname": "BOX0",
                "port": 0,  # (0, "r")
                "runits": {0},
                "background_noise_threshold": 200.0,
            },
            "config": {
                "lo_freq": 8e9,
                "cnco_freq": 1e9,
                "fnco_freq": 0.0,
                "rfswitch": "loop",
            },
        },
        "CAPTURER_MON0": {
            "create": {
                "boxname": "BOX0",
                "port": 4,  # (0, "m")
                "runits": {0},
                "background_noise_threshold": 200.0,
            },
            "config": {
                "lo_freq": 5e9,
                "cnco_freq": 1e9,
                "fnco_freq": 0.0,
                "rfswitch": "loop",
            },
        },
        "CAPTURER_MON1": {
            "create": {
                "boxname": "BOX0",
                "port": 10,  # (1, "m")
                "runits": {0},
                "background_noise_threshold": 200.0,
            },
            "config": {
                "lo_freq": 5e9,
                "cnco_freq": 1e9,
                "fnco_freq": 0.0,
                "rfswitch": "loop",
            },
        },
        "SENDER00": {
            "create": {
                "boxname": "BOX0",
                "port": 1,
                "channel": 0,
            },
            "config": {
                "lo_freq": 8e9,
                "cnco_freq": 1e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
                "sideband": "L",
                "vatt": 0xC00,
            },
            "cw": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (2, 1),
                "num_wait_samples": (0, 80),
            },
        },
        "SENDER01": {
            "create": {
                "boxname": "BOX0",
                "port": (1, 1),
                "channel": 0,
            },
            "config": {
                "cnco_freq": 4e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
            },
            "cw": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (2, 1),
                "num_wait_samples": (0, 80),
            },
        },
        "SENDER02": {
            "create": {
                "boxname": "BOX0",
                "port": 2,
                "channel": 0,
            },
            "config": {
                "lo_freq": 8e9,
                "cnco_freq": 2e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
                "sideband": "L",
                "vatt": 0xC00,
            },
            "cw": {
                "amplitude": 32767.0,
                "num_wave_sample": 128,
                "num_repeats": (2, 1),
                "num_wait_samples": (256, 80),
            },
        },
        "SENDER03": {
            "create": {
                "boxname": "BOX0",
                "port": 3,
                "channel": 0,
            },
            "config": {
                "cnco_freq": 4e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
            },
            "cw": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (2, 1),
                "num_wait_samples": (768, 80),
            },
        },
        "SENDER06": {
            "create": {
                "boxname": "BOX0",
                "port": 6,
                "channel": 0,
            },
            "config": {
                "cnco_freq": 4e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
            },
            "cw": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (2, 1),
                "num_wait_samples": (0, 80),
            },
        },
        "SENDER07": {
            "create": {
                "boxname": "BOX0",
                "port": 7,
                "channel": 0,
            },
            "config": {
                "cnco_freq": 4e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
            },
            "cw": {
                "amplitude": 32767.0,
                "num_wave_sample": 128,
                "num_repeats": (2, 1),
                "num_wait_samples": (256, 80),
            },
        },
        "SENDER08": {
            "create": {
                "boxname": "BOX0",
                "port": 8,
                "channel": 0,
            },
            "config": {
                "cnco_freq": 4e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
            },
            "cw": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (2, 1),
                "num_wait_samples": (768, 80),
            },
        },
        "SENDER09": {
            "create": {
                "boxname": "BOX0",
                "port": 9,
                "channel": 0,
            },
            "config": {
                "cnco_freq": 4e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
            },
            "cw": {
                "amplitude": 32767.0,
                "num_wave_sample": 128,
                "num_repeats": (2, 1),
                "num_wait_samples": (1024, 80),
            },
        },
    }

    boxpool0 = BoxPool(DEVICE_SETTINGS)
    boxpool0.init(resync=True)
    pgs0 = create_pulsegen(VPORT_SETTINGS, boxpool0)
    cps0 = create_pulsecap(VPORT_SETTINGS, boxpool0)
    cp_read = cps0["CAPTURER_READ"]
    cp_mon0 = cps0["CAPTURER_MON0"]
    cp_mon1 = cps0["CAPTURER_MON1"]

    boxpool0.measure_timediff(cp_mon1)

    # Notes: close loop before checking the noise
    box0, sqc0 = boxpool0.get_box("BOX0")
    box0.config_rfswitch(port=0, rfswitch="loop")  # TODO: capturer should control its loop switch
    box0.config_rfswitch(port=4, rfswitch="loop")
    box0.config_rfswitch(port=10, rfswitch="loop")
    cp_read.check_noise(show_graph=False)
    cp_mon0.check_noise(show_graph=False)
    cp_mon1.check_noise(show_graph=False)

    # Notes: do not open the loop for this script
    iqs0 = single_schedule(
        cp_read, pgs0["SENDER00"], {pgs0[f"SENDER{idx:02d}"] for idx in (0,)}, boxpool0, power_thr=2000
    )
    plot_iqs({"read-loop: port-#01": iqs0[0]})

    iqs1 = single_schedule(
        cp_mon0, pgs0["SENDER01"], {pgs0[f"SENDER{idx:02d}"] for idx in (1, 2, 3)}, boxpool0, power_thr=200
    )
    plot_iqs({"monitor0-loop: port-#01, port-#02, port-#03": iqs1[0]})

    iqs2 = single_schedule(
        cp_mon1, pgs0["SENDER06"], {pgs0[f"SENDER{idx:02d}"] for idx in (6, 7, 8, 9)}, boxpool0, power_thr=200
    )
    plot_iqs({"monitor1-loop: port-#06, port-#07, port-#08, port-#09": iqs2[0]})
