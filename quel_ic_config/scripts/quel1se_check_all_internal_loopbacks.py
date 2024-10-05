import argparse
import logging
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict, Mapping, Set

import matplotlib

from quel_ic_config import CaptureReturnCode, Quel1BoxType
from quel_ic_config_utils.common_arguments import complete_ipaddrs
from testlibs.general_looptest_common_updated import (
    BoxPool,
    PulseCap,
    PulseGen,
    VportSettingType,
    find_chunks,
    plot_iqs,
)

logger = logging.getLogger()


def check_background_noise(cp_0: PulseCap, power_thr: float):
    noise_max, noise_avg, _ = cp_0.measure_background_noise()
    if noise_max > power_thr * 0.75:
        logger.warning(
            f"the input port-#{cp_0.port:02d} of the box {cp_0.box.wss._wss_addr} is too noise for the given power "
            "threshold of pulse detection, you may see sprious pulses in the results"
        )


def single_schedule(cp: PulseCap, pg_trigger: PulseGen, pgs: Set[PulseGen], boxpool: BoxPool, power_thr: float):
    if pg_trigger not in pgs:
        raise ValueError("trigerring pulse generator is not included in activated pulse generators")
    thunk = cp.capture_at_single_trigger_of(pg=pg_trigger)
    boxpool.emit_at(cp=cp, pgs=pgs, min_time_offset=125_000_000, time_counts=(0,))

    s0, iqs = thunk.result()
    iq = iqs[0]
    assert s0 == CaptureReturnCode.SUCCESS
    chunks = find_chunks(iq, power_thr=power_thr)
    return iq, chunks


def test_loopback(cp: PulseCap, pg_trigger: PulseGen, pgs: Set[PulseGen], boxpool: BoxPool, power_thr: float):
    check_background_noise(cp, power_thr)
    iq, chunks = single_schedule(cp, pg_trigger, pgs, boxpool, power_thr)
    if len(chunks) != len(pgs):
        logger.error(
            f"the number of pulses captured by the port-#{cp.port:02d} of the box {cp.box.wss._wss_addr} is "
            f"expected to be {len(pgs)} but is actually {len(chunks)}, something wrong"
        )
    return iq, chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    for lgrname, lgr in logging.root.manager.loggerDict.items():
        if lgrname in {"root"}:
            pass
        elif lgrname.startswith("testlibs."):
            pass
        else:
            if isinstance(lgr, logging.Logger):
                lgr.setLevel(logging.WARNING)

    matplotlib.use("Gtk3Agg")

    parser = argparse.ArgumentParser(
        "observing the output of port 1, 2, 3, 6, 7, 8, and 9 via internal loop-back paths"
    )
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
    parser.add_argument(
        "--png",
        action="store_true",
        help="write graphs into a png file",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="don't show graphs on screen",
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

    CAP_VPORT_SETTINGS: Dict[str, Mapping[str, VportSettingType]] = {
        "READ": {
            "create": {
                "boxname": "BOX0",
                "port": 0,  # (0, "r")
                "runits": {0},
            },
            "config": {
                "lo_freq": 8e9,
                "cnco_freq": 1e9,
                "fnco_freq": 0.0,
                "rfswitch": "loop",
            },
            "simple_parameters": {
                0: {
                    "num_delay_sample": 0,
                    "num_integration_section": 1,
                    "num_capture_samples": [3072],
                    "num_blank_samples": [4],
                },
            },
        },
        "MON0": {
            "create": {
                "boxname": "BOX0",
                "port": 4,  # (0, "m")
                "runits": {0},
            },
            "config": {
                "lo_freq": 5e9,
                "cnco_freq": 1e9,
                "fnco_freq": 0.0,
                "rfswitch": "loop",
            },
            "simple_parameters": {
                0: {
                    "num_delay_sample": 0,
                    "num_integration_section": 1,
                    "num_capture_samples": [3072],
                    "num_blank_samples": [4],
                },
            },
        },
        "MON1": {
            "create": {
                "boxname": "BOX0",
                "port": 10,  # (1, "m")
                "runits": {0},
            },
            "config": {
                "lo_freq": 5e9,
                "cnco_freq": 1e9,
                "fnco_freq": 0.0,
                "rfswitch": "loop",
            },
            "simple_parameters": {
                0: {
                    "num_delay_sample": 0,
                    "num_integration_section": 1,
                    "num_capture_samples": [3072],
                    "num_blank_samples": [4],
                },
            },
        },
    }

    GEN_VPORT_SETTINGS: Dict[str, Mapping[str, VportSettingType]] = {
        "GEN00": {
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
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (1, 1),
                "num_wait_samples": (0, 0),
            },
        },
        "GEN01": {
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
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (1, 1),
                "num_wait_samples": (0, 0),
            },
        },
        "GEN02": {
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
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 128,
                "num_repeats": (1, 1),
                "num_wait_samples": (512, 0),
            },
        },
        "GEN03": {
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
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (1, 1),
                "num_wait_samples": (1024, 0),
            },
        },
        "GEN06": {
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
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (1, 1),
                "num_wait_samples": (0, 0),
            },
        },
        "GEN07": {
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
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 128,
                "num_repeats": (1, 1),
                "num_wait_samples": (512, 0),
            },
        },
        "GEN08": {
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
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (1, 1),
                "num_wait_samples": (1024, 0),
            },
        },
        "GEN09": {
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
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 128,
                "num_repeats": (1, 1),
                "num_wait_samples": (1536, 0),
            },
        },
    }

    boxpool0 = BoxPool(DEVICE_SETTINGS)
    boxpool0.init(resync=False)
    pgs0 = PulseGen.create(GEN_VPORT_SETTINGS, boxpool0)
    cps0 = PulseCap.create(CAP_VPORT_SETTINGS, boxpool0)

    boxpool0.measure_timediff(cps0["MON1"])

    iq0, chunk0 = test_loopback(
        cps0["READ"], pgs0["GEN00"], {pgs0[f"GEN{idx:02d}"] for idx in (0,)}, boxpool0, power_thr=2000
    )

    iq1, chunk1 = test_loopback(
        cps0["MON0"], pgs0["GEN01"], {pgs0[f"GEN{idx:02d}"] for idx in (1, 2, 3)}, boxpool0, power_thr=200
    )

    iq2, chunk2 = test_loopback(
        cps0["MON1"], pgs0["GEN06"], {pgs0[f"GEN{idx:02d}"] for idx in (6, 7, 8, 9)}, boxpool0, power_thr=200
    )

    plot_iqs(
        {
            "read-loop: port-#01": iq0,
            "monitor0-loop: port-#01, port-#02, port-#03": iq1,
            "monitor1-loop: port-#06, port-#07, port-#08, port-#09": iq2,
        },
        same_range=False,
        show_graph=not args.headless,
        output_filename=Path(str(args.ipaddr_wss) + ".png") if args.png else None,
    )

    del pgs0
    del cps0
    del boxpool0
