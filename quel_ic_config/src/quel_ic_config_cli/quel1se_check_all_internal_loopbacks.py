import argparse
import logging
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from quel_ic_config import Quel1BoxType
from quel_ic_config_utils import BoxPool, BoxSettingType, VportSettingType, complete_ipaddrs, plot_iqs, single_schedule

logger = logging.getLogger()


def test_loopback(cp: str, pgs: set[str], boxpool: BoxPool, power_thr: dict[str, float]):
    # check_background_noise(cp, power_thr)
    _, iq, chunks = single_schedule(cp, pgs, boxpool, power_thr[cp])
    if len(chunks) != len(pgs):
        logger.error(
            f"the number of pulses captured by the runit '{cp}` is "
            f"expected to be {len(pgs)} but is actually {len(chunks)}, something wrong"
        )
    return iq, chunks


def main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    for lgrname, lgr in logging.root.manager.loggerDict.items():
        if lgrname in {"root"}:
            pass
        elif lgrname.startswith("quel_ic_config_utils.simple_multibox_framework"):
            pass
        else:
            if isinstance(lgr, logging.Logger):
                lgr.setLevel(logging.WARNING)

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

    CLOCKMASTER_SETTINGS: dict[str, Any] = {}

    BOX_SETTINGS: dict[str, BoxSettingType] = {
        "BOX0": {
            "ipaddr_wss": str(args.ipaddr_wss),
            "ipaddr_sss": str(args.ipaddr_sss),
            "ipaddr_css": str(args.ipaddr_css),
            "boxtype": Quel1BoxType.QuEL1SE_RIKEN8,
            "ignore_crc_error_of_mxfe": {0, 1},
        },
    }

    CAP_VPORT_SETTINGS: dict[str, dict[str, VportSettingType]] = {
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

    GEN_VPORT_SETTINGS: dict[str, dict[str, VportSettingType]] = {
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

    boxpool0 = BoxPool(CLOCKMASTER_SETTINGS, BOX_SETTINGS, CAP_VPORT_SETTINGS, GEN_VPORT_SETTINGS)
    boxpool0.initialize(allow_resync=False)

    bgnoise = boxpool0.check_background_noise({"READ", "MON0", "MON1"})
    bgnoise_thr = {
        "READ": 2000.0,
        "MON0": 200.0,
        "MON1": 200.0,
    }
    for runit_name in bgnoise:
        if bgnoise[runit_name] > bgnoise_thr[runit_name] * 0.75:
            logger.warning(
                f"the runit {runit_name} is too noisy for the given power threshold of "
                f"pulse detection (= {bgnoise_thr[runit_name]}), you may see spurious pulses in the results"
            )

    boxpool0.measure_timediff("MON1")

    iq0, chunk0 = test_loopback("READ", {f"GEN{idx:02d}" for idx in (0,)}, boxpool0, bgnoise_thr)
    iq1, chunk1 = test_loopback("MON0", {f"GEN{idx:02d}" for idx in (1, 2, 3)}, boxpool0, bgnoise_thr)
    iq2, chunk2 = test_loopback("MON1", {f"GEN{idx:02d}" for idx in (6, 7, 8, 9)}, boxpool0, bgnoise_thr)

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

    del boxpool0


if __name__ == "__main__":
    main()
