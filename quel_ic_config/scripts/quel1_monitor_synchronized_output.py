import argparse
import logging
from ipaddress import ip_address
from typing import Any, Dict, Final, Mapping, Set

import matplotlib

from quel_ic_config import QUEL1_BOXTYPE_ALIAS, CaptureReturnCode, Quel1Box, Quel1BoxType
from testlibs.general_looptest_common_updated import (
    BoxPool,
    PulseCap,
    PulseGen,
    VportSettingType,
    find_chunks,
    plot_iqs,
)

logger = logging.getLogger()

SUPPORTED_BOXTYPES: Final[Set[Quel1BoxType]] = {
    Quel1BoxType.QuBE_OU_TypeA,
    Quel1BoxType.QuBE_OU_TypeB,
    Quel1BoxType.QuBE_RIKEN_TypeA,
    Quel1BoxType.QuBE_RIKEN_TypeB,
    Quel1BoxType.QuEL1_TypeA,
    Quel1BoxType.QuEL1_TypeB,
    Quel1BoxType.QuEL1_NEC,
}

DEFAULT_CAPTURE_DURATION: Final[int] = 1024  # in samples
MINIMUM_CAPTURE_DURATION: Final[int] = 1024  # in samples
DEFAULT_PULSE_DETECTION_THRESHOLD: Final[float] = 4000.0  # a.u.
TRIGGER_PULSE_DURATION: Final[int] = 64  # in sample
DEFAULT_OUTPUT_DELAY: Final[int] = 256  # in samples
MINIMUM_OUTPUT_DELAY: Final[int] = TRIGGER_PULSE_DURATION + 64  # a.u.  64 is margin


def complete_ipaddrs(args: argparse.Namespace):
    if int(args.ipaddr_sss_a) == 0:
        args.ipaddr_sss_a = args.ipaddr_wss_a + (1 << 16)
    if int(args.ipaddr_css_a) == 0:
        args.ipaddr_css_a = args.ipaddr_wss_a + (4 << 16)
    if int(args.ipaddr_sss_b) == 0:
        args.ipaddr_sss_b = args.ipaddr_wss_b + (1 << 16)
    if int(args.ipaddr_css_b) == 0:
        args.ipaddr_css_b = args.ipaddr_wss_b + (4 << 16)


def parse_boxtype(boxtypename: str) -> Quel1BoxType:
    if boxtypename not in QUEL1_BOXTYPE_ALIAS:
        raise ValueError
    return Quel1BoxType.fromstr(boxtypename)


def validate_input_port(box: Quel1Box, port_idx: int) -> bool:
    judge: bool = False

    try:
        judge = box.is_input_port(port_idx)
        if not judge:
            logger.error(f"port:{port_idx} of box:{box.wss._wss_addr} is not an input port")
    except ValueError:
        logger.error(f"an invalid index of port:{port_idx} of box:{box.wss._wss_addr}")

    return judge


def validate_output_port(box: Quel1Box, port_idx: int) -> bool:
    judge: bool = False

    try:
        judge = box.is_output_port(port_idx)
        if not judge:
            logger.error(f"port:{port_idx} of box:{box.wss._wss_addr} is not an input port")
    except ValueError:
        logger.error(f"an invalid index of port:{port_idx} of box:{box.wss._wss_addr}")

    return judge


def single_schedule(cp: PulseCap, pg_trigger: PulseGen, pgs: Set[PulseGen], boxpool: BoxPool, power_thr: float):
    if pg_trigger not in pgs:
        raise ValueError("trigerring pulse generator is not included in activated pulse generators")
    thunk = cp.capture_at_single_trigger_of(pg=pg_trigger)
    boxpool.emit_at(cp=cp, pgs=pgs, min_time_offset=125_000_000, time_counts=(0,))

    s0, iqs = thunk.result()
    iq0 = iqs[0]
    assert s0 == CaptureReturnCode.SUCCESS
    chunks = find_chunks(iq0, power_thr=power_thr)
    return iq0, chunks


if __name__ == "__main__":
    import sys

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
        "monitoring the output of box-B with the input of box-A via an external RF path",
    )

    parser.add_argument(
        "--ipaddr_clk",
        type=ip_address,
        required=True,
        help="IP address of clock master",
    )
    parser.add_argument("--resync", action="store_true", help="resynchronizing boxes before start monitoring")

    parser.add_argument(
        "--ipaddr_wss_a",
        type=ip_address,
        required=True,
        help="IP address of the wave generation/capture subsystem of the target box A",
    )
    parser.add_argument(
        "--ipaddr_sss_a",
        type=ip_address,
        default=0,
        help="IP address of the synchronization subsystem of the target box A",
    )
    parser.add_argument(
        "--ipaddr_css_a",
        type=ip_address,
        default=0,
        help="IP address of the configuration subsystem of the target box A",
    )
    parser.add_argument(
        "--boxtype_a",
        type=parse_boxtype,
        required=True,
        help=f"a type of the target box A: either of "
        f"{', '.join([t for t in QUEL1_BOXTYPE_ALIAS if not t.startswith('x_')])}",
    )
    parser.add_argument(
        "--input_port",
        type=int,
        required=True,
        help="an input SMA port of the target box A",
    )
    parser.add_argument(
        "--trigger_port",
        type=int,
        required=True,
        help="an output SMA port of the target box A to trigger the input port",
    )
    parser.add_argument(
        "--capture_duration",
        type=int,
        default=DEFAULT_CAPTURE_DURATION,
        help=f"capture duration, default value (= {DEFAULT_CAPTURE_DURATION}) should be enough"
        "if the boxes are synchronized properly",
    )
    parser.add_argument(
        "--pulse_detection_threshold",
        type=int,
        default=DEFAULT_PULSE_DETECTION_THRESHOLD,
        help="threshold of amplitude for detecting the pulses, "
        f"default value (= {DEFAULT_PULSE_DETECTION_THRESHOLD}) is fine for QuEL-1",
    )

    parser.add_argument(
        "--ipaddr_wss_b",
        type=ip_address,
        required=True,
        help="IP address of the wave generation/capture subsystem of the target box B",
    )
    parser.add_argument(
        "--ipaddr_sss_b",
        type=ip_address,
        default=0,
        help="IP address of the synchronization subsystem of the target box B",
    )
    parser.add_argument(
        "--ipaddr_css_b",
        type=ip_address,
        default=0,
        help="IP address of the configuration subsystem of the target box B",
    )
    parser.add_argument(
        "--boxtype_b",
        type=parse_boxtype,
        required=True,
        help=f"a type of the target box B: either of "
        f"{', '.join([t for t in QUEL1_BOXTYPE_ALIAS if not t.startswith('x_')])}",
    )
    parser.add_argument(
        "--output_port",
        type=int,
        required=True,
        help="an output SMA port of the target box B",
    )
    parser.add_argument(
        "--output_delay",
        type=int,
        default=DEFAULT_OUTPUT_DELAY,
        help="delay in samples before emitting signal from the output port of the target box B, "
        f"default value is {DEFAULT_OUTPUT_DELAY}",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)

    if args.capture_duration < MINIMUM_CAPTURE_DURATION:
        logger.error(f"capture duration is too short, it must be more than or equal to {MINIMUM_CAPTURE_DURATION}")
        sys.exit(1)

    if args.capture_duration % 4 != 0:
        logger.error("capture duration must be multiples of 4")
        sys.exit(1)

    if args.boxtype_a not in SUPPORTED_BOXTYPES:
        logger.error(f"boxtype '{Quel1BoxType.tostr(args.boxtype_a)}' is not supported")
        sys.exit(1)

    if args.boxtype_b not in SUPPORTED_BOXTYPES:
        logger.error(f"boxtype '{Quel1BoxType.tostr(args.boxtype_b)}' is not supported")
        sys.exit(1)

    if args.pulse_detection_threshold <= 0:
        logger.error("pulse detection threshold must be positive")
        sys.exit(1)

    if args.output_delay < MINIMUM_OUTPUT_DELAY:
        logger.error(f"output delay must not be shorter than {MINIMUM_OUTPUT_DELAY}")
        sys.exit(1)

    if args.output_delay % 4 != 0:
        logger.error("output delay must be multiples of 4")
        sys.exit(1)

    same_box: bool = args.ipaddr_wss_a == args.ipaddr_wss_b
    same_port: bool = same_box and (args.trigger_port == args.output_port)

    num_expected_pulses = 2
    if same_port:
        num_expected_pulses = 1
        logger.warning("the output port is identical to the trigger port, only single pulse is generated")

    DEVICE_SETTINGS: Dict[str, Mapping[str, Any]] = {
        "CLOCK_MASTER": {
            "ipaddr": str(args.ipaddr_clk),
        },
        "BOX_A": {
            "ipaddr_wss": str(args.ipaddr_wss_a),
            "ipaddr_sss": str(args.ipaddr_sss_a),
            "ipaddr_css": str(args.ipaddr_css_a),
            "boxtype": args.boxtype_a,
            "config_root": None,
            "config_options": [],
            "ignore_crc_error_of_mxfe": {0, 1},
        },
    }

    if not same_box:
        DEVICE_SETTINGS["BOX_B"] = {
            "ipaddr_wss": str(args.ipaddr_wss_b),
            "ipaddr_sss": str(args.ipaddr_sss_b),
            "ipaddr_css": str(args.ipaddr_css_b),
            "boxtype": args.boxtype_b,
            "config_root": None,
            "config_options": [],
            "ignore_crc_error_of_mxfe": {0, 1},
        }

    CAP_VPORT_SETTINGS: Dict[str, Mapping[str, VportSettingType]] = {
        "CAP_A": {
            "create": {
                "boxname": "BOX_A",
                "port": args.input_port,
                "runits": {0},
            },
            "config": {
                "lo_freq": 8.0e9,
                "cnco_freq": 1.5e9,
                "fnco_freq": 0.0,
                "rfswitch": "open",
            },
            "simple_parameters": {
                0: {
                    "num_delay_sample": 0,
                    "num_integration_section": 1,
                    "num_capture_samples": [args.capture_duration],
                    "num_blank_samples": [4],
                },
            },
        },
    }

    GEN_VPORT_SETTINGS: Dict[str, Mapping[str, VportSettingType]] = {
        "TRIG_A": {
            "create": {
                "boxname": "BOX_A",
                "port": args.trigger_port,
                "channel": 0,
            },
            "config": {
                "lo_freq": 8.0e9,
                "cnco_freq": 1.5e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
                "sideband": "U",
                "vatt": 0xA00,
                "rfswitch": "pass",
            },
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 64,
                "num_repeats": (1, 1),
                "num_wait_samples": (0, 0),
            },
        },
    }

    if not same_port:
        GEN_VPORT_SETTINGS["GEN_0"] = {
            "create": {
                "boxname": "BOX_A" if same_box else "BOX_B",
                "port": args.output_port,
                "channel": 0,
            },
            "config": {
                "lo_freq": 8.0e9,
                "cnco_freq": 1.5e9,
                "fnco_freq": 0.0,
                "fullscale_current": 40000,
                "sideband": "U",
                "vatt": 0xA00,
                "rfswitch": "pass",
            },
            "cw_parameter": {
                "amplitude": 32767.0,
                "num_wave_sample": 128,
                "num_repeats": (1, 1),
                "num_wait_samples": (args.output_delay, 0),
            },
        }

    boxpool0 = BoxPool(DEVICE_SETTINGS)
    boxpool0.init(resync=args.resync)
    box_a = boxpool0.get_box("BOX_A")
    box_b = boxpool0.get_box("BOX_A") if same_box else boxpool0.get_box("BOX_B")

    if not validate_input_port(box_a, args.input_port):
        sys.exit(1)
    if not validate_output_port(box_a, args.trigger_port):
        sys.exit(1)
    if not validate_output_port(box_b, args.output_port):
        sys.exit(1)

    pgs0 = PulseGen.create(GEN_VPORT_SETTINGS, boxpool0)
    tr_0 = pgs0["TRIG_A"]

    cps0 = PulseCap.create(CAP_VPORT_SETTINGS, boxpool0)
    cp_0 = cps0["CAP_A"]

    boxpool0.measure_timediff(cp_0)

    noise_max, noise_avg, _ = cp_0.measure_background_noise()
    if noise_max > args.pulse_detection_threshold * 0.75:
        logger.warning(
            f"the input port-#{cp_0.port:02d} of the box {cp_0.box.wss._wss_addr} is too noise for the given power "
            "threshold of pulse detection, you may see sprious pulses in the results"
        )

    iqs0, chunks0 = single_schedule(cp_0, tr_0, set(pgs0.values()), boxpool0, power_thr=args.pulse_detection_threshold)
    if len(chunks0) != num_expected_pulses:
        logger.error(
            f"the number of expected pulses is {num_expected_pulses} but actually is {len(chunks0)}, something wrong"
        )

    label = f"capture of port:{args.input_port} of box:{args.ipaddr_wss_a}"
    plot_iqs({label: iqs0})