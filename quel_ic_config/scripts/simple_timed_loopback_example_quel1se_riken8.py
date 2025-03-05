import logging
import sys
from typing import Final

import numpy as np

from e7awghal import AwgParam, CapParam, CapSection, WaveChunk
from quel_ic_config import Quel1Box, Quel1BoxType, Quel1PortType
from quel_ic_config_utils import add_common_arguments, add_common_workaround_arguments, complete_ipaddrs, plot_iqs

logger = logging.getLogger()

SUPPORTED_BOXTYPES: Final[set[Quel1BoxType]] = {
    Quel1BoxType.QuEL1SE_RIKEN8,
}

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(description="check signal generation and capture of QuEL-1 SE RIKEN 8GHz model")
    add_common_arguments(
        parser,
        use_boxtype=False,
        default_boxtype="quel1se-riken8",
        use_config_root=False,
        use_config_options=False,
        use_mxfe=False,
    )
    add_common_workaround_arguments(parser, use_ignore_crc_error_of_mxfe=True)
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show verbose log",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if args.boxtype not in SUPPORTED_BOXTYPES:
        logger.error(f"boxtype '{Quel1BoxType.tostr(args.boxtype)}' is not supported")
        sys.exit(1)

    box = Quel1Box.create(
        ipaddr_wss=str(args.ipaddr_wss),
        ipaddr_sss=str(args.ipaddr_sss),
        ipaddr_css=str(args.ipaddr_css),
        boxtype=args.boxtype,
        ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
    )
    status = box.reconnect()
    for mxfe_idx, s in status.items():
        if not s:
            logger.error(f"be aware that mxfe-#{mxfe_idx} is not linked-up properly")

    cfg: dict[Quel1PortType, dict[str, object]] = {
        0: {
            "cnco_freq": 1000000000.0,
            "lo_freq": 8000000000,
            "rfswitch": "loop",
            "runits": {0: {"fnco_freq": 0.0}},
        },
        (1, 1): {
            "channels": {0: {"fnco_freq": 0.0}},
            "cnco_freq": 4000000000.0,
            "fullscale_current": 39990,
            "rfswitch": "block",
        },
        1: {
            "channels": {0: {"fnco_freq": 0.0}},
            "cnco_freq": 1000000000.0,
            "fullscale_current": 39990,
            "lo_freq": 8000000000,
            "rfswitch": "block",
            "sideband": "L",
            "vatt": 3072,
        },
        2: {
            "channels": {0: {"fnco_freq": 0.0}},
            "cnco_freq": 2000000000.0,
            "fullscale_current": 39990,
            "lo_freq": 8000000000,
            "rfswitch": "block",
            "sideband": "L",
            "vatt": 3072,
        },
        3: {
            "channels": {0: {"fnco_freq": 0.0}},
            "cnco_freq": 4000000000.0,
            "fullscale_current": 39990,
            "rfswitch": "block",
        },
        4: {
            "cnco_freq": 1000000000.0,
            "lo_freq": 5000000000,
            "rfswitch": "loop",
            "runits": {0: {"fnco_freq": 0.0}},
        },
        6: {
            "channels": {0: {"fnco_freq": 0.0}},
            "cnco_freq": 4000000000.0,
            "fullscale_current": 39990,
            "rfswitch": "block",
        },
        7: {
            "channels": {0: {"fnco_freq": 0.0}},
            "cnco_freq": 4000000000.0,
            "fullscale_current": 39990,
            "rfswitch": "block",
        },
        8: {
            "channels": {0: {"fnco_freq": 0.0}},
            "cnco_freq": 4000000000.0,
            "fullscale_current": 39990,
            "rfswitch": "block",
        },
        9: {
            "channels": {0: {"fnco_freq": 0.0}},
            "cnco_freq": 4000000000.0,
            "fullscale_current": 39990,
            "rfswitch": "block",
        },
        10: {
            "cnco_freq": 1000000000.0,
            "lo_freq": 5000000000,
            "rfswitch": "loop",
            "runits": {0: {"fnco_freq": 0.0}},
        },
    }

    box.config_box(cfg)

    cw_iq = np.zeros(64, dtype=np.complex64)
    cw_iq[:] = 32767.0 + 0.0j
    for port in box.get_output_ports():
        box.register_wavedata(port, 0, "cw32767", cw_iq)

    cp = CapParam(num_repeat=1)
    cp.sections.append(CapSection(name="s0", num_capture_word=(2048 + 512) // 4, num_blank_word=4 // 4))

    # READ-#0
    box.config_runit(0, 0, capture_param=cp)

    ap0 = AwgParam(num_wait_word=0, num_repeat=1)
    ap0.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
    box.config_channel(1, 0, awg_param=ap0)

    # MONITOR-#0
    box.config_runit(4, 0, capture_param=cp)

    ap1 = AwgParam(num_wait_word=0, num_repeat=1)
    ap1.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
    box.config_channel((1, 1), 0, awg_param=ap1)

    ap2 = AwgParam(num_wait_word=512 // 4, num_repeat=1)
    ap2.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
    box.config_channel(2, 0, awg_param=ap2)

    ap3 = AwgParam(num_wait_word=1024 // 4, num_repeat=1)
    ap3.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
    box.config_channel(3, 0, awg_param=ap3)

    # MONITOR-#1
    box.config_runit(10, 0, fnco_freq=0, capture_param=cp)

    ap6 = AwgParam(num_wait_word=0, num_repeat=1)
    ap6.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
    box.config_channel(6, 0, awg_param=ap6)

    ap7 = AwgParam(num_wait_word=512 // 4, num_repeat=1)
    ap7.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
    box.config_channel(7, 0, awg_param=ap7)

    ap8 = AwgParam(num_wait_word=1024 // 4, num_repeat=1)
    ap8.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
    box.config_channel(8, 0, awg_param=ap8)

    ap9 = AwgParam(num_wait_word=1536 // 4, num_repeat=1)
    ap9.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
    box.config_channel(9, 0, awg_param=ap9)

    cur = box.get_current_timecounter()
    c_task, g_task = box.start_capture_by_awg_trigger(
        {(0, 0), (4, 0), (10, 0)},
        {(1, 0), ((1, 1), 0), (2, 0), (3, 0), (6, 0), (7, 0), (8, 0), (9, 0)},
        cur + 125_000_000 // 10,
    )

    assert g_task.result() is None
    rdr2 = c_task.result()

    data0 = rdr2[0, 0].as_wave_dict()
    data1 = rdr2[4, 0].as_wave_dict()
    data2 = rdr2[10, 0].as_wave_dict()

    iq0 = data0["s0"][0]
    iq1 = data1["s0"][0]
    iq2 = data2["s0"][0]

    del g_task
    del c_task
    del rdr2
    del box

    plot_iqs({"READ0": iq0, "MON0": iq1, "MON1": iq2}, same_range=False)
