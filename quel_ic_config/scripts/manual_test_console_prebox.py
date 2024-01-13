import logging
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from quel_ic_config_utils import CaptureReturnCode, create_box_objects, linkup, reconnect
from testlibs.spa_helper import SpectrumAnalyzer, init_e440xb, init_ms2xxxx

logger = logging.getLogger()


def spa_trace():
    global spa

    if spa is None:
        raise RuntimeError("no spa is available")

    t0 = spa.trace_get()
    print(t0[np.argmax(t0[:, 1])])

    plt.cla()
    plt.plot(t0[:, 0], t0[:, 1])
    plt.show()
    return t0


def conf_line(group: int, line: int, lo_mult: int = 110, sideband: str = "L", vatt: int = 0xA00):
    global css

    if not (hasattr(css, "set_sideband") and hasattr(css, "set_vatt")):
        raise TypeError(f"not applicable to {type(css)}")

    css.set_lo_multiplier(group, line, lo_mult)
    css.set_sideband(group, line, sideband)
    css.set_vatt(group, line, vatt)


def start_cw(group: int, line: int, channel: int, cnco_mhz: Union[float, None], fnco_mhz: float = 0):
    global css, wss, rmap

    if cnco_mhz is not None:
        css.set_dac_cnco(group, line, freq_in_hz=int(cnco_mhz * 1000000 + 0.5))
        logger.info(f"setting CNCO frequency to {cnco_mhz}MHz")
    css.set_dac_fnco(group, line, channel, freq_in_hz=int(fnco_mhz * 1000000 + 0.5))
    wss.simple_cw_gen(
        rmap.get_awg_of_channel(group, line, channel), amplitude=16383.0, num_repeats=(0xFFFFFFFF, 0xFFFFFFF)
    )
    logger.info(f"start emitting CW of {fnco_mhz}MHz at ({group}, {line}, {channel})")


def stop_cw(mxfe: int, line: int, channel: int):
    global wss, rmap

    wss.stop_emission({rmap.get_awg_of_channel(mxfe, line, channel)})


def capture(
    group: int,
    rline: Union[str, None] = None,
    cnco_mhz: Union[float, None] = None,
    num_samples: int = 4096,
) -> npt.NDArray[np.complex64]:
    global css, wss, rmap

    rline = rmap.resolve_rline(group, rline)
    if cnco_mhz is not None:
        css.set_adc_cnco(group, rline, freq_in_hz=int(cnco_mhz * 1000000 + 0.5))

    if num_samples % 4 != 0:
        num_samples = ((num_samples + 3) // 4) * 4
        logger.warning(f"num_samples is extended to multiples of 4: {num_samples}")

    if num_samples > 0:
        status, iq = wss.simple_capture(rmap.get_capture_module_of_rline(group, rline), num_words=num_samples // 4)
        if status == CaptureReturnCode.SUCCESS:
            return iq
        else:
            if status == CaptureReturnCode.CAPTURE_ERROR:
                raise RuntimeError("failed to capture due to internal error of FPGA")
            elif status == CaptureReturnCode.CAPTURE_TIMEOUT:
                raise RuntimeError("failed to capture due to capture timeout")
            elif status == CaptureReturnCode.BROKEN_DATA:
                raise RuntimeError("failed to capture due to broken capture data")
            else:
                raise AssertionError
    elif num_samples == 0:
        return np.zeros(0, dtype=np.complex64)
    else:
        raise ValueError(f"nagative num_samples (= {num_samples}) is not allowed")


if __name__ == "__main__":
    import argparse

    from quel_ic_config_utils.common_arguments import (
        add_common_arguments,
        add_common_workaround_arguments,
        complete_ipaddrs,
    )

    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="check the basic functionalities of QuEL-1 without using SimpleBox object",
    )
    add_common_arguments(parser, use_mxfe=True, allow_implicit_mxfe=True)
    add_common_workaround_arguments(
        parser, use_ignore_crc_error_of_mxfe=True, use_ignore_extraordinary_converter_select_of_mxfe=True
    )
    parser.add_argument("--dev", action="store_true", help="use (group, line, channel) instead of port")
    parser.add_argument("--verbose", action="store_true", default=False, help="show verbose log")
    parser.add_argument("--linkup", action="store_true", help="conducting link-up just after the initialization")
    parser.add_argument("--spa", type=str, default="", help="name of spectrum analyzer to use")
    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    css, wss, rmap, linkupper, box = create_box_objects(
        ipaddr_wss=str(args.ipaddr_wss),
        ipaddr_sss=str(args.ipaddr_sss),
        ipaddr_css=str(args.ipaddr_css),
        boxtype=args.boxtype,
        config_root=args.config_root,
        config_options=args.config_options,
        refer_by_port=not args.dev,
    )

    if args.linkup:
        status = linkup(
            linkupper=linkupper,
            mxfe_list=[0, 1],
            ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
            ignore_extraordinal_converter_select_of_mxfe=args.ignore_extraordinary_converter_select_of_mxfe,
        )
    else:
        status = reconnect(
            css=css,
            wss=wss,
            rmap=rmap,
            mxfe_list=[0, 1],
            ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select_of_mxfe=args.ignore_extraordinary_converter_select_of_mxfe,
        )

    spa: Union[SpectrumAnalyzer, None]
    if args.spa != "":
        if args.spa.startswith("ms2090a-") or args.spa.startswith("ms2720t-"):
            spa = init_ms2xxxx(args.spa)
        elif args.spa == "E4407B":
            spa = init_e440xb(args.spa)
        else:
            raise ValueError(f"unsupported spectrum analyzer name or model: {args.spa}")
        matplotlib.use("Qt5Agg")
    else:
        spa = None

    for mxfe_idx, s in status.items():
        if not s:
            logger.error(f"be aware that mxfe-#{mxfe_idx} is not linked-up properly")
