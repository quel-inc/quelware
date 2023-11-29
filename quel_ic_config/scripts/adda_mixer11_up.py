import argparse
import logging
from typing import Union

import numpy as np
import numpy.typing as npt

from quel_ic_config import Quel1SeProto11ConfigSubsystem
from quel_ic_config_utils import CaptureReturnCode, E7HwType, LinkupFpgaMxfe, Quel1E7ResourceMapper, Quel1WaveSubsystem
from quel_ic_config_utils.common_arguments import add_common_arguments

logger = logging.getLogger()

if __name__ == "__main__":

    def start_cw(mxfe: int, line: int, channel: int, cnco_mhz: Union[float, None], fnco_mhz: float = 0):
        global css, wss, rmap

        if cnco_mhz is not None:
            css.set_dac_cnco(mxfe, line, freq_in_hz=int(cnco_mhz * 1000000 + 0.5))
            logger.info(f"setting CNCO frequency to {cnco_mhz}MHz")
        css.set_dac_fnco(mxfe, line, channel, freq_in_hz=int(fnco_mhz * 1000000 + 0.5))
        wss.simple_cw_gen(
            rmap.get_awg_of_channel(mxfe, line, channel), amplitude=16383.0, num_repeats=(0xFFFFFFFF, 0xFFFFFFF)
        )
        logger.info(f"start emitting CW of {fnco_mhz}MHz at ({mxfe}, {line}, {channel})")

    def stop_cw(mxfe: int, line: int, channel: int):
        global wss, rmap

        wss.stop_emission({rmap.get_awg_of_channel(mxfe, line, channel)})

    def capture(
        mxfe: int,
        rline: Union[str, None] = None,
        cnco_mhz: Union[float, None] = None,
        num_samples: int = 4096,
    ) -> npt.NDArray[np.complex64]:
        global css, wss, rmap

        rline = rmap.resolve_rline(mxfe, rline)

        if cnco_mhz is not None:
            css.set_adc_cnco(mxfe, rline, freq_in_hz=int(cnco_mhz * 1000000 + 0.5))

        if num_samples % 4 != 0:
            num_samples = ((num_samples + 3) // 4) * 4
            logger.warning(f"num_samples is extended to multiples of 4: {num_samples}")

        if num_samples > 0:
            status, iq = wss.simple_capture(rmap.get_capture_module_of_rline(mxfe, rline), num_words=num_samples // 4)
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

    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.INFO)

    default_boxtype = "x-quel1se-proto11"
    default_config_root = None
    default_config_options = "use_read_in_mxfe0,use_read_in_mxfe1"  # because no pathselector board is available.
    mxfe_to_linkup = (0, 1)
    available_mixer_boards = (0,)  # if you have two mixer boards, it should be (0, 1)

    # For cli where the parameter are passed as arguments
    parser = argparse.ArgumentParser()
    add_common_arguments(
        parser,
        use_boxtype=False,
        default_boxtype=default_boxtype,
        use_config_root=False,
        default_config_root=default_config_root,
        use_config_options=False,
        default_config_options=default_config_options,
    )
    args = parser.parse_args()

    css = Quel1SeProto11ConfigSubsystem(str(args.ipaddr_css), args.boxtype, args.config_root, args.config_options)
    wss = Quel1WaveSubsystem(str(args.ipaddr_wss), E7HwType.SIMPLE_MULTI_CLASSIC)
    rmap = Quel1E7ResourceMapper(css, wss)

    #
    # Config !
    #
    # configure all the ICs other than AD9082s
    css.configure_peripherals(available_mixer_boards=available_mixer_boards)

    # hard reset AD9082s
    css.gpio_helper[1].write_field(0, b00=False, b01=False)
    css.gpio_helper[1].flush()

    # configure clocks of AD9082s
    css.configure_all_mxfe_clocks()

    # release reset of AD9082s
    css.gpio_helper[1].write_field(0, b00=True, b01=True)
    css.gpio_helper[1].flush()

    #
    # Link up
    #
    linkupper = LinkupFpgaMxfe(css, wss, rmap)
    for mxfe in css.get_all_groups():
        if mxfe in mxfe_to_linkup:
            assert linkupper.linkup_and_check(mxfe, soft_reset=True, hard_reset=False)
        else:
            css.configure_mxfe(mxfe)  # initialize host-side objects based on the current hardware configurations.

    #
    # Specific Settings
    #
    # css.set_sideband(0, 0, "L")
    # css.set_lo_multiplier(0, 0, 80)
    # css.set_vatt(0, 0, 0xA00)

    #
    #  Wave - pilot signal
    #
    # mxfe = 0
    # assert mxfe in mxfe_to_linkup
    # line = 0
    # cnco_mhz = 2000
    # start_cw(mxfe, line, 0, cnco_mhz, 0)
    # stop_cw(mxfe, line, 0)
