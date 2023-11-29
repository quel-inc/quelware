import logging
import time
from typing import Union

from quel_ic_config import Quel1BoxType, Quel1ConfigOption, Quel1SeProto8ConfigSubsystem
from quel_ic_config_utils import E7HwType, LinkupFpgaMxfe, Quel1E7ResourceMapper, Quel1WaveSubsystem

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

    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.INFO)
    #
    # Config Information
    #
    css_addr = "10.5.0.49"
    wss_addr = "10.1.0.49"
    config_options = {
        Quel1ConfigOption.USE_READ_IN_MXFE0,
        Quel1ConfigOption.USE_READ_IN_MXFE1,
    }
    mxfe_to_linkup = [0, 1]
    #
    # Config !
    #
    css = Quel1SeProto8ConfigSubsystem(css_addr, Quel1BoxType.fromstr("x-quel1se-proto8"), None, config_options)

    # configure all the ICs other than AD9082s
    css.configure_peripherals()

    # hard reset AD9082s
    css.gpio_helper[1].write_field(0, b00=False, b01=False)
    css.gpio_helper[1].flush()

    # configure clocks of AD9082s
    css.configure_all_mxfe_clocks()

    # release reset of AD9082s
    css.gpio_helper[1].write_field(0, b00=True, b01=True)
    css.gpio_helper[1].flush()

    #
    # Wave
    #
    wss = Quel1WaveSubsystem(wss_addr, E7HwType.SIMPLE_MULTI_CLASSIC)
    rmap = Quel1E7ResourceMapper(css, wss)

    #
    # Link up
    #
    linkupper = LinkupFpgaMxfe(css, wss, rmap)
    for mxfe in mxfe_to_linkup:
        assert linkupper.linkup_and_check(mxfe, soft_reset=True, hard_reset=False)

    #
    # Specific Settings (to be reflected to default settings)
    #
    css.set_sideband(0, 0, "L")
    css.set_lo_multiplier(0, 0, 80)
    css.set_vatt(0, 0, 0xA00)

    # css.set_sideband(0, 2, "U")
    # css.set_lo_multiplier(0, 2, 120)
    # css.set_vatt(0, 2, 0xA00)

    #
    #  Wave - pilot signal
    #
    mxfe = 0
    assert mxfe in mxfe_to_linkup
    line = 0
    cnco_mhz = 2000
    start_cw(mxfe, line, 0, cnco_mhz, 0)
    # start_cw(mxfe, line, 1, None, 200)
    # start_cw(mxfe, line, 2, None, -200)
    time.sleep(0)
    # stop_cw(mxfe, line, 0)
    # stop_cw(mxfe, line, 1)
    # stop_cw(mxfe, line, 2)
