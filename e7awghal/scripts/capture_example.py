import logging

from e7awghal import AbstractQuel1Au50Hal, CapParam, CapSection
from quel_ic_config import Quel1BoxType, Quel1Feature, Quel1TypeAConfigSubsystem
from testlibs.capunit_with_hlapi import CapUnitSimplifiedHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    boxtype = Quel1BoxType.QuEL1_TypeA
    features = {Quel1Feature.BOTH_ADC}

    # initializing proxy objects
    css = Quel1TypeAConfigSubsystem("10.5.0.58", boxtype)
    css.initialize(features)
    for mxfe_idx in css.get_all_mxfes():
        css.configure_mxfe(mxfe_idx)

    proxy: AbstractQuel1Au50Hal = create_quel1au50hal_for_test(ipaddr_wss="10.1.0.58", auth_callback=lambda: True)
    proxy.initialize()

    cp0 = CapParam(num_repeat=5)
    cp0.sections.append(CapSection(name="s0", num_capture_word=256, num_blank_word=128))
    cp0.sections.append(CapSection(name="s1", num_capture_word=128, num_blank_word=1))

    cc = proxy.capctrl
    cu0 = proxy.capunit(0)
    assert isinstance(cu0, CapUnitSimplifiedHL)
    cu0.load_parameter(cp0)

    fut0 = cu0.start_now()
    rdr0 = fut0.result()
    data0 = rdr0.as_wave_dict()
