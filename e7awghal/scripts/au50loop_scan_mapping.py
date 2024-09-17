import logging

import numpy as np
import numpy.typing as npt

from e7awghal import AbstractQuel1Au50Hal, AwgParam, AwgUnit, CapCtrlStandard, CapParam, CapSection, CapUnit, WaveChunk
from testlibs.awgctrl_with_hlapi import AwgUnitHL
from testlibs.capunit_with_hlapi import CapUnitSimplifiedHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger()


def find_au50loop_mapping(auidx, w, valid_data):
    au: AwgUnit = proxy.awgunit(auidx)
    assert isinstance(au, AwgUnitHL)
    au.register_wavedata_from_complex64vector("w", w)

    param_w256 = AwgParam(num_wait_word=0, num_repeat=1)
    param_w256.chunks.append(WaveChunk(name_of_wavedata="w", num_blank_word=0, num_repeat=1))
    au.load_parameter(param_w256)

    cp0 = CapParam(num_repeat=1)
    cp0.sections.append(CapSection(name="s0", num_capture_word=256, num_blank_word=1))

    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)

    cus: list[CapUnit] = []
    for mod in cc.modules:
        cc.set_triggering_awgunit_idx(capmod_idx=mod, awgunit_idx=au.unit_index)
        cu = proxy.capunit(cc.units_of_module(mod)[0])
        cu.load_parameter(cp0)
        cc.add_triggerable_unit(cu.unit_index)
        cus.append(cu)

    futs = []
    for cu in cus:
        assert isinstance(cu, CapUnitSimplifiedHL)
        futs.append(cu.wait_for_triggered_capture())

    au.start_now().result()
    au.wait_done().result()
    for modidx, fut in enumerate(futs):
        rdr = fut.result()
        data = rdr.as_wave_dict()
        if not (data["s0"] == 0.0).all():
            logger.info(f"awg_unix-#{auidx} --> cap_mod-#{modidx}")
            valid_data[modidx] = data["s0"]


if __name__ == "__main__":
    """
    awg_unix-#2 --> cap_mod-#0
    awg_unix-#3 --> cap_mod-#2
    awg_unix-#4 --> cap_mod-#3
    awg_unix-#15 --> cap_mod-#1
    """
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    proxy: AbstractQuel1Au50Hal = create_quel1au50hal_for_test(ipaddr_wss="10.1.0.74")
    proxy.initialize()

    w = np.zeros(16384, dtype=np.complex64)
    w[:] = np.arange(16384)
    valid_data: dict[int, npt.NDArray[np.complex64]] = {}
    for i in proxy.awgctrl.units:
        logger.debug(f"testing awg_unit-#{i}")
        find_au50loop_mapping(i, w, valid_data)
