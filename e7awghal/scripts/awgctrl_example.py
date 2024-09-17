import logging

import numpy as np

from e7awghal import AbstractQuel1Au50Hal, AwgParam, AwgUnit, WaveChunk
from testlibs.awgctrl_with_hlapi import AwgUnitHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    proxy: AbstractQuel1Au50Hal = create_quel1au50hal_for_test(ipaddr_wss="10.1.0.58")
    proxy.initialize()
    ac = proxy.awgctrl
    au0: AwgUnit = proxy.awgunit(0)
    assert isinstance(au0, AwgUnitHL)

    cw = np.zeros(64, dtype=np.complex64)
    cw[:] = 32767 + 0j
    au0.register_wavedata_from_complex64vector("cw", cw)

    cw_param = AwgParam(num_wait_word=0, num_repeat=1)
    cw_param.chunks.append(WaveChunk(name_of_wavedata="cw", num_blank_word=0, num_repeat=1))
    au0.load_parameter(cw_param)

    # the following methods are for debugging. You should use APIs at wss or higher-layer in your empirical code.
    au0.start_now().result()
    au0.wait_done().result()
