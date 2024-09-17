import numpy as np

from quel_ic_config.quel1_box import Quel1Box
from quel_ic_config.quel1_box_intrinsic import Quel1BoxIntrinsic


def register_cw_to_all_lines(boxi: Quel1BoxIntrinsic) -> None:
    cw_iq = np.zeros(64, dtype=np.complex64)
    cw_iq[:] = 32767.0 + 0.0j

    cww_iq = np.zeros(64, dtype=np.complex64)
    cww_iq[:] = 16383.0 + 0.0j

    cwww_iq = np.zeros(64, dtype=np.complex64)
    cwww_iq[:] = 8191.0 + 0.0j

    cwwww_iq = np.zeros(64, dtype=np.complex64)
    cwwww_iq[:] = 4095.0 + 0.0j

    for gr, ln in boxi.get_output_lines():
        for ch in boxi.get_channels_of_line(gr, ln):
            boxi.register_wavedata(gr, ln, ch, "test_wave_generation:cw32767", cw_iq)
            boxi.register_wavedata(gr, ln, ch, "test_wave_generation:cw16383", cww_iq)
            boxi.register_wavedata(gr, ln, ch, "test_wave_generation:cw8191", cwww_iq)
            boxi.register_wavedata(gr, ln, ch, "test_wave_generation:cw4095", cwww_iq)


def register_cw_to_all_ports(box: Quel1Box) -> None:
    cw_iq = np.zeros(64, dtype=np.complex64)
    cw_iq[:] = 32767.0 + 0.0j

    cww_iq = np.zeros(64, dtype=np.complex64)
    cww_iq[:] = 16383.0 + 0.0j

    cwww_iq = np.zeros(64, dtype=np.complex64)
    cwww_iq[:] = 8191.0 + 0.0j

    cwwww_iq = np.zeros(64, dtype=np.complex64)
    cwwww_iq[:] = 4095.0 + 0.0j

    for port in box.get_output_ports():
        for ch in box.get_channels_of_port(port):
            box.register_wavedata(port, ch, "test_wave_generation:cw32767", cw_iq)
            box.register_wavedata(port, ch, "test_wave_generation:cw16383", cww_iq)
            box.register_wavedata(port, ch, "test_wave_generation:cw8191", cwww_iq)
            box.register_wavedata(port, ch, "test_wave_generation:cw4095", cwwww_iq)
