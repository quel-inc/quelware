from typing import Optional

from e7awghal import AwgParam, WaveChunk
from quel_ic_config.quel1_box import Quel1Box, Quel1PortType
from quel_ic_config.quel1_box_intrinsic import Quel1BoxIntrinsic
from quel_ic_config.quel1_wave_subsystem import AbstractStartAwgunitsTask


def boxi_gen_cw(
    boxi: Quel1BoxIntrinsic,
    group: int,
    line: int,
    channel: int,
    *,
    fnco_freq: float,
    cnco_freq: float,
    fullscale_current: int,
    lo_freq: Optional[float] = None,
    sideband: Optional[str] = None,
    vatt: Optional[int] = None,
    via_monitor: bool,
) -> AbstractStartAwgunitsTask:
    boxi.config_line(
        group,
        line,
        lo_freq=lo_freq,
        vatt=vatt,
        sideband=sideband,
        cnco_freq=cnco_freq,
        fullscale_current=fullscale_current,
        rfswitch="pass" if not via_monitor else "block",
    )

    p = AwgParam(num_repeat=0xFFFF_FFFF)
    p.chunks.append(
        WaveChunk(name_of_wavedata="test_wave_generation:cw32767", num_blank_word=0, num_repeat=0xFFFF_FFFF)
    )
    boxi.config_channel(group, line, channel, fnco_freq=fnco_freq, awg_param=p)

    return boxi.start_wavegen({(group, line, channel)})


def box_gen_cw(
    box: Quel1Box,
    port: Quel1PortType,
    channel: int,
    *,
    fnco_freq: float,
    cnco_freq: float,
    fullscale_current: int,
    lo_freq: Optional[float] = None,
    sideband: Optional[str] = None,
    vatt: Optional[int] = None,
    via_monitor: bool,
) -> AbstractStartAwgunitsTask:
    box.config_port(
        port,
        lo_freq=lo_freq,
        vatt=vatt,
        sideband=sideband,
        cnco_freq=cnco_freq,
        fullscale_current=fullscale_current,
        rfswitch="pass" if not via_monitor else "block",
    )

    p = AwgParam(num_repeat=0xFFFF_FFFF)
    p.chunks.append(
        WaveChunk(name_of_wavedata="test_wave_generation:cw32767", num_blank_word=0, num_repeat=0xFFFF_FFFF)
    )
    box.config_channel(port, channel, fnco_freq=fnco_freq, awg_param=p)
    return box.start_wavegen({(port, channel)})
