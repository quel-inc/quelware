import logging
import time
from collections.abc import Collection
from typing import Final, Union

import numpy as np

from quel_ic_config import AwgParam, Quel1Box, Quel1BoxIntrinsic, Quel1PortType, Quel1WaveSubsystem, WaveChunk

logger = logging.getLogger(__name__)

_DEFAULT_AMPLITUDE: Final[float] = 32767.0
_DEFAULT_NUM_REPEATS: Final[tuple[int, int]] = (0xFFFFFFFF, 0xFFFFFFFF)


def box_easy_start_cw(
    box: Quel1Box,
    port: Quel1PortType,
    *,
    channel: int = 0,
    amplitude: float = _DEFAULT_AMPLITUDE,
    num_repeats: tuple[int, int] = _DEFAULT_NUM_REPEATS,
) -> None:
    group, line = box._convert_output_port(port)
    boxi_easy_start_cw(box._dev, group, line, channel=channel, amplitude=amplitude, num_repeats=num_repeats)


def boxi_easy_start_cw(
    boxi: Quel1BoxIntrinsic,
    group: int,
    line: int,
    *,
    channel: int = 0,
    amplitude: float = _DEFAULT_AMPLITUDE,
    num_repeats: tuple[int, int] = _DEFAULT_NUM_REPEATS,
) -> None:
    awg_idx = boxi._get_awg_from_channel(group, line, channel)
    wss_easy_start_cw(boxi.wss, awg_idx, amplitude=amplitude, num_repeats=num_repeats)


def wss_easy_start_cw(
    wss: Quel1WaveSubsystem,
    awg_idx: int,
    *,
    amplitude: float = _DEFAULT_AMPLITUDE,
    num_repeats: tuple[int, int] = _DEFAULT_NUM_REPEATS,
) -> None:
    # XXX: just for debugging, do not use this function in your quantum experiment code.
    wname: str = "wss_easy_start_cw.cw"

    cw_iq = np.zeros(64, dtype=np.complex64)
    cw_iq[:] = (1.0 + 0.0j) * amplitude
    wss.register_wavedata(awg_idx, wname, cw_iq)

    p = AwgParam(num_repeat=num_repeats[0])
    p.chunks.append(WaveChunk(name_of_wavedata=wname, num_blank_word=0, num_repeat=num_repeats[1]))
    wss.config_awgunit(awg_idx, p)

    fut = wss.start_awgunits_now(awgunit_idxs={awg_idx})
    try:
        print(f"Ctrl+C to stop wave generation from awgunit-#{awg_idx}: ", end="", flush=True)
        while True:
            print(".", end="", flush=True)
            time.sleep(2.5)
    except KeyboardInterrupt:
        print()
        fut.cancel()
        fut.exception()


def box_easy_stop(box: Quel1Box, port: Quel1PortType, channel: Union[int, None] = None) -> None:
    # XXX: just for debugging, do not use this function in your quantum experiment code.
    group, line = box._convert_output_port(port)
    boxi_easy_stop(box._dev, group, line, channel)


def boxi_easy_stop(boxi: Quel1BoxIntrinsic, group: int, line: int, channel: Union[int, None] = None) -> None:
    # XXX: just for debugging, do not use this function in your quantum experiment code.
    if channel is None:
        channels = boxi.get_channels_of_line(group, line)
    else:
        channels = {channel}

    for ch in channels:
        awg_idx = boxi._get_awg_from_channel(group, line, ch)
        wss_easy_stop(boxi.wss, {awg_idx})


def wss_easy_stop(wss: Quel1WaveSubsystem, awg_idxs: Collection[int]) -> None:
    # XXX: just for debugging, do not use this function in your quantum experiment code.
    for awg_idx in awg_idxs:
        u = wss.hal.awgunit(awg_idx)
        u.terminate().result()


def box_easy_stop_all(box: Quel1Box) -> None:
    boxi_easy_stop_all(box._dev)


def boxi_easy_stop_all(boxi: Quel1BoxIntrinsic) -> None:
    wss_easy_stop_all(boxi.wss)


def wss_easy_stop_all(wss) -> None:
    # XXX: just for debugging, do not use this function in your quantum experiment code.
    rs = []

    for i in wss.hal.awgctrl.num_unit:
        u = wss.hal.awgunit(i)
        rs.append(u.terminate())
    for r in rs:
        r.result()
