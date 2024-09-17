import logging
from typing import Final, Optional

import numpy as np
import numpy.typing as npt

from quel_ic_config import CapParam, CapSection, Quel1Box, Quel1BoxIntrinsic, Quel1PortType, Quel1WaveSubsystem

logger = logging.getLogger(__name__)

_DEFAULT_AMPLITUDE: Final[float] = 32767.0
_DEFAULT_NUM_REPEATS: Final[tuple[int, int]] = (0xFFFFFFFF, 0xFFFFFFFF)


def box_easy_capture(
    box: Quel1Box,
    port: Quel1PortType,
    *,
    runit: int = 0,
    num_capture_sample: int,
    lo_freq: Optional[float] = None,
    cnco_freq: Optional[float] = None,
    fnco_freq: Optional[float] = None,
    activate_internal_loop: Optional[bool] = None,
) -> npt.NDArray[np.complex64]:
    if num_capture_sample % 4 != 0:
        raise ValueError("num_capture_sample must be multiple of 4")

    rfswitch_value = None if activate_internal_loop is None else ("loop" if activate_internal_loop else "open")
    box.config_port(port, cnco_freq=cnco_freq, lo_freq=lo_freq, rfswitch=rfswitch_value)

    param = CapParam(num_repeat=1)
    param.sections.append(CapSection(name="s0", num_capture_word=num_capture_sample // 4, num_blank_word=1))
    box.config_runit(port, runit, fnco_freq=fnco_freq, capture_param=param)

    task = box.start_capture_now({(port, runit)})
    reader = task.result()[(port, runit)]
    cap_data = reader.as_wave_dict()["s0"][0]
    return cap_data


def boxi_easy_capture(
    boxi: Quel1BoxIntrinsic,
    group: int,
    rline: str,
    *,
    runit: int = 0,
    num_capture_sample: int,
    lo_freq: Optional[float] = None,
    cnco_freq: Optional[float] = None,
    fnco_freq: Optional[float] = None,
    activate_internal_loop: Optional[bool] = None,
) -> npt.NDArray[np.complex64]:
    if num_capture_sample % 4 != 0:
        raise ValueError("num_capture_sample must be multiple of 4")

    rfswitch_value = None if activate_internal_loop is None else ("loop" if activate_internal_loop else "open")
    boxi.config_rline(group, rline, cnco_freq=cnco_freq, lo_freq=lo_freq, rfswitch=rfswitch_value)

    param = CapParam(num_repeat=1)
    param.sections.append(CapSection(name="s0", num_capture_word=num_capture_sample // 4, num_blank_word=1))
    boxi.config_runit(group, rline, runit, fnco_freq=fnco_freq, capture_param=param)

    task = boxi.start_capture_now({(group, rline, runit)})
    reader = task.result()[(group, rline, runit)]
    cap_data = reader.as_wave_dict()["s0"][0]
    return cap_data


def wss_easy_capture(
    wss: Quel1WaveSubsystem, capmod_idx: int, *, capunit_idx: int = 0, num_capture_sample: int
) -> npt.NDArray[np.complex64]:
    if num_capture_sample % 4 != 0:
        raise ValueError("num_capture_sample must be multiple of 4")

    param = CapParam(num_repeat=1)
    param.sections.append(CapSection(name="s0", num_capture_word=num_capture_sample // 4, num_blank_word=1))
    task = wss.start_capunits_now({(capmod_idx, capunit_idx)})
    reader = task.result()[(capmod_idx, capunit_idx)]
    cap_data = reader.as_wave_dict()["s0"][0]
    return cap_data
