import logging

import numpy as np
import numpy.typing as npt

from e7awghal.common_defs import DECIMATION_RATE, SAMPLING_FREQ
from e7awghal_utils.fir_coefficient import _folded_frequency_by_decimation

logger = logging.getLogger(__name__)


def table_for_demodulation(
    ro_freq: float, ro_duration: float, decimated_input: bool, phase_offset: float = 0.0
) -> npt.NDArray[np.complex128]:
    if decimated_input:
        ro_freq = _folded_frequency_by_decimation(ro_freq)
        n = int(ro_duration * SAMPLING_FREQ / DECIMATION_RATE)
    else:
        n = int(ro_duration * SAMPLING_FREQ)

    t = np.linspace(0, ro_duration, n, endpoint=False, dtype=np.float64)

    return np.exp(-1j * (2.0 * np.pi * ro_freq * t + phase_offset))
