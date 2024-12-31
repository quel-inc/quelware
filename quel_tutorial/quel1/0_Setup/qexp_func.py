import numpy as np
import numpy.typing as npt


def square_pulse(
    num_blank1: int,
    num_flat: int,
    num_blank2: int,
    amplitude: float = 16383.0,
) -> npt.NDArray[np.complex64]:
    if not (0.0 <= amplitude <= 32767.0):
        raise ValueError("amplitude must be positive value not greater than 32767.0")

    num_total = num_blank1 + num_flat + num_blank2
    wave = np.zeros(num_total, dtype=np.complex64)
    wave[0:num_blank1] = 0.0 + 0.0j
    wave[num_blank1 : num_blank1 + num_flat] = amplitude + 0.0j
    wave[num_blank1 + num_flat : num_total] = 0.0 + 0.0j

    return wave * amplitude
