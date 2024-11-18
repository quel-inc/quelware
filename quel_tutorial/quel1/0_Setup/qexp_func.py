import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Union
from IPython.display import display


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


def plot_iq(
    iq: npt.NDArray[np.complex64],
    scale: Union[float, np.floating] = 1.0e4,
    title: str = "wave data",
):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.set_title(title)
    ax.set_xlim(0, len(iq))
    ax.set_ylim(-scale, scale)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid()
    ax.plot(iq.real, label="I")
    ax.plot(iq.imag, label="Q")
    ax.legend()
    plt.show()