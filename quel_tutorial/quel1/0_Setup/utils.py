from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_iq(
    iq: npt.NDArray[np.complex64],
    scale: Union[float, np.floating] = 1.0e4,
    title: str = "wave data",
):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.set_title(title)
    ax.set_xlim(0, len(iq))
    scale = float(scale)
    ax.set_ylim(-scale, scale)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid()
    ax.plot(iq.real, label="I")
    ax.plot(iq.imag, label="Q")
    ax.legend()
    plt.show()
