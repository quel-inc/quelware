from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def plot_iqs(
    iq_dict,
    *,
    t_offset: int = 0,
    same_range: bool = True,
    show_graph: bool = True,
    output_filename: Union[Path, None] = None,
) -> None:
    n_plot = len(iq_dict)

    m = 0
    if same_range:
        for _, iq in iq_dict.items():
            m = max(m, np.max(np.abs(iq)))

    fig, axs = plt.subplots(n_plot, sharex="col")
    if n_plot == 1:
        axs = [axs]
    fig.set_size_inches(10.0, 2.0 * n_plot)
    fig.subplots_adjust(bottom=max(0.025, 0.125 / n_plot), top=min(0.975, 1.0 - 0.05 / n_plot))
    for idx, (title, iq) in enumerate(iq_dict.items()):
        if not same_range:
            m = np.max(np.abs(iq))
        t = np.arange(0, len(iq)) - t_offset
        axs[idx].plot(t, np.real(iq))
        axs[idx].plot(t, np.imag(iq))
        axs[idx].set_ylim((-m * 1.1, m * 1.1))
        axs[idx].text(0.05, 0.1, title, transform=axs[idx].transAxes)
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_graph:
        plt.show()
