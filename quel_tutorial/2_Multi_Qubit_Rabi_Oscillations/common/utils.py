import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from common.qexp_func_multi_qubit_rabi import fit_and_rotate_iq, rabi_fitting, rabi_model, rabi_resonance_fit


def plot_iq_ns(
    iq: npt.NDArray[np.complex64],
    scale: float = 1.0e4,
    title: str = "Port 0",
    figsize: tuple[int, int] = (8, 2),
    constant: float = 2.0,
):
    x = np.arange(len(iq)) * constant
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    ax.set_xlim(0, len(iq) * constant)
    ax.set_ylim(-scale, scale)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Amplitude")
    ax.grid()
    ax.plot(x, iq.real, label="I")
    ax.plot(x, iq.imag, label="Q")
    ax.legend()
    plt.show()
    plt.close(fig)


def plot_iq_scatter_marker(
    iq: npt.NDArray[np.complex64],
    scale: float = 1.0e4,
    title: str = "Port 0",
    markersize: float = 5.0,
    figsize: tuple[int, int] = (4, 4),
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    ax.set_xlim(-1 * scale, scale)
    ax.set_ylim(-1 * scale, scale)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid()
    ax.scatter(iq.real, iq.imag, color="red", s=markersize)
    plt.show()
    plt.close(fig)


def plot_iq_rot_scatter_marker(
    iq_sum: npt.NDArray[np.complex64],
    iq_rot: npt.NDArray[np.complex64],
    scale: float = 1.0e4,
    title: str = "Port 0",
    markersize: float = 5.0,
    figsize: tuple[int, int] = (4, 4),
    label_sum: str = "Before rotation",
    label_rot: str = "After rotation",
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    ax.set_xlim(-1 * scale, scale)
    ax.set_ylim(-1 * scale, scale)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid()
    ax.scatter(
        iq_sum.real,
        iq_sum.imag,
        color="red",
        s=markersize,
        label=label_sum,
    )
    ax.scatter(
        iq_rot.real,
        iq_rot.imag,
        color="blue",
        s=markersize,
        label=label_rot,
    )
    ax.legend()
    plt.show()
    plt.close(fig)


def plot_rabi(
    iq_sum_dict: dict[str, npt.NDArray[np.complex64]],
    ctrl_pulse_width_list: npt.NDArray[np.float64],
    title_prefix: str = "Rabi Oscillation",
    figsize: tuple[int, int] = (8, 4),
) -> dict[str, np.ndarray]:

    fitting_results = {}

    for qubit, iq_sum in iq_sum_dict.items():
        iq_rot = fit_and_rotate_iq(iq_sum)
        title = f"{title_prefix}: {qubit}"

        plot_iq_rot_scatter_marker(
            iq_sum,
            iq_rot,
            np.max(np.abs(iq_rot)),
            title,
            markersize=5,
            figsize=(4, 4),
        )

        popt, q_norm = rabi_fitting(ctrl_pulse_width_list, iq_rot)
        fitting_results[qubit] = popt
        fitted_q = rabi_model(ctrl_pulse_width_list, *popt)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel("Pulse length [ns]")
        ax.set_ylabel("Normalized Q")
        ax.grid(True)
        ax.plot(ctrl_pulse_width_list, q_norm, "o", label="Normalized Data")
        ax.plot(ctrl_pulse_width_list, fitted_q, "r--", label="Fitted Model")
        ax.legend()
        plt.show()
        plt.close(fig)

        rabi_freq_mhz = popt[2] * 1.0e3
        print(f"Rabi frequency for {qubit}: {rabi_freq_mhz:.2f} MHz")

    return fitting_results


def plot_rabi_freq(
    x_list: npt.NDArray[np.float64],
    y_list: npt.NDArray[np.float64],
    title: str = "Rabi Oscillation",
    figsize: tuple[int, int] = (4, 4),
) -> npt.NDArray[np.float64]:

    parameter_initial = np.array([15, 7500, 1])
    popt, pcov = curve_fit(rabi_resonance_fit, x_list, y_list, p0=parameter_initial, maxfev=100000)
    rabi_freq_fit = rabi_resonance_fit(x_list, *popt)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel("Drive Frequency [MHz]")
    ax.set_ylabel("Rabi Frequency [MHz]")
    ax.grid()
    ax.plot(x_list, y_list, "o", label="Normalized Data")
    ax.plot(x_list, rabi_freq_fit, "r--", label="Fitted Model")
    ax.legend()
    plt.show()
    return popt


def plot_chevron_pattern(
    chevron_pattern: npt.NDArray[np.float64],
    x_list: npt.NDArray[np.float64],
    y_list: npt.NDArray[np.float64],
    title: str = "Rabi Oscillation",
    figsize: tuple[int, int] = (8, 4),
):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(
        chevron_pattern,
        aspect="auto",
        cmap="viridis",
        extent=(x_list[0], x_list[-1], y_list[0], y_list[-1]),
        origin="lower",
    )
    ax.set_xticks(x_list)
    ax.set_xticklabels([f"{int(label)}" for label in x_list])
    ax.set_yticks(y_list)
    ax.set_yticklabels([f"{int(label)}" for label in y_list])

    if ax.figure is None:
        raise RuntimeError("ax.figure is None unexpectedly")
    cbar = ax.figure.colorbar(ax.images[0], ax=ax)
    cbar.set_label("Q oscillation")
    ax.set_title(title)
    ax.set_xlabel("Pulse Width (ns)")
    ax.set_ylabel("Drive Frequency (MHz)")
    plt.show()
    plt.close(fig)


def plot_fft_pattern(
    fft_pattern: npt.NDArray[np.float64],
    x_list: npt.NDArray[np.float64],
    y_list: npt.NDArray[np.float64],
    figsize: tuple[int, int] = (8, 4),
):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(fft_pattern, aspect="auto", cmap="viridis")
    ax.set_yticks(np.linspace(0, fft_pattern.shape[0] - 1, len(y_list)))
    ax.set_yticklabels([f"{int(label)}" for label in y_list])
    if ax.figure is None:
        raise RuntimeError("ax.figure is None unexpectedly")
    cbar = ax.figure.colorbar(ax.images[0], ax=ax)
    cbar.set_label("Q oscillation")
    ax.set_xlim(-0, 50)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Drive Frequency (MHz)")
    plt.show()
    plt.close(fig)
