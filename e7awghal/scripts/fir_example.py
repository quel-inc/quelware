import argparse
import logging
from pprint import pprint
from typing import Dict, Final, Tuple, Union

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import freqz
from scipy.signal.windows import get_window

from e7awghal.common_defs import DECIMATION_RATE, SAMPLING_FREQ
from e7awghal_utils import complex_fir_bpf, real_fir_bpf

DEFAULT_PASS_BAND_WIDTH: Final[float] = 25.0e6


def plot_transfer_function(
    axs: Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes],
    freqs: npt.NDArray[np.float64],
    amplitudes: npt.NDArray[np.float64],
    phases: npt.NDArray[np.float64],
    title: str,
):

    # Plot the magnitude response
    axs[0].plot(freqs, amplitudes, linewidth=0, marker="o", markersize=1)
    axs[0].set_title(f"{title}-amplitude")
    axs[0].set_xlabel("Frequency (MHz)")
    axs[0].set_ylabel("Amplitude (dB)")
    axs[0].set_xlim(-250, 250)
    axs[0].set_ylim(-80, 0)
    axs[0].grid(True)

    axs[1].plot(freqs, phases, linewidth=0, marker="o", markersize=1)
    axs[1].set_title(f"{title}-phase")
    axs[1].set_xlabel("Frequency (MHz)")
    axs[1].set_ylabel("Phase (radians)")
    axs[1].set_xlim(-250, 250)
    axs[1].grid(True)


def get_transfer_function(
    coeffs: Union[npt.NDArray[np.float32], npt.NDArray[np.complex64]], sampling_freq: float
) -> Dict[str, npt.NDArray[np.float64]]:
    w, h = freqz(coeffs[::-1], whole=True)
    phase_response = np.angle(h)
    bias = round(np.mean(phase_response) / (2 * np.pi))
    #  w is converted from [0, 2*pi] to [-pi, pi] for frequency vector to be in ascending order
    return {
        "freqs": np.fft.fftshift(np.where(w < np.pi, w, -2 * np.pi + w)) * (sampling_freq / (2 * np.pi)) / 1e6,
        "amplitudes": 20 * np.log10(np.abs(np.fft.fftshift(h))),
        "phases": np.unwrap(np.fft.fftshift(phase_response)) - 2 * bias * np.pi,
    }


def unfold_transfer_function(folded_func: Dict[str, npt.NDArray[np.float64]]) -> Dict[str, npt.NDArray[np.float64]]:
    freqvec = folded_func["freqs"]
    ampvec = folded_func["amplitudes"]
    phasevec = folded_func["phases"]

    if not np.all(np.diff(freqvec) >= 0):
        raise ValueError("Frequency vector (freqvec) must be sorted in ascending order.")

    fs = SAMPLING_FREQ / DECIMATION_RATE / 1e6

    # Initialize lists to hold the extended data
    freqvec_extended = []
    ampvec_extended = []
    phasevec_extended = []

    # extend freqvec upto 5th nyquist zone
    # need only upto  4th nyquist zone. but easier to extend upto 5th zone
    for shift in [-2 * fs, -fs, 0, fs, 2 * fs]:
        # Shift the frequency vector
        freq_shifted = freqvec + shift
        freqvec_extended.append(freq_shifted[abs(freq_shifted) < 2 * fs])
        # Repeat the amplitude response
        ampvec_extended.append(ampvec[abs(freq_shifted) < 2 * fs])
        phasevec_extended.append(phasevec[abs(freq_shifted) < 2 * fs])

    # sort the arrays to maintain frequency order
    return {
        "freqs": np.concatenate(freqvec_extended),
        "amplitudes": np.concatenate(ampvec_extended),
        "phases": np.concatenate(phasevec_extended),
    }


def combine_transfer_functions(
    cfir_func: Dict[str, npt.NDArray[np.float64]], rfir_func_unfolded: Dict[str, npt.NDArray[np.float64]]
) -> Dict[str, npt.NDArray[np.float64]]:
    amps_combined = []
    phases_combined = []
    # interpolate the rfir response to be evaluated at same freqs as cfir response
    for idx, freq in enumerate(cfir_func["freqs"]):
        amp = np.interp(freq, rfir_func_unfolded["freqs"], rfir_func_unfolded["amplitudes"])
        phase = np.interp(freq, rfir_func_unfolded["freqs"], rfir_func_unfolded["phases"])
        amps_combined.append(amp + cfir_func["amplitudes"][idx])
        phases_combined.append(phase + cfir_func["phases"][idx])
    return {
        "freqs": cfir_func["freqs"],
        "amplitudes": np.array(amps_combined, dtype=np.float64),
        "phases": np.array(phases_combined, dtype=np.float64),
    }


def draw_responses(
    complex_coeffs: npt.NDArray[np.complex64],
    real_coeffs: npt.NDArray[np.float32],
    without_decimation: bool,
):
    _, axs = plt.subplots(2, 3, figsize=(10, 6))
    response_cfir = get_transfer_function(complex_coeffs, SAMPLING_FREQ)
    if without_decimation:
        response_rfir = get_transfer_function(real_coeffs, SAMPLING_FREQ)
        response_rfir_unfolded = response_rfir
    else:
        response_rfir = get_transfer_function(real_coeffs, SAMPLING_FREQ / DECIMATION_RATE)

        # extend the rfir response upto at least 4th nyquist zone in case of 1/4 decimation applied
        response_rfir_unfolded = unfold_transfer_function(response_rfir)

    plot_transfer_function(axs=(axs[0][0], axs[1][0]), **response_cfir, title="CFIR")
    plot_transfer_function(axs=(axs[0][1], axs[1][1]), **response_rfir_unfolded, title="RFIR")
    plot_transfer_function(
        axs=(axs[0][2], axs[1][2]),
        **combine_transfer_functions(response_cfir, response_rfir_unfolded),
        title="Combined",
    )

    plt.tight_layout()
    plt.show()


def parse_windowfunc(wf: str) -> str:
    # Notes: ValueError is raised for an invalid name of window function.
    _ = get_window(wf, 8)
    return wf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(description="generating coefficient complex bandpass fir filter")
    parser.add_argument("--target_freq", type=float, required=True, help="target frequency [Hz]")
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_PASS_BAND_WIDTH, help="pass band width [Hz]")
    parser.add_argument("--without_decimation", action="store_true", help="real FIR filters for no-decimated signal")
    # Notes: only window functions without additional parameters can be specified here.
    #        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html for further info.
    parser.add_argument(
        "--cfir_windowfunc",
        type=parse_windowfunc,
        default="hamming",
        help="window function of complex bandpass fir filter",
    )
    parser.add_argument(
        "--rfirs_windowfunc", type=parse_windowfunc, default="hamming", help="window function of real fir filters"
    )

    args = parser.parse_args()
    cfir_coeffs = complex_fir_bpf(
        target_freq=args.target_freq,
        bandwidth=args.bandwidth,
        window=args.cfir_windowfunc,
    )
    target_freq, rfir_coeffs = real_fir_bpf(
        target_freq=args.target_freq,
        bandwidth=args.bandwidth,
        decimated_input=not args.without_decimation,
        window=args.rfirs_windowfunc,
    )

    pprint("*** CFIR coefficients ***")
    pprint(cfir_coeffs)
    pprint("*** RFIR coefficients ***")
    pprint(rfir_coeffs)

    matplotlib.use("Gtk3Agg")
    draw_responses(cfir_coeffs, rfir_coeffs, args.without_decimation)
