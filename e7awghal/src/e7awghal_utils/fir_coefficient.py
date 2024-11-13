import logging
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.signal import firwin

from e7awghal.common_defs import _CFIR_NTAPS, _RFIRS_NTAPS, DECIMATION_RATE, SAMPLING_FREQ

logger = logging.getLogger(__name__)

WindowType = Union[float, str, tuple[str, float]]


def complex_fir_bpf(
    target_freq: float,
    bandwidth: float,
    *,
    window: WindowType = "hamming",
) -> npt.NDArray[np.complex64]:
    """
    Design a complex band pass filter.

    Args:
        target_freq (float): Target frequency [Hz]
        bandwidth (float): Pass band [Hz]
    Returns:
        npt.NDArray[np.complex64]: FIR coefficients
    """

    # Notes: since low-pass filter in [0,fs/2] is used as band-pass filter [-fs/2, fs/2], 
    # the cutoff frequency must be half of the bandwidth. 
    #           f_cutoff = bandwidth / 2 [Hz]
    # f_cutoff must be normalized by nyquist frequency fs/2 for firwin
    #           cutoff = f_cutoff / (fs/2) = bandwidth / fs
    coeff = firwin(_CFIR_NTAPS, cutoff=bandwidth / SAMPLING_FREQ, pass_zero="lowpass", window=window)

    # Generate complex exponential to shift the filter to the target frequency
    t = np.arange(_CFIR_NTAPS)
    complex_exp = np.exp(1j * 2 * np.pi * target_freq * t / SAMPLING_FREQ)

    # Multiply low-pass filter by complex exponential to shift its frequency response
    complex_coeff = coeff * complex_exp

    return complex_coeff[::-1]  # reverse list to be argument for CaptureParam


def real_fir_bpf(
    target_freq: float,
    bandwidth: float,
    *,
    window: WindowType = "hamming",
    decimated_input: bool = True,
) -> tuple[float, npt.NDArray[np.float32]]:
    """
    Design a real band pass filter.

    Args:
        target_freq (float): Target frequency [Hz]
        bandwidth (float): Pass band [Hz]
        decimated_input (bool): True for 1/4 decimated input. both target_freq and span are converted automatically.
    Returns:
        npt.NDArray[np.float32]: FIR coefficients
    """

    sampling_freq = SAMPLING_FREQ
    if decimated_input:
        target_freq = _folded_frequency_by_decimation(target_freq)
        sampling_freq /= DECIMATION_RATE

    # Notes: cutoff must be normalized by nyquist frequency which is half of sampling frequency
    #   low_cutoff = ( target_freq - span / 2 ) / ( sampling_freq / 2 )
    #   high_cutoff = ( target_freq + span / 2 ) / ( sampling_freq / 2 )
    low_cutoff = (2.0 * abs(target_freq) - bandwidth) / sampling_freq
    high_cutoff = (2.0 * abs(target_freq) + bandwidth) / sampling_freq
    logger.debug(f"low_cutoff = {low_cutoff:.3f}, high_cutoff = {high_cutoff:.3f}")
    if 0.0 < low_cutoff and high_cutoff < 1.0:
        coeff = firwin(_RFIRS_NTAPS, cutoff=[low_cutoff, high_cutoff], pass_zero="bandpass", window=window)
    elif low_cutoff <= 0.0 and high_cutoff < 1.0:
        coeff = firwin(_RFIRS_NTAPS, cutoff=high_cutoff, pass_zero="lowpass", window=window)
    elif 0.0 < low_cutoff and 1.0 <= high_cutoff:
        # Notes: it is impossible to make highpass filter with even number of taps.
        coeff = firwin(_RFIRS_NTAPS - 1, cutoff=low_cutoff, pass_zero="highpass", window=window)
    else:
        logger.warning(
            f"specified bandwidth {bandwidth} is wider than nyquist frequency {sampling_freq / 2}, "
            "generating identity coefficients for RFIR"
        )
        coeff = np.zeros(_RFIRS_NTAPS, dtype=np.float32)
        coeff[0] = 1.0

    return target_freq, coeff[::-1]  # reverse list to be argument for CaptureParam


def _folded_frequency_by_decimation(frequency: float) -> float:
    """
    Convert frequency by downsampling 1/4.

    Args:
        frequency (float): Frequency before downsampling [Hz]
    Returns:
        float: Converted frequency [Hz]
    """
    sign = np.sign(frequency)
    new_sampling_freq = SAMPLING_FREQ / DECIMATION_RATE
    new_nyquist_freq = new_sampling_freq / 2

    folded_freq = abs(frequency) % new_sampling_freq

    if folded_freq > new_nyquist_freq:
        folded_freq = -(new_sampling_freq - folded_freq)

    return sign * folded_freq
