from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


def frequency_calc_ctrl(
    freq_target: int,
    sideband: str,
    lo_unit: int = 500_000_000,
    nco_unit: int = 23_437_500,
    lo_min: int = 10_000_000_000,
    lo_max: int = 11_500_000_000,
    cnco_min: int = 700_000_000,
    cnco_max: int = 3_500_000_000,
) -> tuple[int, int, int]:

    freq_cnco = 3_000_000_000  # CNCO の周波数をこの値の近くに設定したい
    if sideband == "L":
        freq_lo = round((freq_target + freq_cnco) / lo_unit) * lo_unit
        if freq_lo < lo_min or freq_lo > lo_max:
            raise ValueError(f"LO frequency {freq_lo} Hz is out of range ({lo_min} Hz to {lo_max} Hz).")
        freq_cnco = round((freq_lo - freq_target) / nco_unit) * nco_unit
        if freq_cnco < cnco_min or freq_cnco > cnco_max:
            raise ValueError(f"CNCO frequency {freq_cnco} Hz is out of range ({cnco_min} Hz to {cnco_max} Hz).")
        freq_awg = freq_lo - (freq_target + freq_cnco)
    elif sideband == "U":
        freq_lo = round((freq_target - freq_cnco) / lo_unit) * lo_unit
        if freq_lo < lo_min or freq_lo > lo_max:
            raise ValueError(f"LO frequency {freq_lo} Hz is out of range ({lo_min} Hz to {lo_max} Hz).")
        freq_cnco = round((freq_target - freq_lo) / nco_unit) * nco_unit
        if freq_cnco < cnco_min or freq_cnco > cnco_max:
            raise ValueError(f"CNCO frequency {freq_cnco} Hz is out of range ({cnco_min} Hz to {cnco_max} Hz).")
        freq_awg = freq_target - (freq_lo + freq_cnco)
    else:
        raise ValueError("Invalid sideband value. Use 'L' for lower sideband or 'U' for upper sideband.")

    return freq_lo, freq_cnco, freq_awg


def frequency_calc_ro(
    freq_target: int,
    sideband: str,
    lo_unit: int = 500_000_000,
    nco_unit: int = 23_437_500,
    lo_min: int = 7_500_000_000,
    lo_max: int = 9_000_000_000,
    cnco_min: int = 700_000_000,
    cnco_max: int = 3_500_000_000,
) -> tuple[int, int]:

    freq_cnco = 2_000_000_000  # CNCO の周波数をこの値の近くに設定したい
    if sideband == "L":
        freq_lo = round((freq_target + freq_cnco) / lo_unit) * lo_unit
        if freq_lo < lo_min or freq_lo > lo_max:
            raise ValueError(f"LO frequency {freq_lo} Hz is out of range ({lo_min} Hz to {lo_max} Hz).")
        freq_cnco = round((freq_lo - freq_target) / nco_unit) * nco_unit
        if freq_cnco < cnco_min or freq_cnco > cnco_max:
            raise ValueError(f"CNCO frequency {freq_cnco} Hz is out of range ({cnco_min} Hz to {cnco_max} Hz).")
    elif sideband == "U":
        freq_lo = round((freq_target - freq_cnco) / lo_unit) * lo_unit
        if freq_lo < lo_min or freq_lo > lo_max:
            raise ValueError(f"LO frequency {freq_lo} Hz is out of range ({lo_min} Hz to {lo_max} Hz).")
        freq_cnco = round((freq_target - freq_lo) / nco_unit) * nco_unit
        if freq_cnco < cnco_min or freq_cnco > cnco_max:
            raise ValueError(f"CNCO frequency {freq_cnco} Hz is out of range ({cnco_min} Hz to {cnco_max} Hz).")
    else:
        raise ValueError("Invalid sideband value. Use 'L' for lower sideband or 'U' for upper sideband.")

    return freq_lo, freq_cnco


def frequency_calc_ro_awg(
    freq_target: int,
    freq_lo: int,
    freq_cnco: int,
    sideband: str,
    awg_min: int = -250_000_000,
    awg_max: int = 250_000_000,
) -> int:

    if sideband not in ("L", "U"):
        raise ValueError("Invalid sideband value. Use 'L' for lower sideband or 'U' for upper sideband.")

    if sideband == "L":
        freq_awg = freq_lo - (freq_target + freq_cnco)
    elif sideband == "U":
        freq_awg = freq_target - (freq_lo + freq_cnco)

    if freq_awg < awg_min or freq_awg > awg_max:
        raise ValueError("AWG frequency is out of range.")

    return freq_awg


def square_pulse(
    num_blank1: int,
    num_flat: int,
    num_blank2: int,
    amplitude: float = 1.0,
) -> npt.NDArray[np.complex64]:

    num_total = num_blank1 + num_flat + num_blank2
    wave = np.zeros(num_total, dtype=np.complex64)
    wave[0:num_blank1] = 0.0
    wave[num_blank1 : num_blank1 + num_flat] = 1.0
    wave[num_blank1 + num_flat : num_total] = 0.0

    return wave * amplitude * 16383


def raised_cosine_flat_top_pulse(
    flat_duration: int,
    rise_time: int,
    amplitude: float = 1.0,
) -> npt.NDArray[np.complex64]:

    time_total = flat_duration + 2 * rise_time
    time_list = np.arange(0, time_total, 2)

    t1 = 0
    t2 = t1 + rise_time
    t3 = t2 + flat_duration
    t4 = t3 + rise_time

    cond_12 = (t1 <= time_list) & (time_list < t2)
    cond_23 = (t2 <= time_list) & (time_list < t3)
    cond_34 = (t3 <= time_list) & (time_list < t4)

    t_12 = time_list[cond_12]
    t_34 = time_list[cond_34]

    waveform_rcft = 0.0 * time_list + 0.0 * 1j
    waveform_rcft[cond_12] = (1.0 - np.cos(np.pi * (t_12 - t1) / rise_time)) / 2
    waveform_rcft[cond_23] = 1.0
    waveform_rcft[cond_34] = (1.0 - np.cos(np.pi * (t4 - t_34) / rise_time)) / 2

    return waveform_rcft * amplitude * 16383.0


def embed_array(
    array_length: int,
    array: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    if len(array) > array_length:
        raise ValueError("The length of ar must not exceed array_length.")

    result = np.zeros(array_length, dtype=np.complex64)
    result[-len(array) :] = array

    return result


def awg_output(
    iq: npt.NDArray[np.complex64],
    freq_if: float,
    phase: float = 0.0,
) -> npt.NDArray[np.complex64]:
    dt = 2.0e-9
    result = iq * np.exp(1j * (2 * np.pi * freq_if * np.arange(len(iq)) * dt + phase))
    return result.astype(np.complex64)


def demodulate(
    iq: npt.NDArray[np.complex64],
    freq_if: float,
    phase: float = 0.0,
) -> npt.NDArray[np.complex64]:
    dt = 2.0e-9
    return iq * np.exp(1j * (2 * np.pi * (-freq_if) * np.arange(len(iq)) * dt + phase))


def fit_and_rotate_iq(iq_sum: np.ndarray) -> np.ndarray:

    A = np.vstack([iq_sum.real, np.ones(len(iq_sum))]).T
    slope, _ = np.linalg.lstsq(A, iq_sum.imag, rcond=None)[0]
    theta_fit = np.arctan(slope)
    theta_rotation = np.pi / 2 - theta_fit
    iq_rot = iq_sum * np.exp(1j * theta_rotation)

    if np.mean(iq_rot.real) < 0:
        iq_rot *= np.exp(1j * np.pi)

    return iq_rot


def rabi_model(x, A, tau, f, phi, B):
    return A * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi) + B


def rabi_fitting(
    ctrl_pulse_width_list: npt.NDArray[np.float64],
    iq_rot: npt.NDArray[np.complex64],
) -> tuple[Any, npt.NDArray[np.float64]]:

    q_norm = 2 * (iq_rot.imag - np.min(iq_rot.imag)) / (np.max(iq_rot.imag) - np.min(iq_rot.imag)) - 1

    param_init = [1.0, 100.0, 15.0e-3, 0.0, 0.0]
    param_bounds = ([0.9, 0.0, 0.0, -np.pi, -0.2], [1.1, 1.5e9, 40.0e-3, np.pi, 0.2])

    popt, pcov = curve_fit(
        rabi_model,
        ctrl_pulse_width_list,
        q_norm,
        p0=param_init,
        bounds=param_bounds,
    )

    return popt, q_norm


def rabi_resonance_fit(f_drive, f_rabi, f_reso, cons):
    return cons * np.sqrt(f_rabi**2 + (f_drive - f_reso) ** 2)
