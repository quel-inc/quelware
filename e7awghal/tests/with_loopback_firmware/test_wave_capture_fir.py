import logging
import os
import shutil
from pathlib import Path
from typing import Final, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from e7awghal.awgunit import AwgUnit
from e7awghal.capctrl import CapCtrlStandard
from e7awghal.capparam import CapParam, CapSection
from e7awghal.common_defs import DECIMATION_RATE
from e7awghal.fwtype import E7FwType
from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal
from e7awghal.wavedata import AwgParam, WaveChunk
from e7awghal_utils.fir_coefficient import _folded_frequency_by_decimation, complex_fir_bpf, real_fir_bpf
from testlibs.awgctrl_with_hlapi import AwgUnitHL
from testlibs.capunit_with_hlapi import CapUnitHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger(__name__)

TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.74",
            "auidx": 2,
            "cmidx": 0,
        },
    },
)

OUTPUT_SETTING = {
    "wave_image_path": Path("./artifacts/dsp"),
}

# -150.0 MHz and -100 MHz can be chosen by complex FIR BPF but cannot be
# chosen by real FIR BPF after downsampling. This is because the -150 MHz and -100 MHz
# are converted to -25 MHz and 25 MHz by downsampling, and the real FIR BPF cannot
# separate these two frequencies.
# On the other hand, 200 MHz and 230 MHz is hardly selected by complex FIR BPF since
# the separation is too small. However, real FIR BPF after downsampling can separate
# 200 MHz and 230 MHz.

RO_FREQS: List[float] = [-150.0e6, -100.0e6, 200.0e6, 230.0e6]
RO_DURATION: Final[float] = 1.536e-6
DT: Final[float] = 2.0e-9
PASS_BAND_WIDTH: Final[float] = 25.0e6


def make_outdir(label: str):
    mpl.use("Gtk3Agg")  # TODO: reconsider where to execute.

    dirpath = OUTPUT_SETTING["wave_image_path"] / label
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    return dirpath


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def proxy_au_cm_w(request):

    proxy = create_quel1au50hal_for_test(
        ipaddr_wss=request.param["box_config"]["ipaddr_wss"], auth_callback=lambda: True
    )
    assert proxy.fw_type() == E7FwType.SIMPLEMULTI_STANDARD

    proxy.initialize()

    au: AwgUnit = proxy.awgunit(request.param["box_config"]["auidx"])
    cmidx = request.param["box_config"]["cmidx"]

    t = np.linspace(0, RO_DURATION, int(RO_DURATION / DT), endpoint=False, dtype=np.float32)
    w_sum = np.sum([4096.0 * np.exp(1j * 2.0 * np.pi * freq * t) for freq in RO_FREQS], axis=0)
    au.register_wavedata_from_complex64vector("w", w_sum.astype(np.complex64))
    param_w = AwgParam(num_wait_word=16)  # Notes: loopback firmware has delay of 64 samples
    param_w.chunks.append(WaveChunk(name_of_wavedata="w"))
    au.load_parameter(param_w)
    yield proxy, au.unit_index, cmidx, w_sum


def au50loopback(
    proxy: AbstractQuel1Au50Hal, auidx: int, cmidx: int, cps: List[CapParam]
) -> list[npt.NDArray[np.complex64]]:
    cc = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard)
    cus = [proxy.capunit(unit) for unit in cc.units_of_module(cmidx)]
    for cu in cus:
        assert isinstance(cu, CapUnitHL)
    cc.set_triggering_awgunit_idx(capmod_idx=cmidx, awgunit_idx=auidx)
    for i, cu in enumerate(cus):
        cc.add_triggerable_unit(cu.unit_index)
        cu.load_parameter(cps[i])
    futs = [cu.wait_for_triggered_capture() for cu in cus]

    au = proxy.awgunit(auidx)
    assert isinstance(au, AwgUnitHL)
    au.start_now().result()
    au.wait_done().result()

    rdrs = [fut.result() for fut in futs]
    return [rdr.as_wave_list() for rdr in rdrs]


def test_cfir(proxy_au_cm_w):
    proxy, auidx, cmidx, w = proxy_au_cm_w
    cps = []
    for freq in RO_FREQS:
        cp = CapParam(
            num_wait_word=0,
            num_repeat=1,
            complexfir_enable=True,
            complexfir_coeff=complex_fir_bpf(target_freq=freq, bandwidth=PASS_BAND_WIDTH),
            complexfir_exponent_offset=15,
        )
        cp.sections.append(CapSection(name="s0", num_capture_word=int(RO_DURATION / DT / 4)))
        cps.append(cp)

    data = au50loopback(proxy, auidx, cmidx, cps)
    outdir = make_outdir("cfir")

    p = abs(np.fft.fft(w))
    f = np.fft.fftfreq(len(p), DT)

    plt.cla()
    plt.plot(f, p / (RO_DURATION / DT), marker="o", linestyle="None")
    plt.savefig(outdir / "Spectrum_Raw.png")

    largest_peak_indices = np.argsort(p)[-4:][::-1]  # indices of 4 largest peaks

    for i, freq in enumerate(RO_FREQS):
        d00 = data[i][0]
        assert len(d00) == 1
        assert len(d00[0]) == int(RO_DURATION / DT)

        p = abs(np.fft.fft(d00[0]))
        f = np.fft.fftfreq(len(p), DT)

        max_idx = np.argmax(p)

        for peak_idx in largest_peak_indices:
            if peak_idx == max_idx:
                continue
            if abs(abs(f[peak_idx] - f[max_idx]) - 50e6) < abs(f[1] - f[0]):
                assert p[peak_idx] / p[max_idx] < 0.2  # seperation between-150 MHz and -100 MHz
            elif abs(abs(f[peak_idx] - f[max_idx]) - 30e6) < abs(f[1] - f[0]):
                assert p[peak_idx] / p[max_idx] < 0.6  # seperation between 200 MHz and 230 MHz
            else:
                assert p[peak_idx] / p[max_idx] < 2e-2

        assert abs(f[max_idx] - freq) < abs(f[1] - f[0])

        plt.cla()
        plt.plot(f, p / (RO_DURATION / DT), marker="o", linestyle="None")
        plt.savefig(outdir / f"Spectrum{int(freq / 1e6):03d}MHZ.png")


def test_folded_freq():
    folded_freqs = [_folded_frequency_by_decimation(freq) for freq in RO_FREQS]
    assert abs(folded_freqs[0] + 25e6) < 1e-6  # -150 MHz -> - 25 MHz
    assert abs(folded_freqs[1] - 25e6) < 1e-6  # -100 MHz -> 25 MHz
    assert abs(folded_freqs[2] + 50e6) < 1e-6  # 200 MHz -> -50 MHz
    assert abs(folded_freqs[3] + 20e6) < 1e-6  # 230 MHz -> -20 MHz


def test_rfir(proxy_au_cm_w):
    proxy, auidx, cmidx, _ = proxy_au_cm_w
    folded_freqs = [_folded_frequency_by_decimation(freq) for freq in RO_FREQS]
    cps = []
    phase_shifts = []
    for i, freq in enumerate(RO_FREQS):
        _, phase_shift, rfir_coeff = real_fir_bpf(target_freq=freq, bandwidth=PASS_BAND_WIDTH, decimated_input=True)
        cp = CapParam(
            num_wait_word=0,
            num_repeat=1,
            complexfir_enable=True,
            complexfir_coeff=complex_fir_bpf(target_freq=freq, bandwidth=PASS_BAND_WIDTH),
            complexfir_exponent_offset=15,
            decimation_enable=True,
            realfirs_enable=True,
            realfirs_real_coeff=rfir_coeff,
            realfirs_imag_coeff=rfir_coeff,
            realfirs_exponent_offset=15,
        )
        cp.sections.append(CapSection(name="s0", num_capture_word=int(RO_DURATION / DT / 4)))
        cps.append(cp)
        phase_shifts.append(phase_shift)

    data = au50loopback(proxy, auidx, cmidx, cps)
    outdir = make_outdir("rfir")
    peak_indices = []
    for i, freq in enumerate(RO_FREQS):
        d00 = data[i][0]
        assert len(d00) == 1
        assert len(d00[0]) == int(RO_DURATION / DT / DECIMATION_RATE)
        p = abs(np.fft.fft(d00[0]))
        f = np.fft.fftfreq(len(p), DECIMATION_RATE * DT)

        max_idx = np.argmax(p)
        peak_indices.append(max_idx)
        assert abs(f[max_idx] - folded_freqs[i]) < abs(f[1] - f[0])

        # phases are obtained with demodulation since fft includes phase offset.
        t = np.linspace(0, RO_DURATION, int(RO_DURATION / DT / DECIMATION_RATE), endpoint=False, dtype=np.float32)
        iq = np.sum(d00[0] * np.exp(-1j * 2.0 * np.pi * folded_freqs[i] * t))
        # check if detected phase shift matches the expected wthin 1 degree.
        assert abs(np.arctan2(iq.imag, iq.real) - phase_shifts[i]) < np.pi / 180.0

        plt.cla()
        plt.plot(f, p / (RO_DURATION / DT / 4), marker="o", linestyle="None")
        plt.savefig(outdir / f"Spectrum{int(freq / 1e6):03d}MHZ.png")

    for i in range(len(RO_FREQS)):
        d00 = data[i][0]

        p = abs(np.fft.fft(d00[0]))
        f = np.fft.fftfreq(len(p), DECIMATION_RATE * DT)

        # seperation between -150 MHz and -100 MHz does not improve by real FIR BPF but other seperation improves
        for idx in peak_indices:
            if idx == peak_indices[i]:
                continue
            if abs(f[idx] + f[peak_indices[i]]) < abs(f[1] - f[0]):
                assert p[idx] / p[peak_indices[i]] < 0.2  # seperation between -150 MHz and -100 MHz
            elif abs(abs(f[idx] - f[peak_indices[i]]) - 30e6) < abs(f[1] - f[0]):
                assert p[idx] / p[peak_indices[i]] < 1e-1  # seperation between 200 MHz and 230 MHz
            elif abs(abs(f[idx] - f[peak_indices[i]]) - 5e6) < abs(f[1] - f[0]):
                continue
            else:
                assert p[idx] / p[peak_indices[i]] < 1e-2
