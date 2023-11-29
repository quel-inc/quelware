import logging
import os
import time
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quel_inst_tool import E440xb, E440xbTraceMode, Ms2xxxxTraceMode, Ms2720t, SpectrumAnalyzer, SynthHDSweepParams

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def outsubdir(outdir) -> Path:
    os.makedirs(outdir / "synthHD")
    return outdir / "synthHD"


def test_basic_params(signal_generator):
    signal_generator.channel[0].enable = False
    assert signal_generator.channel[0].enable is False
    signal_generator.channel[1].enable = False
    assert signal_generator.channel[1].enable is False

    signal_generator.channel[0].frequency = 100000000
    assert signal_generator.channel[0].frequency == 100000000
    signal_generator.channel[1].frequency = 10000000000
    assert signal_generator.channel[1].frequency == 10000000000
    signal_generator.channel[0].power = 0
    assert signal_generator.channel[0].power == 0
    signal_generator.channel[1].power = -10
    assert signal_generator.channel[1].power == -10

    signal_generator.channel[0].enable = True
    assert signal_generator.channel[0].enable is True
    signal_generator.channel[1].enable = True
    assert signal_generator.channel[1].enable is True

    signal_generator.channel[0].enable = False
    assert signal_generator.channel[0].enable is False
    signal_generator.channel[1].enable = False
    assert signal_generator.channel[1].enable is False

    sweep_params_write = SynthHDSweepParams(
        sweep_freq_low=100000000,
        sweep_freq_high=10000000000.0,
        sweep_freq_step=100000000.0,
        sweep_time_step=100.0,
        sweep_power_high=0.0,
        sweep_power_low=0.0,
        sweep_direction=1,
        sweep_type=0,
        sweep_cont=False,
    )
    sweep_params_write.update_device_parameter(signal_generator.channel[0])
    sweep_params_read = SynthHDSweepParams.from_synthHD(signal_generator.channel[0])
    assert sweep_params_read.sweep_freq_low == sweep_params_write.sweep_freq_low
    assert sweep_params_read.sweep_freq_high == sweep_params_write.sweep_freq_high
    assert sweep_params_read.sweep_freq_step == sweep_params_write.sweep_freq_step
    assert sweep_params_read.sweep_time_step == sweep_params_write.sweep_time_step
    assert sweep_params_read.sweep_power_low == sweep_params_write.sweep_power_low
    assert sweep_params_read.sweep_power_high == sweep_params_write.sweep_power_high
    assert sweep_params_read.sweep_direction == sweep_params_write.sweep_direction
    assert sweep_params_read.sweep_cont == sweep_params_write.sweep_cont


def _gen_spectrum_image(trace, filepath):
    plt.cla()
    plt.plot(trace[:, 0], trace[:, 1])
    plt.savefig(filepath)


def test_generation(signal_generator, spectrum_analyzer, outsubdir):
    spa: SpectrumAnalyzer
    _, spa = spectrum_analyzer
    signal_generator.channel[0].frequency = 9000000000
    assert signal_generator.channel[0].frequency == 9000000000
    signal_generator.channel[0].power = -10
    assert signal_generator.channel[0].power == -10

    signal_generator.channel[1].frequency = 8000000000
    assert signal_generator.channel[1].frequency == 8000000000
    signal_generator.channel[1].power = -20
    assert signal_generator.channel[1].power == -20

    if isinstance(spa, Ms2720t):
        spa.trace_mode = Ms2xxxxTraceMode.NORM
    elif isinstance(spa, E440xb):
        spa.trace_mode = E440xbTraceMode.WRITE
    else:
        raise AssertionError

    freq_center = 9e9
    freq_span = 3e9
    spa.freq_range_set(freq_center, freq_span)

    if spa.prod_id != "MS2720T":  # for MS2720T, The number of sweep points is fixed at 551
        spa.sweep_points = 4001

    spa.resolution_bandwidth = 1e5

    spa.average_enable = False
    signal_generator.channel[0].enable = True
    fd, pk = spa.trace_and_peak_get(minimum_power=-40.0)
    _gen_spectrum_image(fd, outsubdir / "CH0")
    assert np.abs(pk[0, 0] - signal_generator.channel[0].frequency) < freq_span / (spa.sweep_points - 1) + 1
    # may need to be changed due to the attenuation from SG to the analyzer
    assert pk[0, 1] > signal_generator.channel[0].power - 10.0
    signal_generator.channel[0].enable = False

    signal_generator.channel[1].enable = True
    fd, pk = spa.trace_and_peak_get(minimum_power=-40.0)
    _gen_spectrum_image(fd, outsubdir / "CH1")
    assert np.abs(pk[0, 0] - signal_generator.channel[1].frequency) < freq_span / (spa.sweep_points - 1) + 1
    # may need to be changed due to the attenuation from SG to the analyzer
    assert pk[0, 1] > signal_generator.channel[1].power - 10.0
    signal_generator.channel[1].enable = False


def test_sweep_params(signal_generator, spectrum_analyzer):
    spa: SpectrumAnalyzer
    spa_name, spa = spectrum_analyzer
    sweep_params_write = SynthHDSweepParams(
        sweep_freq_low=100000000,
        sweep_freq_high=10000000000.0,
        sweep_freq_step=100000000.0,
        sweep_time_step=100.0,
        sweep_power_high=0.0,
        sweep_power_low=0.0,
        sweep_direction=1,
        sweep_type=0,
        sweep_cont=False,
    )
    sweep_params_write.update_device_parameter(signal_generator.channel[0])
    sweep_params_read = SynthHDSweepParams.from_synthHD(signal_generator.channel[0])
    assert sweep_params_read.sweep_freq_low == sweep_params_write.sweep_freq_low
    assert sweep_params_read.sweep_freq_high == sweep_params_write.sweep_freq_high
    assert sweep_params_read.sweep_freq_step == sweep_params_write.sweep_freq_step
    assert sweep_params_read.sweep_time_step == sweep_params_write.sweep_time_step
    assert sweep_params_read.sweep_power_low == sweep_params_write.sweep_power_low
    assert sweep_params_read.sweep_power_high == sweep_params_write.sweep_power_high
    assert sweep_params_read.sweep_direction == sweep_params_write.sweep_direction
    assert sweep_params_read.sweep_cont == sweep_params_write.sweep_cont

    freq_center = 5.1e9
    freq_span = 1e10
    spa.freq_range_set(freq_center, freq_span)
    if spa.prod_id == "MS2720T":
        cast(Ms2720t, spa).continuous_sweep = True
        # Note: for MS2720T, The number of sweep points is fixed at 551
    else:
        spa.sweep_points = 4001
        # Note: continuous_sweep feature is not implemented for E440xB yet.

    spa.resolution_bandwidth = 1e5

    signal_generator.channel[0].enable = True
    signal_generator.channel[0].run_sweep = True  # Need to check with analyzer
    assert signal_generator.channel[0].enable is True
    assert signal_generator.channel[0].run_sweep is True

    for _ in range(10):
        time.sleep(1)
        if not signal_generator.channel[0].run_sweep:
            break

    signal_generator.channel[0].enable = False
