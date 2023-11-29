import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quel_inst_tool import Ms2xxxx, Ms2xxxxTraceMode

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def check_environment(spectrum_analyzer) -> None:
    _, obj = spectrum_analyzer
    if obj.prod_id != "MS2720T" and obj.prod_id != "MS2090A":
        pytest.skip("The spectrum analyzer must be MS2720T for this test.")
    return None


@pytest.fixture(scope="module")
def outsubdir(outdir) -> Path:
    os.makedirs(outdir / "ms2xxxx_peakgen")
    return outdir / "ms2xxxx_peakgen"


@pytest.mark.usefixtures("check_environment")
class TestMs2xxxxPeak:
    def _gen_spectrum_image(self, trace, filepath):
        plt.cla()
        plt.plot(trace[:, 0], trace[:, 1])
        plt.savefig(filepath)

    def test_trace_capture_normal(self, signal_generator, spectrum_analyzer, outsubdir):
        obj: Ms2xxxx
        name, obj = spectrum_analyzer
        signal_generator.channel[0].frequency = 1e10
        signal_generator.channel[1].frequency = 8e9
        signal_generator.channel[0].power = -20
        signal_generator.channel[1].power = -30
        signal_generator.channel[0].enable = True
        signal_generator.channel[1].enable = True
        time.sleep(0.1)  # need to wait for SG ouputs because trace capture is so fast for MS2090A.
        obj.continuous_sweep = False
        obj.trace_mode = Ms2xxxxTraceMode.NORM
        freq_span = 5e9
        obj.freq_range_set(9e9, 5e9)
        obj.resolution_bandwidth = 1e5
        obj.average_enable = False

        fd, pk = obj.trace_and_peak_get(minimum_power=-40.0)
        assert fd.shape == (obj.sweep_points, 2)
        assert len(pk) == 2
        assert np.abs(pk[0, 0] - signal_generator.channel[1].frequency) < freq_span / (obj.sweep_points - 1) + 1
        # may need to be changed due to the attenuation from SG to the analyzer
        assert pk[0, 1] > signal_generator.channel[1].power - 10.0
        assert np.abs(pk[1, 0] - signal_generator.channel[0].frequency) < freq_span / (obj.sweep_points - 1) + 1
        # may need to be changed due to the attenuation from SG to the analyzer
        assert pk[1, 1] > signal_generator.channel[0].power - 10.0
        self._gen_spectrum_image(fd, outsubdir / Path(name + "_typical"))
        signal_generator.channel[0].enable = False
        signal_generator.channel[1].enable = False
