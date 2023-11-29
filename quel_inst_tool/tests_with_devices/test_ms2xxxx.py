import logging
import os
from pathlib import Path
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quel_inst_tool import Ms2xxxx, Ms2xxxxTraceMode, Ms2090a, Ms2720t

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def check_environment(spectrum_analyzer) -> None:
    _, obj = spectrum_analyzer
    if obj.prod_id != "MS2720T" and obj.prod_id != "MS2090A":
        pytest.skip("The spectrum analyzer must be MS2720T for this test.")
    return None


@pytest.fixture(scope="module")
def outsubdir(outdir) -> Path:
    os.makedirs(outdir / "ms2xxxx")
    return outdir / "ms2xxxx"


@pytest.mark.usefixtures("check_environment")
class TestMs2xxxx:
    def test_basic(self, spectrum_analyzer):
        obj: Ms2xxxx
        _, obj = spectrum_analyzer

        obj.freq_range_set(8e9, 4e9)

        assert obj.freq_center == 8e9
        assert obj.freq_span == 4e9
        assert obj.freq_start == 6e9

        if isinstance(obj, Ms2720t):
            with pytest.raises(ValueError):
                obj.sweep_points = 1000

        if isinstance(obj, Ms2090a):
            obj.sweep_points = 551
            assert obj.sweep_points == 551

        freqs = obj.freq_points
        assert len(freqs) == obj.sweep_points

        obj.display_enable = True
        assert obj.display_enable
        obj.display_enable = False
        assert not obj.display_enable

        obj.average_enable = True
        assert obj.average_enable
        obj.average_enable = False
        assert not obj.average_enable

        obj.input_attenuation = 20
        assert obj.input_attenuation == 20
        obj.input_attenuation = 10
        assert obj.input_attenuation == 10

        obj.average_count = 5
        assert obj.average_count == 5
        obj.average_count = 10
        assert obj.average_count == 10

        if isinstance(obj, Ms2720t):
            with pytest.raises(ValueError):
                obj.trace_mode = Ms2xxxxTraceMode.RMAXHOLD

        if isinstance(obj, Ms2090a):
            obj.trace_mode = Ms2xxxxTraceMode.RMAXHOLD
            assert obj.trace_mode == Ms2xxxxTraceMode.RMAXHOLD

        obj.trace_mode = Ms2xxxxTraceMode.NORM
        assert obj.trace_mode == Ms2xxxxTraceMode.NORM

        obj.peak_threshold = 20
        assert obj.peak_threshold == 20
        obj.peak_threshold = 10
        assert obj.peak_threshold == 10

        obj.max_peaksearch_trials = 2000
        assert obj.max_peaksearch_trials == 2000
        obj.max_peaksearch_trials = 10
        assert obj.max_peaksearch_trials == 10

        assert obj.holdmode_nsweeps == 10
        obj.holdmode_nsweeps = 20
        assert obj.holdmode_nsweeps == 20
        obj.holdmode_nsweeps = 10
        assert obj.holdmode_nsweeps == 10

        obj.continuous_sweep = False
        assert obj.continuous_sweep is False
        assert obj.init_cont is False

        obj.continuous_sweep = True
        assert obj.continuous_sweep is True
        assert obj.init_cont is True

    def _gen_spectrum_image(self, trace, filepath):
        plt.cla()
        plt.plot(trace[:, 0], trace[:, 1])
        plt.savefig(filepath)

    def test_trace_capture_exp(self, spectrum_analyzer, outsubdir):
        obj: Ms2xxxx
        name, obj = spectrum_analyzer

        obj.continuous_sweep = True
        obj.freq_range_set(9e9, 5e9)
        obj.resolution_bandwidth = 1e5
        obj.average_enable = False

        fd, pk = obj.trace_and_peak_get(minimum_power=-40.0)
        assert fd.shape == (obj.sweep_points, 2)
        assert len(pk) == 0
        self._gen_spectrum_image(fd, outsubdir / Path(name + "_experimental"))

    def test_trace_capture_normal(self, spectrum_analyzer, outsubdir):
        obj: Ms2xxxx
        name, obj = spectrum_analyzer

        obj.continuous_sweep = False
        obj.trace_mode = Ms2xxxxTraceMode.NORM
        obj.freq_range_set(9e9, 5e9)
        obj.resolution_bandwidth = 1e5
        obj.average_enable = False

        fd, pk = obj.trace_and_peak_get(minimum_power=-40.0)
        assert fd.shape == (obj.sweep_points, 2)
        assert len(pk) == 0
        self._gen_spectrum_image(fd, outsubdir / Path(name + "_typical"))

    def test_trace_capture_maximum(self, spectrum_analyzer, outsubdir):
        obj: Ms2xxxx
        name, obj = spectrum_analyzer

        obj.continuous_sweep = False
        obj.freq_range_set(9e9, 5e9)
        obj.resolution_bandwidth = 1e5
        if isinstance(obj, Ms2720t):
            obj.trace_mode = Ms2xxxxTraceMode.MAXHOLD
        elif isinstance(obj, Ms2090a):
            obj.trace_mode = Ms2xxxxTraceMode.RMAXHOLD
        else:
            raise AssertionError

        obj.holdmode_nsweeps = 15
        fd, pk = obj.trace_and_peak_get(minimum_power=-80.0)
        assert fd.shape == (obj.sweep_points, 2)
        assert np.abs(fd[np.argmax(fd[:, 1]), 0] - pk[np.argmax(pk[:, 1]), 0]) < 1
        self._gen_spectrum_image(fd, outsubdir / Path(name + "_maximum"))

    def test_trace_capture_minimum(self, spectrum_analyzer, outsubdir):
        obj: Ms2xxxx
        name, obj = spectrum_analyzer

        obj.continuous_sweep = False
        obj.freq_range_set(9e9, 5e9)
        obj.resolution_bandwidth = 1e5
        if isinstance(obj, Ms2720t):
            obj.trace_mode = Ms2xxxxTraceMode.MINHOLD
        elif isinstance(obj, Ms2090a):
            obj.trace_mode = Ms2xxxxTraceMode.RMINHOLD
        else:
            raise AssertionError

        obj.holdmode_nsweeps = 10

        fd, pk = obj.trace_and_peak_get(minimum_power=-40.0)
        assert fd.shape == (obj.sweep_points, 2)
        assert len(pk) == 0
        self._gen_spectrum_image(fd, outsubdir / Path(name + "_minimum"))

    def test_trace_capture_average(self, spectrum_analyzer, outsubdir):
        obj: Ms2xxxx
        name, obj = spectrum_analyzer

        obj.continuous_sweep = False
        obj.trace_mode = Ms2xxxxTraceMode.AVER
        obj.freq_range_set(9e9, 5e9)
        obj.resolution_bandwidth = 1e5
        obj.average_enable = True

        fd, pk = obj.trace_and_peak_get(minimum_power=-40.0)
        assert fd.shape == (obj.sweep_points, 2)
        assert len(pk) == 0
        self._gen_spectrum_image(fd, outsubdir / Path(name + "_average"))

    @pytest.mark.parametrize(
        ("freq_center", "freq_span", "valid_parameter", "out_of_range"),
        [
            (9e9, 5e9, True, {}),
            (9e9, 10e9, True, {}),
            (9e9, None, False, {}),
            (33e9, 8e9, True, {"MS2720T", "MS2090A"}),
            (None, 10e9, False, {}),
        ],
    )
    def test_app_interface(
        self, freq_center: float, freq_span: float, valid_parameter: bool, out_of_range: Set[str], spectrum_analyzer
    ):
        obj: Ms2720t
        _, obj = spectrum_analyzer

        obj.trace_mode = Ms2xxxxTraceMode.NORM
        obj.resolution_bandwidth = 1e5
        obj.average_enable = False

        if valid_parameter:
            is_ok = obj.freq_range_check(freq_center, freq_span)
            if obj.prod_id in out_of_range:
                assert not is_ok
                with pytest.raises(ValueError):
                    obj.freq_range_set(freq_center, freq_span)
            else:
                assert is_ok
                obj.freq_range_set(freq_center, freq_span)
                assert obj.freq_center == freq_center
                assert obj.freq_span == freq_span
