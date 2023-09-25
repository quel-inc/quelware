import logging
from typing import Final, List, Set

import pytest
from device_availablity import QuelInstDevice

from quel_inst_tool import E440xbTraceMode, E440xbWritableParams

logger = logging.getLogger(__name__)

DEVICES: Final[List[str]] = [QuelInstDevice.E4405B, QuelInstDevice.E4407B]


# Notes: scope="session" disrupts the test results. (TODO: investigate the reason)
@pytest.mark.parametrize("spectrum_analyzer", DEVICES, indirect=True)
def test_basic(spectrum_analyzer):
    name, obj = spectrum_analyzer

    obj.freq_range_set(8e9, 4e9)
    assert obj.freq_center == 8e9
    assert obj.freq_span == 4e9
    assert obj.freq_start == 6e9

    obj.sweep_points = 1001
    freqs = obj.freq_points
    assert len(freqs) == 1001
    freqs[0] = 4e9
    freqs[-1] = 10e9

    obj.display_enable = True
    assert obj.display_enable
    obj.display_enable = False
    assert not obj.display_enable

    obj.input_attenuation = 20
    assert obj.input_attenuation == 20
    obj.input_attenuation = 10
    assert obj.input_attenuation == 10

    obj.trace_mode = E440xbTraceMode.MAXHOLD
    assert obj.trace_mode == E440xbTraceMode.MAXHOLD
    obj.trace_mode = E440xbTraceMode.WRITE
    assert obj.trace_mode == E440xbTraceMode.WRITE


@pytest.mark.parametrize("spectrum_analyzer", DEVICES, indirect=True)
def test_trace_capture(spectrum_analyzer):
    name, obj = spectrum_analyzer

    obj.trace_mode = E440xbTraceMode.WRITE
    obj.freq_range_set(9e9, 5e9)
    obj.sweep_points = 4001
    obj.resolution_bandwidth = 1e5
    obj.average_enable = False

    fd, pk = obj.trace_and_peak_get(minimum_power=-40.0)
    assert fd.shape == (4001, 2)
    assert len(pk) == 0


@pytest.mark.parametrize("spectrum_analyzer", DEVICES, indirect=True)
@pytest.mark.parametrize(
    ("freq_center", "freq_span", "valid_parameter", "out_of_range"),
    [
        (9e9, 5e9, True, {}),
        (9e9, 10e9, True, {QuelInstDevice.E4405B}),
        (9e9, None, False, {}),
        (23e9, 8e9, True, {QuelInstDevice.E4405B, QuelInstDevice.E4407B}),
        (None, 10e9, False, {}),
    ],
)
def test_app_interface(
    freq_center: float, freq_span: float, valid_parameter: bool, out_of_range: Set[str], spectrum_analyzer
):
    name, obj = spectrum_analyzer

    param0 = E440xbWritableParams(
        trace_mode=E440xbTraceMode.WRITE,
        freq_center=freq_center,
        freq_span=freq_span,
        sweep_points=801,
        resolution_bandwidth=1e5,
        average_enable=False,
    )

    if valid_parameter:
        is_ok = obj.freq_range_check(freq_center, freq_span)
        if name in out_of_range:
            assert not is_ok
            with pytest.raises(ValueError):
                obj.freq_range_set(freq_center, freq_span)
            assert not param0.update_device_parameter(obj)
        else:
            assert is_ok
            obj.freq_range_set(freq_center, freq_span)
            assert obj.freq_center == freq_center
            assert obj.freq_span == freq_span

            # clear the settings.
            obj.freq_range_set(8e9, 1e9)

            assert param0.update_device_parameter(obj)
            assert obj.freq_center == freq_center
            assert obj.freq_span == freq_span
    else:
        assert not param0.update_device_parameter(obj)
