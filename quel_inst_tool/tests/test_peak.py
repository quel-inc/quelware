import pytest

from quel_inst_tool import ExpectedSpectrumPeaks, MeasuredSpectrumPeak


def test_normal():
    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (8991e6, -20)])

    max_freq_error = 2e6
    m0 = [
        MeasuredSpectrumPeak(8992.0e6, -14.5, max_freq_error),
        MeasuredSpectrumPeak(9988.0e6, -15.4, max_freq_error),
    ]

    e0.validate_with_measurement_condition(max_freq_error)
    j0, s0, w0 = e0.match(m0)
    assert all(j0)
    assert len(s0) == 0
    assert len(w0) == 0


def test_missing():
    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (8991e6, -20)])

    max_freq_error = 2e6
    m0 = [
        MeasuredSpectrumPeak(9988.0e6, -15.4, max_freq_error),
    ]

    assert e0.validate_with_measurement_condition(max_freq_error)
    j0, s0, w0 = e0.match(m0)
    assert j0 == (True, False)  # in the order of e0
    assert len(s0) == 0
    assert len(w0) == 0


def test_sprious():
    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (8991e6, -20)])

    max_freq_error = 2e6
    m0 = [
        MeasuredSpectrumPeak(8992.0e6, -14.5, max_freq_error),
        MeasuredSpectrumPeak(9988.0e6, -15.4, max_freq_error),
        MeasuredSpectrumPeak(8000.0e6, -25.4, max_freq_error),
    ]

    assert e0.validate_with_measurement_condition(max_freq_error)
    j0, s0, w0 = e0.match(m0)
    assert all(j0)
    assert len(s0) == 1 and s0[0] == m0[2]
    assert len(w0) == 0


def test_too_week():
    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (8991e6, -20)])

    max_freq_error = 2e6
    m0 = [
        MeasuredSpectrumPeak(8992.0e6, -14.5, max_freq_error),
        MeasuredSpectrumPeak(9988.0e6, -25.4, max_freq_error),
    ]

    assert e0.validate_with_measurement_condition(max_freq_error)
    j0, s0, w0 = e0.match(m0)
    assert j0 == (False, True)  # in the order of e0
    assert len(s0) == 0
    assert len(w0) == 1 and w0[0] == m0[1]


def test_duplicated():
    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (8991e6, -20)])

    # this measurement can't happen since the difference of two peaks must be more than twice of max_freq_error.
    max_freq_error = 2e6
    m0 = [
        MeasuredSpectrumPeak(8992.0e6, -14.5, max_freq_error),
        MeasuredSpectrumPeak(8990.0e6, -15.4, max_freq_error),
        MeasuredSpectrumPeak(9988.0e6, -15.4, max_freq_error),
    ]

    assert e0.validate_with_measurement_condition(max_freq_error)
    with pytest.raises(ValueError):
        _, _, _ = e0.match(m0)


def test_invalid_expected():
    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (9986e6, -20)])

    max_freq_error = 2e6
    assert not e0.validate_with_measurement_condition(max_freq_error)

    # notes: it should be stopped to apply measurements to invalid e0, usually.
    m0 = [
        MeasuredSpectrumPeak(8992.0e6, -14.5, max_freq_error),
        MeasuredSpectrumPeak(9988.0e6, -15.4, max_freq_error),
    ]
    with pytest.raises(ValueError):
        _, _, _ = e0.match(m0)


def test_extract_matched_normal():
    e0 = ExpectedSpectrumPeaks([(9987e6, -20)])

    max_freq_error = 2e6
    m0 = [
        MeasuredSpectrumPeak(8992.0e6, -14.5, max_freq_error),
        MeasuredSpectrumPeak(9988.0e6, -15.4, max_freq_error),
    ]

    e0.validate_with_measurement_condition(max_freq_error)
    d0 = e0.extract_matched(m0)

    assert len(d0) == 1
    d00 = d0.pop()
    assert d00.power == -15.4


def test_extract_matched_missing():
    e0 = ExpectedSpectrumPeaks([(9987e6, -20)])

    max_freq_error = 2e6
    m0 = [
        MeasuredSpectrumPeak(8992.0e6, -14.5, max_freq_error),
    ]

    e0.validate_with_measurement_condition(max_freq_error)
    d0 = e0.extract_matched(m0)

    assert len(d0) == 0


if __name__ == "__main__":
    max_freq_error = 2e6
    m0 = [
        MeasuredSpectrumPeak(8992.0e6, -14.5, max_freq_error),
        MeasuredSpectrumPeak(9988.0e6, -25.4, max_freq_error),
    ]

    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (8991e6, -20)])
    j0, s0, w0 = e0.match(m0)
