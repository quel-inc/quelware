import numpy as np
import pytest

from quel_ic_config.quel1_thermistor import Quel1NormalThermistor, Quel1PathSelectorThermistor, _TfpltConv


def test_table():
    assert np.isclose(_TfpltConv.convert(0.705), -54.0, rtol=0.005)
    assert np.isclose(_TfpltConv.convert(1.0), 25.0, rtol=0.005)
    assert np.isclose(_TfpltConv.convert(1.595), 150.0, rtol=0.005)

    with pytest.raises(ValueError):
        _ = _TfpltConv.convert(1.5950001)

    with pytest.raises(ValueError):
        _ = _TfpltConv.convert(2.0)

    with pytest.raises(ValueError):
        _ = _TfpltConv.convert(0.7019999)

    with pytest.raises(ValueError):
        _ = _TfpltConv.convert(0.5)

    with pytest.raises(ValueError):
        _ = _TfpltConv.convert(0.0)

    with pytest.raises(ValueError):
        _ = _TfpltConv.convert(-0.5)


def test_normal():
    th00 = Quel1NormalThermistor("th00")
    assert abs(th00.convert(3700) - 31.2) < 0.1
    assert abs(th00.convert(3800) - 44.9) < 0.1
    assert abs(th00.convert(3900) - 58.7) < 0.1
    assert abs(th00.convert(4000) - 73.1) < 0.1


def test_pathsel():
    th26 = Quel1PathSelectorThermistor("th26")
    assert abs(th26.convert(1800) - 22.6) < 0.1
    assert abs(th26.convert(1900) - 40.5) < 0.1
    assert abs(th26.convert(2000) - 58.2) < 0.1
    assert abs(th26.convert(2100) - 75.8) < 0.1
