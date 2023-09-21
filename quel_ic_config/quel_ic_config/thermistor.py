from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.typing as npt


class _TfpltConv:
    """Converting relative resistor value to temperature. The table is based on
    https://www.vishay.com/docs/33027/tfptl.pdf.
    """

    _MinTemp: float = -55.0  # degC
    _MaxTemp: float = 150.0  # degC
    _TfpltConvTable: npt.NDArray[np.float_] = np.array(
        (
            0.702,  # -55 [degC]
            0.705,
            0.708,
            0.712,
            0.715,
            0.719,  # -50
            0.722,
            0.725,
            0.729,
            0.732,
            0.736,
            0.739,
            0.743,
            0.746,
            0.749,
            0.753,  # -40
            0.756,
            0.760,
            0.763,
            0.767,
            0.771,
            0.774,
            0.778,
            0.781,
            0.785,
            0.788,  # -30
            0.792,
            0.796,
            0.799,
            0.803,
            0.806,
            0.810,
            0.814,
            0.817,
            0.821,
            0.825,  # -20
            0.828,
            0.832,
            0.836,
            0.839,
            0.843,
            0.847,
            0.851,
            0.854,
            0.858,
            0.862,  # -10
            0.866,
            0.869,
            0.873,
            0.877,
            0.881,
            0.885,
            0.889,
            0.892,
            0.896,
            0.900,  # 0
            0.904,
            0.908,
            0.912,
            0.916,
            0.920,
            0.924,
            0.927,
            0.931,
            0.935,
            0.939,  # 10
            0.943,
            0.947,
            0.951,
            0.955,
            0.959,
            0.963,
            0.967,
            0.971,
            0.975,
            0.980,  # 20
            0.984,
            0.988,
            0.992,
            0.996,
            1.000,
            1.004,
            1.008,
            1.012,
            1.017,
            1.021,  # 30
            1.025,
            1.029,
            1.033,
            1.037,
            1.042,
            1.046,
            1.050,
            1.054,
            1.059,
            1.063,  # 40
            1.067,
            1.071,
            1.076,
            1.080,
            1.084,
            1.089,
            1.093,
            1.097,
            1.102,
            1.106,  # 50
            1.110,
            1.115,
            1.119,
            1.124,
            1.128,
            1.133,
            1.137,
            1.141,
            1.146,
            1.150,  # 60
            1.155,
            1.159,
            1.164,
            1.168,
            1.173,
            1.177,
            1.182,
            1.186,
            1.191,
            1.196,  # 70
            1.200,
            1.205,
            1.209,
            1.214,
            1.219,
            1.223,
            1.228,
            1.232,
            1.237,
            1.242,  # 80
            1.246,
            1.251,
            1.256,
            1.261,
            1.265,
            1.270,
            1.275,
            1.280,
            1.284,
            1.289,  # 90
            1.294,
            1.299,
            1.303,
            1.308,
            1.313,
            1.318,
            1.323,
            1.328,
            1.333,
            1.337,  # 100
            1.342,
            1.347,
            1.352,
            1.357,
            1.362,
            1.367,
            1.372,
            1.377,
            1.382,
            1.387,  # 110
            1.392,
            1.397,
            1.402,
            1.407,
            1.412,
            1.417,
            1.422,
            1.427,
            1.432,
            1.437,  # 120
            1.442,
            1.448,
            1.453,
            1.458,
            1.463,
            1.468,
            1.473,
            1.478,
            1.484,
            1.489,  # 130
            1.494,
            1.499,
            1.505,
            1.510,
            1.515,
            1.520,
            1.526,
            1.531,
            1.536,
            1.541,  # 140
            1.547,
            1.552,
            1.557,
            1.563,
            1.568,
            1.574,
            1.579,
            1.584,
            1.590,
            1.595,  # 150
        )
    )
    _TfpltConvTableSize = _TfpltConvTable.shape[0]

    @classmethod
    def convert(cls, r: float) -> float:
        """convert a relative resistor value into temperature in degC
        :param r: a relative resitor of the thermistor
        :return: temperature at the resitor in degC
        """
        idx = np.searchsorted(cls._TfpltConvTable, r)
        if idx == 0 or idx == cls._TfpltConvTableSize:
            raise ValueError(f"the given relative resistor value {r:.3f} is out of range")

        t_frac = (r - cls._TfpltConvTable[idx]) / (cls._TfpltConvTable[idx] - cls._TfpltConvTable[idx - 1])
        return float(idx) + cls._MinTemp + t_frac


class Thermistor(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def convert(self, v: int) -> float:
        pass


# the following two classws will be moved to the right place.
class Quel1NormalThermistor(Thermistor):
    def convert(self, v: int) -> float:
        # Notes: 7303 comes from strange Vdd of 4.45V.
        return _TfpltConv.convert(v / (7307 - v))


class Quel1PathSelectorThermistor(Thermistor):
    def convert(self, v: int) -> float:
        # Notes: thermistor is read by AD7490 on power board v13.
        return _TfpltConv.convert(v / (7307 - v) * 3.0303)


class Quel1SeProtoThermistor(Thermistor):
    def convert(self, v: int) -> float:
        # TODO: check whether this is OK or not.
        # Notes: Vref = 2.5V, range = Vref
        return _TfpltConv.convert(v / (8192 - v) * 3.0303)


class Quel1SeProtoExternalThermistor(Thermistor):
    def convert(self, v: int) -> float:
        # Notes: Vref = 2.5V, range = 2 * Vref
        return _TfpltConv.convert(v / (4096 - v))


if __name__ == "__main__":
    th00 = Quel1NormalThermistor("th00")
    th26 = Quel1PathSelectorThermistor("th26")
    thse = Quel1SeProtoThermistor("thse")
