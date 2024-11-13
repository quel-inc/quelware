from abc import ABCMeta, abstractmethod


class _TfpltConv:
    """Converting relative resistor value to temperature. The formula is based on page 4 of
    https://www.vishay.com/docs/33027/tfptl.pdf.
    """

    _MinR: float = 0.702  # a.u.
    _MaxR: float = 1.595  # a.u.

    @classmethod
    def convert(cls, r: float) -> float:
        """converting a relative resistor value into temperature in degC

        :param r: a relative resitor of the thermistor
        :return: temperature at the resitor in degC
        """

        if not (cls._MinR <= r <= cls._MaxR):
            raise ValueError(f"the given relative resistor value {r:.3f} is out of range")
        return ((28.54 * r - 158.5) * r + 474.8) * r - 319.85


class Quel1Thermistor(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def convert(self, v: int) -> float:
        pass


# the following two classws will be moved to the right place.
class Quel1NormalThermistor(Quel1Thermistor):
    def convert(self, v: int) -> float:
        # Notes: 7303 comes from strange Vdd of 4.45V.
        return _TfpltConv.convert(v / (7307 - v))


class Quel1PathSelectorThermistor(Quel1Thermistor):
    def convert(self, v: int) -> float:
        # Notes: thermistor is read by AD7490 on power board v13.
        return _TfpltConv.convert(v / (7307 - v) * 3.0303)


class Quel1seOnboardThermistor(Quel1Thermistor):
    """330Ohm thermistor + 1kOhm resistor"""

    def convert(self, v: int) -> float:
        # TODO: check whether this is OK or not.
        # Notes: Vref = 2.5V, range = Vref
        return _TfpltConv.convert(v / (8192 - v) * 3.0303)


class Quel1seExternalThermistor(Quel1Thermistor):
    """1.00kOhm thermistor + 4.70kOhm resistor"""

    def convert(self, v: int) -> float:
        # Notes: Vref = 2.5V, range = Vref
        return _TfpltConv.convert(v / (8192 - v) * 4.7)
