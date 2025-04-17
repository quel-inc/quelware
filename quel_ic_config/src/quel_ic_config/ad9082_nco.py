import logging
from fractions import Fraction
from typing import Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


class AbstractNcoFtw(BaseModel):
    # notes: x + a/b
    ftw: int = Field(ge=-0x8000_0000_0000, le=0x7FFF_FFFF_FFFF)
    delta_b: int = Field(ge=0x0000_0000_0000, le=0xFFFF_FFFF_FFFF)
    modulus_a: int = Field(ge=0x0000_0000_0000, le=0xFFFF_FFFF_FFFF)
    enable_fraction: bool = Field(default=True)

    @model_validator(mode="after")
    def check_numerator(self):
        if self.enable_fraction:
            if self.delta_b >= self.modulus_a:
                raise ValueError("improper fraction is not allowed")
        else:
            if not (self.delta_b == 0 and self.modulus_a == 1):
                raise ValueError("delta_b and modulus_a must be 0 and 1 when fractional mode is disabled")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractNcoFtw):
            return False
        return self.ftw == other.ftw and Fraction(self.delta_b, self.modulus_a) == Fraction(
            other.delta_b, other.modulus_a
        )

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return f"<ftw: {self.ftw} + {self.delta_b} / {self.modulus_a}>"

    @staticmethod
    def _encode_s48_as_u48(v: int):
        return v if v >= 0 else ((1 << 48) + v)

    @staticmethod
    def _decode_u48_as_s48(w: int):
        return w if w < (1 << 47) else w - (1 << 48)

    @staticmethod
    def _converter_frequency_as_interger(converter_freq_hz_: Union[int, float]) -> int:
        if isinstance(converter_freq_hz_, float):
            if not converter_freq_hz_.is_integer():
                raise ValueError("converter_freq_hz must be integer value")
            converter_freq_hz = int(converter_freq_hz_)
        elif isinstance(converter_freq_hz_, int):
            converter_freq_hz = converter_freq_hz_
        else:
            raise TypeError("illegal object for converter_freq_hz")

        return converter_freq_hz

    def to_frequency(self, converter_freq_hz_: Union[int, float]) -> float:
        converter_freq_hz = self._converter_frequency_as_interger(converter_freq_hz_)
        if self.enable_fraction:
            t = float((self.ftw + Fraction(self.delta_b, self.modulus_a)) * converter_freq_hz / (1 << 48))
        else:
            t = self.ftw * converter_freq_hz / (1 << 48)
        return t

    def truncate(self):
        self.delta_b = 0
        self.modulus_a = 1
        self.enable_fraction = False

    def round(self):
        frac = self.delta_b / self.modulus_a
        if (frac > 0.5) or (frac == 0.5 and self.ftw % 2 == 1):
            self.ftw += 1
        self.delta_b = 0
        self.modulus_a = 1
        self.enable_fraction = False

    def is_zero(self):
        return self.ftw == 0 and self.delta_b == 0

    def multiply(self, r: int) -> Self:
        x = self.ftw
        b = self.delta_b
        negative: bool = x < 0
        if negative:
            x = -x
            b = -b

        b *= r
        x = x * r + b // self.modulus_a
        b = b % self.modulus_a

        if negative:
            x = -x
            b = -b

        return self.__class__(ftw=x, delta_b=b, modulus_a=self.modulus_a)

    @classmethod
    def from_frequency(
        cls, nco_freq_hz_: Union[Fraction, float], converter_freq_hz_: Union[int, float], epsilon: float = 0.0
    ) -> Self:
        converter_freq_hz: int = cls._converter_frequency_as_interger(converter_freq_hz_)

        hcf: int = converter_freq_hz // 2
        if not (-hcf <= nco_freq_hz_ < hcf):
            raise ValueError(f"the given nco_frequency (= {nco_freq_hz_:f}Hz) is out of range")

        if not isinstance(converter_freq_hz, int):
            raise TypeError("converter_freq_hz must be integer")

        if isinstance(nco_freq_hz_, Fraction):
            nco_freq_hz: Fraction = nco_freq_hz_
        else:
            nco_freq_hz = Fraction(nco_freq_hz_)

        negative: bool = nco_freq_hz < 0
        if negative:
            t: Fraction = -nco_freq_hz * (1 << 48) / converter_freq_hz
        else:
            t = nco_freq_hz * (1 << 48) / converter_freq_hz
        t = t.limit_denominator(0xFFFF_FFFF_FFFF)

        x: int = int(t)
        if negative and t.denominator > 1:
            x += 1
        delta_b: int = t.numerator - t.denominator * x
        modulus_a: int = t.denominator
        if negative:
            x = -x
            delta_b = -delta_b  # Notes: delta_b becomes non-negative, finally.
        obj = cls(ftw=x, delta_b=delta_b, modulus_a=modulus_a)

        conv_error = abs(obj.to_frequency(converter_freq_hz) - nco_freq_hz_)
        if conv_error > epsilon:
            raise ValueError(
                f"large conversion error (= {conv_error}Hz) of the given nco frequency (= {nco_freq_hz:f}Hz)"
            )

        return obj
