import logging
from typing import TYPE_CHECKING, Union

import numpy as np
from pydantic import Field, conint

from e7awghal.abstract_register import AbstractFpgaReg, b_nbf, p_nbf

logger = logging.getLogger(__name__)

my_array_like = Union[bytes, bytearray, memoryview]


class E7awgVersion(AbstractFpgaReg):
    if TYPE_CHECKING:
        ver_char: int = Field(default=0)  # [31:24]
        ver_year: int = Field(default=0)  # [23:16]
        ver_month: int = Field(default=0)  # [15:12]
        ver_day: int = Field(default=0)  # [11:4]
        ver_id: int = Field(default=0)  # [3:0]
    else:
        ver_char: conint(ge=0x00, le=0xFF) = Field(default=0)  # [31:24]
        ver_year: conint(ge=0x00, le=0xFF) = Field(default=0)  # [23:16]
        ver_month: conint(ge=0x0, le=0xF) = Field(default=0)  # [15:12]
        ver_day: conint(ge=0x0, le=0xFF) = Field(default=0)  # [11:4]
        ver_id: conint(ge=0x0, le=0xF) = Field(default=0)  # [3:0]

    def _parse(self, v):
        self.ver_char = p_nbf(v, 31, 24)
        self.ver_year = p_nbf(v, 23, 16)
        self.ver_month = p_nbf(v, 15, 12)
        self.ver_day = p_nbf(v, 11, 4)
        self.ver_id = p_nbf(v, 3, 0)

    def build(self) -> np.uint32:
        return np.uint32(
            b_nbf(self.ver_char, 31, 24)
            | b_nbf(self.ver_year, 23, 16)
            | b_nbf(self.ver_month, 15, 12)
            | b_nbf(self.ver_day, 11, 4)
            | b_nbf(self.ver_id, 3, 0)
        )

    @classmethod
    def parse(cls, v: np.uint32) -> "E7awgVersion":
        r = cls()
        r._parse(v)
        return r
