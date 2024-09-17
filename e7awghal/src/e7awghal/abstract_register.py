import logging
from abc import ABCMeta, abstractmethod

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Notes: the word size of registers of E7AwgHw is 32-bit.
#        the word size of HBM of E7AwgHw is 256-bit, but it is represented as an array of complex64 ((32+32) * 4).


def p_1bf_bool(v: np.uint32, bidx: int) -> bool:
    return bool((v >> bidx) & 0x1)


def p_nbf(v: np.uint32, bidx0: int, bidx1: int) -> int:
    # Notes: return type should be int rather than np.signedinteger for the accordance with pydantic
    return int((v >> bidx1) & ((1 << (bidx0 - bidx1 + 1)) - 1))


def b_1bf_bool(f: bool, bidx: int) -> np.uint32:
    return np.uint32(f << bidx)


def b_nbf(f: int, bidx0: int, bidx1: int) -> np.uint32:
    return np.uint32((f & ((1 << (bidx0 - bidx1 + 1)) - 1)) << bidx1)


class AbstractFpgaReg(BaseModel, validate_assignment=True, extra="forbid", metaclass=ABCMeta):
    @abstractmethod
    def _parse(self, v: np.uint32) -> None:
        pass

    @abstractmethod
    def build(self) -> np.uint32:
        pass

    @classmethod
    def num_bytes(cls) -> int:
        return 4

    def __int__(self) -> np.uint32:
        return self.build()
