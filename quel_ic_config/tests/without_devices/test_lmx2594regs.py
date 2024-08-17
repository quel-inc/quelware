import pytest

from quel_ic_config import Lmx2594Regs

bits_to_check = {
    0: (15, 14, 9, 8, 7, 6, 5, 3, 2, 1, 0),
    1: (2, 1, 0),
    7: (14,),
    9: (12,),
    10: (11, 10, 9, 8, 7),
    11: (11, 10, 9, 8, 7, 6, 5, 4),
    12: (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    14: (6, 5, 4),
    4: (15, 14, 13, 12, 11, 10, 9, 8),
    8: (14, 11),
    16: (8, 7, 6, 5, 4, 3, 2, 1, 0),
    17: (8, 7, 6, 5, 4, 3, 2, 1, 0),
    19: (7, 6, 5, 4, 3, 2, 1, 0),
    20: (13, 12, 11, 10),
    34: (2, 1, 0),
    36: range(16),
    37: (15, 13, 12, 11, 10, 9, 8),
    38: range(16),
    39: range(16),
    40: range(16),
    41: range(16),
    42: range(16),
    43: range(16),
    44: (13, 12, 11, 10, 9, 8, 7, 6, 5, 2, 1, 0),
    45: (12, 11, 10, 9, 5, 4, 3, 2, 1, 0),
    46: (1, 0),
    58: (15, 14, 13, 12, 11, 10, 9),
    59: (0,),
    60: range(16),
    69: range(16),
    70: range(16),
    71: (7, 6, 5, 4, 3, 2),
    72: (10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    73: (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    74: range(16),
    31: (14,),
    75: (10, 9, 8, 7, 6),
    78: (11, 9, 8, 7, 6, 5, 4, 3, 2, 1),
    110: (10, 9, 7, 6, 5),
    111: (7, 6, 5, 4, 3, 2, 1, 0),
    112: (8, 7, 6, 5, 4, 3, 2, 1, 0),
}


@pytest.mark.parametrize(
    "idx,",
    [reg for reg in Lmx2594Regs],
)
def test_bitfield(idx: int):
    regobj = Lmx2594Regs[idx]()
    constant = regobj.build()
    for i in range(16):
        v = 1 << i
        if i in bits_to_check[idx]:
            assert v & constant == 0
            regobj.parse(v)
            assert v == regobj.build() & (~constant)
        else:
            regobj.parse(v)
            assert regobj.build() == constant
