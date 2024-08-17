import pytest

from quel_ic_config import Adrf6780Regs

bits_to_check = {
    0: (15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    1: (15, 14, 13, 12),
    2: (15, 14, 13, 12),
    3: (8, 7, 6, 5, 4, 3, 2, 1, 0),
    4: (7, 6, 5, 4, 3, 2, 1, 0),
    5: (10, 7, 6, 5, 4, 3, 2, 1, 0),
    6: (3, 2, 1, 0),
    12: (8, 7, 6, 5, 4, 3, 2, 1, 0),
}


@pytest.mark.parametrize(
    "addr,cls",
    [(k, v) for k, v in Adrf6780Regs.items()],
)
def test_bitfield(addr: int, cls: type):
    regobj = cls()
    for i in range(16):
        v = 1 << i
        if i in bits_to_check[addr]:
            regobj.parse(v)
            assert v == regobj.build()
        else:
            regobj.parse(v)
            assert 0 == regobj.build()
