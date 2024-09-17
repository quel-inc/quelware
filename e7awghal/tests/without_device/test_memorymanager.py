import gc
import itertools
import logging
from typing import List, Tuple

import pytest

from e7awghal.e7awg_memoryobj import E7awgMemoryObj, E7awgPrimitiveMemoryManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def mm0() -> Tuple[E7awgPrimitiveMemoryManager, List[E7awgMemoryObj]]:
    mm = E7awgPrimitiveMemoryManager("testmm", 0x0_0000_0000, 0x0_0000_1000)
    o0 = mm.allocate(64, address_top=0)
    o1 = mm.allocate(64, address_top=128)
    o2 = mm.allocate(64, address_top=256)
    o3 = mm.allocate(64, address_top=320)
    o4 = mm.allocate(64, address_top=384)
    return mm, [o0, o1, o2, o3, o4]


def test_mm0(mm0):
    mm, mmobjs = mm0
    assert mm._freekey == [64, 192, 448]


@pytest.mark.parametrize(
    ["scenario"],
    [(x,) for x in itertools.permutations([0, 1, 2, 3, 4], 5)],
)
def test_mm0_scenario0(scenario, mm0):
    mm, mmobjs = mm0

    for idx in scenario:
        mm._deallocate_liveobj(mmobjs[idx])
    assert mm._freekey == [0] and mm._freelist[0]._size == 0x1000


def test_abnormal_cases(mm0):
    mm, mmobjs = mm0

    # Notes: negative address
    with pytest.raises(ValueError):
        _ = mm.allocate(-1, address_top=512)

    # Notes: used memory region (full)
    with pytest.raises(ValueError):
        _ = mm.allocate(64, address_top=0)

    # Notes: used memory region (partly)
    with pytest.raises(ValueError):
        _ = mm.allocate(64, address_top=32)

    # Notes: used memory region (partly)
    with pytest.raises(ValueError):
        _ = mm.allocate(64, address_top=96)

    # Notes: used memory region (partly)
    with pytest.raises(ValueError):
        _ = mm.allocate(128, address_top=96)

    # Notes: used memory region (partly)
    with pytest.raises(ValueError):
        _ = mm.allocate(4096, address_top=0)

    # Notes: out of address space (partly)
    with pytest.raises(ValueError):
        _ = mm.allocate(4096, address_top=512)

    # Notes: out of address space (full)
    with pytest.raises(ValueError):
        _ = mm.allocate(4096, address_top=8192)


def test_reset():
    # Notes: the references to the fixture objects looks to be kept somewhere...

    mm = E7awgPrimitiveMemoryManager("testmm", 0x0_0000_0000, 0x0_0000_1000)
    o0 = mm.allocate(64, address_top=0)
    o1 = mm.allocate(64, address_top=128)
    o2 = mm.allocate(64, address_top=256)
    o3 = mm.allocate(64, address_top=320)
    o4 = mm.allocate(64, address_top=384)

    for o in (o0, o1, o2, o3, o4):
        assert o._live

    assert len(mm._issued_obj) == 5
    del o1
    gc.collect()

    n = 0
    for o in mm._issued_obj:
        assert o._live
        n += 1
    assert n == 4

    mm.reset()
    assert len(mm._issued_obj) == 0
    for o in (o0, o2, o3, o4):
        assert not o._live

    o5 = mm.allocate(4096)
    for o in (o0, o2, o3, o4):
        with pytest.raises(RuntimeError):
            _ = o.address_top

    assert o5.address_top == 0x0000_0000
    assert len(mm._issued_obj) == 1

    del o0
    del o2
    del o3
    del o4
    gc.collect()


def test_reset_twice():
    mm = E7awgPrimitiveMemoryManager("testmm", 0x0_0000_0000, 0x0_0000_1000)
    o0 = mm.allocate(64, address_top=0)
    o1 = mm.allocate(64, address_top=128)
    o2 = mm.allocate(64, address_top=256)
    o3 = mm.allocate(64, address_top=320)
    o4 = mm.allocate(64, address_top=384)
    mm.reset()

    o5 = mm.allocate(4096)
    mm.reset()

    for o in (o0, o1, o2, o3, o4, o5):
        with pytest.raises(RuntimeError):
            _ = o.address_top

    o6 = mm.allocate(4096)
    assert o6.address_top == 0x0000_0000
