import logging
from typing import List, Union

import pytest

from e7awghal.common_defs import E7awgMemoryError
from e7awghal.e7awg_memoryobj import E7awgMemoryObj, E7awgPrimitiveMemoryManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def mm() -> E7awgPrimitiveMemoryManager:
    return E7awgPrimitiveMemoryManager("testmm", 0x0_0000_0000, 0x0_0000_1000)


def test_scenario0(mm):
    obj = mm.allocate(0x1000)
    assert obj.address_top == 0x0_0000_0000

    with pytest.raises(E7awgMemoryError):
        _ = mm.allocate(1)

    del obj

    with pytest.raises(E7awgMemoryError):
        _ = mm.allocate(0x1001)


def test_scenario1(mm):
    objs: List[Union[E7awgMemoryObj, None]] = []
    for i in range(16):
        objs.append(mm.allocate(0x100))

    for i, obj in enumerate(objs):
        assert obj and obj.address_top == i * 0x0_0000_0100
    obj = None

    with pytest.raises(E7awgMemoryError):
        _ = mm.allocate(0x100)

    for i in range(1, 16, 2):
        objs[i] = None

    new_obj1 = mm.allocate(0x100)
    assert new_obj1.address_top == 0x0_0000_0100

    with pytest.raises(E7awgMemoryError):
        _ = mm.allocate(0x101)

    new_obj2 = mm.allocate(0x100)
    assert new_obj2.address_top == 0x0_0000_0300

    del objs
    del new_obj1
    del new_obj2
    logger.info(mm._freekey)

    new_obj3 = mm.allocate(0x1000)
    assert new_obj3 and new_obj3.address_top == 0x0_0000_0000


def test_scenario2(mm):
    objs: List[Union[E7awgMemoryObj, None]] = []
    for i in range(64):
        objs.append(mm.allocate(1, minimum_align=64))

    for i, obj in enumerate(objs):
        assert obj and obj.address_top == i * 64

    with pytest.raises(E7awgMemoryError):
        _ = mm.allocate(1, minimum_align=64)

    objs2: List[Union[E7awgMemoryObj, None]] = []
    for i in range(64):
        objs2.append(mm.allocate(1, minimum_align=32))

    for i, obj in enumerate(objs2):
        assert obj and obj.address_top == i * 64 + 32
    obj = None

    with pytest.raises(E7awgMemoryError):
        _ = mm.allocate(33, minimum_align=64)

    objs.clear()
    for i in range(64):
        objs.append(mm.allocate(32, minimum_align=64))

    for i, obj in enumerate(objs):
        assert obj and obj.address_top == i * 64
    obj = None

    del objs
    del objs2
    obj3 = mm.allocate(0x1000)
    assert obj3 and obj3.address_top == 0x0_0000_0000
