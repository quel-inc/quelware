import logging
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from quel_ic_config import BoxLockError, Quel1Box, Quel1BoxType, force_unlock_all_boxes
from tests.with_devices.conftest import BoxProvider

logger = logging.getLogger(__name__)


TARGET_BOXES = [
    ("10.1.0.50", Quel1BoxType.QuEL1_TypeA),
    ("10.1.0.94", Quel1BoxType.QuEL1SE_RIKEN8),
    ("10.1.0.157", Quel1BoxType.QuEL1SE_FUJITSU11_TypeA),
]

EXTENDED_TARGET_BOXES = [
    ("10.1.0.50", Quel1BoxType.QuEL1_TypeA),
    ("10.1.0.60", Quel1BoxType.QuEL1_TypeB),
    ("10.1.0.94", Quel1BoxType.QuEL1SE_RIKEN8),
    ("10.1.0.157", Quel1BoxType.QuEL1SE_FUJITSU11_TypeA),
    ("10.1.0.164", Quel1BoxType.QuEL1SE_FUJITSU11_TypeB),
]

TARGET_BOX_TYPES = [Quel1BoxType.QuEL1_TypeA, Quel1BoxType.QuEL1SE_RIKEN8, Quel1BoxType.QuEL1SE_FUJITSU11_TypeA]
EXTENDED_TARGET_BOX_TYPES = [
    Quel1BoxType.QuEL1_TypeA,
    Quel1BoxType.QuEL1_TypeB,
    Quel1BoxType.QuEL1SE_RIKEN8,
    Quel1BoxType.QuEL1SE_FUJITSU11_TypeA,
    Quel1BoxType.QuEL1SE_FUJITSU11_TypeB,
]


@pytest.fixture(scope="function")
def clean_box_provider(box_provider: BoxProvider) -> Generator[BoxProvider, None, None]:
    box_provider.clean_up()
    yield box_provider
    box_provider.clean_up()


@pytest.mark.parametrize("boxtype", TARGET_BOX_TYPES)
def test_unlock_normal(boxtype, clean_box_provider: BoxProvider):
    boxconf = next(clean_box_provider.find_boxconf_from_type(boxtype))

    box1 = Quel1Box.create(ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)
    p0 = box1.dump_port(0)
    del box1

    box2 = Quel1Box.create(ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)
    p1 = box2.dump_port(0)
    assert p0 == p1


@pytest.mark.parametrize("boxtype", TARGET_BOX_TYPES)
def test_duplicated_lock(boxtype, clean_box_provider: BoxProvider):
    boxconf = next(clean_box_provider.find_boxconf_from_type(boxtype))
    box1 = Quel1Box.create(ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)
    ipaddr_css = str(boxconf.ipaddr + 0x00040000)
    with pytest.raises(BoxLockError, match=f"failed to acquire lock of {ipaddr_css}"):
        box2 = Quel1Box.create(ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)
        box2.dump_port(0)

    box1.dump_port(0)
    del box1


@pytest.mark.parametrize("boxtype", TARGET_BOX_TYPES)
def test_unlocked_lock(boxtype, clean_box_provider: BoxProvider):
    boxconf = next(clean_box_provider.find_boxconf_from_type(boxtype))
    box1 = Quel1Box.create(ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)
    p0 = box1.dump_port(0)
    rfsw0 = box1.dump_rfswitches()
    t0 = box1.get_current_timecounter()
    force_unlock_all_boxes()
    with pytest.raises(BoxLockError):
        p1 = box1.dump_port(0)  # Notes: raises here!
        assert p0 == p1
    with pytest.raises(BoxLockError):
        t1 = box1.get_current_timecounter()  # Notes: raises here!
        assert t1 > t0
    with pytest.raises(BoxLockError):
        rfsw1 = box1.dump_rfswitches()  # Notes: raises here!
        assert rfsw0 == rfsw1
    del box1


@pytest.mark.parametrize("boxtype", TARGET_BOX_TYPES)
def test_lock_release_race_condition(boxtype, clean_box_provider: BoxProvider):
    boxconf = next(clean_box_provider.find_boxconf_from_type(boxtype))
    box1 = Quel1Box.create(ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)
    p0 = box1.dump_port(0)
    force_unlock_all_boxes()

    for i in range(15):
        box1 = Quel1Box.create(ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)
        force_unlock_all_boxes()

    with pytest.raises(BoxLockError):
        p1 = box1.dump_port(0)  # Notes: raises here!
        assert p0 == p1

    del box1


def test_lock_release_race_condition_multiple(clean_box_provider: BoxProvider):
    for i in range(10):
        boxes: dict[str, Quel1Box] = {}
        for boxtype in TARGET_BOX_TYPES:
            boxconf = next(clean_box_provider.find_boxconf_from_type(boxtype))
            boxes[boxconf.name] = Quel1Box.create(ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)

        for _, box in boxes.items():
            t0 = box.get_current_timecounter()
            t1 = box.get_current_timecounter()
            # Notes: just want to check some random API requiring the lock works correctly.
            assert t1 > t0

        force_unlock_all_boxes()


"""
def test_lock_release_race_condition_multiple_del():
    targets = EXTENDED_TARGET_BOXES
    pool = ThreadPoolExecutor(max_workers=len(targets))

    for i in range(2 ** len(targets)):
        futs: dict[str, Future[Quel1Box]] = {}
        for ipaddr_wss, boxtype in targets:
            futs[ipaddr_wss] = pool.submit(Quel1Box.create, ipaddr_wss=ipaddr_wss, boxtype=boxtype)
        boxes: dict[str, Quel1Box] = {k: v.result() for k, v in futs.items()}

        for _, box in boxes.items():
            t0 = box.get_current_timecounter()
            t1 = box.get_current_timecounter()
            # Notes: just want to check some random API requiring the lock works correctly.
            assert t1 > t0

        to_del: list[str] = []
        for idx, ipaddr_wss in enumerate(boxes):
            if (1 << idx) & i != 0:
                to_del.append(ipaddr_wss)

        for ipaddr_wss in to_del:
            logger.info(f"DELETING {ipaddr_wss}")
            del boxes[ipaddr_wss]

        force_unlock_all_boxes()
"""


def _del_box(boxes: dict[str, Quel1Box], to_del: str):
    logger.info(f"DELETING {to_del}")
    del boxes[to_del]


def test_lock_release_race_condition_multiple_del_parallel(clean_box_provider: BoxProvider):
    targets = EXTENDED_TARGET_BOX_TYPES
    pool = ThreadPoolExecutor(max_workers=len(targets))

    for i in range(2 ** len(targets)):
        futs: dict[str, Future[Quel1Box]] = {}
        for boxtype in targets:
            boxconf = next(clean_box_provider.find_boxconf_from_type(boxtype))
            futs[boxconf.name] = pool.submit(Quel1Box.create, ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype)
        boxes: dict[str, Quel1Box] = {k: v.result() for k, v in futs.items()}

        for _, box in boxes.items():
            t0 = box.get_current_timecounter()
            t1 = box.get_current_timecounter()
            # Notes: just want to check some random API requiring the lock works correctly.
            assert t1 > t0

        to_del: list[str] = []
        for idx, name in enumerate(boxes):
            if (1 << idx) & i != 0:
                to_del.append(name)

        futs_del = []
        for name in to_del:
            futs_del.append(pool.submit(_del_box, boxes, name))
        futs_del.append(pool.submit(force_unlock_all_boxes))

        for fut in futs_del:
            fut.result()


@pytest.mark.parametrize("boxtype", TARGET_BOX_TYPES[0:1])
def test_lock_lock_race_condition(boxtype, clean_box_provider: BoxProvider):
    n = 3
    pool = ThreadPoolExecutor(max_workers=n)

    futs: list[Future[Quel1Box]] = []
    for i in range(n):
        boxconf = next(clean_box_provider.find_boxconf_from_type(boxtype))
        futs.append(pool.submit(Quel1Box.create, ipaddr_wss=str(boxconf.ipaddr), boxtype=boxtype))

    boxes: list[Quel1Box] = []
    for i, fut in enumerate(futs):
        try:
            boxes.append(fut.result())
        except BoxLockError:
            pass

    assert len(boxes) == 1
    for i, box in enumerate(boxes):
        t0 = box.get_current_timecounter()
        logger.info(f"{box}, {i}: {t0}")
