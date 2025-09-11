import logging
import time
from concurrent.futures import CancelledError

import numpy as np
import pytest

from quel_ic_config.quel1_wave_subsystem import (
    StartAwgunitsNowTask,
    StartAwgunitsTimedTask,
    StartCapunitsByTriggerTask,
    StartCapunitsNowTask,
)
from testlibs.gencap_quel1_utils import config_lines, config_rlines
from testlibs.gencap_utils import (
    check_awgs_are_clear,
    check_caps_are_clear,
    config_awgs_gen_seconds,
    config_caps_cap_now_seconds,
)
from tests.with_devices.conftest import BoxProvider

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def test_awg_normal(box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 1})
    assert boxi.wss._hal.awgunit(list(awg_idxs)[0]).get_wave_duration() == 125_000_000

    task: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)
    for _ in range(10):
        time.sleep(0.01)
        if task.running():
            break
    else:
        assert task.running()
    assert not task.done()
    assert not task.cancelled()

    assert task.result() is None

    assert not task.running()
    assert task.done()
    assert not task.cancelled()
    check_awgs_are_clear(boxi, awg_idxs)


@pytest.mark.parametrize(
    ["timeout"],
    [
        (0.5,),
        (2.5,),
    ],
)
def test_awg_timeout(timeout, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 3, (0, 3, 1): 2})
    task: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs, timeout=timeout)
    for _ in range(10):
        time.sleep(0.01)
        if task.running():
            break
    else:
        assert task.running()

    with pytest.raises(TimeoutError):
        task.result()

    assert not task.running()
    assert task.done()
    assert not task.cancelled()
    check_awgs_are_clear(boxi, awg_idxs)


@pytest.mark.parametrize(
    ["sleep_time"],
    [
        (0.1,),
        (2.0,),
    ],
)
def test_awg_cancelled(sleep_time: float, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 3, (0, 3, 2): 1})
    task: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)
    time.sleep(sleep_time)
    task.cancel()
    with pytest.raises(CancelledError):
        task.result()

    assert not task.running()
    assert not task.done()
    assert task.cancelled()
    check_awgs_are_clear(boxi, awg_idxs)


@pytest.mark.parametrize(
    ["sleep_time"],
    [
        (0.0,),
        (1.5,),
    ],
)
def test_awg_busy(sleep_time, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 2, (0, 3, 2): 1})

    task_0: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)
    if sleep_time > 0.0:
        time.sleep(sleep_time)

    with pytest.raises(RuntimeError):
        # Notes: default value of return_after_starting_emission is True for start_awgunits_now()
        #        during the checking of starting, the task dies due to busy flag check, that causes exception in the
        #        checking loop:
        #            RuntimeError: task has stopped, never start emission.
        task_1: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)
        task_1.result()

    assert task_0.result() is None


@pytest.mark.parametrize(
    ["sleep_time"],
    [
        (0.0,),
        (1.5,),
    ],
)
def test_awg_busy_no_wait_starting(sleep_time, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 2, (0, 3, 2): 1})

    task_0: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)
    if sleep_time > 0.0:
        time.sleep(sleep_time)

    task_1: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs, return_after_start_emission=False)
    with pytest.raises(RuntimeError):
        task_1.result()

    assert task_0.result() is None


def test_cap_normal(box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    targets = {(0, "r", 0): (256, 3), (1, "r", 3): (128, 2)}
    capunit_idxs = config_caps_cap_now_seconds(boxi, targets)
    capunit_map = {target: boxi._get_capunit_from_runit(*target) for target in targets}
    hwidx_map = {target: boxi.wss._capunit_idx_to_hwidx(boxi._get_capunit_from_runit(*target)) for target in targets}
    for target, hwidx in hwidx_map.items():
        assert boxi.wss._hal.capunit(hwidx).get_capture_duration() == targets[target][1] * 125_000_000

    task: StartCapunitsNowTask = boxi.wss.start_capunits_now(capunit_idxs)
    for _ in range(10):
        time.sleep(0.01)
        if task.running():
            break
    else:
        assert task.running()
    assert not task.done()
    assert not task.cancelled()
    rdrs = task.result()

    assert not task.running()
    assert task.done()
    assert not task.cancelled()
    check_caps_are_clear(boxi, set(hwidx_map.values()))

    for runit, (num_word, num_repeat) in targets.items():
        capunit_idx = capunit_map[runit]
        assert capunit_idx in rdrs
        rdr = rdrs[capunit_idx]
        waves = rdr.as_wave_dict()
        assert "s0" in waves
        assert waves["s0"].shape == (num_repeat, num_word * 4)


@pytest.mark.parametrize(
    ["timeout"],
    [
        (1.0,),
        (1.1,),
        (2.5,),
    ],
)
def test_cap_timeout(timeout, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    targets = {(0, "r", 0): (256, 3), (1, "r", 3): (128, 2)}
    capunit_idxs = config_caps_cap_now_seconds(boxi, targets)
    hwidx_map = {target: boxi.wss._capunit_idx_to_hwidx(boxi._get_capunit_from_runit(*target)) for target in targets}

    task: StartCapunitsNowTask = boxi.wss.start_capunits_now(capunit_idxs, timeout=timeout)
    for _ in range(10):
        time.sleep(0.01)
        if task.running():
            break
    else:
        assert task.running()
    assert not task.done()
    assert not task.cancelled()
    with pytest.raises(TimeoutError):
        _ = task.result()

    assert not task.running()
    assert task.done()
    assert not task.cancelled()
    check_caps_are_clear(boxi, set(hwidx_map.values()))


@pytest.mark.parametrize(
    ["sleep_time"],
    [
        (1.0,),
        (2.5,),
    ],
)
def test_cap_cancelled(sleep_time: float, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    targets = {(0, "r", 0): (256, 3), (1, "r", 3): (128, 2)}
    capunit_idxs = config_caps_cap_now_seconds(boxi, targets)
    hwidx_map = {target: boxi.wss._capunit_idx_to_hwidx(boxi._get_capunit_from_runit(*target)) for target in targets}

    task: StartCapunitsNowTask = boxi.wss.start_capunits_now(capunit_idxs)
    for _ in range(10):
        time.sleep(0.01)
        if task.running():
            break
    else:
        assert task.running()
    time.sleep(sleep_time)
    task.cancel()
    with pytest.raises(CancelledError):
        _ = task.result()

    assert not task.running()
    assert not task.done()
    assert task.cancelled()
    check_caps_are_clear(boxi, set(hwidx_map.values()))


@pytest.mark.parametrize(
    ["sleep_time"],
    [(0.25,), (0.75,), (1.25,), (2.50,)],
)
def test_captrig_normal_and_timeout(sleep_time: float, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    cap_targets = {(0, "r", 0): (256, 3), (1, "r", 3): (128, 2)}
    capunit_idxs = config_caps_cap_now_seconds(boxi, cap_targets)
    capunit_map = {target: boxi._get_capunit_from_runit(*target) for target in cap_targets}
    hwidx_map = {
        target: boxi.wss._capunit_idx_to_hwidx(boxi._get_capunit_from_runit(*target)) for target in cap_targets
    }
    for target, hwidx in hwidx_map.items():
        assert boxi.wss._hal.capunit(hwidx).get_capture_duration() == cap_targets[target][1] * 125_000_000

    awg_targets = {(0, 0, 0): 1, (1, 0, 0): 1}
    awg_idxs = config_awgs_gen_seconds(boxi, awg_targets)
    awgunit_map = {target: boxi._get_awg_from_channel(*target) for target in awg_targets}
    assert boxi.wss._hal.awgunit(list(awg_idxs)[0]).get_wave_duration() == 125_000_000

    # Notes: connecting CapUnit to Awg, this operation is encapsulated in upper layers.
    boxi.wss.set_triggering_awg_to_line(capunit_map[0, "r", 0][0], awgunit_map[(0, 0, 0)])
    boxi.wss.set_triggering_awg_to_line(capunit_map[1, "r", 3][0], awgunit_map[(1, 0, 0)])

    cap_task: StartCapunitsByTriggerTask = boxi.wss.start_capunits_by_trigger(capunit_idxs, timeout_before_trigger=1.0)
    time.sleep(sleep_time)
    awg_task: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)

    # Notes: be aware that cap_task may already be timeout'ed.
    for _ in range(10):
        time.sleep(0.01)
        if awg_task.running():
            break
    else:
        assert awg_task.running()
    assert not cap_task.cancelled()

    if sleep_time > 1.0:
        with pytest.raises(TimeoutError):
            _ = cap_task.result()
        assert not cap_task.running()
        assert cap_task.done()
        assert not cap_task.cancelled()
        check_caps_are_clear(boxi, set(hwidx_map.values()))
    else:
        rdrs = cap_task.result()
        assert not cap_task.running()
        assert cap_task.done()
        assert not cap_task.cancelled()
        check_caps_are_clear(boxi, set(hwidx_map.values()))

        for runit, (num_word, num_repeat) in cap_targets.items():
            capunit_idx = capunit_map[runit]
            assert capunit_idx in rdrs
            rdr = rdrs[capunit_idx]
            waves = rdr.as_wave_dict()
            assert "s0" in waves
            assert waves["s0"].shape == (num_repeat, num_word * 4)

    assert awg_task.result() is None
    for runit in cap_targets:
        assert boxi.wss.get_triggering_awg_to_line(capunit_map[runit][0]) is None


@pytest.mark.parametrize(
    ["sleep_time"],
    [(0.05,), (0.0,), (0.25,), (0.75,), (1.25,), (2.50,)],  # non-first 0.0 can cancel future before starting.
)
def test_captrig_cancel_before_and_after_trigger(sleep_time: float, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    cap_targets = {(0, "r", 0): (256, 3), (1, "r", 3): (128, 2)}
    capunit_idxs = config_caps_cap_now_seconds(boxi, cap_targets)
    capunit_map = {target: boxi._get_capunit_from_runit(*target) for target in cap_targets}
    hwidx_map = {
        target: boxi.wss._capunit_idx_to_hwidx(boxi._get_capunit_from_runit(*target)) for target in cap_targets
    }
    for target, hwidx in hwidx_map.items():
        assert boxi.wss._hal.capunit(hwidx).get_capture_duration() == cap_targets[target][1] * 125_000_000

    awg_targets = {(0, 0, 0): 1, (1, 0, 0): 1}
    awg_idxs = config_awgs_gen_seconds(boxi, awg_targets)
    awgunit_map = {target: boxi._get_awg_from_channel(*target) for target in awg_targets}
    assert boxi.wss._hal.awgunit(list(awg_idxs)[0]).get_wave_duration() == 125_000_000

    # Notes: connecting CapUnit to Awg, this operation is encapsulated in upper layers.
    boxi.wss.set_triggering_awg_to_line(capunit_map[0, "r", 0][0], awgunit_map[(0, 0, 0)])
    boxi.wss.set_triggering_awg_to_line(capunit_map[1, "r", 3][0], awgunit_map[(1, 0, 0)])

    cap_task: StartCapunitsByTriggerTask = boxi.wss.start_capunits_by_trigger(capunit_idxs, timeout_before_trigger=1.0)
    if sleep_time == 0.0:
        cap_task.cancel()
    elif sleep_time <= 0.75:
        time.sleep(sleep_time)
        cap_task.cancel()
    else:
        time.sleep(0.75)
        awg_task: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)
        time.sleep(sleep_time - 0.75)
        cap_task.cancel()
        assert awg_task.result() is None

    with pytest.raises(CancelledError):
        _ = cap_task.result()

    assert not cap_task.running()
    assert not cap_task.done()
    assert cap_task.cancelled()
    check_caps_are_clear(boxi, set(hwidx_map.values()))

    for runit in cap_targets:
        assert boxi.wss.get_triggering_awg_to_line(capunit_map[runit][0]) is None


@pytest.mark.parametrize(
    ["connect"],
    [({(0, "r", 0)},), ({(1, "r", 3)},)],
)
def test_captrig_partial_trigger_failure(connect: list[tuple[int, str, int]], box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    cap_targets = {(0, "r", 0): (256, 3), (1, "r", 3): (128, 2)}
    capunit_idxs = config_caps_cap_now_seconds(boxi, cap_targets)
    capunit_map = {target: boxi._get_capunit_from_runit(*target) for target in cap_targets}
    hwidx_map = {
        target: boxi.wss._capunit_idx_to_hwidx(boxi._get_capunit_from_runit(*target)) for target in cap_targets
    }
    for target, hwidx in hwidx_map.items():
        assert boxi.wss._hal.capunit(hwidx).get_capture_duration() == cap_targets[target][1] * 125_000_000

    awg_targets = {(0, 0, 0): 1, (1, 0, 0): 1}
    awg_idxs = config_awgs_gen_seconds(boxi, awg_targets)
    awgunit_map = {target: boxi._get_awg_from_channel(*target) for target in awg_targets}
    assert boxi.wss._hal.awgunit(list(awg_idxs)[0]).get_wave_duration() == 125_000_000

    # Notes: connecting CapUnit to Awg, this operation is encapsulated in upper layers.
    if (0, "r", 0) in connect:
        boxi.wss.set_triggering_awg_to_line(capunit_map[0, "r", 0][0], awgunit_map[(0, 0, 0)])
    if (1, "r", 3) in connect:
        boxi.wss.set_triggering_awg_to_line(capunit_map[1, "r", 3][0], awgunit_map[(1, 0, 0)])

    cap_task: StartCapunitsByTriggerTask = boxi.wss.start_capunits_by_trigger(capunit_idxs, timeout_before_trigger=0.01)
    awg_task: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)

    with pytest.raises(RuntimeError):
        _ = cap_task.result()

    assert not cap_task.running()
    assert cap_task.done()
    assert not cap_task.cancelled()
    check_caps_are_clear(boxi, set(hwidx_map.values()))

    assert awg_task.result() is None
    for runit in cap_targets:
        assert boxi.wss.get_triggering_awg_to_line(capunit_map[runit][0]) is None


@pytest.mark.parametrize(
    ["delta_second"],
    [(2,), (4,)],
)
def test_awgtimed_normal(delta_second: float, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    cap_targets = {(0, "r", 0): (256, 5 * 10)}
    capunit_map = {target: boxi._get_capunit_from_runit(*target) for target in cap_targets}
    capunit_idxs = config_caps_cap_now_seconds(boxi, cap_targets, period=125_000_000 // 10)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 1})
    curcntr, _ = boxi.wss._hal.clkcntr.read_counter()
    awg_task: StartAwgunitsTimedTask = boxi.wss.start_awgunits_timed(
        awg_idxs, curcntr + round(125_000_000 * delta_second)
    )
    cap_task: StartCapunitsNowTask = boxi.wss.start_capunits_now(capunit_idxs)  # Notes: capture now
    rdr = cap_task.result()
    data = rdr[capunit_map[0, "r", 0]].as_wave_dict()
    for i in range(50):
        m = np.max(np.abs(data["s0"][i]))
        if delta_second * 10 + 1 <= i <= delta_second * 10 + 8:
            assert m > 1200.0
        elif i in {delta_second * 10 - 1, delta_second * 10, delta_second * 10 + 9, delta_second * 10 + 10}:
            pass
        else:
            assert m < 400.0

    assert awg_task.result() is None
    assert not awg_task.running()
    assert awg_task.done()
    assert not awg_task.cancelled()
    check_awgs_are_clear(boxi, awg_idxs)


@pytest.mark.parametrize(
    ["sleep_time"],
    [(0.25,), (0.55,), (0.65,), (0.75,), (0.85,), (0.95,), (1.05,), (1.15,), (1.25,), (1.35,), (1.45,), (1.75,)],
)
def test_awgtimed_cancel(sleep_time: float, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 1})
    curcntr, _ = boxi.wss._hal.clkcntr.read_counter()
    awg_task: StartAwgunitsTimedTask = boxi.wss.start_awgunits_timed(awg_idxs, curcntr + round(125_000_000 * 1.0))
    for _ in range(10):
        time.sleep(0.01)
        if awg_task.running():
            break
    else:
        assert awg_task.running()

    assert not awg_task.done()
    assert not awg_task.cancelled()

    time.sleep(sleep_time)
    assert not awg_task.done()

    awg_task.cancel()
    with pytest.raises(CancelledError):
        _ = awg_task.result()

    assert not awg_task.running()
    assert not awg_task.done()
    assert awg_task.cancelled()
    check_awgs_are_clear(boxi, awg_idxs)


@pytest.mark.parametrize(
    ["scheduled_time"],
    [(2,), (5,), (10,), (30,), (100,), (300,)],
)
def test_awgtimed_cancel_far(scheduled_time: float, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 1})
    curcntr, _ = boxi.wss._hal.clkcntr.read_counter()
    awg_task: StartAwgunitsTimedTask = boxi.wss.start_awgunits_timed(
        awg_idxs, curcntr + round(125_000_000 * scheduled_time)
    )
    for _ in range(10):
        time.sleep(0.01)
        if awg_task.running():
            break
    else:
        assert awg_task.running()

    assert not awg_task.done()
    assert not awg_task.cancelled()

    time.sleep(0.5)
    assert not awg_task.done()

    awg_task.cancel()
    with pytest.raises(CancelledError):
        _ = awg_task.result()

    assert not awg_task.running()
    assert not awg_task.done()
    assert awg_task.cancelled()
    check_awgs_are_clear(boxi, awg_idxs)


@pytest.mark.parametrize(
    ["schedule_time"],
    [
        (0.5,),
        (1.5,),
    ],
)
def test_awgtimed_busy(schedule_time, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 2, (0, 3, 2): 1})

    task_0: StartAwgunitsNowTask = boxi.wss.start_awgunits_now(awg_idxs)

    curcntr, _ = boxi.wss._hal.clkcntr.read_counter()

    task_1: StartAwgunitsTimedTask = boxi.wss.start_awgunits_timed(
        awg_idxs, curcntr + round(schedule_time * 125_000_000)
    )
    with pytest.raises(RuntimeError):
        task_1.result()

    assert task_0.result() is None


@pytest.mark.parametrize(
    ["schedule_time"],
    [
        (-1,),
        (0.01,),
    ],
)
def test_awgtimed_past(schedule_time, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 2, (0, 3, 2): 1})

    curcntr, _ = boxi.wss._hal.clkcntr.read_counter()
    task_0: StartAwgunitsTimedTask = boxi.wss.start_awgunits_timed(
        awg_idxs, curcntr + round(schedule_time) * 125_000_000
    )
    with pytest.raises(RuntimeError):
        task_0.result()


@pytest.mark.parametrize(
    ["schedule_time"],
    [(0.25,), (0.75,), (1.50,), (2.50,)],
)
def test_captimed_normal_and_timeout(schedule_time: float, box_provider: BoxProvider):
    box = box_provider.get_box_from_type("quel1-a")
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    cap_targets = {(0, "r", 0): (256, 3), (1, "r", 3): (128, 2)}
    capunit_idxs = config_caps_cap_now_seconds(boxi, cap_targets)
    capunit_map = {target: boxi._get_capunit_from_runit(*target) for target in cap_targets}
    hwidx_map = {
        target: boxi.wss._capunit_idx_to_hwidx(boxi._get_capunit_from_runit(*target)) for target in cap_targets
    }

    awg_targets = {(0, 0, 0): 1, (1, 0, 0): 1}
    awg_idxs = config_awgs_gen_seconds(boxi, awg_targets)
    awgunit_map = {target: boxi._get_awg_from_channel(*target) for target in awg_targets}

    # Notes: connecting CapUnit to Awg, this operation is encapsulated in upper layers.
    boxi.wss.set_triggering_awg_to_line(capunit_map[0, "r", 0][0], awgunit_map[(0, 0, 0)])
    boxi.wss.set_triggering_awg_to_line(capunit_map[1, "r", 3][0], awgunit_map[(1, 0, 0)])

    cap_task: StartCapunitsByTriggerTask = boxi.wss.start_capunits_by_trigger(capunit_idxs, timeout_before_trigger=1.0)
    curcntr, _ = boxi.wss._hal.clkcntr.read_counter()
    awg_task: StartAwgunitsTimedTask = boxi.wss.start_awgunits_timed(
        awg_idxs, curcntr + round(125_000_000 * schedule_time)
    )

    for _ in range(10):
        time.sleep(0.01)
        if cap_task.running() and awg_task.running():
            break
    else:
        assert cap_task.running()
        assert awg_task.running()
    assert not cap_task.done()
    assert not cap_task.cancelled()

    if schedule_time > 1.25:  # timeout_before_trigger + margin
        with pytest.raises(TimeoutError):
            _ = cap_task.result()
        assert not cap_task.running()
        assert cap_task.done()
        assert not cap_task.cancelled()
        check_caps_are_clear(boxi, set(hwidx_map.values()))
    else:
        rdrs = cap_task.result()
        assert not cap_task.running()
        assert cap_task.done()
        assert not cap_task.cancelled()
        check_caps_are_clear(boxi, set(hwidx_map.values()))

        for runit, (num_word, num_repeat) in cap_targets.items():
            capunit_idx = capunit_map[runit]
            assert capunit_idx in rdrs
            rdr = rdrs[capunit_idx]
            waves = rdr.as_wave_dict()
            assert "s0" in waves
            assert waves["s0"].shape == (num_repeat, num_word * 4)

    assert awg_task.result() is None
    for runit in cap_targets:
        assert boxi.wss.get_triggering_awg_to_line(capunit_map[runit][0]) is None
