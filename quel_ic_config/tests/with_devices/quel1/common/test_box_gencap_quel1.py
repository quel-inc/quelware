import logging
import time

import numpy as np
import pytest

from quel_ic_config.quel1_box import BoxStartCapunitsByTriggerTask, BoxStartCapunitsNowTask
from quel_ic_config.quel1_wave_subsystem import AbstractStartAwgunitsTask
from testlibs.gencap_quel1_utils import config_lines, config_rlines
from testlibs.gencap_utils import (
    check_awgs_are_clear,
    check_caps_are_clear,
    config_awgs_gen_seconds,
    config_caps_cap_now_seconds,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def test_awg_normal(fixtures1):
    box, params, dpath = fixtures1
    if params["label"] not in {"staging-050"}:
        pytest.skip()
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 1})
    assert boxi.wss._hal.awgunit(list(awg_idxs)[0]).get_wave_duration() == 125_000_000

    task: AbstractStartAwgunitsTask = box.start_wavegen({(1, 0)})  # Port-#1
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
    ["delta_second"],
    [(2,), (4,)],
)
def test_awgtimed_normal(delta_second: float, fixtures1):
    box, params, dpath = fixtures1
    if params["label"] not in {"staging-050"}:
        pytest.skip()
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    # Port-#01  --> Port-#00
    # (0, 0, 0) --> (0, "r", 0)
    cap_targets = {(0, "r", 0): (256, 5 * 10)}
    config_caps_cap_now_seconds(boxi, cap_targets, period=125_000_000 // 10)

    awg_idxs = config_awgs_gen_seconds(boxi, {(0, 0, 0): 1})
    curcntr = box.get_current_timecounter()
    awg_task: AbstractStartAwgunitsTask = box.start_wavegen(
        {(1, 0)}, curcntr + round(boxi.wss._hal.clkcntr.CLOCK_FREQUENCY * delta_second)  # Port-#01
    )
    cap_task: BoxStartCapunitsNowTask = box.start_capture_now({(0, 0)})  # Notes: Port-#00
    rdr = cap_task.result()
    data = rdr[(0, 0)].as_wave_dict()
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
    ["schedule_time"],
    [(0.25,), (8.0,)],
)
def test_captimed_normal_and_timeout(schedule_time: float, fixtures1):
    box, params, dpath = fixtures1
    if params["label"] not in {"staging-050"}:
        pytest.skip()
    boxi = box._dev
    config_lines(boxi)
    config_rlines(boxi)

    # Port-#01  --> Port-#00,    Port-#08  --> Port-#07
    # (0, 0, 0) --> (0, "r", 0)  (1, 0, 0) --> (1, "r", 3)
    cap_targets = {(0, "r", 0): (256, 3), (1, "r", 3): (128, 2)}
    config_caps_cap_now_seconds(boxi, cap_targets)
    capunit_map = {target: boxi._get_capunit_from_runit(*target) for target in cap_targets}
    hwidx_map = {
        target: boxi.wss._capunit_idx_to_hwidx(boxi._get_capunit_from_runit(*target)) for target in cap_targets
    }

    awg_targets = {(0, 0, 0): 1, (1, 0, 0): 1}
    config_awgs_gen_seconds(boxi, awg_targets)

    cap_task, awg_task = box.start_capture_by_awg_trigger(
        runits={(0, 0), (7, 3)},
        channels={(1, 0), (8, 0)},
        timecounter=boxi.wss.get_current_timecounter() + round(boxi.wss._hal.clkcntr.CLOCK_FREQUENCY * schedule_time),
    )
    assert isinstance(cap_task, BoxStartCapunitsByTriggerTask)
    assert isinstance(awg_task, AbstractStartAwgunitsTask)

    for _ in range(10):
        time.sleep(0.01)
        if cap_task.running() and awg_task.running():
            break
    else:
        assert cap_task.running()
        assert awg_task.running()
    assert not cap_task.done()
    assert not cap_task.cancelled()

    rdrs = cap_task.result()
    assert not cap_task.running()
    assert cap_task.done()
    assert not cap_task.cancelled()
    check_caps_are_clear(boxi, set(hwidx_map.values()))

    # TODO: rewrite test case in terms of (port, channel)
    rline_to_port = {
        (0, "r", 0): (0, 0),
        (1, "r", 3): (7, 3),
    }
    for runit_, (num_word, num_repeat) in cap_targets.items():
        runit = rline_to_port[runit_]
        assert runit in rdrs
        rdr = rdrs[runit]
        waves = rdr.as_wave_dict()
        assert "s0" in waves
        assert waves["s0"].shape == (num_repeat, num_word * 4)

    assert awg_task.result() is None
    for runit_ in cap_targets:
        assert boxi.wss.get_triggering_awg_to_line(capunit_map[runit_][0]) is None
