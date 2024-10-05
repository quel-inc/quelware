import copy
import logging
import time
from typing import Dict, List, Tuple, Union

import pytest

from quel_ic_config.exstickge_coap_tempctrl_client import _ExstickgeCoapClientQuel1seTempctrlBase
from quel_ic_config.quel1_box import Quel1BoxIntrinsic
from quel_ic_config.quel1se_riken8_config_subsystem import Quel1seRiken8DebugConfigSubsystem
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.94",
            "ipaddr_sss": "10.2.0.94",
            "ipaddr_css": "10.5.0.94",
            "boxtype": Quel1BoxType.fromstr("x-quel1se-riken8"),
        },
    },
)


def diff_temperatures(
    css: Quel1seRiken8DebugConfigSubsystem,
    t0: Dict[Tuple[int, int], float],
    t1: Union[Dict[Tuple[int, int], float], None] = None,
    threshold: float = 0.4,
):
    if t1 is None:
        t1 = css.get_temperatures()

    diff: Dict[Tuple[int, int], float] = {}
    for k in t0:
        d = t1[k] - t0[k]
        if threshold <= abs(d):
            diff[k] = d
    return diff


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def fixtures(request):
    param0 = request.param

    box = Quel1BoxIntrinsic.create(**param0["box_config"])
    assert isinstance(box.css, Quel1seRiken8DebugConfigSubsystem)
    box.css.start_tempctrl_external()
    time.sleep(15)
    _ = box.css.get_tempctrl_temperature().result()
    box.css.stop_tempctrl()
    _ = box.css.get_tempctrl_temperature().result()
    yield box

    hh: List[float] = box.css.get_tempctrl_actuator_output()["heater"]
    for idx in range(len(hh)):
        if hh[idx] != 0.0:
            logger.error(
                f"heater-{idx} is not turned-off (h[{idx}] = {hh[idx]} after the completion of the test, "
                "fix it right now!"
            )
            hh[idx] = 0.0
    box.css.set_tempctrl_actuator_output(fan=[0.5, 0.5], heater=hh)


def test_fan_settings(fixtures):
    css = fixtures.css
    assert isinstance(css, Quel1seRiken8DebugConfigSubsystem)

    proxy = css._proxy
    assert isinstance(proxy, _ExstickgeCoapClientQuel1seTempctrlBase)

    t0 = css.get_tempctrl_temperature().result()
    a0 = css.get_tempctrl_actuator_output()
    logger.info(f"current actuator settings: {a0}")

    fan_idx = 0
    num_fan = 2
    while fan_idx < num_fan + 1:
        # wait for the next loop count
        t1 = css.get_tempctrl_temperature().result()
        logger.info(f"temperature acquired at loop {proxy.read_tempctrl_loop_count()}")
        logger.info(f"diff temperature: {diff_temperatures(css, t0, t1, 0.5)}")
        t0 = t1

        # check heater settings
        a0 = css.get_tempctrl_actuator_output()
        logger.info(f"current actuator settings: {a0}")
        h = a0["fan"]

        for i in range(num_fan):
            if i == fan_idx - 1:
                assert h[i] == 1.0, f"unexpected setting of fan[{i}]: {h[i]} (!= 1.0)"
            else:
                assert h[i] == 0.5, f"unexpected setting of fan[{i}]: {h[i]} (!= 0.5)"

        # update heater settings
        if fan_idx > 0:
            a0["fan"][fan_idx - 1] = 0.5
        if fan_idx < num_fan:
            a0["fan"][fan_idx] = 1.0
        css.set_tempctrl_actuator_output(**a0)

        fan_idx += 1

    # Notes: to ensure the last set_thermal_actuator_output() becomes effective.
    _ = css.get_tempctrl_temperature().result()


def test_heater_settings(fixtures):
    css = fixtures.css
    assert isinstance(css, Quel1seRiken8DebugConfigSubsystem)

    proxy = css._proxy
    assert isinstance(proxy, _ExstickgeCoapClientQuel1seTempctrlBase)

    t0 = css.get_tempctrl_temperature().result()
    a0 = css.get_tempctrl_actuator_output()
    logger.info(f"current actuator settings: {a0}")

    num_heater = 40
    heater_prev_idx = -1
    # Notes: choose heater index randomly to reduce time.
    for heater_idx in (0, 1, 10, 11, 20, 21, 30, 31, 39, 40):
        # wait for the next loop count
        t1 = css.get_tempctrl_temperature().result()
        logger.info(f"temperature acquired at loop {proxy.read_tempctrl_loop_count()}")
        logger.info(f"diff temperature: {diff_temperatures(css, t0, t1, 0.5)}")
        t0 = t1

        # check heater settings
        a0 = css.get_tempctrl_actuator_output()
        logger.info(f"current actuator settings: {a0}")
        h = a0["heater"]
        hh = css.get_heater_outputs()  # Notes: this API is only for test.

        for i in range(num_heater):
            if i == heater_prev_idx:
                assert h[i] == 0.3, f"unexpected setting of heater[{i}]: {h[i]} (!= 0.3)"
                if i in hh:
                    assert hh[i] == 0.3, f"unexpected register value of heater[{i}]: {hh[i]} (!= 0.3)"
            else:
                assert h[i] == 0, f"unexpected setting of heater[{i}]: {h[i]} (!= 0.0)"
                if i in hh:
                    assert hh[i] == 0.0, f"unexpected register value of heater[{i}]: {hh[i]} (!= 0.0)"

        # update heater settings
        if heater_prev_idx >= 0:
            a0["heater"][heater_prev_idx] = 0
        if heater_idx < num_heater:
            a0["heater"][heater_idx] = 0.3
        css.set_tempctrl_actuator_output(**a0)

        heater_prev_idx = heater_idx

    # Notes: to ensure the last set_thermal_actuator_output() becomes effective.
    _ = css.get_tempctrl_temperature().result()


def test_setpoint(fixtures):
    css = fixtures.css
    assert isinstance(css, Quel1seRiken8DebugConfigSubsystem)

    v0 = css.get_tempctrl_setpoint()
    v = copy.deepcopy(v0)
    for i, _ in enumerate(v["fan"]):
        v["fan"][i] = 10.0 + i
    for i, _ in enumerate(v["heater"]):
        v["heater"][i] = 20.0 + i

    css.set_tempctrl_setpoint(**v)
    w = css.get_tempctrl_setpoint()

    for i, u in enumerate(w["fan"]):
        assert u == v["fan"][i], f"unexpected setpoint of fan[{i}]: {u} (!= {v['fan'][i]})"

    for i, u in enumerate(v["heater"]):
        assert u == v["heater"][i], f"unexpected setpoint of heater[{i}]: {u} (!= {v['heater'][i]})"

    css.set_tempctrl_setpoint(**v0)


def test_gain(fixtures):
    css = fixtures.css
    assert isinstance(css, Quel1seRiken8DebugConfigSubsystem)

    v0 = css.get_tempctrl_gain()

    v = copy.deepcopy(v0)
    v["fan"]["Kp"] = 0.001
    v["fan"]["Ki"] = 0.002
    v["heater"]["Kp"] = 0.003
    v["heater"]["Ki"] = 0.004
    css.set_tempctrl_gain(**v)

    w = css.get_tempctrl_gain()

    for atype in ("fan", "heater"):
        for k in ("Kp", "Ki"):
            assert abs(v[atype][k] - w[atype][k]) <= 1 / (
                1 << 22
            ), f"unexpected coefficient '{k}' of '{atype}': {w[atype][k]} (!= {v[atype][k]}"

    css.set_tempctrl_gain(**v0)
