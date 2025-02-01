import copy
import logging
import time
from typing import Dict, Union

import pytest

from quel_ic_config.exstickge_coap_tempctrl_client import _ExstickgeCoapClientQuel1seTempctrlBase
from quel_ic_config.quel1se_riken8_config_subsystem import Quel1seRiken8ConfigSubsystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def diff_temperatures(
    css: Quel1seRiken8ConfigSubsystem,
    t0: Dict[str, float],
    t1: Union[Dict[str, float], None] = None,
    threshold: float = 0.4,
):
    if t1 is None:
        t1 = css.get_tempctrl_temperature_now()

    diff: Dict[str, float] = {}
    for k in t0:
        d = t1[k] - t0[k]
        if threshold <= abs(d):
            diff[k] = d
    return diff


@pytest.fixture(scope="module")
def fixtures_local(fixtures8):
    box, _, _ = fixtures8

    css = box.css
    assert isinstance(css, Quel1seRiken8ConfigSubsystem)
    css.start_tempctrl_external()  # Notes: enter into test mode, normal thermal control is halted.
    time.sleep(15)
    _ = css.get_tempctrl_temperature().result()
    css.stop_tempctrl()
    _ = css.get_tempctrl_temperature().result()
    yield box

    hh: Dict[str, float] = css.get_tempctrl_actuator_output()["heater"]
    for name, hhh in hh.items():
        if hhh != 0.0:
            logger.error(f"heater['{name}'] is not turned-off ({hhh}) after the completion of the test, fixing...")
            hh[name] = 0.0
    css.set_tempctrl_actuator_output(fan={"adda_lmx2594_0": 0.5, "adda_lmx2594_1": 0.5}, heater=hh)
    time.sleep(15)

    css.start_tempctrl()
    time.sleep(15)


def test_fan_settings(fixtures_local):
    css = fixtures_local.css
    assert isinstance(css, Quel1seRiken8ConfigSubsystem)

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
            name = f"adda_lmx2594_{i}"
            if i == fan_idx - 1:
                assert h[name] == 1.0, f"unexpected setting of fan[{name}]: {h[name]} (!= 1.0)"
            else:
                assert h[name] == 0.5, f"unexpected setting of fan[{name}]: {h[name]} (!= 0.5)"

        # update heater settings
        if fan_idx > 0:
            a0["fan"][f"adda_lmx2594_{fan_idx - 1}"] = 0.5
        if fan_idx < num_fan:
            a0["fan"][f"adda_lmx2594_{fan_idx}"] = 1.0
        css.set_tempctrl_actuator_output(**a0)

        fan_idx += 1

    # Notes: to ensure the last set_thermal_actuator_output() becomes effective.
    _ = css.get_tempctrl_temperature().result()


def test_heater_settings(fixtures_local):
    css = fixtures_local.css
    assert isinstance(css, Quel1seRiken8ConfigSubsystem)

    proxy = css._proxy
    assert isinstance(proxy, _ExstickgeCoapClientQuel1seTempctrlBase)

    t0 = css.get_tempctrl_temperature().result()
    a0 = css.get_tempctrl_actuator_output()
    logger.info(f"current actuator settings: {a0}")

    heater_prev_name = ""
    # Notes: choose heater index randomly to reduce time.
    for heater_name in (
        "mx0_adrf6780_0",
        "mx0_amp_1",
        "ps0_lna_readin",
        "ps0_lna_readout",
        "mx1_amp_0",
        "mx1_amp_1",
        "ps1_lna_readin",
        "ps1_lna_readout",
        "",
    ):
        # wait for the next loop count
        t1 = css.get_tempctrl_temperature().result()
        logger.info(f"temperature acquired at loop {proxy.read_tempctrl_loop_count()}")
        logger.info(f"diff temperature: {diff_temperatures(css, t0, t1, 0.5)}")
        t0 = t1

        # check heater settings
        a0 = css.get_tempctrl_actuator_output()
        logger.info(f"current actuator settings: {a0}")
        h = a0["heater"]

        for hn, hv in h.items():
            if hn == heater_prev_name:
                assert hv == 0.3, f"unexpected setting of heater['{hn}']: {hv} (!= 0.3)"
            else:
                assert hv == 0.0, f"unexpected setting of heater['{hn}']: {hv} (!= 0.0)"

        # update heater settings
        if heater_prev_name != "":
            a0["heater"][heater_prev_name] = 0.0
        if heater_name != "":
            a0["heater"][heater_name] = 0.3
        css.set_tempctrl_actuator_output(**a0)

        heater_prev_name = heater_name

    # Notes: to ensure the last set_thermal_actuator_output() becomes effective.
    _ = css.get_tempctrl_temperature().result()


def test_setpoint(fixtures_local):
    css = fixtures_local.css
    assert isinstance(css, Quel1seRiken8ConfigSubsystem)

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


def test_gain(fixtures_local):
    css = fixtures_local.css
    assert isinstance(css, Quel1seRiken8ConfigSubsystem)

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
