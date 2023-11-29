import time

from quel_inst_tool import Pe6108ava, PeSwitchState

TARGET_IPADDR = "10.250.0.102"
SWITCH_FOR_TEST = 4


def test_pe6108ava():
    # TODO: check the availability of the target resource as in the same as the other devices.
    # Notes: PE6108AVA (10.250.0.102) is assumed to be always available because it is a part of the test environment.
    obj = Pe6108ava(TARGET_IPADDR)

    obj.turn_switch_off(SWITCH_FOR_TEST)
    assert obj.check_switch(SWITCH_FOR_TEST) == PeSwitchState.OFF
    obj.turn_switch_on(SWITCH_FOR_TEST)
    assert obj.check_switch(SWITCH_FOR_TEST) == PeSwitchState.ON

    obj.turn_switch(SWITCH_FOR_TEST, PeSwitchState.OFF)
    assert obj.check_switch(SWITCH_FOR_TEST) == PeSwitchState.OFF
    obj.turn_switch(SWITCH_FOR_TEST, PeSwitchState.ON)
    assert obj.check_switch(SWITCH_FOR_TEST) == PeSwitchState.ON

    t0 = time.perf_counter()
    obj.powercycle_switch(SWITCH_FOR_TEST, 10.0)
    assert time.perf_counter() - t0 > 10.0
    assert obj.check_switch(SWITCH_FOR_TEST) == PeSwitchState.ON

    # Notes: you can comment out the following test cases for casual tests,
    #        but don't forget re-enable it before making PR.
    time.sleep(130)
    # session should be re-established after 120 seconds with no communication.
    assert obj.check_switch(SWITCH_FOR_TEST) == PeSwitchState.ON
    obj.turn_switch_off(SWITCH_FOR_TEST)
    assert obj.check_switch(SWITCH_FOR_TEST) == PeSwitchState.OFF
