import time
from typing import Tuple

import pytest

from quel_inst_tool import Pe4104aj, Pe6108ava, PeSwitchState, Pexxxx


def test_pexxxx(pexxxx_ch_and_device):
    ch_and_device: Tuple[int, Pexxxx] = pexxxx_ch_and_device
    ch_for_test = ch_and_device[0]
    obj = ch_and_device[1]

    with pytest.raises(ValueError):
        if isinstance(obj, Pe6108ava):
            obj.turn_switch_off(9)
        elif isinstance(obj, Pe4104aj):
            obj.turn_switch_off(5)
        else:
            raise AssertionError

    obj.turn_switch_off(ch_for_test)
    assert obj.check_switch(ch_for_test) == PeSwitchState.OFF
    obj.turn_switch_on(ch_for_test)
    assert obj.check_switch(ch_for_test) == PeSwitchState.ON

    obj.turn_switch(ch_for_test, PeSwitchState.OFF)
    assert obj.check_switch(ch_for_test) == PeSwitchState.OFF
    obj.turn_switch(ch_for_test, PeSwitchState.ON)
    assert obj.check_switch(ch_for_test) == PeSwitchState.ON

    t0 = time.perf_counter()
    obj.powercycle_switch(ch_for_test, 10.0)
    assert time.perf_counter() - t0 > 10.0
    assert obj.check_switch(ch_for_test) == PeSwitchState.ON

    # Notes: you can comment out the following test cases for casual tests,
    #        but don't forget re-enable it before making PR.
    time.sleep(130)
    # session should be re-established after 120 seconds with no communication.
    assert obj.check_switch(ch_for_test) == PeSwitchState.ON
    obj.turn_switch_off(ch_for_test)
    assert obj.check_switch(ch_for_test) == PeSwitchState.OFF
