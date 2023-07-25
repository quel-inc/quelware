import time
from typing import Any, Dict, Final, Tuple

import pytest

from quel_clock_master import QuBEMasterClient, SequencerClient

TIME_PRECISION: Final[float] = 0.005  # ~ 5ms
CLOCK_RATE: Final[float] = 125000000.0  # Hz

DEVICE_SETTINGS: Final[Dict[str, Dict[str, Any]]] = {
    "MASTER": {
        "ipaddr": "10.3.0.13",
    },
    "TARGET0": {
        "ipaddr": "10.2.0.42",
    },
    "TARGET1": {
        "ipaddr": "10.2.0.58",
    },
}


@pytest.fixture(scope="session", params=(DEVICE_SETTINGS,))
def proxies(request) -> Tuple[QuBEMasterClient, Dict[str, SequencerClient]]:
    param0 = request.param
    master = QuBEMasterClient(param0["MASTER"]["ipaddr"])
    retcode = master.reset()
    assert retcode

    target0_ipaddr = param0["TARGET0"]["ipaddr"]
    target0 = SequencerClient(target0_ipaddr)
    target1_ipaddr = param0["TARGET1"]["ipaddr"]
    target1 = SequencerClient(target1_ipaddr)

    return master, {target0_ipaddr: target0, target1_ipaddr: target1}


def test_clear(proxies: Tuple[QuBEMasterClient, Dict[str, SequencerClient]]):
    master, _ = proxies

    t0 = (1 << 32) - (3 * int(CLOCK_RATE))
    retcode0 = master.clear_clock(t0)
    assert retcode0

    retcode1, t1 = master.read_clock()
    assert retcode1

    assert abs((t1 - t0) / CLOCK_RATE) < TIME_PRECISION


def test_read_clock_master(proxies: Tuple[QuBEMasterClient, Dict[str, SequencerClient]]):
    master, _ = proxies
    duration = 1

    t0 = master.read_clock()
    assert t0[0]
    time.sleep(duration)
    t1 = master.read_clock()
    assert t1[0]
    assert abs((t1[1] - t0[1]) / CLOCK_RATE - duration) < TIME_PRECISION


def test_ipaddress(proxies: Tuple[QuBEMasterClient, Dict[str, SequencerClient]]):
    _, targets = proxies

    for ipaddress, target in targets.items():
        assert target.ipaddress == ipaddress


def test_read_clock(proxies: Tuple[QuBEMasterClient, Dict[str, SequencerClient]]):
    _, targets = proxies
    duration = 1

    for _, target in targets.items():
        t0 = target.read_clock()
        assert t0[0]
        time.sleep(duration)
        t1 = target.read_clock()
        assert t1[0]
        assert abs((t1[1] - t0[1]) / CLOCK_RATE - duration) < TIME_PRECISION


def test_kick(proxies: Tuple[QuBEMasterClient, Dict[str, SequencerClient]]):
    master, targets = proxies

    time.sleep(0.1)
    retcode1: bool = master.kick_clock_synch([ipaddr for ipaddr, proxy in targets.items()])
    assert retcode1
    time.sleep(0.1)
    for target_addr, target_proxy in targets.items():
        t0 = master.read_clock()
        t1 = target_proxy.read_clock()
        assert t0[0]
        assert t1[0]
        assert abs((t1[1] - t0[1]) / CLOCK_RATE) < TIME_PRECISION


def test_nonexistent_target():
    target = SequencerClient("10.193.194.195")
    retcode, _ = target.read_clock()
    assert not retcode


def test_nonexistent_master():
    master = QuBEMasterClient("10.193.194.195")
    retcode0, _ = master.read_clock()
    assert not retcode0

    retcode1 = master.reset()
    assert not retcode1

    retcode2 = master.clear_clock()
    assert not retcode2

    retcode3 = master.kick_clock_synch(["10.193.194.196"])
    assert not retcode3


def test_too_long_packet():
    master = QuBEMasterClient("10.193.194.195")

    with pytest.raises(ValueError):
        _ = master.kick_clock_synch([f"10.193.194.{x}" for x in range(1, 253)])
