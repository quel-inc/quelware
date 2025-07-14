import logging
import time

import pytest

from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal, create_quel1au50hal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.50",
        },
    },
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def proxy(request) -> AbstractQuel1Au50Hal:
    proxy = create_quel1au50hal(ipaddr_wss=request.param["box_config"]["ipaddr_wss"], auth_callback=lambda: True)
    proxy.initialize()
    return proxy


def test_counter(proxy):
    cnt1 = proxy.clkcntr.read_counter()
    time.sleep(0.5)
    cnt2 = proxy.clkcntr.read_counter()

    assert 125_000_000 * 0.5 * 0.95 < (cnt2[0] - cnt1[0]) < 125_000_000 * 0.5 * 1.15  # Notes: -5% ~ +15%
    # Notes: only very old firmware returns None.
    assert cnt1[1] is not None
    assert cnt2[1] is not None
    delta = (cnt2[1] - cnt1[1]) % 2000
    if delta >= 1000:
        delta -= 2000
    assert abs(delta) <= 2
