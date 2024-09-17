import logging

import pytest

from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal, create_quel1au50hal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.58",
        },
    },
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def proxy(request) -> AbstractQuel1Au50Hal:
    proxy = create_quel1au50hal(ipaddr_wss=request.param["box_config"]["ipaddr_wss"], auth_callback=lambda: True)
    proxy.initialize()
    return proxy


def test_version(proxy):
    assert proxy.awgctrl.version == "a:2024/01/25-1"
