import logging

import pytest

from quel_ic_config.exstickge_coap_client import _ExstickgeCoapClientBase
from testlibs.create_css_proxy import ProxyType, create_proxy

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = (
    {
        "label": "staging-132",
        "config": {
            "ipaddr_css": "10.5.0.132",
        },
    },
)


@pytest.fixture(scope="session", params=DEVICE_SETTINGS)
def fixtures(request) -> ProxyType:
    param0 = request.param

    proxy = create_proxy(param0["config"]["ipaddr_css"])
    return proxy


def test_basic(fixtures):
    proxy = fixtures
    if not isinstance(proxy, _ExstickgeCoapClientBase):
        assert False, "unexpected type of the proxy object"

    assert proxy.read_boxtype() == "quel1se-riken8"
    # TODO: add more test which doesn't disrupt the link status.
