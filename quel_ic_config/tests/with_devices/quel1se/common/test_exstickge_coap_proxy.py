import logging
from typing import Any, Dict, Tuple

import pytest

from quel_ic_config.exstickge_coap_client import _ExstickgeCoapClientBase, get_exstickge_server_info
from testlibs.create_css_proxy import ProxyType, create_proxy

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = (
    {
        "label": "staging-094",
        "config": {
            "ipaddr_css": "10.5.0.94",
        },
        "expected": {
            "boxtype": "quel1se-riken8",
            "version": {"v1.2.1", "v1.3.0"},
        },
    },
    {
        "label": "staging-158",
        "config": {
            "ipaddr_css": "10.5.0.158",
        },
        "expected": {
            "boxtype": "quel1se-fujitsu11-a",
            "version": {"v1.2.1"},
        },
    },
)


@pytest.fixture(scope="module", params=DEVICE_SETTINGS)
def fixtures_local(request) -> Tuple[ProxyType, Dict[str, Any], Dict[str, Any]]:
    param0 = request.param

    proxy = create_proxy(param0["config"]["ipaddr_css"])
    return proxy, param0["config"], param0["expected"]


def test_basic(fixtures_local):
    proxy, _, expected = fixtures_local
    if not isinstance(proxy, _ExstickgeCoapClientBase):
        assert False, "unexpected type of the proxy object"

    assert proxy.read_boxtype() in expected["boxtype"]
    # TODO: add more test which doesn't disrupt the link status.


def test_with_coap_server(fixtures_local):
    _, config, expected = fixtures_local

    is_coap, version, _ = get_exstickge_server_info(config["ipaddr_css"])
    assert is_coap
    assert version in expected["version"]
