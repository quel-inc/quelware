import logging

import pytest

from quel_ic_config.exstickge_coap_client import _ExstickgeCoapClientBase, get_exstickge_server_info
from testlibs.create_css_proxy import create_proxy

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
        "label": "staging-157",
        "config": {
            "ipaddr_css": "10.5.0.157",
        },
        "expected": {
            "boxtype": "quel1se-fujitsu11-a",
            "version": {"v1.2.1"},
        },
    },
)


@pytest.mark.parametrize(
    ("param0",),
    [
        (DEVICE_SETTINGS[0],),
        (DEVICE_SETTINGS[1],),
    ],
)
def test_basic(param0):
    proxy = create_proxy(param0["config"]["ipaddr_css"])
    expected = param0["expected"]

    if not isinstance(proxy, _ExstickgeCoapClientBase):
        assert False, "unexpected type of the proxy object"

    assert proxy.read_boxtype() in expected["boxtype"]
    # TODO: add more test which doesn't disrupt the link status.

    del proxy


@pytest.mark.parametrize(
    ("param0",),
    [
        (DEVICE_SETTINGS[0],),
        (DEVICE_SETTINGS[1],),
    ],
)
def test_with_coap_server(param0):
    config = param0["config"]
    expected = param0["expected"]

    is_coap, version, _ = get_exstickge_server_info(config["ipaddr_css"])
    assert is_coap
    assert version in expected["version"]
