import ipaddress
import logging

import pytest

from quel_ic_config.exstickge_coap_client import _ExstickgeCoapClientBase, get_exstickge_server_info
from testlibs.create_css_proxy import create_proxy
from tests.with_devices.conftest import BoxProvider

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def _find_ipaddr_css(box_provider: BoxProvider, boxtype: str) -> str:
    for boxconf in box_provider.find_boxconf_from_type(boxtype):
        if "queltest_exclude_from_css_test" in boxconf.marks:
            continue
        return str(ipaddress.IPv4Address(int(boxconf.ipaddr) + 0x00_04_00_00))
    else:
        raise ValueError(f"target box with type '{boxtype}' not found.")


DEVICE_SETTINGS = (
    {
        "boxtype": "quel1se-riken8",
        "expected_versions": {"v1.2.1", "v1.3.0"},
    },
    {
        "boxtype": "quel1se-fujitsu11-a",
        "expected_versions": {"v1.2.1"},
    },
)


@pytest.mark.parametrize(
    list(DEVICE_SETTINGS[0].keys()),
    [
        list(DEVICE_SETTINGS[0].values()),
        list(DEVICE_SETTINGS[1].values()),
    ],
)
def test_basic(box_provider: BoxProvider, boxtype, expected_versions):
    ipaddr_css = _find_ipaddr_css(box_provider, boxtype)
    proxy = create_proxy(ipaddr_css)

    if not isinstance(proxy, _ExstickgeCoapClientBase):
        assert False, "unexpected type of the proxy object"

    assert proxy.read_boxtype() in boxtype
    # TODO: add more test which doesn't disrupt the link status.

    del proxy


@pytest.mark.parametrize(
    list(DEVICE_SETTINGS[0].keys()),
    [
        list(DEVICE_SETTINGS[0].values()),
        list(DEVICE_SETTINGS[1].values()),
    ],
)
def test_with_coap_server(box_provider: BoxProvider, boxtype, expected_versions):
    ipaddr_css = _find_ipaddr_css(box_provider, boxtype)
    is_coap, version, _ = get_exstickge_server_info(ipaddr_css)
    assert is_coap
    assert version in expected_versions
