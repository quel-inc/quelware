import pytest

from quel_ic_config import get_exstickge_server_info


def test_with_nonexistent_server():
    with pytest.raises(RuntimeError):
        is_coap, version, _ = get_exstickge_server_info("10.255.254.253")


def test_with_coap_server():
    is_coap, version, _ = get_exstickge_server_info("10.5.0.94")
    assert is_coap
    assert version == "v1.0.3"


def test_with_udp_server():
    is_coap, version, _ = get_exstickge_server_info("10.5.0.74")
    assert not is_coap
    assert version == ""
