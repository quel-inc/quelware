import pytest

from quel_ic_config import get_exstickge_server_info


def test_with_nonexistent_server():
    with pytest.raises(RuntimeError):
        is_coap, version, _ = get_exstickge_server_info("10.255.254.253")
