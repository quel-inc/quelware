from quel_ic_config import get_exstickge_server_info


# Notes: this test case is almost identical to tests/with_device/quel1/common/test_exstickge_version_quel1.py
def test_with_nonexistent_server():
    is_coap, version, _ = get_exstickge_server_info("10.255.254.253")
    assert not is_coap
    assert version == ""
