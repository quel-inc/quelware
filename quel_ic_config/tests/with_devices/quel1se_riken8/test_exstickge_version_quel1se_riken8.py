from quel_ic_config import get_exstickge_server_info


def test_with_coap_server():
    is_coap, version, _ = get_exstickge_server_info("10.5.0.94")
    assert is_coap
    assert version == "v1.2.1"
