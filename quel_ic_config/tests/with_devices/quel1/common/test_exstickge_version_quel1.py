from quel_ic_config import get_exstickge_server_info


def test_with_udp_server():
    is_coap, version, _ = get_exstickge_server_info("10.5.0.74")
    assert not is_coap
    assert version == ""
