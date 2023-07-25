import socket
from concurrent.futures import ThreadPoolExecutor

import pytest

from quel_clock_master.simpleudpclient import SimpleUdpClient


# Notes: packet from illegal host is filtered well even when receiver_limit_by_bind is False.
#        "filtering with bind()" may improve performance slightly. At least it is confirmed to be harmless.
@pytest.mark.parametrize(
    "with_bind",
    [
        (False,),
        (True,),
    ],
)
def test_bogus_host(with_bind):
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock1.bind(("127.0.0.1", port))
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2.bind(("127.1.0.1", 0))
    tgt = SimpleUdpClient("127.0.0.1", receiver_limit_by_bind=with_bind)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt._send_recv_generic, port, b"hello")
        rpl, sender = sock1.recvfrom(8)
        bogus_pkt = b"bogus"
        for _ in range(100):
            sock2.sendto(bogus_pkt, sender)
        sock1.sendto(b"genuine", sender)

    sock1.close()
    sock2.close()
    reply, raddr = future.result()
    assert raddr is not None
    assert reply == b"genuine"


# Notes: this kind of "attack" cannot be avoided based on bind(). At least it is harmless in this conditions.
@pytest.mark.parametrize(
    "with_bind",
    [
        (False,),
        (True,),
    ],
)
def test_loopback_read_bogus_port(with_bind):
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock1.bind(("127.0.0.1", port))
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2.bind(("127.0.0.1", 0))
    tgt = SimpleUdpClient("127.0.0.1", receiver_limit_by_bind=with_bind)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt._send_recv_generic, port, b"hello")
        rpl, sender = sock1.recvfrom(8)
        bogus_pkt = b"bogus"
        genuine_pkt = b"genuine"
        for _ in range(100):
            sock2.sendto(bogus_pkt, sender)
        sock1.sendto(genuine_pkt, sender)

    sock1.close()
    sock2.close()
    reply, raddr = future.result()
    assert raddr is not None
    assert reply == genuine_pkt
