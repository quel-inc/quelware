import socket
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from quel_ic_config.exstickge_proxy import LsiKindId
from quel_ic_config.quel1_config_subsystem import ExstickgeSockClientQuel1WithDummyLock


# jig for unittest
def spkt(hexstring):
    return bytes([int(x, 16) for x in hexstring.split()])


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value,pkt",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x4000, spkt("82 02 00 00 00 00 40 00")),
        (LsiKindId.ADRF6780, 1, 3, 0x0167, spkt("82 02 00 00 00 43 01 67")),
        (LsiKindId.ADRF6780, 2, 5, 0x0400, spkt("82 02 00 00 00 85 04 00")),
        (LsiKindId.ADRF6780, 5, 0, 0x4000, spkt("82 03 00 00 00 40 40 00")),
        (LsiKindId.ADRF6780, 6, 0, 0x0000, spkt("82 03 00 00 00 80 00 00")),
        (LsiKindId.ADRF6780, 7, 5, 0x0400, spkt("82 03 00 00 00 c5 04 00")),
        (LsiKindId.LMX2594, 8, 0x00, 0x6612, spkt("82 05 00 00 01 80 66 12")),
        (LsiKindId.LMX2594, 8, 0x42, 0x01F4, spkt("82 05 00 00 01 c2 01 f4")),
        (LsiKindId.LMX2594, 9, 0x49, 0x003F, spkt("82 05 00 00 02 49 00 3f")),
        (LsiKindId.LMX2594, 9, 0x40, 0x1388, spkt("82 05 00 00 02 40 13 88")),
        (LsiKindId.LMX2594, 0, 0x46, 0xC350, spkt("82 04 00 00 00 46 c3 50")),
        (LsiKindId.LMX2594, 1, 0x4B, 0x0840, spkt("82 04 00 00 00 cb 08 40")),
        (LsiKindId.LMX2594, 2, 0x2D, 0xC8DF, spkt("82 04 00 00 01 2d c8 df")),
        (LsiKindId.LMX2594, 3, 0x31, 0x4180, spkt("82 04 00 00 01 b1 41 80")),
        (LsiKindId.LMX2594, 4, 0x44, 0x03E8, spkt("82 04 00 00 02 44 03 e8")),
        (LsiKindId.LMX2594, 5, 0x47, 0x0081, spkt("82 05 00 00 00 47 00 81")),
        (LsiKindId.LMX2594, 6, 0x4B, 0x0840, spkt("82 05 00 00 00 cb 08 40")),
        (LsiKindId.LMX2594, 7, 0x3E, 0x0322, spkt("82 05 00 00 01 3e 03 22")),
        (LsiKindId.AD5328, 0, 0x8, 0x00C, spkt("82 06 00 00 00 08 00 0c")),
        (LsiKindId.AD5328, 0, 0x3, 0xC00, spkt("82 06 00 00 00 03 0c 00")),
        (LsiKindId.GPIO, 0, 0x0, 0xFFFF, spkt("82 07 00 00 00 00 ff ff")),
        (LsiKindId.AD9082, 0, 0x1234, 0x5678, spkt("82 01 00 00 12 34 56 78")),
        (LsiKindId.AD9082, 1, 0x1234, 0x5678, spkt("82 01 00 00 92 34 56 78")),
    ],
)
def test_wpakcet(lsitype, lsiidx, addr, value, pkt):
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1")
    tgt.initialize()
    assert tgt._make_writepkt(lsitype, lsiidx, addr, value) == pkt


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,pkt",
    [
        (LsiKindId.ADRF6780, 0, 0, spkt("80 02 00 00 00 00 00 00")),
        (LsiKindId.ADRF6780, 1, 3, spkt("80 02 00 00 00 43 00 00")),
        (LsiKindId.ADRF6780, 2, 5, spkt("80 02 00 00 00 85 00 00")),
        (LsiKindId.ADRF6780, 5, 0, spkt("80 03 00 00 00 40 00 00")),
        (LsiKindId.ADRF6780, 6, 0, spkt("80 03 00 00 00 80 00 00")),
        (LsiKindId.ADRF6780, 7, 5, spkt("80 03 00 00 00 c5 00 00")),
        (LsiKindId.LMX2594, 8, 0x00, spkt("80 05 00 00 01 80 00 00")),
        (LsiKindId.LMX2594, 8, 0x42, spkt("80 05 00 00 01 c2 00 00")),
        (LsiKindId.LMX2594, 9, 0x49, spkt("80 05 00 00 02 49 00 00")),
        (LsiKindId.LMX2594, 9, 0x40, spkt("80 05 00 00 02 40 00 00")),
        (LsiKindId.LMX2594, 0, 0x46, spkt("80 04 00 00 00 46 00 00")),
        (LsiKindId.LMX2594, 1, 0x4B, spkt("80 04 00 00 00 cb 00 00")),
        (LsiKindId.LMX2594, 2, 0x2D, spkt("80 04 00 00 01 2d 00 00")),
        (LsiKindId.LMX2594, 3, 0x31, spkt("80 04 00 00 01 b1 00 00")),
        (LsiKindId.LMX2594, 4, 0x44, spkt("80 04 00 00 02 44 00 00")),
        (LsiKindId.LMX2594, 5, 0x47, spkt("80 05 00 00 00 47 00 00")),
        (LsiKindId.LMX2594, 6, 0x4B, spkt("80 05 00 00 00 cb 00 00")),
        (LsiKindId.LMX2594, 7, 0x3E, spkt("80 05 00 00 01 3e 00 00")),
        (LsiKindId.AD5328, 0, 0x8, spkt("80 06 00 00 00 08 00 00")),
        (LsiKindId.AD5328, 0, 0x3, spkt("80 06 00 00 00 03 00 00")),
        (LsiKindId.GPIO, 0, 0x0, spkt("80 07 00 00 00 00 00 00")),
        (LsiKindId.AD9082, 0, 0x1234, spkt("80 01 00 00 12 34 00 00")),
        (LsiKindId.AD9082, 1, 0x1234, spkt("80 01 00 00 92 34 00 00")),
    ],
)
def test_rpakcet(lsitype, lsiidx, addr, pkt):
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1")
    tgt.initialize()
    assert tgt._make_readpkt(lsitype, lsiidx, addr) == pkt


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value,xrpl",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x1234, spkt("83 02 00 00 00 00 12 34")),
        (LsiKindId.LMX2594, 8, 0x00, 0x5678, spkt("83 05 00 00 01 80 56 78")),
    ],
)
def test_loopback_write_normal(lsitype, lsiidx, addr, value, xrpl):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock.bind(("127.0.0.1", port))
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port)
    tgt.initialize()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt.write_reg, lsitype, lsiidx, addr, value)
        rpl, sender = sock.recvfrom(8)
        sock.sendto(xrpl, sender)

    sock.close()
    assert future.result()


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value,xrpl",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x1234, spkt("81 02 00 00 00 00 12 34")),
        (LsiKindId.LMX2594, 8, 0x00, 0x5678, spkt("81 05 00 00 01 80 56 78")),
    ],
)
def test_loopback_read_normal(lsitype, lsiidx, addr, value, xrpl):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock.bind(("127.0.0.1", port))
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port)
    tgt.initialize()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt.read_reg, lsitype, lsiidx, addr)
        rpl, sender = sock.recvfrom(8)
        sock.sendto(xrpl, sender)

    sock.close()
    assert future.result() == value


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x1234),
    ],
)
def test_loopback_write_timeout(lsitype, lsiidx, addr, value):
    port = 16384
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port, timeout=0.5)
    tgt.initialize()
    assert not tgt.write_reg(lsitype, lsiidx, addr, value)


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr",
    [
        (LsiKindId.ADRF6780, 0, 0),
    ],
)
def test_loopback_read_timeout(lsitype, lsiidx, addr):
    port = 16384
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port, timeout=0.5)
    tgt.initialize()
    assert tgt.read_reg(lsitype, lsiidx, addr) is None


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value,xrpl",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x1234, spkt("81 02 00 00 00 00 12 34")),
        (LsiKindId.LMX2594, 8, 0x00, 0x5678, spkt("81 05 00 00 01 80 56 78")),
    ],
)
def test_loopback_read_broken(lsitype, lsiidx, addr, value, xrpl):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock.bind(("127.0.0.1", port))
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port, timeout=2.0)
    tgt.initialize()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt.read_reg, lsitype, lsiidx, addr)
        rpl, sender = sock.recvfrom(8)
        for i in range(20):
            xrpl_broken = bytearray(xrpl)
            xrpl_broken[i % 6] += 1
            sock.sendto(xrpl_broken, sender)
            time.sleep(0.1)
        sock.sendto(xrpl, sender)

    sock.close()
    assert future.result() == value


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value,xrpl",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x1234, spkt("83 02 00 00 00 00 12 34")),
        (LsiKindId.LMX2594, 8, 0x00, 0x5678, spkt("83 05 00 00 01 80 56 78")),
    ],
)
def test_loopback_write_broken(lsitype, lsiidx, addr, value, xrpl):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock.bind(("127.0.0.1", port))
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port, timeout=2.0)
    tgt.initialize()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt.write_reg, lsitype, lsiidx, addr, value)
        rpl, sender = sock.recvfrom(8)
        for i in range(20):
            xrpl_broken = bytearray(xrpl)
            xrpl_broken[i % 6] += 1
            sock.sendto(xrpl_broken, sender)
            time.sleep(0.1)
        sock.sendto(xrpl, sender)

    sock.close()
    assert future.result()


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value,xrpl",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x1234, spkt("83 02 00 00 00 00 12 34")),
    ],
)
def test_loopback_write_broken_toomany(lsitype, lsiidx, addr, value, xrpl):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock.bind(("127.0.0.1", port))
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port)
    tgt.initialize()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt.write_reg, lsitype, lsiidx, addr, value)
        rpl, sender = sock.recvfrom(8)
        for i in range(32):
            xrpl_broken = bytearray(xrpl)
            xrpl_broken[i % 8] += 1
            sock.sendto(xrpl_broken, sender)
        # NOTE: this right reply packet should not be received
        sock.sendto(xrpl, sender)

    sock.close()
    assert not future.result()


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value,xrpl",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x1234, spkt("81 02 00 00 00 00 12 34")),
        (LsiKindId.LMX2594, 8, 0x00, 0x5678, spkt("81 05 00 00 01 80 56 78")),
    ],
)
def test_loopback_read_bogus_host(lsitype, lsiidx, addr, value, xrpl):
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock1.bind(("127.0.0.1", port))
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2.bind(("127.1.0.1", 0))
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port)
    tgt.initialize()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt.read_reg, lsitype, lsiidx, addr)
        rpl, sender = sock1.recvfrom(8)
        bogus_pkt = bytearray(xrpl)
        bogus_pkt[7] += 1
        for _ in range(100):
            sock2.sendto(bogus_pkt, sender)
        sock1.sendto(xrpl, sender)

    sock1.close()
    sock2.close()
    assert future.result() == value


@pytest.mark.parametrize(
    "lsitype,lsiidx,addr,value,xrpl",
    [
        (LsiKindId.ADRF6780, 0, 0, 0x1234, spkt("81 02 00 00 00 00 12 34")),
        (LsiKindId.LMX2594, 8, 0x00, 0x5678, spkt("81 05 00 00 01 80 56 78")),
    ],
)
def test_loopback_read_bogus_port(lsitype, lsiidx, addr, value, xrpl):
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 16384
    sock1.bind(("127.0.0.1", port))
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2.bind(("127.0.0.1", 0))
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1", port)
    tgt.initialize()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tgt.read_reg, lsitype, lsiidx, addr)
        rpl, sender = sock1.recvfrom(8)
        bogus_pkt = bytearray(xrpl)
        bogus_pkt[7] += 1
        for _ in range(100):
            sock2.sendto(bogus_pkt, sender)
        sock1.sendto(xrpl, sender)

    sock1.close()
    sock2.close()
    assert future.result() == value


def test_not_initialized():
    tgt = ExstickgeSockClientQuel1WithDummyLock("127.0.0.1")
    # Notes: usually should call tgt.initialize() here
    with pytest.raises(RuntimeError):
        _ = tgt.read_reg(LsiKindId.AD9082, 0, 0x0003)
