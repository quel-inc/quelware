import logging

import numpy as np
import pytest

from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal, create_quel1au50hal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.58",
        },
    },
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def proxy(request) -> AbstractQuel1Au50Hal:
    proxy = create_quel1au50hal(ipaddr_wss=request.param["box_config"]["ipaddr_wss"], auth_callback=lambda: True)
    proxy.initialize()
    return proxy


def test_simple(proxy):
    hc = proxy.hbmctrl

    data = bytes(range(256))
    hc._write_simple(0x00001000, 256, memoryview(data))
    d0 = bytes(hc._read_simple(0x00001000, 256))
    assert data == d0

    d1 = bytes(hc._read_simple(0x00001020, 256))
    assert data[0x20:] == d1[:-0x20]

    d2 = bytes(hc._read_simple(0x00001080, 32))
    assert data[0x80:0xA0] == d2

    with pytest.raises(AssertionError):
        _ = bytes(hc._read_simple(0x00001000, 16))


@pytest.mark.parametrize(
    ["size", "offset"],
    [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 359),
        (2, 0),
        (2, 1),
        (2, 258),
        (16, 0),
        (16, 16),
        (16, 360 - 16),
        (64, 0),
        (64, 64),
        (64, 359 - 64),
        (64, 360 - 64),
        (359 - 64, 0),
        (359 - 64, 1),
        (360 - 64, 0),
        (359, 0),
        (359, 1),
        (358, 1),
        (358, 2),
    ],
)
def test_read_iq32_start_and_end(size, offset, proxy):
    hc = proxy.hbmctrl

    raw_data = bytes([i % 256 for i in range(1440)])
    hc._write_simple(0x00001000, 1440, memoryview(raw_data))
    expected = np.frombuffer(raw_data, dtype=np.int16).reshape((360, 2))

    d0 = hc.read_iq32(0x00001000 + offset * 4, size)
    assert (d0 == expected[offset : offset + size]).all()


def test_read_iq32_start_and_end_edge(proxy):
    hc = proxy.hbmctrl

    d = hc.read_iq32(0x1_FFFF_FFFC, 1)
    assert len(d) == 1


@pytest.mark.parametrize(
    ["size", "offset"],
    [
        (360, 0),
        (361, 0),
        (360, 1),
        (718, 0),
        (718, 1),
        (718, 2),
        (718, 3),
        (718, 4),
        (719, 0),
        (719, 1),
        (719, 2),
        (720, 0),
        (720, 1),
        (721, 0),
        (721, 1),
        (11518, 1),
        (11519, 0),
        (11519, 1),
        (11520, 0),
    ],
)
def test_read_iq32_general(size, offset, proxy):
    hc = proxy.hbmctrl

    raw_data = memoryview(bytes([i % 256 for i in range(1440 * 32)]))
    for i in range(32):
        hc._write_simple(0x00001000 + 1440 * i, 1440, raw_data[1440 * i : 1440 * (i + 1)])
    expected = np.frombuffer(raw_data, dtype=np.int16).reshape((1440 * 8, 2))

    d0 = hc.read_iq32(0x00001000 + offset * 4, size)
    assert (d0 == expected[offset : offset + size]).all()


def test_read_iq32_abnormal(proxy):
    hc = proxy.hbmctrl

    # alignment error
    with pytest.raises(ValueError):
        _ = hc.read_iq32(0x00001001, 8)

    # negative address
    with pytest.raises(ValueError):
        _ = hc.read_iq32(-1, 256)

    # too large address
    with pytest.raises(ValueError):
        _ = hc.read_iq32(0x1_FFFF_FFFC, 2)

    # too large address
    with pytest.raises(ValueError):
        _ = hc.read_iq32(0x2_FFFF_FFFC, 1)

    # zero length
    with pytest.raises(ValueError):
        _ = hc.read_iq32(0x0_0000_1000, 0)


@pytest.mark.parametrize(
    ["size"],
    [
        (1,),
        (2,),
        (304,),
        (359,),
        (360,),
        (361,),
        (610,),
        (720 - 65,),
        (720 - 64,),
        (720 - 63,),
        (718,),
        (719,),
        (720,),
        (721,),
        (722,),
        (12500,),
    ],
)
def test_write_iq32_general(size: int, proxy):
    hc = proxy.hbmctrl

    size_aligned = ((size + 359) // 360 + 1) * 360

    data_iq = np.array([[i, -i * 2] for i in range(size)], dtype=np.int16)
    zero_iq = np.array([[0, 0] for _ in range(size_aligned)], dtype=np.int16)

    hc.write_iq32(0x0_0000_2000, size, data_iq)
    d0 = hc.read_iq32(0x0_0000_2000, size)
    assert (d0 == data_iq).all()

    for j in range(1, 9):
        for i in range(1, 9):
            if size - i <= 0:
                continue
            hc.write_iq32(0x0_0000_2000, size_aligned, zero_iq)  # Notes: clear !
            hc.write_iq32(0x0_0000_2000 + 4 * j, size - i, data_iq[:-i])
            d0 = hc.read_iq32(0x0_0000_2000 + 4 * j, size - i)
            assert (d0 == data_iq[:-i]).all()

            d1 = hc.read_iq32(0x0_0000_2000, size_aligned)
            assert (d1[j : j + size - i] == data_iq[:-i]).all()
            assert (d1[:j] == (0, 0)).all()
            assert (d1[j + size - i :] == (0, 0)).all()


def test_write_iq32_abnormal(proxy):
    hc = proxy.hbmctrl
    zero_iq = np.array([[0, 0] for _ in range(256)], dtype=np.int16)

    # alignment error
    with pytest.raises(ValueError):
        _ = hc.write_iq32(0x0_0000_1001, 8, zero_iq[:8])

    # negative address
    with pytest.raises(ValueError):
        _ = hc.write_iq32(-1, 256, zero_iq)

    # too large address
    with pytest.raises(ValueError):
        _ = hc.write_iq32(0x1_FFFF_FFFC, 2, zero_iq[:2])

    # too large address
    with pytest.raises(ValueError):
        _ = hc.write_iq32(0x2_FFFF_FFFC, 1, zero_iq[:1])

    # size mismatch
    with pytest.raises(ValueError):
        _ = hc.write_iq32(0x0_0000_1000, 8, zero_iq)

    # zero length
    with pytest.raises(ValueError):
        _ = hc.write_iq32(0x0_0000_1000, 0, zero_iq[:0])
