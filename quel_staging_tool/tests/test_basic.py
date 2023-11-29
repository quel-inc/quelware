import hashlib
import logging

from quel_staging_tool import Au50Programmer, ExstickgeProgrammer

logger = logging.getLogger(__name__)


def test_basic_exstickge():
    obj = ExstickgeProgrammer()
    answer = """\
@00000000
1400050a
000000ff
0100000a
0100000a
ee1a1b00
0000ec00
00000000
00000000
00000000
00000000
00000000
00000000
00000000
00000000
00000000
00000000
"""
    memfile_path = obj.make_mem(
        macaddr="00-1B-1A-EE-00-EC", ipaddr="10.5.0.20", netmask="255.0.0.0", default_gateway="10.0.0.1"
    )
    with open(memfile_path) as f:
        generated = f.read()
    assert generated == answer

    bitfiles = obj.get_bits()
    assert "quel1seproto11_20230804" in bitfiles

    e_path = obj.make_embedded_bit(bitpath=bitfiles["quel1seproto11_20230804"], mempath=memfile_path)
    logger.info(f"the generated bit file embedding configuration info: {e_path}")
    with open(e_path, "rb") as f:
        c = f.read()
        # Notes: 0x50:0x7f varies (depends on time?)
        h = hashlib.md5(c[0x80:]).hexdigest()
        assert h == "0ad3d8c0c8430163dd41f7ee205916a5"

    m_path = obj.make_mcs(e_path)
    logger.info(f"the generated mcs file: {m_path}")


def test_basic_au50():
    obj = Au50Programmer()
    answer = """\
@00000000
2000010a
000000ff
0100000a
0100000a
06350a00
00001e7d
2000020a
000000ff
0100000a
0100000a
06350a00
00001f7d
00000000
00000000
00000000
00000000
"""
    memfile_path = obj.make_mem(
        macaddr="00-0A-35-06-7D-1E", ipaddr="10.1.0.32", netmask="255.0.0.0", default_gateway="10.0.0.1"
    )
    with open(memfile_path) as f:
        generated = f.read()
    assert generated == answer

    bitfiles = obj.get_bits()
    assert "simplemulti_20230820" in bitfiles

    e_path = obj.make_embedded_bit(bitpath=bitfiles["simplemulti_20230820"], mempath=memfile_path)
    logger.info(f"the generated bit file embedding configuration info: {e_path}")
    with open(e_path, "rb") as f:
        c = f.read()
        # Notes: 0x50:0x7f varies (depends on time?)
        h = hashlib.md5(c[0x80:]).hexdigest()
        assert h == "02d97079f2546ed840843efd090b34bf"

    m_path = obj.make_mcs(e_path)
    logger.info(f"the generated mcs file: {m_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    """
    from ping3 import ping

    adapter_id = "210249AEC362"
    obj.program(m_path, adapter_id)
    obj.reboot(adapter_id)

    time.sleep(5)
    assert ping("10.5.0.20")
    """
