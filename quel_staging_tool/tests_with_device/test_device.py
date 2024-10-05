import logging
import time
from ipaddress import IPv4Address

from ping3 import ping

from quel_ic_config import ExstickgeCoapClientQuel1seRiken8
from quel_staging_tool import Au50Programmer, ExstickgeProgrammer, QuelXilinxFpgaProgrammerZephyr

logger = logging.getLogger(__name__)

AU50_TEST_HOST = "172.30.2.203"
EXSTICKGE_TEST_HOST = "172.30.2.204"


def test_device_exstickge():
    obj = ExstickgeProgrammer()

    macaddr = "00-1B-1A-EE-00-EC"
    ipaddr = "10.5.0.20"
    adapter = "210249B87F82"
    version = "quel1seproto11_20230804"

    memfile_path = obj.make_mem(macaddr=macaddr, ipaddr=ipaddr, netmask="255.0.0.0", default_gateway="10.0.0.1")
    bitfiles = obj.get_bits()
    e_path = obj.make_embedded_bit(bitpath=bitfiles[version], mempath=memfile_path)
    logger.info(f"the generated bit file embedding configuration info: {e_path}")

    m_path = obj.make_mcs(e_path)
    logger.info(f"the generated mcs file: {m_path}")

    obj.program(m_path, EXSTICKGE_TEST_HOST, 5121, adapter)
    obj.reboot(EXSTICKGE_TEST_HOST, 5121, adapter)

    logger.info("waiting 10 seconds to boot up...")
    time.sleep(10)
    assert ping("10.5.0.20")


def test_device_exstickge_zephyr():
    obj = QuelXilinxFpgaProgrammerZephyr()

    macaddr = "00-0b-0a-ee-01-87"
    ipaddr = IPv4Address("10.5.0.83")
    adapter = "210249B87F82"
    version = "quel1se-riken8_20240206"

    bitfile = obj.get_bits(bitfile_name="quel1_config.bit")[version]
    elffile = bitfile.parent / "zephyr.elf"
    mmifile = bitfile.parent / "itcm.mmi"

    eelfpath = obj.make_embedded_elf(elfpath=elffile, ipaddr=ipaddr, patch_dict={})
    macaddrpath = obj.make_macaddr_bin(macaddr)
    ebitpath = obj.make_embedded_bit(bitpath=bitfile, mmipath=mmifile, elfpath=eelfpath)

    mcspath = obj.make_mcs_with_macaddr(bitpath=ebitpath, macaddrpath=macaddrpath)
    obj.program(mcspath, "localhost", 5121, adapter)
    obj.reboot("localhost", 5121, adapter)

    logger.info("waiting 10 seconds to boot up...")
    ipaddr_str = str(ipaddr).split("/")[0]
    time.sleep(10)
    assert ping(ipaddr_str)

    proxy = ExstickgeCoapClientQuel1seRiken8(target_addr=ipaddr_str)
    ver, chash = proxy.read_fpga_version()
    assert ver == "v0.3.0"
    assert chash == "0xd6a188fb"


def test_device_au50():
    obj = Au50Programmer()
    adapter_id = "500202A50TIAA"  # staging-074
    assert obj.dry_run(AU50_TEST_HOST, 3121, adapter_id) == 0
