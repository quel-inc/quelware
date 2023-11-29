import logging
import time

from ping3 import ping

from quel_staging_tool import ExstickgeProgrammer

logger = logging.getLogger(__name__)


def test_device_exstickge():
    obj = ExstickgeProgrammer()

    memfile_path = obj.make_mem(
        macaddr="00-1B-1A-EE-00-EC", ipaddr="10.5.0.20", netmask="255.0.0.0", default_gateway="10.0.0.1"
    )
    bitfiles = obj.get_bits()
    e_path = obj.make_embedded_bit(bitpath=bitfiles["quel1seproto11_20230804"], mempath=memfile_path)
    logger.info(f"the generated bit file embedding configuration info: {e_path}")

    m_path = obj.make_mcs(e_path)
    logger.info(f"the generated mcs file: {m_path}")

    adapter_id = "210249AEC362"
    obj.program(m_path, "localhost", 3121, adapter_id)
    obj.reboot("localhost", 3121, adapter_id)

    logger.info("waiting 10 seconds to boot up...")
    time.sleep(10)
    assert ping("10.5.0.20")
