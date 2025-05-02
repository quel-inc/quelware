import logging
import time
from typing import Final

import pytest
from quel_pyxsdb import XsctClient

from quel_cmod_scripting import Quel1SeProtoCmod

logger = logging.getLogger()

CMOD_XSDB_PORT: Final[int] = 36335
CMOD_HWSVR_PORT: Final[int] = 6121
CMOD_ADAPTER_ID: Final[str] = "210328B7923BA"


@pytest.fixture(scope="session")
def cmod():
    global CMOD_ADAPTER_ID

    clt = XsctClient(xsdb_port=CMOD_XSDB_PORT, hwsvr_port=CMOD_HWSVR_PORT)
    clt.connect()
    port = clt.get_jtagterminal_by_adapter_id(CMOD_ADAPTER_ID)
    logger.info(f"jtag terminal port is {port}")

    cmod = Quel1SeProtoCmod("localhost", port)
    cmod.init()
    logger.info("cmod is initialized")
    return cmod


def test_read(cmod):
    adda_temp = cmod.read_adda_ad7490()
    logger.info(f"adda temp: {adda_temp}")
    assert "lmx2594" in adda_temp and len(adda_temp["lmx2594"]) == 2
    for i in range(2):
        assert 20.0 <= adda_temp["lmx2594"][i] <= 80.0

    mx0_temp = cmod.read_mx0_ad7490()
    logger.info(f"mixer0 temp: {mx0_temp}")
    assert "lmx2594" in mx0_temp and len(mx0_temp["lmx2594"]) == 4
    assert "adrf6780" in mx0_temp and len(mx0_temp["adrf6780"]) == 4
    for i in range(4):
        assert 20.0 <= mx0_temp["lmx2594"][i] < 80.0
        assert 20.0 <= mx0_temp["adrf6780"][i] < 80.0

    mx0_cur = cmod.read_mx0_current()
    logger.info(f"mixer0 current: {mx0_cur}")
    assert "lmx2594" in mx0_cur and len(mx0_cur["lmx2594"]) == 4
    assert "6V" in mx0_cur and len(mx0_cur["6V"]) == 1
    assert "4V" in mx0_cur and len(mx0_cur["4V"]) == 1
    for i in range(4):
        assert 100.0 <= mx0_cur["lmx2594"][i] <= 400.0
    assert 0 <= mx0_cur["6V"][0] <= 1000.0
    assert sum(mx0_cur["lmx2594"]) <= mx0_cur["4V"][0] <= 4000.0

    mx1_temp = cmod.read_mx1_ad7490()
    logger.info(f"mixer1 temp: {mx1_temp}")
    assert "lmx2594" in mx1_temp and len(mx1_temp["lmx2594"]) == 4
    assert "adrf6780" in mx1_temp and len(mx1_temp["adrf6780"]) == 4
    for i in range(4):
        assert 20.0 <= mx1_temp["lmx2594"][i] < 80.0
        assert 20.0 <= mx1_temp["adrf6780"][i] < 80.0

    mx1_cur = cmod.read_mx1_current()
    logger.info(f"mixer1 current: {mx1_cur}")
    assert "lmx2594" in mx1_cur and len(mx1_cur["lmx2594"]) == 4
    assert "6V" in mx1_cur and len(mx1_cur["6V"]) == 1
    assert "4V" in mx1_cur and len(mx1_cur["4V"]) == 1
    for i in range(4):
        assert 100.0 <= mx1_cur["lmx2594"][i] <= 400.0
    assert 0 <= mx1_cur["6V"][0] <= 1000.0
    assert sum(mx1_cur["lmx2594"]) <= mx1_cur["4V"][0] <= 4000.0


def test_write(cmod):
    cmod.neutral()
    logger.info("any heating activity is stopped")
    time.sleep(2)
    cmod.set_fan(0)
    logger.info("fan speed is set to minimum value")
    cmod.set_adrf6780_mx0_pwdn(False)
    cmod.set_adrf6780_mx1_pwdn(False)
    logger.info("adrf6780_pwdn() completes without error")

    for i in range(2):
        cmod.set_b_heater(f"adda:lmx2594:{i}", 0.0)
        logger.info(
            f"onboard heater control of adda:lmx2594:{i} completes without error"
        )
    with pytest.raises(ValueError):
        cmod.set_b_heater("adda:lmx2594:3", 0.0)
    with pytest.raises(ValueError):
        cmod.set_b_heater("adda:lmx2595:0", 0.0)

    for b in range(2):
        for ic in ("lmx2594", "adrf6780"):
            for k in range(4):
                cmod.set_b_heater(f"mx{b}:{ic}:{k}", 0.0)
                logger.info(
                    f"onboard heater control of mx{b}:{ic}:{k} completes without error"
                )

    with pytest.raises(ValueError):
        cmod.set_b_heater("mx0:lmx2594:5", 0.0)

    with pytest.raises(ValueError):
        cmod.set_b_heater("mx1:lmx2593:0", 0.0)

    with pytest.raises(ValueError):
        cmod.set_b_heater("mx2:lmx2594:0", 0.0)

    for i in range(6):
        cmod.set_x_heater(f"adda:{i}", 0.0)
        logger.info(f"external heaterl control of adda:{i} completes without error")

    with pytest.raises(ValueError):
        cmod.set_x_heater("adda:6", 0.0)

    for i in range(2):
        for j in range(4):
            cmod.set_x_heater(f"mx{i}:{j}", 0.0)
            logger.info(
                f"external heaterl control of mx{i}:{j} completes without error"
            )

    with pytest.raises(ValueError):
        cmod.set_x_heater("mx0:4", 0.0)

    with pytest.raises(ValueError):
        cmod.set_x_heater("mx1:4", 0.0)

    cmod.neutral()
    logger.info("any heating activity is stopped, again.")
    time.sleep(2)
    cmod.set_fan(0)
