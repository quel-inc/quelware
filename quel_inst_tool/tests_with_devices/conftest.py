import logging
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Dict, Final, List, Tuple, Type

import matplotlib as mpl
import pytest
from device_availablity import QuelInstDevice, get_available_devices

from quel_inst_tool import E4405b, E4407b, InstDevManager, Ms2xxxx, Ms2090a, Ms2720t, SpectrumAnalyzer, SynthHDMaster

logger = logging.getLogger(__name__)

DEVICE_CLASS_MAPPING_HACHIOJI: Final[Dict[QuelInstDevice, Type[Ms2xxxx]]] = {
    QuelInstDevice.MS2720T_1: Ms2720t,
    QuelInstDevice.MS2090A_1: Ms2090a,
}


DEVICES: Final[List[str]] = [
    QuelInstDevice.E4405B,
    QuelInstDevice.E4407B,
    QuelInstDevice.MS2720T_1,
    QuelInstDevice.MS2090A_1,
]

OUTDIR: Final[Path] = Path("./artifacts/device/")


MAX_RETRY_VISA_LOOKUP: Final = 5


@pytest.fixture(scope="module", params=DEVICES)
def spectrum_analyzer(request) -> Tuple[QuelInstDevice, SpectrumAnalyzer]:
    available = get_available_devices()
    spa_name = request.param

    logger.info(f"try to acquire {spa_name}")
    if spa_name not in available:
        pytest.skip(f"{spa_name} is not available")

    if spa_name is not None:
        if spa_name == QuelInstDevice.MS2090A_1 or spa_name == QuelInstDevice.MS2720T_1:
            # SG must be on blacklist. otherwise SG parameters goes wrong.
            bl = [f"ASRL/dev/ttyACM{k}::INSTR" for k in range(16)]
            im = InstDevManager(ivi="@py", blacklist=bl)
            if spa_name in DEVICE_CLASS_MAPPING_HACHIOJI:
                dev = im.get_inst_device(
                    DEVICE_CLASS_MAPPING_HACHIOJI[spa_name].get_visa_name(ipaddr=socket.gethostbyname(spa_name))
                )
                obj: SpectrumAnalyzer = DEVICE_CLASS_MAPPING_HACHIOJI[spa_name](dev)
            else:
                raise ValueError(f"Invalid device name: {spa_name}")
        elif spa_name == QuelInstDevice.E4405B or spa_name == QuelInstDevice.E4407B:
            n: int = MAX_RETRY_VISA_LOOKUP
            for i in range(n):
                if i > 0:
                    time.sleep(1)
                    logger.warning(f"failed to connect to the spectrum analyzer, retrying... ({i}/{n})")
                else:
                    im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so")
                    im.scan()
                    dev = im.lookup(prod_id=spa_name)
                    if dev is not None:
                        if spa_name == QuelInstDevice.E4405B:
                            obj = E4405b(dev)
                        else:
                            obj = E4407b(dev)
                        break
            else:
                pytest.fail(f"failed to connect the spectrum analyzer {spa_name} too many times.")
                assert False  # for preventing PyCharm from generating warning, never executed.
        else:
            raise AssertionError

    return request.param, obj


@pytest.fixture(scope="module")
def signal_generator() -> SynthHDMaster:
    available = get_available_devices()

    logger.info("try to acquire SYNTH_HD")
    if "SYNTH_HD" not in available:
        pytest.skip("SYNTH_HD is not available")

    SynthHD = SynthHDMaster()
    return SynthHD


@pytest.fixture(scope="session")
def outdir() -> Path:
    mpl.use("Qt5Agg")
    if os.path.exists(OUTDIR):
        shutil.rmtree(OUTDIR)
        os.makedirs(OUTDIR)
    return Path(OUTDIR)
