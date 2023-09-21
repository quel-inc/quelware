import logging
import os
import shutil
import time
from pathlib import Path
from typing import Final, List, Tuple

import matplotlib as mpl
import pytest
from device_availablity import QuelInstDevice, get_available_devices

from quel_inst_tool import E4405b, E4407b, InstDevManager, Ms2720t, SpectrumAnalyzer, SynthHDMaster

logger = logging.getLogger(__name__)
DEVICES: Final[List[str]] = [QuelInstDevice.E4405B, QuelInstDevice.E4407B, QuelInstDevice.MS2720T]
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
        n: int = MAX_RETRY_VISA_LOOKUP
        for i in range(n):
            if i > 0:
                time.sleep(1)
                logger.warning(f"failed to connect to the spectrum analyzer, retrying... ({i}/{n})")
            if spa_name == QuelInstDevice.MS2720T:
                # SG must be on blacklist. otherwise SG parameters goes wrong.
                im = InstDevManager(ivi="@py", blacklist=[f"ASRL/dev/ttyACM{k}::INSTR" for k in range(16)])
            else:
                im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so")
            dev = im.lookup(prod_id=spa_name)
            if dev is not None:
                break
        else:
            pytest.fail(f"failed to connect the spectrum analyzer {spa_name} too many times.")
            assert False  # for preventing PyCharm from generating warning, never executed.

        if spa_name == QuelInstDevice.E4405B:
            obj: SpectrumAnalyzer = E4405b(dev)
        elif spa_name == QuelInstDevice.E4407B:
            obj = E4407b(dev)
        elif spa_name == QuelInstDevice.MS2720T:
            obj = Ms2720t(dev)
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
