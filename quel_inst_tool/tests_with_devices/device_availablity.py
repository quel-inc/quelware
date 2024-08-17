import logging
import os
from enum import Enum
from typing import Set

logger = logging.getLogger(__name__)


class QuelInstDevice(str, Enum):
    E4405B = "E4405B"
    E4407B = "E4407B"
    MS2720T_1 = "MS2720T-1"
    MS2090A_1 = "MS2090A-1"
    SYNTHHD = "SYNTH_HD"  # tentative
    PE6108AVA_1 = "PE6108AVA-1"
    PE4104AJ_1 = "PE4104AJ-1"

    def get_visa_name(self):
        if self in {self.E4405B, self.E4407B, self.MS2720T_1, self.MS2090A_1}:
            return self.value
        else:
            raise ValueError(f"{self.value} is not a visa device")


def get_available_devices() -> Set[QuelInstDevice]:
    devs = os.getenv("QUEL_INST_AVAILABLE_DEVICES")

    if devs is None:
        r: Set[QuelInstDevice] = {v for v in QuelInstDevice}
    else:
        r = {QuelInstDevice(n) for n in devs.split(":")}

    logger.info(f"available devices = {', '.join([str(x) for x in r])}")
    return r
