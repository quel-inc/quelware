import logging
import os
from enum import Enum
from typing import Set

logger = logging.getLogger(__name__)


class QuelInstDevice(str, Enum):
    E4405B = "E4405B"
    E4407B = "E4407B"
    MS2720T = "MS2720T"
    SYNTHHD = "SYNTH_HD"  # tentative

    def get_visa_name(self):
        if self in {self.E4405B, self.E4407B, self.MS2720T}:
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
