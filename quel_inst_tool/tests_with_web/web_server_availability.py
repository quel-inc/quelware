import logging
import os
from enum import Enum
from typing import Set

logger = logging.getLogger(__name__)


class QuelInstWebServer(str, Enum):
    E4405B = "E4405B"
    E4407B = "E4407B"

    def get_visa_name(self):
        if self in {self.E4405B, self.E4407B}:
            return self.value
        else:
            raise ValueError(f"{self.value} is not a visa device")


def get_available_devices() -> Set[QuelInstWebServer]:
    devs = os.getenv("QUEL_INST_AVAILABLE_WEBSERVERS")
    if devs is None:
        r: Set[QuelInstWebServer] = {v for v in QuelInstWebServer}
    else:
        rlist = []
        for n in devs.split(":"):
            if n in (QuelInstWebServer.E4405B, QuelInstWebServer.E4407B):
                rlist.append(QuelInstWebServer(n))
        r = {n for n in rlist}

    logger.info(f"available web servers = {', '.join([str(x) for x in r])}")
    return r
