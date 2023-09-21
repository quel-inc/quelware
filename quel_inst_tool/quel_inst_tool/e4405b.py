import logging
from typing import Final, Set

from quel_inst_tool.e440xb import E440xb
from quel_inst_tool.spectrum_analyzer import InstDev

logger = logging.getLogger(__name__)


class E4405b(E440xb):
    _FREQ_MAX: Final[float] = 1.32e10
    _FREQ_MIN: Final[float] = 9e3
    _SUPPORTED_PROD_ID: Final[Set[str]] = {"E4405B"}

    __slots__ = ()

    def __init__(self, dev: InstDev):
        super().__init__(dev)
