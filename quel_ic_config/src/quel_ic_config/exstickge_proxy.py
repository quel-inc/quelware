import logging
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import Final, Mapping, Union

logger = logging.getLogger(__name__)


class LsiKindId(IntEnum):
    AD9082 = 1
    ADRF6780 = 2
    LMX2594 = 4
    AD5328 = 6
    GPIO = 7
    # Notes: the followings are only for QuEL-1 SE (SockClient doesn't support them)
    AD7490 = 16
    INA239 = 17
    MIXERBOARD_GPIO = 18
    PATHSELECTORBOARD_GPIO = 19
    POWERBOARD_PWM = 20

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class _ExstickgeProxyBase(metaclass=ABCMeta):
    _ADDR_MASKS: Final[Mapping[LsiKindId, int]] = {
        LsiKindId.AD9082: 0x7FFF,
        LsiKindId.ADRF6780: 0x003F,
        LsiKindId.LMX2594: 0x007F,
        LsiKindId.AD5328: 0x000F,
        LsiKindId.AD7490: 0x0000,
        LsiKindId.GPIO: 0x0000,
        LsiKindId.MIXERBOARD_GPIO: 0x0003,
        LsiKindId.PATHSELECTORBOARD_GPIO: 0x0000,
        LsiKindId.POWERBOARD_PWM: 0x007F,
    }

    def __init__(
        self,
        target_address: str,
        target_port: int,
        timeout: float,
    ):
        self._target = (target_address, target_port)
        self._timeout = timeout
        self._dump_enable = False

    def __del__(self):
        self.terminate()

    def dump_enable(self):
        self._dump_enable = True

    def dump_disable(self):
        self._dump_enable = False

    @abstractmethod
    def read_reg(self, kind, idx, addr) -> Union[int, None]:
        pass

    @abstractmethod
    def write_reg(self, kind, idx, addr, value) -> bool:
        pass

    @property
    def has_lock(self) -> bool:
        return False

    @abstractmethod
    def terminate(self):
        pass
