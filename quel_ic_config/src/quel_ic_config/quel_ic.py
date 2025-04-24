import logging
from abc import ABCMeta
from typing import Tuple, Union

from quel_ic_config.ad5328 import Ad5328Mixin
from quel_ic_config.ad7490 import Ad7490Mixin
from quel_ic_config.ad9082 import Ad9082Mixin
from quel_ic_config.adrf6780 import Adrf6780Mixin
from quel_ic_config.exstickge_proxy import LsiKindId, _ExstickgeProxyBase
from quel_ic_config.lmx2594 import Lmx2594Mixin
from quel_ic_config.mixerboard_gpio import MixerboardGpioMixin
from quel_ic_config.pathselectorboard_gpio import PathselectorboardGpioMixin
from quel_ic_config.powerboard_pwm import PowerboardPwmMixin
from quel_ic_config.rfswitcharray import (
    QubeRfSwitchArrayMixin,
    Quel1TypeARfSwitchArrayMixin,
    Quel1TypeBRfSwitchArrayMixin,
)

logger = logging.getLogger(__name__)


class Quel1Ic(metaclass=ABCMeta):
    def __init__(self, proxy: _ExstickgeProxyBase, kind: LsiKindId, idx: int):
        self.proxy = proxy
        self.kind = kind
        self.idx = idx
        # TODO: it may be fine if ExstickgeProxyBase conducts range check of (kind, idx) here.

    def _read_reg(self, addr: int) -> int:
        """A universal method for reading registers of the corresponding IC. You should not use this method
        if an equivalent specialized method is available.
        :param addr: the address of the register to read
        :return: the reading of the register
        """
        v = self.proxy.read_reg(self.kind, self.idx, addr)
        if v is None:
            raise RuntimeError(f"failed to read register 0x{addr:02x} of {self.kind.name}[{self.idx}]")
        return v & 0xFFFF

    def _read_reg_no_except(self, addr: int) -> Union[None, int]:
        """A universal method for reading registers of the corresponding IC. You should not use this method
        if an equivalent specialized method is available.
        :param addr: the address of the register to read
        :return: the reading of the register, it can be None for failure.
        """
        v = self.proxy.read_reg(self.kind, self.idx, addr)
        if v is None:
            return None
        else:
            return v & 0xFFFF

    def _write_reg(self, addr: int, data: int) -> None:
        """A universal method for writing registers of the corresponding IC. You should not use this method
        if an equivalent specialized method is available.
        :param addr: the address of the register to write
        :param data: a value to be written into the specified register
        """
        r = self.proxy.write_reg(self.kind, self.idx, addr, data & 0xFFFF)
        if not r:
            raise RuntimeError(
                f"failed to write 0x{data:04x} into register 0x{addr:02x} of {self.kind.name}[{self.idx}]"
            )
        return

    def _write_reg_no_except(self, addr: int, data: int) -> bool:
        """A universal method for writing registers of the corresponding IC. You should not use this method
        if an equivalent specialized method is available.
        :param addr: the address of the register to write
        :param data: a value to be written into the specified register
        :return: returns true if success
        """
        return self.proxy.write_reg(self.kind, self.idx, addr, data & 0xFFFF)


class Adrf6780(Adrf6780Mixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        Adrf6780Mixin.__init__(self, f"ADRF6780[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.ADRF6780, idx)

    def read_reg(self, addr: int) -> int:
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        return Quel1Ic._write_reg(self, addr, data)


class Ad5328(Ad5328Mixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        Ad5328Mixin.__init__(self, f"AD5328[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.AD5328, idx)

    def read_reg(self, addr: int) -> int:
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        self._carbon_copy_not_updated[addr] = data
        return Quel1Ic._write_reg(self, addr, data)


class Ad7490(Ad7490Mixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        Ad7490Mixin.__init__(self, f"AD7490[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.AD7490, idx)

    def read_reg(self, addr: int) -> int:
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        return Quel1Ic._write_reg(self, addr, data)


# TODO: consider to subclass it into ForLo and ForRefClk or not.
class Lmx2594(Lmx2594Mixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        Lmx2594Mixin.__init__(self, f"LMX2594[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.LMX2594, idx)

    def read_reg(self, addr: int) -> int:
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        return Quel1Ic._write_reg(self, addr, data)


class Ad9082Generic(Ad9082Mixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        Ad9082Mixin.__init__(self, f"AD9082[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.AD9082, idx)

    def _read_reg_cb(self, addr: int) -> Tuple[bool, int]:
        value = Quel1Ic._read_reg_no_except(self, addr)
        if value is None:
            logger.debug(f"failed to read from reg[{addr:04x}] of ad9082-#{self.idx:d}")
            return False, 0
        else:
            logger.debug(f"read {value:02x} from reg[{addr:04x}] of ad9082-#{self.idx:d}")
            return True, value

    def _write_reg_cb(self, addr: int, data: int) -> Tuple[bool]:
        logger.debug(f"write {data:02x} into reg[{addr:04x}] of ad9082-#{self.idx:d}")
        return (Quel1Ic._write_reg_no_except(self, addr, data),)


class QubeRfSwitchArray(QubeRfSwitchArrayMixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        QubeRfSwitchArrayMixin.__init__(self, f"QubeRfSwitchArray[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.GPIO, idx)

    def read_reg(self, addr: int) -> int:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        Quel1Ic._write_reg(self, addr, data)


class Quel1TypeARfSwitchArray(Quel1TypeARfSwitchArrayMixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        Quel1TypeARfSwitchArrayMixin.__init__(self, f"Quel1TypeARfSwitchArray[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.GPIO, idx)

    def read_reg(self, addr: int) -> int:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        Quel1Ic._write_reg(self, addr, data)


class Quel1TypeBRfSwitchArray(Quel1TypeBRfSwitchArrayMixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        Quel1TypeBRfSwitchArrayMixin.__init__(self, f"Quel1TypeBRfSwitchArray[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.GPIO, idx)

    def read_reg(self, addr: int) -> int:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        if addr != 0:
            raise ValueError(f"invalid address of {self.name}: {addr}")
        Quel1Ic._write_reg(self, addr, data)


class MixerboardGpio(MixerboardGpioMixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        MixerboardGpioMixin.__init__(self, f"MixerboardGpio[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.MIXERBOARD_GPIO, idx)

    def read_reg(self, addr: int) -> int:
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        return Quel1Ic._write_reg(self, addr, data)


class PathselectorboardGpio(PathselectorboardGpioMixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        PathselectorboardGpioMixin.__init__(self, f"PathselectorboardGpio[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.PATHSELECTORBOARD_GPIO, idx)

    def read_reg(self, addr: int) -> int:
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        return Quel1Ic._write_reg(self, addr, data)


class PowerboardPwm(PowerboardPwmMixin, Quel1Ic):
    def __init__(self, proxy: _ExstickgeProxyBase, idx: int):
        PowerboardPwmMixin.__init__(self, f"PowerboardPwm[{idx}]")
        Quel1Ic.__init__(self, proxy, LsiKindId.POWERBOARD_PWM, idx)

    def read_reg(self, addr: int) -> int:
        return Quel1Ic._read_reg(self, addr)

    def write_reg(self, addr: int, data: int) -> None:
        return Quel1Ic._write_reg(self, addr, data)
