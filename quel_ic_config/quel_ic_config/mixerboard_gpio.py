import logging
from typing import Dict

from quel_ic_config.abstract_ic import AbstractIcConfigHelper, AbstractIcMixin, Gpio4, Gpio16

logger = logging.getLogger(__name__)


MixerboardGpioRegs: Dict[int, type] = {
    0: Gpio16,
    1: Gpio16,
    2: Gpio4,
    3: Gpio4,
}


MixerboardGpioRegNames: Dict[str, int] = {
    "GPO_Enable": 0,
    "GPIO": 1,
    "ADRF6780_RSTn": 2,
    "ADRF6780_PWDN": 3,
}


class MixerboardGpioMixin(AbstractIcMixin):
    Regs = MixerboardGpioRegs
    RegNames = MixerboardGpioRegNames

    def __init__(self, name):
        super().__init__(name)

    def dump_regs(self) -> Dict[int, int]:
        """dump all the available registers.
        :return: a mapping between an address and a value of the registers
        """
        regs = {}
        for addr in self.Regs:
            regs[addr] = self.read_reg(addr)
        return regs


class MixerboardGpioConfigHelper(AbstractIcConfigHelper):
    """Helper class for programming GPIO with convenient notations. It also provides caching capability that
    keep modifications on the registers to write them at once with flash_updated() in a right order.
    """

    def __init__(self, ic: MixerboardGpioMixin):
        super().__init__(ic)

    def flush(self, discard_after_flush=True):
        for addr in self.ic.Regs:
            if addr in self.updated:
                self.ic.write_reg(addr, self.updated[addr])
        if discard_after_flush:
            self.discard()
