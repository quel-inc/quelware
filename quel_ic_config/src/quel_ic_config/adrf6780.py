import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Final, Set, Tuple, Union, cast

from quel_ic_config.abstract_ic import (
    AbstractIcConfigHelper,
    AbstractIcMixin,
    AbstractIcReg,
    b_1bf_bool,
    b_nbf,
    p_1bf_bool,
    p_nbf,
)

logger = logging.getLogger(__name__)


@dataclass
class Adrf6780Control(AbstractIcReg):
    parity_en: bool = field(default=False)  # [15]
    soft_reset: bool = field(default=False)  # [14]
    chip_id: int = field(default=0)  # [11:4]
    chip_rev: int = field(default=0)  # [3:0]

    def parse(self, v: int) -> None:
        self.parity_en = p_1bf_bool(v, 15)
        self.soft_reset = p_1bf_bool(v, 14)
        self.chip_id = p_nbf(v, 11, 4)
        self.chip_rev = p_nbf(v, 3, 0)

    def build(self) -> int:
        return (
            b_1bf_bool(self.parity_en, 15)
            | b_1bf_bool(self.soft_reset, 14)
            | b_nbf(self.chip_id, 11, 4)
            | b_nbf(self.chip_rev, 3, 0)
        )


@dataclass
class Adrf6780Alarm(AbstractIcReg):
    parity_error: bool = field(default=False)  # [15]
    too_few_errors: bool = field(default=False)  # [14]
    too_many_errors: bool = field(default=False)  # [13], 7@reset
    address_range_error: bool = field(default=False)  # [12], 6@reset, but may be incremented in the future.

    def parse(self, v: int) -> None:
        self.parity_error = p_1bf_bool(v, 15)
        self.too_few_errors = p_1bf_bool(v, 14)
        self.too_many_errors = p_1bf_bool(v, 13)
        self.address_range_error = p_1bf_bool(v, 12)

    def build(self) -> int:
        return (
            b_1bf_bool(self.parity_error, 15)
            | b_1bf_bool(self.too_few_errors, 14)
            | b_1bf_bool(self.too_many_errors, 13)
            | b_1bf_bool(self.address_range_error, 12)
        )


class Adrf6780AlarmReadBack(Adrf6780Alarm):
    pass


class Adrf6780AlarmMask(Adrf6780Alarm):
    pass


@dataclass
class Adrf6780Enable(AbstractIcReg):
    vga_buffer_enable: bool = field(default=False)  # [8]
    detector_enable: bool = field(default=False)  # [7]
    lo_buffer_enable: bool = field(default=False)  # [6]
    if_mode_enable: bool = field(default=False)  # [5]
    iq_mode_enable: bool = field(default=False)  # [4]
    lo_x2_enable: bool = field(default=False)  # [3]
    lo_ppf_enable: bool = field(default=False)  # [2]
    lo_enable: bool = field(default=False)  # [1]
    uc_bias_enable: bool = field(default=False)  # [0]

    def parse(self, v: int) -> None:
        self.vga_buffer_enable = p_1bf_bool(v, 8)
        self.detector_enable = p_1bf_bool(v, 7)
        self.lo_buffer_enable = p_1bf_bool(v, 6)
        self.if_mode_enable = p_1bf_bool(v, 5)
        self.iq_mode_enable = p_1bf_bool(v, 4)
        self.lo_x2_enable = p_1bf_bool(v, 3)
        self.lo_ppf_enable = p_1bf_bool(v, 2)
        self.lo_enable = p_1bf_bool(v, 1)
        self.uc_bias_enable = p_1bf_bool(v, 0)

    def build(self) -> int:
        return (
            b_1bf_bool(self.vga_buffer_enable, 8)
            | b_1bf_bool(self.detector_enable, 7)
            | b_1bf_bool(self.lo_buffer_enable, 6)
            | b_1bf_bool(self.if_mode_enable, 5)
            | b_1bf_bool(self.iq_mode_enable, 4)
            | b_1bf_bool(self.lo_x2_enable, 3)
            | b_1bf_bool(self.lo_ppf_enable, 2)
            | b_1bf_bool(self.lo_enable, 1)
            | b_1bf_bool(self.uc_bias_enable, 0)
        )


@dataclass
class Adrf6780Linearize(AbstractIcReg):
    rdac_linearize: int = field(default=0)  # [7:0]

    def parse(self, v: int) -> None:
        self.rdac_linearize = p_nbf(v, 7, 0)

    def build(self) -> int:
        return b_nbf(self.rdac_linearize, 7, 0)


class Adrf6780LoSideband(IntEnum):
    Usb = 0
    Lsb = 1


@dataclass
class Adrf6780LoPath(AbstractIcReg):
    lo_sideband: Adrf6780LoSideband = field(default=Adrf6780LoSideband.Usb)  # [10]
    q_path_phase_accuracy: int = field(default=0)  # [7:4]
    i_path_phase_accuracy: int = field(default=0)  # [3:0]

    def parse(self, v: int) -> None:
        self.lo_sideband = Adrf6780LoSideband.Lsb if p_1bf_bool(v, 10) else Adrf6780LoSideband.Usb
        self.q_path_phase_accuracy = p_nbf(v, 7, 4)
        self.i_path_phase_accuracy = p_nbf(v, 3, 0)

    def build(self) -> int:
        return (
            b_nbf(int(self.lo_sideband), 10, 10)
            | b_nbf(self.q_path_phase_accuracy, 7, 4)
            | b_nbf(self.i_path_phase_accuracy, 3, 0)
        )


@dataclass
class Adrf6780AdcControl(AbstractIcReg):
    vdet_output_select: bool = field(default=False)  # [3]
    adc_start: bool = field(default=False)  # [2]
    adc_enable: bool = field(default=False)  # [1]
    adc_clock_enable: bool = field(default=False)  # [0]

    def parse(self, v: int) -> None:
        self.vdet_output_select = p_1bf_bool(v, 3)
        self.adc_start = p_1bf_bool(v, 2)
        self.adc_enable = p_1bf_bool(v, 1)
        self.adc_clock_enable = p_1bf_bool(v, 0)

    def build(self) -> int:
        return (
            b_1bf_bool(self.vdet_output_select, 3)
            | b_1bf_bool(self.adc_start, 2)
            | b_1bf_bool(self.adc_enable, 1)
            | b_1bf_bool(self.adc_clock_enable, 0)
        )


@dataclass
class Adrf6780AdcOutput(AbstractIcReg):
    adc_status: bool = field(default=False)  # [8]
    adc_value: int = field(default=0)  # [7:0]

    def parse(self, v: int) -> None:
        self.adc_status = p_1bf_bool(v, 8)
        self.adc_value = p_nbf(v, 7, 0)

    def build(self) -> int:
        return b_1bf_bool(self.adc_status, 8) | b_nbf(self.adc_value, 7, 0)


Adrf6780Regs: Dict[int, type] = {
    0: Adrf6780Control,
    1: Adrf6780AlarmReadBack,
    2: Adrf6780AlarmMask,
    3: Adrf6780Enable,
    4: Adrf6780Linearize,
    5: Adrf6780LoPath,
    6: Adrf6780AdcControl,
    12: Adrf6780AdcOutput,
}


Adrf6780RegNames: Dict[str, int] = {
    "Control": 0,
    "AlarmReadBack": 1,
    "AlarmMask": 2,
    "Enable": 3,
    "Linearize": 4,
    "LoPath": 5,
    "AdcControl": 6,
    "AdcOutput": 12,
}


class Adrf6780Mixin(AbstractIcMixin):
    Regs = Adrf6780Regs
    RegNames = Adrf6780RegNames
    _CHIP_ID: Final[int] = 7
    _DEFAULT_EXPECTED_REVISION: Final[Set[int]] = {6}
    DEFAULT_SIDEBAND_SWITCH_WAIT = 0.001

    def __init__(self, name):
        super().__init__(name)

    def dump_regs(self) -> Dict[int, int]:
        """dumping all the available registers.

        :return: a mapping between an address and a value of the registers
        """
        regs = {}
        for addr in (0, 1, 2, 3, 4, 5, 6, 12):
            regs[addr] = self.read_reg(addr)
        return regs

    def soft_reset(self, parity_enable: bool = False) -> None:
        """committing soft reset.

        :return: None
        """
        addr, reg = cast(Tuple[int, Adrf6780Control], self._read_and_parse_reg("Control"))
        reg.parity_en = parity_enable
        reg.soft_reset = True
        self._build_and_write_reg(addr, reg)
        reg.soft_reset = False
        self._build_and_write_reg(addr, reg)

    def check_id(self, expected_revision: Union[Set[int], None] = None) -> Tuple[bool, Tuple[int, int]]:
        """confirming the chip version and chip revision, or check the soundness of the register access.

        :param expected_revision: a set of valid revisions.
        :return: an actual chip revision
        """
        if expected_revision is None:
            expected_revision = self._DEFAULT_EXPECTED_REVISION
        addr, reg = cast(Tuple[int, Adrf6780Control], self._read_and_parse_reg("Control"))
        if reg.chip_id == self._CHIP_ID and reg.chip_rev in expected_revision:
            return True, (reg.chip_id, reg.chip_rev)
        else:
            return False, (reg.chip_id, reg.chip_rev)

    def read_alarm(self) -> Dict[str, bool]:
        # don't use helper to confirm type safety.
        _, reg = cast(Tuple[int, Adrf6780AlarmReadBack], self._read_and_parse_reg("AlarmReadBack"))
        return {
            "parity": reg.parity_error,
            "too_few": reg.too_few_errors,
            "too_many": reg.too_many_errors,
            "address_range": reg.address_range_error,
        }

    def get_lo_sideband(self) -> Adrf6780LoSideband:
        """setting LO Sideband to either of LSB or USB.

        :return: None
        """
        # don't use helper to confirm type safety.
        _, reg = self._read_and_parse_reg("LoPath")
        return cast(Adrf6780LoPath, reg).lo_sideband

    def set_lo_sideband(self, sideband: Adrf6780LoSideband, wait=DEFAULT_SIDEBAND_SWITCH_WAIT) -> None:
        """setting LO Sideband to either of LSB or USB.

        :param sideband: either of LoSideband.Lsb or LoSideband.Usb.
        :return: None
        """
        # don't use helper to confirm type safety.
        if sideband != Adrf6780LoSideband.Lsb and sideband != Adrf6780LoSideband.Usb:
            raise ValueError(f"invalid sideband specifier {sideband:d} is given")

        addr, reg = self._read_and_parse_reg("LoPath")
        cast(Adrf6780LoPath, reg).lo_sideband = sideband
        self.write_reg(addr, reg.build())
        time.sleep(wait)

    def enable_detector_adc(self):
        # don't use helper to confirm type safety.
        addr_enable, reg_enable = self._read_and_parse_reg("Enable")
        cast(Adrf6780Enable, reg_enable).detector_enable = True
        self.write_reg(addr_enable, reg_enable.build())

        addr_adc_control, reg_adc_control = cast(Tuple[int, Adrf6780AdcControl], self._read_and_parse_reg("AdcControl"))
        reg_adc_control.adc_clock_enable = True
        self._build_and_write_reg(addr_adc_control, reg_adc_control)
        reg_adc_control.adc_enable = True
        self._build_and_write_reg(addr_adc_control, reg_adc_control)

    def disable_detector_adc(self):
        # don't use helper to confirm type safety.
        addr_adc_control, reg_adc_control = cast(Tuple[int, Adrf6780AdcControl], self._read_and_parse_reg("AdcControl"))
        reg_adc_control.adc_clock_enable = False
        reg_adc_control.adc_enable = False
        reg_adc_control.adc_start = False
        self._build_and_write_reg(addr_adc_control, reg_adc_control)

        addr_enable, reg_enable = cast(Tuple[int, Adrf6780Enable], self._read_and_parse_reg("Enable"))
        reg_enable.detector_enable = False
        self._build_and_write_reg(addr_enable, reg_enable)

    def read_detector_adc(self) -> int:
        """reading the value of the internal ADC. This method assumes the ADC is already enabled with the steps 1 -- 3
        described at the p.23 of the revision D datasheet.

        :return: the reading of the ADC (8bit unsigned).
        """
        # don't use helper to confirm type safety.
        addr_adc_control, reg_adc_control = cast(Tuple[int, Adrf6780AdcControl], self._read_and_parse_reg("AdcControl"))
        addr_adc_output, reg_adc_output = cast(Tuple[int, Adrf6780AdcOutput], self._read_and_parse_reg("AdcOutput"))

        reg_adc_control.adc_clock_enable = True
        reg_adc_control.adc_enable = True
        reg_adc_control.adc_start = True
        self._build_and_write_reg(addr_adc_control, reg_adc_control)
        time.sleep(200e-6)
        for i in range(10):
            reg_adc_output.parse(self.read_reg(addr_adc_output))
            if reg_adc_output.adc_status:
                break
            time.sleep(20e-6)
        else:
            raise RuntimeError(f"Timeout for ADC_STATUS of {self.name}")

        reg_adc_control.adc_start = False
        self._build_and_write_reg(addr_adc_control, reg_adc_control)
        reg_adc_output.parse(self.read_reg(addr_adc_output))
        return reg_adc_output.adc_value


class Adrf6780ConfigHelper(AbstractIcConfigHelper):
    """Helper class for programming ADRF6780 with convenient notations. It also provides caching capability that
    keep modifications on the registers to write them at once with flash_updated() in a right order.
    """

    def __init__(self, ic: Adrf6780Mixin):
        super().__init__(ic)

    def flush(self, discard_after_flush=True):
        """updating the modified registers in the order of addresses 0, 2, 3, 4, 5, 6. The modifications on addresses 1
        and 12 are ignored because they are read-only registers."""
        for addr in (0, 2, 3, 4, 5, 6):
            if addr in self.updated:
                self.ic.write_reg(addr, self.updated[addr])
        if discard_after_flush:
            self.discard()
