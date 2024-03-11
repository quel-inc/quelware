import logging
from dataclasses import dataclass, field
from typing import Dict, Final, Union

from quel_ic_config.abstract_ic import (
    AbstractIcConfigHelper,
    AbstractIcMixin,
    AbstractIcReg,
    Gpio2,
    Gpio4,
    Gpio8,
    Gpio16,
    Uint16,
    b_1bf_bool,
    b_nbf,
    p_1bf_bool,
    p_nbf,
)

logger = logging.getLogger(__name__)


@dataclass
class PowerboardPwmFanMaster(AbstractIcReg):
    en: bool = field(default=True)

    def parse(self, v: int) -> None:
        self.en = p_1bf_bool(v, 0)

    def build(self) -> int:
        return b_1bf_bool(self.en, 0)


@dataclass
class PowerboardPwmHeaterMaster(AbstractIcReg):
    en: bool = field(default=False)

    def parse(self, v: int) -> None:
        self.en = p_1bf_bool(v, 0)

    def build(self) -> int:
        return b_1bf_bool(self.en, 0)


@dataclass
class PowerboardPwmFanHighCount(AbstractIcReg):
    h_count: int = field(default=499)

    def parse(self, v: int) -> None:
        self.h_count = p_nbf(v, 9, 0)

    def build(self) -> int:
        return b_nbf(self.h_count, 9, 0)


@dataclass
class PowerboardPwmHeaterHighCount(AbstractIcReg):
    h_count: int = field(default=0)

    def parse(self, v: int) -> None:
        self.h_count = p_nbf(v, 9, 0)

    def build(self) -> int:
        return b_nbf(self.h_count, 9, 0)


@dataclass
class PowerboardPwmHeaterDelayCount(AbstractIcReg):
    d_count: int = field(default=0)

    def parse(self, v: int) -> None:
        self.d_count = p_nbf(v, 9, 0)

    def build(self) -> int:
        return b_nbf(self.d_count, 9, 0)


PowerboardPwmRegs: Dict[int, type] = {
    # Fan
    0x00: PowerboardPwmFanMaster,
    0x01: Gpio2,
    0x04: PowerboardPwmFanHighCount,
    0x05: PowerboardPwmFanHighCount,
    # Heater
    0x08: PowerboardPwmHeaterMaster,
    0x09: Gpio16,
    0x0A: Gpio16,
    0x0B: Gpio8,
    0x0C: Uint16,
    0x0D: Uint16,
    0x0E: PowerboardPwmHeaterHighCount,
    0x0F: PowerboardPwmHeaterHighCount,
    0x10: PowerboardPwmHeaterHighCount,
    0x11: PowerboardPwmHeaterHighCount,
    0x12: PowerboardPwmHeaterHighCount,
    0x13: PowerboardPwmHeaterHighCount,
    0x14: PowerboardPwmHeaterHighCount,
    0x15: PowerboardPwmHeaterHighCount,
    0x16: PowerboardPwmHeaterHighCount,
    0x17: PowerboardPwmHeaterHighCount,
    0x18: PowerboardPwmHeaterHighCount,
    0x19: PowerboardPwmHeaterHighCount,
    0x1A: PowerboardPwmHeaterHighCount,
    0x1B: PowerboardPwmHeaterHighCount,
    0x1C: PowerboardPwmHeaterHighCount,
    0x1D: PowerboardPwmHeaterHighCount,
    0x1E: PowerboardPwmHeaterHighCount,
    0x1F: PowerboardPwmHeaterHighCount,
    0x20: PowerboardPwmHeaterHighCount,
    0x21: PowerboardPwmHeaterHighCount,
    0x22: PowerboardPwmHeaterHighCount,
    0x23: PowerboardPwmHeaterHighCount,
    0x24: PowerboardPwmHeaterHighCount,
    0x25: PowerboardPwmHeaterHighCount,
    0x26: PowerboardPwmHeaterHighCount,
    0x27: PowerboardPwmHeaterHighCount,
    0x28: PowerboardPwmHeaterHighCount,
    0x29: PowerboardPwmHeaterHighCount,
    0x2A: PowerboardPwmHeaterHighCount,
    0x2B: PowerboardPwmHeaterHighCount,
    0x2C: PowerboardPwmHeaterHighCount,
    0x2D: PowerboardPwmHeaterHighCount,
    0x2E: PowerboardPwmHeaterHighCount,
    0x2F: PowerboardPwmHeaterHighCount,
    0x30: PowerboardPwmHeaterHighCount,
    0x31: PowerboardPwmHeaterHighCount,
    0x32: PowerboardPwmHeaterHighCount,
    0x33: PowerboardPwmHeaterHighCount,
    0x34: PowerboardPwmHeaterHighCount,
    0x35: PowerboardPwmHeaterHighCount,
    0x36: PowerboardPwmHeaterDelayCount,
    0x37: PowerboardPwmHeaterDelayCount,
    0x38: PowerboardPwmHeaterDelayCount,
    0x39: PowerboardPwmHeaterDelayCount,
    0x3A: PowerboardPwmHeaterDelayCount,
    0x3B: PowerboardPwmHeaterDelayCount,
    0x3C: PowerboardPwmHeaterDelayCount,
    0x3D: PowerboardPwmHeaterDelayCount,
    0x3E: PowerboardPwmHeaterDelayCount,
    0x3F: PowerboardPwmHeaterDelayCount,
    0x40: PowerboardPwmHeaterDelayCount,
    0x41: PowerboardPwmHeaterDelayCount,
    0x42: PowerboardPwmHeaterDelayCount,
    0x43: PowerboardPwmHeaterDelayCount,
    0x44: PowerboardPwmHeaterDelayCount,
    0x45: PowerboardPwmHeaterDelayCount,
    0x46: PowerboardPwmHeaterDelayCount,
    0x47: PowerboardPwmHeaterDelayCount,
    0x48: PowerboardPwmHeaterDelayCount,
    0x49: PowerboardPwmHeaterDelayCount,
    0x4A: PowerboardPwmHeaterDelayCount,
    0x4B: PowerboardPwmHeaterDelayCount,
    0x4C: PowerboardPwmHeaterDelayCount,
    0x4D: PowerboardPwmHeaterDelayCount,
    0x4E: PowerboardPwmHeaterDelayCount,
    0x4F: PowerboardPwmHeaterDelayCount,
    0x50: PowerboardPwmHeaterDelayCount,
    0x51: PowerboardPwmHeaterDelayCount,
    0x52: PowerboardPwmHeaterDelayCount,
    0x53: PowerboardPwmHeaterDelayCount,
    0x54: PowerboardPwmHeaterDelayCount,
    0x55: PowerboardPwmHeaterDelayCount,
    0x56: PowerboardPwmHeaterDelayCount,
    0x57: PowerboardPwmHeaterDelayCount,
    0x58: PowerboardPwmHeaterDelayCount,
    0x59: PowerboardPwmHeaterDelayCount,
    0x5A: PowerboardPwmHeaterDelayCount,
    0x5B: PowerboardPwmHeaterDelayCount,
    0x5C: PowerboardPwmHeaterDelayCount,
    0x5D: PowerboardPwmHeaterDelayCount,
    # External Switch
    0x5E: Gpio4,
}


PowerboardPwmRegsName: Dict[str, int] = {
    # Fan
    "FanMaster": 0x00,
    "FanEnable": 0x01,
    "FanHighCount_0": 0x04,
    "FanHighCount_1": 0x05,
    # Heater,
    "HeaterMaster": 0x08,
    "HeaterEnable_L16": 0x09,
    "HeaterEnable_M16": 0x0A,
    "HeaterEnable_H8": 0x0B,
    "HeaterCycleCount_L": 0x0C,
    "HeaterCycleCount_H": 0x0D,
    "HeaterHighCount_00": 0x0E,
    "HeaterHighCount_01": 0x0F,
    "HeaterHighCount_02": 0x10,
    "HeaterHighCount_03": 0x11,
    "HeaterHighCount_04": 0x12,
    "HeaterHighCount_05": 0x13,
    "HeaterHighCount_06": 0x14,
    "HeaterHighCount_07": 0x15,
    "HeaterHighCount_08": 0x16,
    "HeaterHighCount_09": 0x17,
    "HeaterHighCount_10": 0x18,
    "HeaterHighCount_11": 0x19,
    "HeaterHighCount_12": 0x1A,
    "HeaterHighCount_13": 0x1B,
    "HeaterHighCount_14": 0x1C,
    "HeaterHighCount_15": 0x1D,
    "HeaterHighCount_16": 0x1E,
    "HeaterHighCount_17": 0x1F,
    "HeaterHighCount_18": 0x20,
    "HeaterHighCount_19": 0x21,
    "HeaterHighCount_20": 0x22,
    "HeaterHighCount_21": 0x23,
    "HeaterHighCount_22": 0x24,
    "HeaterHighCount_23": 0x25,
    "HeaterHighCount_24": 0x26,
    "HeaterHighCount_25": 0x27,
    "HeaterHighCount_26": 0x28,
    "HeaterHighCount_27": 0x29,
    "HeaterHighCount_28": 0x2A,
    "HeaterHighCount_29": 0x2B,
    "HeaterHighCount_30": 0x2C,
    "HeaterHighCount_31": 0x2D,
    "HeaterHighCount_32": 0x2E,
    "HeaterHighCount_33": 0x2F,
    "HeaterHighCount_34": 0x30,
    "HeaterHighCount_35": 0x31,
    "HeaterHighCount_36": 0x32,
    "HeaterHighCount_37": 0x33,
    "HeaterHighCount_38": 0x34,
    "HeaterHighCount_39": 0x35,
    "HeaterDelayCount_00": 0x36,
    "HeaterDelayCount_01": 0x37,
    "HeaterDelayCount_02": 0x38,
    "HeaterDelayCount_03": 0x39,
    "HeaterDelayCount_04": 0x3A,
    "HeaterDelayCount_05": 0x3B,
    "HeaterDelayCount_06": 0x3C,
    "HeaterDelayCount_07": 0x3D,
    "HeaterDelayCount_08": 0x3E,
    "HeaterDelayCount_09": 0x3F,
    "HeaterDelayCount_10": 0x40,
    "HeaterDelayCount_11": 0x41,
    "HeaterDelayCount_12": 0x42,
    "HeaterDelayCount_13": 0x43,
    "HeaterDelayCount_14": 0x44,
    "HeaterDelayCount_15": 0x45,
    "HeaterDelayCount_16": 0x46,
    "HeaterDelayCount_17": 0x47,
    "HeaterDelayCount_18": 0x48,
    "HeaterDelayCount_19": 0x49,
    "HeaterDelayCount_20": 0x4A,
    "HeaterDelayCount_21": 0x4B,
    "HeaterDelayCount_22": 0x4C,
    "HeaterDelayCount_23": 0x4D,
    "HeaterDelayCount_24": 0x4E,
    "HeaterDelayCount_25": 0x4F,
    "HeaterDelayCount_26": 0x50,
    "HeaterDelayCount_27": 0x51,
    "HeaterDelayCount_28": 0x52,
    "HeaterDelayCount_29": 0x53,
    "HeaterDelayCount_30": 0x54,
    "HeaterDelayCount_31": 0x55,
    "HeaterDelayCount_32": 0x56,
    "HeaterDelayCount_33": 0x57,
    "HeaterDelayCount_34": 0x58,
    "HeaterDelayCount_35": 0x59,
    "HeaterDelayCount_36": 0x5A,
    "HeaterDelayCount_37": 0x5B,
    "HeaterDelayCount_38": 0x5C,
    "HeaterDelayCount_39": 0x5D,
    # External Switch
    "ExtSw": 0x5E,
}


class PowerboardPwmMixin(AbstractIcMixin):
    Regs = PowerboardPwmRegs
    RegNames = PowerboardPwmRegsName

    NUM_HEATER: Final[int] = 40

    def __init__(self, name: str):
        super().__init__(name)

    def dump_regs(self) -> Dict[int, int]:
        """dump all the available registers.
        :return: a mapping between an address and a value of the registers
        """
        regs = {}
        for addr in self.Regs:
            regs[addr] = self.read_reg(addr)
        return regs

    def _validate_fan_idx(self, idx: int) -> None:
        if idx not in {0, 1}:
            raise ValueError(f"illegal index of fan: {idx}")

    def get_fan_speed(self, idx: int) -> float:
        self._validate_fan_idx(idx)
        r = PowerboardPwmFanHighCount()
        r.parse(self.read_reg(self.RegNames[f"FanHighCount_{idx}"]))
        return r.h_count / 1000.0

    def set_fan_speed(self, idx: int, ratio: float) -> None:
        self._validate_fan_idx(idx)
        if not 0.0 <= ratio <= 1.0:
            clipped_ratio = max(0.0, min(ratio, 1.0))
            logger.warning(f"illegal ratio of fan[{idx}]: {ratio:.3f}, clipped to {clipped_ratio:.3f}")
            ratio = clipped_ratio
        self.write_reg(
            self.RegNames[f"FanHighCount_{idx}"], PowerboardPwmFanHighCount(h_count=round(ratio * 1000)).build()
        )

    def _validate_heater_idx(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise TypeError(f"unexpected type of value for heater index: '{idx}'")
        if not (0 <= idx < self.NUM_HEATER):
            raise ValueError(f"illegal index of heater index: {idx}")

    def get_heater_settings(self, idx: int) -> Dict[str, float]:
        self._validate_heater_idx(idx)
        hr = PowerboardPwmHeaterHighCount()
        hr.parse(self.read_reg(self.RegNames[f"HeaterHighCount_{idx:02}"]))
        dr = PowerboardPwmHeaterDelayCount()
        dr.parse(self.read_reg(self.RegNames[f"HeaterDelayCount_{idx:02}"]))
        return {"high_ratio": hr.h_count / 1000.0, "delay_ratio": dr.d_count / 1000.0}

    def set_heater_settings(
        self, idx: int, *, high_ratio: Union[float, None] = None, delay_ratio: Union[float, None] = None
    ) -> None:
        self._validate_heater_idx(idx)
        if high_ratio is not None:
            if not 0.0 <= high_ratio <= 1.0:
                clipped_high_count = max(0.0, min(high_ratio, 1.0))
                logger.warning(
                    f"illegal on-ratio of heater[{idx}]: {high_ratio:.3f}, clipped to {clipped_high_count:.3f}"
                )
                high_ratio = clipped_high_count
        if delay_ratio is not None:
            if not 0.0 <= delay_ratio <= 1.0:
                clipped_delay_count = max(0.0, min(delay_ratio, 1.0))
                logger.warning(
                    f"illegal delay-ratio of heater[{idx}]: {delay_ratio:.3f}, clipped to {clipped_delay_count:.3f}"
                )
                delay_ratio = clipped_delay_count

        if high_ratio is not None:
            hr = PowerboardPwmHeaterHighCount(h_count=round(high_ratio * 1000))
            self.write_reg(self.RegNames[f"HeaterHighCount_{idx:02}"], hr.build())
        if delay_ratio is not None:
            dr = PowerboardPwmHeaterDelayCount(d_count=round(delay_ratio * 1000))
            self.write_reg(self.RegNames[f"HeaterDelayCount_{idx:02}"], dr.build())

    def get_heater_enable(self, idx: int) -> bool:
        self._validate_heater_idx(idx)
        reg_addr = self.RegNames["HeaterEnable_L16"] + (idx // 16)
        bit_pos = idx % 16
        return (self.read_reg(reg_addr) >> bit_pos) & 0x01 == 0x01

    def set_heater_enable(self, idx: int, enable: bool) -> None:
        self._validate_heater_idx(idx)
        reg_addr = self.RegNames["HeaterEnable_L16"] + (idx // 16)
        bit_pos = idx % 16
        v = self.read_reg(reg_addr) & ~(1 << bit_pos)
        if enable:
            v |= 1 << bit_pos
        self.write_reg(reg_addr, v)

    def clear_heater_enable_all(self):
        reg_addr = self.RegNames["HeaterEnable_L16"]
        self.write_reg(reg_addr, 0)
        self.write_reg(reg_addr + 1, 0)
        self.write_reg(reg_addr + 2, 0)

    def dump_heater_enable_all(self):
        def reverse_bin(v: int) -> str:
            vl = list(f"{v:08b}")
            vl.reverse()
            return "".join(vl)

        reg_addr = self.RegNames["HeaterEnable_L16"]
        logger.info(f"heater 00--07: {reverse_bin(self.read_reg(reg_addr) & 0xff)}")
        logger.info(f"heater 08--15: {reverse_bin(self.read_reg(reg_addr) >> 8)}")
        logger.info(f"heater 16--23: {reverse_bin(self.read_reg(reg_addr+1) & 0xff)}")
        logger.info(f"heater 24--31: {reverse_bin(self.read_reg(reg_addr+1) >> 8)}")
        logger.info(f"heater 32--39: {reverse_bin(self.read_reg(reg_addr+2) & 0xff)}")

    def get_heater_master(self) -> bool:
        r = PowerboardPwmHeaterMaster()
        r.parse(self.read_reg(self.RegNames["HeaterMaster"]))
        return r.en

    def set_heater_master(self, en: bool) -> None:
        r = PowerboardPwmHeaterMaster()
        r.en = en
        self.write_reg(self.RegNames["HeaterMaster"], r.build())


class PowerboardPwmConfigHelper(AbstractIcConfigHelper):
    def __init__(self, ic: PowerboardPwmMixin):
        super().__init__(ic)

    def flush(self, discard_after_flush=True):
        for addr, value in self.updated.items():
            self.ic.write_reg(addr, value)
        if discard_after_flush:
            self.discard()
