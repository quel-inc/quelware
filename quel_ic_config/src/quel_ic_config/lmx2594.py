import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Final, List, Sequence, Tuple, Union, cast

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


# General Registers
@dataclass
class Lmx2594R0(AbstractIcReg):
    ramp_en: bool = field(default=False)  # [15]
    vco_phase_sync_en: bool = field(default=False)  # [14]
    out_mute: bool = field(default=False)  # [9]
    fcal_hpfd_adj: int = field(default=0)  # [8:7]
    fcal_lpfd_adj: int = field(default=0)  # [6:5]
    fcal_en: bool = field(default=False)  # [3]
    muxout_ld_sel: bool = field(default=False)  # [2]
    reset: bool = field(default=False)  # [1]
    powerdown: bool = field(default=False)  # [0]

    def parse(self, v):
        self.ramp_en = p_1bf_bool(v, 15)
        self.vco_phase_sync_en = p_1bf_bool(v, 14)
        self.out_mute = p_1bf_bool(v, 9)
        self.fcal_hpfd_adj = p_nbf(v, 8, 7)
        self.fcal_lpfd_adj = p_nbf(v, 6, 5)
        self.fcal_en = p_1bf_bool(v, 3)
        self.muxout_ld_sel = p_1bf_bool(v, 2)
        self.reset = p_1bf_bool(v, 1)
        self.powerdown = p_1bf_bool(v, 0)

    def build(self):
        return (
            0b0010010000010000
            | b_1bf_bool(self.ramp_en, 15)
            | b_1bf_bool(self.vco_phase_sync_en, 14)
            | b_1bf_bool(self.out_mute, 9)
            | b_nbf(self.fcal_hpfd_adj, 8, 7)
            | b_nbf(self.fcal_lpfd_adj, 6, 5)
            | b_1bf_bool(self.fcal_en, 3)
            | b_1bf_bool(self.muxout_ld_sel, 2)
            | b_1bf_bool(self.reset, 1)
            | b_1bf_bool(self.powerdown, 0)
        )


@dataclass
class Lmx2594R1(AbstractIcReg):
    cal_clk_div: int = field(default=0)  # [2:0], 3@reset

    def parse(self, v):
        self.cal_clk_div = p_nbf(v, 2, 0)

    def build(self):
        return 0b0000100000001000 | b_nbf(self.cal_clk_div, 2, 0)


@dataclass
class Lmx2594R7(AbstractIcReg):
    out_force: bool = field(default=False)  # [14]

    def parse(self, v):
        self.out_force = p_1bf_bool(v, 14)

    def build(self):
        return 0b0000000010110010 | b_1bf_bool(self.out_force, 14)


# Input Path Registers
@dataclass
class Lmx2594R9(AbstractIcReg):
    osc_2x: bool = field(default=False)  # [12]

    def parse(self, v):
        self.osc_2x = p_1bf_bool(v, 12)

    def build(self):
        return 0b0000011000000100 | b_1bf_bool(self.osc_2x, 12)


@dataclass
class Lmx2594R10(AbstractIcReg):
    mult: int = field(default=0)  # [11:7], 1@reset

    def parse(self, v):
        self.mult = p_nbf(v, 11, 7)

    def build(self):
        return 0b0001000001011000 | b_nbf(self.mult, 11, 7)


@dataclass
class Lmx2594R11(AbstractIcReg):
    pll_r: int = field(default=0)  # [11:4], 1@reset

    def parse(self, v):
        self.pll_r = p_nbf(v, 11, 4)

    def build(self):
        return 0b0000000000001000 | b_nbf(self.pll_r, 11, 4)


@dataclass
class Lmx2594R12(AbstractIcReg):
    pll_r_pre: int = field(default=0)  # [11:0], 1@reset

    def parse(self, v):
        self.pll_r_pre = p_nbf(v, 11, 0)

    def build(self):
        return 0b0101000000000000 | b_nbf(self.pll_r_pre, 11, 0)


# Charge Pump Registers
@dataclass
class Lmx2594R14(AbstractIcReg):
    cpg: int = field(default=0)  # [6:4], 7@reset

    def parse(self, v):
        self.cpg = p_nbf(v, 6, 4)

    def build(self):
        return 0b0001111000000000 | b_nbf(self.cpg, 6, 4)


# VCO calibration registers
@dataclass
class Lmx2594R4(AbstractIcReg):
    acal_cmp_dly: int = field(default=0)  # [15:8], 10@reset

    def parse(self, v):
        self.acal_cmp_dly = p_nbf(v, 15, 8)

    def build(self):
        return 0b0000000001000011 | b_nbf(self.acal_cmp_dly, 15, 8)


@dataclass
class Lmx2594R8(AbstractIcReg):
    vco_daciset_force: bool = field(default=False)  # [14]
    vco_capctrl_force: bool = field(default=False)  # [11]

    def parse(self, v):
        self.vco_daciset_force = p_1bf_bool(v, 14)
        self.vco_capctrl_force = p_1bf_bool(v, 11)

    def build(self):
        return 0b0010000000000000 | b_1bf_bool(self.vco_daciset_force, 14) | b_1bf_bool(self.vco_capctrl_force, 11)


@dataclass
class Lmx2594R16(AbstractIcReg):
    vco_daciset: int = field(default=0)  # [8:0], 128@reset

    def parse(self, v):
        self.vco_daciset = p_nbf(v, 8, 0)

    def build(self):
        return 0b0000000000000000 | b_nbf(self.vco_daciset, 8, 0)


@dataclass
class Lmx2594R17(AbstractIcReg):
    vco_daciset_strt: int = field(default=0)  # [8:0]. 250@reset

    def parse(self, v):
        self.vco_daciset_strt = p_nbf(v, 8, 0)

    def build(self):
        return 0b0000000000000000 | b_nbf(self.vco_daciset_strt, 8, 0)


@dataclass
class Lmx2594R19(AbstractIcReg):
    vco_capctrl: int = field(default=0)  # [7:0], 183@reset

    def parse(self, v):
        self.vco_capctrl = p_nbf(v, 7, 0)

    def build(self):
        return 0b0010011100000000 | b_nbf(self.vco_capctrl, 7, 0)


@dataclass
class Lmx2594R20(AbstractIcReg):
    vco_sel: int = field(default=0)  # [13:11], 7@reset
    vco_sel_force: bool = field(default=False)  # [10]

    def parse(self, v):
        self.vco_sel = p_nbf(v, 13, 11)
        self.vco_sel_force = p_1bf_bool(v, 10)

    def build(self):
        return 0b1100000001001000 | b_nbf(self.vco_sel, 13, 11) | b_1bf_bool(self.vco_sel_force, 10)


@dataclass
class Lmx2594R78(AbstractIcReg):
    ramp_thresh_32: int = field(default=0)  # [11:11]
    quick_recal_en: bool = field(default=False)  # [9]
    vco_capctrl_strt: int = field(default=0)  # [8:1]

    def parse(self, v: int) -> None:
        self.ramp_thresh_32 = p_nbf(v, 11, 11)
        self.quick_recal_en = p_1bf_bool(v, 9)
        self.vco_capctrl_strt = p_nbf(v, 8, 1)

    def build(self) -> int:
        return (
            0b0000000000000001
            | b_nbf(self.ramp_thresh_32, 11, 11)
            | b_1bf_bool(self.quick_recal_en, 9)
            | b_nbf(self.vco_capctrl_strt, 8, 1)
        )


# N Divider, MASH, and Output Registers
@dataclass
class Lmx2594R34(AbstractIcReg):
    pll_n_18_16: int = field(default=0)  # [2:0]

    def parse(self, v):
        self.pll_n_18_16 = p_nbf(v, 2, 0)

    def build(self):
        return 0b0000000000000000 | b_nbf(self.pll_n_18_16, 2, 0)


@dataclass
class Lmx2594R36(AbstractIcReg):
    pll_n: int = field(default=0)  # [15:0], 100@reset

    def parse(self, v):
        self.pll_n = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.pll_n, 15, 0)


@dataclass
class Lmx2594R37(AbstractIcReg):
    mash_seed_en: bool = field(default=False)  # [15]
    pfd_dly_sel: int = field(default=0)  # [13:8]  2@reset

    def parse(self, v):
        self.mash_seed_en = p_1bf_bool(v, 15)
        self.pfd_dly_sel = p_nbf(v, 13, 8)

    def build(self):
        return 0b0000000000000100 | b_1bf_bool(self.mash_seed_en, 15) | b_nbf(self.pfd_dly_sel, 13, 8)


@dataclass
class Lmx2594R38(AbstractIcReg):
    pll_den_31_16: int = field(default=0)  # [15:0], 4294967295@reset, with R39

    def parse(self, v):
        self.pll_den_31_16 = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.pll_den_31_16, 15, 0)


@dataclass
class Lmx2594R39(AbstractIcReg):
    pll_den_15_0: int = field(default=0)  # [15:0]

    def parse(self, v):
        self.pll_den_15_0 = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.pll_den_15_0, 15, 0)


@dataclass
class Lmx2594R40(AbstractIcReg):
    mash_seed_31_16: int = field(default=0)  # [15:0]

    def parse(self, v):
        self.mash_seed_31_16 = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.mash_seed_31_16, 15, 0)


@dataclass
class Lmx2594R41(AbstractIcReg):
    mash_seed_15_0: int = field(default=0)  # [15:0]

    def parse(self, v):
        self.mash_seed_15_0 = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.mash_seed_15_0, 15, 0)


@dataclass
class Lmx2594R42(AbstractIcReg):
    pll_num_31_16: int = field(default=0)  # [15:0]

    def parse(self, v):
        self.pll_num_31_16 = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.pll_num_31_16, 15, 0)


@dataclass
class Lmx2594R43(AbstractIcReg):
    pll_num_15_0: int = field(default=0)  # [15:0]

    def parse(self, v):
        self.pll_num_15_0 = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.pll_num_15_0, 15, 0)


@dataclass
class Lmx2594R44(AbstractIcReg):
    outa_pwr: int = field(default=0)  # [13:8], 31@reset
    outb_pd: bool = field(default=False)  # [7], 1@reset
    outa_pd: bool = field(default=False)  # [6]
    mash_reset_n: bool = field(default=False)  # [5] 1@reset
    mash_order: int = field(default=0)  # [2:0]

    def parse(self, v):
        self.outa_pwr = p_nbf(v, 13, 8)
        self.outb_pd = p_1bf_bool(v, 7)
        self.outa_pd = p_1bf_bool(v, 6)
        self.mash_reset_n = p_1bf_bool(v, 5)
        self.mash_order = p_nbf(v, 2, 0)

    def build(self):
        return (
            0b0000000000000000
            | b_nbf(self.outa_pwr, 13, 8)
            | b_1bf_bool(self.outb_pd, 7)
            | b_1bf_bool(self.outa_pd, 6)
            | b_1bf_bool(self.mash_reset_n, 5)
            | b_nbf(self.mash_order, 2, 0)
        )


@dataclass
class Lmx2594R45(AbstractIcReg):
    outa_mux: int = field(default=0)  # [12:11], 1@reset
    out_iset: int = field(default=0)  # [10:9]
    outb_pwr: int = field(default=0)  # [5:0], 31@reset

    def parse(self, v):
        self.outa_mux = p_nbf(v, 12, 11)
        self.out_iset = p_nbf(v, 10, 9)
        self.outb_pwr = p_nbf(v, 5, 0)

    def build(self):
        return (
            0b1100000011000000 | b_nbf(self.outa_mux, 12, 11) | b_nbf(self.out_iset, 10, 9) | b_nbf(self.outb_pwr, 5, 0)
        )


@dataclass
class Lmx2594R46(AbstractIcReg):
    outb_mux: int = field(default=0)  # [1:0], 1@reset

    def parse(self, v):
        self.outb_mux = p_nbf(v, 1, 0)

    def build(self):
        return 0b0000011111111100 | b_nbf(self.outb_mux, 1, 0)


# SYNC and SysRefReq Input Pin Register
@dataclass
class Lmx2594R58(AbstractIcReg):
    inpin_ignore: bool = field(default=False)  # [15], 1@reset
    inpin_hyst: bool = field(default=False)  # [14]
    inpin_lvl: int = field(default=0)  # [13:12]
    inpin_fmt: int = field(default=0)  # [11:9]

    def parse(self, v):
        self.inpin_ignore = p_1bf_bool(v, 15)
        self.inpin_hyst = p_1bf_bool(v, 14)
        self.inpin_lvl = p_nbf(v, 13, 12)
        self.inpin_fmt = p_nbf(v, 11, 9)

    def build(self):
        return (
            0b0000000000000001
            | b_1bf_bool(self.inpin_ignore, 15)
            | b_1bf_bool(self.inpin_hyst, 14)
            | b_nbf(self.inpin_lvl, 13, 12)
            | b_nbf(self.inpin_fmt, 11, 9)
        )


# Lock Detect Registers
@dataclass
class Lmx2594R59(AbstractIcReg):
    ld_type: bool = field(default=False)  # [0], 1@reset

    def parse(self, v):
        self.ld_type = p_1bf_bool(v, 0)

    def build(self):
        return 0b0000000000000000 | b_1bf_bool(self.ld_type, 0)


@dataclass
class Lmx2594R60(AbstractIcReg):
    ld_dly: int = field(default=0)  # [15:0], 1000@reset

    def parse(self, v):
        self.ld_dly = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.ld_dly, 15, 0)


# MASH_RESET
@dataclass
class Lmx2594R69(AbstractIcReg):
    mash_reset_count_31_16: int = field(default=0)  # [15:0], 50000@reset (with R70)

    def parse(self, v):
        self.mash_reset_count_31_16 = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.mash_reset_count_31_16, 15, 0)


@dataclass
class Lmx2594R70(AbstractIcReg):
    mash_reset_count_15_0: int = field(default=0)  # [15:0], 50000@reset (with R69)

    def parse(self, v):
        self.mash_reset_count_15_0 = p_nbf(v, 15, 0)

    def build(self):
        return b_nbf(self.mash_reset_count_15_0, 15, 0)


# SysREF Registers
@dataclass
class Lmx2594R71(AbstractIcReg):
    sysref_div_pre: int = field(default=0)  # [7:5], 4@reset
    sysref_pulse: bool = field(default=False)  # [4]
    sysref_en: bool = field(default=False)  # [3]
    sysref_repeat: bool = field(default=False)  # [2]

    def parse(self, v):
        self.sysref_div_pre = p_nbf(v, 7, 5)
        self.sysref_pulse = p_1bf_bool(v, 4)
        self.sysref_en = p_1bf_bool(v, 3)
        self.sysref_repeat = p_1bf_bool(v, 2)

    def build(self):
        return (
            0b0000000000000001
            | b_nbf(self.sysref_div_pre, 7, 5)
            | b_1bf_bool(self.sysref_pulse, 4)
            | b_1bf_bool(self.sysref_en, 3)
            | b_1bf_bool(self.sysref_repeat, 2)
        )


@dataclass
class Lmx2594R72(AbstractIcReg):
    sysref_div: int = field(default=0)  # [10:0]

    def parse(self, v):
        self.sysref_div = p_nbf(v, 10, 0)

    def build(self):
        return 0b0000000000000000 | b_nbf(self.sysref_div, 10, 0)


@dataclass
class Lmx2594R73(AbstractIcReg):
    jesd_dac2_ctrl: int = field(default=0)  # [11:6]
    jesd_dac1_ctrl: int = field(default=0)  # [5:0], 63@reset

    def parse(self, v):
        self.jesd_dac2_ctrl = p_nbf(v, 11, 6)
        self.jesd_dac1_ctrl = p_nbf(v, 5, 0)

    def build(self):
        return 0b0000000000000000 | b_nbf(self.jesd_dac2_ctrl, 11, 6) | b_nbf(self.jesd_dac1_ctrl, 5, 0)


@dataclass
class Lmx2594R74(AbstractIcReg):
    sysref_pulse_cnt: int = field(default=0)  # [15:12]
    jesd_dac4_ctrl: int = field(default=0)  # [11:6]
    jesd_dac3_ctrl: int = field(default=0)  # [5:0]

    def parse(self, v):
        self.sysref_pulse_cnt = p_nbf(v, 15, 12)
        self.jesd_dac4_ctrl = p_nbf(v, 11, 6)
        self.jesd_dac3_ctrl = p_nbf(v, 5, 0)

    def build(self):
        return (
            0b0000000000000000
            | b_nbf(self.sysref_pulse_cnt, 15, 12)
            | b_nbf(self.jesd_dac4_ctrl, 11, 6)
            | b_nbf(self.jesd_dac3_ctrl, 5, 0)
        )


# CHANNEL Divider Registers
@dataclass
class Lmx2594R31(AbstractIcReg):
    chdiv_div2: bool = field(default=False)  # [14]

    def parse(self, v):
        self.chdiv_div2 = p_1bf_bool(v, 14)

    def build(self):
        return 0b0000001111101100 | b_1bf_bool(self.chdiv_div2, 14)


@dataclass
class Lmx2594R75(AbstractIcReg):
    chdiv: int = field(default=0)  # [10:6]

    def parse(self, v):
        self.chdiv = p_nbf(v, 10, 6)

    def build(self):
        return 0b0000100000000000 | b_nbf(self.chdiv, 10, 6)


# Ramping and Calibration Fields
# TODO: define R79 and R80

# Ramping Registers/Ramp Limits
# TODO: define R81, R82, R83, R84, R85, and R86

# Ramping Registers/Ramping Triggers, Burst Mode, and RAMP0_RST
# TODO define R96 and R97

# Ramping Registers/Ramping Configuration
# TODO: define R98, R99, R100, R101, R102, R103, R104, R105, and R106


# Readback Registers
@dataclass
class Lmx2594R110(AbstractIcReg):
    rb_ld_vtune: int = field(default=0)  # [10:9]
    rb_vco_sel: int = field(default=0)  # [7:5]

    def parse(self, v):
        self.rb_ld_vtune = p_nbf(v, 10, 9)
        self.rb_vco_sel = p_nbf(v, 7, 5)

    def build(self):
        return 0b0000000000000000 | b_nbf(self.rb_ld_vtune, 10, 9) | b_nbf(self.rb_vco_sel, 7, 5)


@dataclass
class Lmx2594R111(AbstractIcReg):
    rb_vco_capctrl: int = field(default=0)  # [7:0], 183@reset

    def parse(self, v):
        self.rb_vco_capctrl = p_nbf(v, 7, 0)

    def build(self):
        return 0b0000000000000000 | b_nbf(self.rb_vco_capctrl, 7, 0)


@dataclass
class Lmx2594R112(AbstractIcReg):
    rb_vco_daciset: int = field(default=0)  # [8:0], 170@reset

    def parse(self, v):
        self.rb_vco_daciset = p_nbf(v, 8, 0)

    def build(self):
        return 0b0000000000000000 | b_nbf(self.rb_vco_daciset, 8, 0)


Lmx2594Regs: Dict[int, type] = {
    0: Lmx2594R0,
    1: Lmx2594R1,
    4: Lmx2594R4,
    7: Lmx2594R7,
    8: Lmx2594R8,
    9: Lmx2594R9,
    10: Lmx2594R10,
    11: Lmx2594R11,
    12: Lmx2594R12,
    14: Lmx2594R14,
    16: Lmx2594R16,
    17: Lmx2594R17,
    19: Lmx2594R19,
    20: Lmx2594R20,
    31: Lmx2594R31,
    34: Lmx2594R34,
    36: Lmx2594R36,
    37: Lmx2594R37,
    38: Lmx2594R38,
    39: Lmx2594R39,
    40: Lmx2594R40,
    41: Lmx2594R41,
    42: Lmx2594R42,
    43: Lmx2594R43,
    44: Lmx2594R44,
    45: Lmx2594R45,
    46: Lmx2594R46,
    58: Lmx2594R58,
    59: Lmx2594R59,
    60: Lmx2594R60,
    69: Lmx2594R69,
    70: Lmx2594R70,
    71: Lmx2594R71,
    72: Lmx2594R72,
    73: Lmx2594R73,
    74: Lmx2594R74,
    75: Lmx2594R75,
    78: Lmx2594R78,
    110: Lmx2594R110,
    111: Lmx2594R111,
    112: Lmx2594R112,
}


Lmx2594RegNames: Dict[str, int] = {f"R{k}": k for k in Lmx2594Regs}


class Lmx2594LockStatus(IntEnum):
    TooLow = 0
    Invalid = 1
    Locked = 2
    TooHigh = 3


class Lmx2594Mixin(AbstractIcMixin):
    Regs: Dict[int, type] = Lmx2594Regs
    RegNames: Dict[str, int] = Lmx2594RegNames
    MIN_FREQ_MULTIPLIER: Final[int] = 75
    MAX_FREQ_MULTIPLIER: Final[int] = 150
    CHDIV2RATIO: Final[Dict[int, int]] = {
        0: 2,
        1: 4,
        2: 6,
        3: 8,
        4: 12,
        5: 16,
        6: 24,
        7: 32,
        8: 48,
        9: 64,
        10: 72,
        11: 96,
        12: 128,
        13: 192,
        14: 256,
        15: 384,
        16: 512,
        17: 768,
    }

    def __init__(self, name):
        super().__init__(name)

    def dump_regs(self) -> Dict[int, int]:
        regs: Dict[int, int] = {}
        for addr in range(0, 113):
            regs[addr] = self.read_reg(addr)
        return regs

    def soft_reset(self) -> None:
        v0 = self.read_reg(0) & ~0x08  # clear calibration_enable once just for the ease of debug
        self.write_reg(0, v0 | 0x2)
        time.sleep(0.01)
        self.write_reg(0, v0 & ~0x2)

    def is_locked(self) -> Lmx2594LockStatus:
        v110 = self.read_reg(110)
        return Lmx2594LockStatus((v110 >> 9) & 0x3)

    def calibrate(self) -> bool:
        # ensure it takes more than 10msec after manipulating any registers.
        time.sleep(0.01)

        # writing 1 on bit3 of R0 activates calibration procedure once. No transition from 0 to 1 is required.
        v0 = self.read_reg(0)
        self.write_reg(0, v0 | 0x08)
        # checking the completion of calibration based on register value.
        t0 = time.perf_counter()
        for i in range(2):
            n = (100, 500)[i]  # 500 is required for the conventional settings of the refclks.
            for _ in range(n):
                time.sleep(0.001)
                v110 = self.read_reg(110)
                if (v110 >> 9) & 0x3 == 2:
                    logger.info(f"calibration of {self.name} is finished in {(time.perf_counter()-t0)*1000:.1f}ms")
                    return True
            else:
                # Maybe the second kick is required due to inappropriate initial values of VCO related settings.
                # TODO: investigate it by make the register settings correct.
                if i == 0:
                    logger.info("raise calibration flag again")
                    self.write_reg(0, v0 | 0x08)
        else:
            logger.warning(f"calibration of {self.name} is not finished within {(time.perf_counter()-t0)*1000:.1f}ms")
        return False

    def _validate_freq_multiplier(self, freq_multiplier: int):
        if not isinstance(freq_multiplier, int):
            raise TypeError(f"unexpected frequency multiplier: {freq_multiplier}")
        if not (self.MIN_FREQ_MULTIPLIER <= freq_multiplier <= self.MAX_FREQ_MULTIPLIER):
            raise TypeError(f"invalid frequency multiplier: {freq_multiplier}")

    def set_lo_multiplier(self, freq_multiplier: int) -> None:
        addr34, reg34 = cast(Tuple[int, Lmx2594R34], self._read_and_parse_reg("R34"))
        addr36, reg36 = cast(Tuple[int, Lmx2594R36], self._read_and_parse_reg("R36"))
        reg34.pll_n_18_16 = (freq_multiplier >> 16) & 0x0007
        reg36.pll_n = freq_multiplier & 0xFFFF
        self._build_and_write_reg(addr34, reg34)
        self._build_and_write_reg(addr36, reg36)
        return

    def get_lo_multiplier(self) -> int:
        _, reg34 = cast(Tuple[int, Lmx2594R34], self._read_and_parse_reg("R34"))
        _, reg36 = cast(Tuple[int, Lmx2594R36], self._read_and_parse_reg("R36"))
        return (reg34.pll_n_18_16 << 16) + (reg36.pll_n)

    def _parse_divide_ratio(
        self, out_mux: Sequence[int], pwdn: Sequence[bool], enable_seg1: bool, chdiv: int
    ) -> Tuple[int, int]:
        ratio: List[int] = [0, 0]
        for i in range(2):
            if pwdn[i]:
                continue

            if out_mux[i] == 0:
                if enable_seg1:
                    if chdiv == 0:
                        raise RuntimeError(
                            f"invalid combination of seg1_enable(={enable_seg1}) and chdiv(= {chdiv}) at {self.name}"
                        )
                    elif chdiv in self.CHDIV2RATIO:
                        ratio[i] = self.CHDIV2RATIO[chdiv]
                    else:
                        raise RuntimeError(f"invalid chdiv(= {chdiv}) at {self.name}")
                else:
                    if chdiv == 0:
                        ratio[i] = 2
                    else:
                        raise RuntimeError(
                            f"invalid combination of seg1_enable(={enable_seg1}) and chdiv(= {chdiv}) at {self.name}"
                        )
            elif out_mux[i] == 1:
                ratio[i] = 1
            elif out_mux[i] == 2:
                # SYSREF output, considered as ration = 0
                pass
            elif out_mux[i] == 3:
                # high_impedance output, equivalent to ratio = 0
                pass
            else:
                # Note: Sysref should not be selected for LO.
                raise AssertionError(f"unexpected mux of outpin:{i} of {self.name}")

        return ratio[0], ratio[1]

    def set_divider_ratio(self, ratio0: Union[int, None], ratio1: Union[int, None]) -> None:
        addr44, reg44 = cast(Tuple[int, Lmx2594R44], self._read_and_parse_reg("R44"))
        addr45, reg45 = cast(Tuple[int, Lmx2594R45], self._read_and_parse_reg("R45"))
        addr46, reg46 = cast(Tuple[int, Lmx2594R46], self._read_and_parse_reg("R46"))
        addr31, reg31 = cast(Tuple[int, Lmx2594R31], self._read_and_parse_reg("R31"))
        addr75, reg75 = cast(Tuple[int, Lmx2594R75], self._read_and_parse_reg("R75"))

        out_mux: Tuple[int, int] = reg45.outa_mux, reg46.outb_mux
        pwdn: Tuple[bool, bool] = reg44.outa_pd, reg44.outb_pd
        enable_seg1: bool = reg31.chdiv_div2
        chdiv: int = reg75.chdiv

        ratios = list(self._parse_divide_ratio(out_mux, pwdn, enable_seg1, chdiv))
        if ratio0 is not None:
            ratios[0] = ratio0
        if ratio1 is not None:
            ratios[1] = ratio1
        if not (ratios[0] in {0, 1} or ratios[1] in {0, 1} or ratios[0] == ratios[1]):
            raise ValueError(f"impossible combination of divide ratios ({ratios[0]}, {ratios[1]}) for {self.name}")

        for i, ratio in enumerate(ratios):
            if ratio == 1:
                if i == 0:
                    reg45.outa_mux = 1
                    reg44.outa_pd = False
                elif i == 1:
                    reg46.outb_mux = 1
                    reg44.outb_pd = False
                else:
                    raise AssertionError
            elif ratio == 0:
                if i == 0:
                    reg45.outa_mux = 3
                    reg44.outa_pd = True
                elif i == 1:
                    reg46.outb_mux = 3
                    reg44.outb_pd = True
                else:
                    raise AssertionError
            else:
                if i == 0:
                    reg45.outa_mux = 0
                    reg44.outa_pd = False
                elif i == 1:
                    reg46.outb_mux = 0
                    reg44.outb_pd = False
                else:
                    raise AssertionError

        for i, ratio in enumerate(ratios):
            if ratio in {0, 1}:
                continue
            else:
                for k, v in self.CHDIV2RATIO.items():
                    if v == ratio:
                        chdiv = k
                        break
                else:
                    raise ValueError(f"invalid frequency divider ratio {ratio} for outpin {i} of {self.name}")
                reg31.chdiv_div2 = ratio != 2
                reg75.chdiv = chdiv
                break

        self._build_and_write_reg(75, reg75)
        self._build_and_write_reg(31, reg31)
        self._build_and_write_reg(46, reg46)
        self._build_and_write_reg(45, reg45)
        self._build_and_write_reg(44, reg44)

    def get_divider_ratio(self) -> Tuple[int, int]:
        addr44, reg44 = cast(Tuple[int, Lmx2594R44], self._read_and_parse_reg("R44"))
        addr45, reg45 = cast(Tuple[int, Lmx2594R45], self._read_and_parse_reg("R45"))
        addr46, reg46 = cast(Tuple[int, Lmx2594R46], self._read_and_parse_reg("R46"))
        addr31, reg31 = cast(Tuple[int, Lmx2594R31], self._read_and_parse_reg("R31"))
        addr75, reg75 = cast(Tuple[int, Lmx2594R75], self._read_and_parse_reg("R75"))

        out_mux: Tuple[int, int] = reg45.outa_mux, reg46.outb_mux
        pwdn: Tuple[bool, bool] = reg44.outa_pd, reg44.outb_pd
        enable_seg1: bool = reg31.chdiv_div2
        chdiv: int = reg75.chdiv
        return self._parse_divide_ratio(out_mux, pwdn, enable_seg1, chdiv)

    def get_sync_enable(self) -> bool:
        addr0, reg0 = cast(Tuple[int, Lmx2594R0], self._read_and_parse_reg("R0"))
        return reg0.vco_phase_sync_en


class Lmx2594ConfigHelper(AbstractIcConfigHelper):
    """Helper class for programming LMX2594 with convenient notations. It also provides caching capability that
    keep modifications on the registers to wirte them at once with flash_updated() in a right order.
    """

    _TopWritableAddress: int = 106

    def __init__(self, ic: Lmx2594Mixin):
        super().__init__(ic)

    def flush(self, discard_after_flush=True):
        for addr in sorted(self.updated, reverse=True):
            if addr <= self._TopWritableAddress:
                self.ic.write_reg(addr, self.updated[addr])
            else:
                logger.warning(f"failed attempt to write reg-{addr} is detected")
        if discard_after_flush:
            self.discard()

    # TODO: will be moved to super class later.
    def load_settings(self, regs: Dict[str, Dict[str, Union[int, str]]]):
        pass
