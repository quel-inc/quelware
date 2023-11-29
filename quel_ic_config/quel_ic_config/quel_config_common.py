from enum import Enum
from typing import Dict, Final


class Quel1BoxType(Enum):
    QuBE_TypeA = ("qube", "type-a")  # TODO: not tested yet
    QuBE_TypeB = ("qube", "type-b")  # TODO: not tested yet
    QuEL1_TypeA = ("quel-1", "type-a")
    QuEL1_TypeB = ("quel-1", "type-b")
    QuEL1Plus_TypeA = ("quel-1", "type-a-plus")
    QuEL1Plus_TypeB = ("quel-1", "type-b-plus")
    QuEL1_NTT = ("quel-1", "ntt")  # TODO: not supported yet
    QuEL1SE_Adda = ("quel-1se", "adda")  # Prototype configuration
    QuEL1SE_Proto8 = ("quel-1se", "proto8")  # Prototype configuration
    QuEL1SE_Proto11 = ("quel-1se", "proto11")  # Prototype configuration

    @classmethod
    def fromstr(cls, label: str) -> "Quel1BoxType":
        return QUEL1_BOXTYPE_ALIAS[label]


QUEL1_BOXTYPE_ALIAS: Final[Dict[str, Quel1BoxType]] = {
    "qube-a": Quel1BoxType.QuEL1_TypeA,
    "qube-b": Quel1BoxType.QuEL1_TypeB,
    "quel1-a": Quel1BoxType.QuEL1_TypeA,
    "quel1-b": Quel1BoxType.QuEL1_TypeB,
    "quel1++-a": Quel1BoxType.QuEL1Plus_TypeA,
    "quel1++-b": Quel1BoxType.QuEL1Plus_TypeB,
    "x-quel1se-adda": Quel1BoxType.QuEL1SE_Adda,
    "x-quel1se-proto8": Quel1BoxType.QuEL1SE_Proto8,
    "x-quel1se-proto11": Quel1BoxType.QuEL1SE_Proto11,
}


class Quel1ConfigOption(str, Enum):
    USE_READ_IN_MXFE0 = "use_read_in_mxfe0"  # TODO: this is removed when dynamic alternation of input port is realized
    USE_MONITOR_IN_MXFE0 = "use_monitor_in_mxfe0"  # TODO: same as above
    REFCLK_12GHz_FOR_MXFE0 = "refclk_12ghz_for_mxfe0"
    X_REFCLK_250MHz_FOR_MXFE0 = "x_refclk_250mhz_for_mxfe0"
    X_REFCLK_100MHz_FOR_MXFE0 = "x_refclk_100mhz_for_mxfe0"
    DAC_CNCO_1500MHz_MXFE0 = "dac_cduc_1500mhz_mxfe0"
    DAC_CNCO_2000MHz_MXFE0 = "dac_cduc_2000mhz_mxfe0"

    USE_READ_IN_MXFE1 = "use_read_in_mxfe1"  # TODO: same as above
    USE_MONITOR_IN_MXFE1 = "use_monitor_in_mxfe1"  # TODO: same as above
    REFCLK_12GHz_FOR_MXFE1 = "refclk_12ghz_for_mxfe1"
    X_REFCLK_250MHz_FOR_MXFE1 = "x_refclk_250mhz_for_mxfe1"
    X_REFCLK_100MHz_FOR_MXFE1 = "x_refclk_100mhz_for_mxfe1"
    DAC_CNCO_1500MHz_MXFE1 = "dac_cduc_1500mhz_mxfe1"
    DAC_CNCO_2000MHz_MXFE1 = "dac_cduc_2000mhz_mxfe1"
