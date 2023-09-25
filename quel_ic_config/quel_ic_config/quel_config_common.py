from enum import Enum
from typing import Dict, Final


class Quel1BoxType(Enum):
    QuBE_TypeA = ("qube", "type-a")  # TODO: not tested yet
    QuBE_TypeB = ("qube", "type-b")  # TODO: not tested yet
    QuEL1_TypeA = ("quel-1", "type-a")
    QuEL1_TypeB = ("quel-1", "type-b")  # TODO: not tested yet
    QuEL1_NTT = ("quel-1", "ntt")  # TODO: not supported yet

    @classmethod
    def fromstr(cls, label: str) -> "Quel1BoxType":
        return QUEL1_BOXTYPE_ALIAS[label]


QUEL1_BOXTYPE_ALIAS: Final[Dict[str, Quel1BoxType]] = {
    "qube-a": Quel1BoxType.QuEL1_TypeA,
    "qube-b": Quel1BoxType.QuEL1_TypeB,
    "quel1-a": Quel1BoxType.QuEL1_TypeA,
    "quel1-b": Quel1BoxType.QuEL1_TypeB,
}


class Quel1ConfigOption(str, Enum):
    USE_READ_IN_MXFE0 = "use_read_in_mxfe0"  # TODO: this is removed when dynamic alternation of input port is realized
    USE_MONITOR_IN_MXFE0 = "use_monitor_in_mxfe0"  # TODO: same as above
    REFCLK_12GHz_FOR_MXFE0 = "refclk_12ghz_for_mxfe0"
    DAC_CNCO_1500MHz_IN_MXFE0 = "dac_cduc_1500mhz_in_mxfe0"
    DAC_CNCO_2000MHz_IN_MXFE0 = "dac_cduc_2000mhz_in_mxfe0"

    USE_READ_IN_MXFE1 = "use_read_in_mxfe1"  # TODO: same as above
    USE_MONITOR_IN_MXFE1 = "use_monitor_in_mxfe1"  # TODO: same as above
    REFCLK_12GHz_FOR_MXFE1 = "refclk_12ghz_for_mxfe1"
    DAC_CNCO_1500MHz_IN_MXFE1 = "dac_cduc_1500mhz_in_mxfe1"
    DAC_CNCO_2000MHz_IN_MXFE1 = "dac_cduc_2000mhz_in_mxfe1"
