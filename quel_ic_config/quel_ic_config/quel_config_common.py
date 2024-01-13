from enum import Enum
from typing import Dict, Final


class Quel1BoxType(Enum):
    QuBE_OU_TypeA = ("qube", "ou-type-a")
    QuBE_OU_TypeB = ("qube", "ou-type-b")
    QuBE_RIKEN_TypeA = ("qube", "riken-type-a")
    QuBE_RIKEN_TypeB = ("qube", "riken-type-b")
    QuEL1_TypeA = ("quel-1", "type-a")
    QuEL1_TypeB = ("quel-1", "type-b")
    QuEL1_NTT = ("quel-1", "ntt")  # Notes: not fully supported yet.  TODO: implement automatic tests
    QuEL1_NEC = ("quel-1", "nec")  # Notes: under development
    QuEL1SE_Adda = ("quel-1se", "adda")  # Prototype configuration
    QuEL1SE_Proto8 = ("quel-1se", "proto8")  # Prototype configuration
    QuEL1SE_Proto11 = ("quel-1se", "proto11")  # Prototype configuration

    @classmethod
    def fromstr(cls, label: str) -> "Quel1BoxType":
        return QUEL1_BOXTYPE_ALIAS[label]


QUEL1_BOXTYPE_ALIAS: Final[Dict[str, Quel1BoxType]] = {
    "qube-ou-a": Quel1BoxType.QuBE_OU_TypeA,
    "qube-ou-b": Quel1BoxType.QuBE_OU_TypeB,
    "qube-riken-a": Quel1BoxType.QuBE_RIKEN_TypeA,
    "qube-riken-b": Quel1BoxType.QuBE_RIKEN_TypeB,
    "quel1-a": Quel1BoxType.QuEL1_TypeA,
    "quel1-b": Quel1BoxType.QuEL1_TypeB,
    "quel1-ntt": Quel1BoxType.QuEL1_NTT,
    "quel1-nec": Quel1BoxType.QuEL1_NEC,
    "x-quel1se-adda": Quel1BoxType.QuEL1SE_Adda,
    "x-quel1se-proto8": Quel1BoxType.QuEL1SE_Proto8,
    "x-quel1se-proto11": Quel1BoxType.QuEL1SE_Proto11,
}


class Quel1Feature(Enum):
    SINGLE_ADC = "single_adc"
    BOTH_ADC = "both_adc"
    # Notes: workaround for a bad design of feedback_20231108 firmware
    BOTH_ADC_EARLY = "both_adc_early"


class Quel1ConfigOption(str, Enum):
    USE_READ_IN_MXFE0 = "use_read_in_mxfe0"  # Notes: only for SIMPLEMULTI_CLASSIC
    USE_MONITOR_IN_MXFE0 = "use_monitor_in_mxfe0"  # Notes: only for SIMPLEMULTI_CLASSIC
    REFCLK_CORRECTED_MXFE0 = "refclk_corrected_mxfe0"
    REFCLK_12GHz_FOR_MXFE0 = "refclk_12ghz_for_mxfe0"
    X_REFCLK_250MHz_FOR_MXFE0 = "x_refclk_250mhz_for_mxfe0"
    X_REFCLK_100MHz_FOR_MXFE0 = "x_refclk_100mhz_for_mxfe0"
    DAC_CNCO_1500MHz_MXFE0 = "dac_cduc_1500mhz_mxfe0"
    DAC_CNCO_2000MHz_MXFE0 = "dac_cduc_2000mhz_mxfe0"

    USE_READ_IN_MXFE1 = "use_read_in_mxfe1"  # Notes: only for SIMPLEMULTI_CLASSIC
    USE_MONITOR_IN_MXFE1 = "use_monitor_in_mxfe1"  # Notes: only for SIMPLEMULTI_CLASSIC
    REFCLK_CORRECTED_MXFE1 = "refclk_corrected_mxfe1"
    REFCLK_12GHz_FOR_MXFE1 = "refclk_12ghz_for_mxfe1"
    X_REFCLK_250MHz_FOR_MXFE1 = "x_refclk_250mhz_for_mxfe1"
    X_REFCLK_100MHz_FOR_MXFE1 = "x_refclk_100mhz_for_mxfe1"
    DAC_CNCO_1500MHz_MXFE1 = "dac_cduc_1500mhz_mxfe1"
    DAC_CNCO_2000MHz_MXFE1 = "dac_cduc_2000mhz_mxfe1"
