from enum import Enum
from pathlib import Path
from typing import Dict, Final

_DEFAULT_LOCK_DIRECTORY: Final[Path] = Path("/run/quelware")


class Quel1BoxType(Enum):
    QuBE_OU_TypeA = ("qube", "ou-type-a")
    QuBE_OU_TypeB = ("qube", "ou-type-b")
    QuBE_RIKEN_TypeA = ("qube", "riken-type-a")
    QuBE_RIKEN_TypeB = ("qube", "riken-type-b")
    QuEL1_TypeA = ("quel-1", "type-a")
    QuEL1_TypeB = ("quel-1", "type-b")
    QuEL1_NTT = ("quel-1", "ntt")  # Notes: not fully supported yet.  TODO: implement automatic tests
    QuEL1_NEC = ("quel-1", "nec")
    QuEL1SE_RIKEN8 = ("quel-1se", "riken8")
    QuEL1SE_RIKEN8DBG = ("quel-1se", "riken8dbg")  # for development purpose
    QuEL1SE_FUJITSU11_TypeA = ("quel-1se", "fujitsu11-type-a")  # not available yet
    QuEL1SE_FUJITSU11_TypeB = ("quel-1se", "fujitsu11-type-b")  # not available yet
    QuEL1SE_FUJITSU11DBG_TypeA = ("quel-1se", "fujitsu11dbg-type-a")  # for development purpose
    QuEL1SE_FUJITSU11DBG_TypeB = ("quel-1se", "fujitsu11dbg-type-b")  # for development purpose
    # Notes: the following boxtypes can be eliminated without the prior notification.
    QuEL1SE_Adda = ("quel-1se", "adda")  # Prototype configuration
    QuEL2_ProtoAdda = ("quel-2", "protoadda")  # Prototype configuration

    @classmethod
    def fromstr(cls, label: str) -> "Quel1BoxType":
        return QUEL1_BOXTYPE_ALIAS[label]

    def tostr(self):
        for a, t in QUEL1_BOXTYPE_ALIAS.items():
            if t is self:
                return a
        else:
            raise AssertionError

    def is_quel1se(self) -> bool:
        # Notes: quel-2 is regarded as being quel-1se since it is in early stage and not so different from quel-1se.
        return self.value[0] in {"quel-1se", "quel-2"}


QUEL1_BOXTYPE_ALIAS: Final[Dict[str, Quel1BoxType]] = {
    "qube-ou-a": Quel1BoxType.QuBE_OU_TypeA,
    "qube-ou-b": Quel1BoxType.QuBE_OU_TypeB,
    "qube-riken-a": Quel1BoxType.QuBE_RIKEN_TypeA,
    "qube-riken-b": Quel1BoxType.QuBE_RIKEN_TypeB,
    "quel1-a": Quel1BoxType.QuEL1_TypeA,
    "quel1-b": Quel1BoxType.QuEL1_TypeB,
    "quel1-ntt": Quel1BoxType.QuEL1_NTT,
    "quel1-nec": Quel1BoxType.QuEL1_NEC,
    "quel1se-riken8": Quel1BoxType.QuEL1SE_RIKEN8,
    "quel1se-fujitsu11-a": Quel1BoxType.QuEL1SE_FUJITSU11_TypeA,
    "quel1se-fujitsu11-b": Quel1BoxType.QuEL1SE_FUJITSU11_TypeB,
    "x-quel1se-adda": Quel1BoxType.QuEL1SE_Adda,
    "x-quel1se-riken8": Quel1BoxType.QuEL1SE_RIKEN8DBG,
    "x-quel1se-fujitsu11-a": Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA,
    "x-quel1se-fujitsu11-b": Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeB,
    "x-quel2-protoadda": Quel1BoxType.QuEL2_ProtoAdda,
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
    SE8_MXFE1_AWG1331 = "se8_mxfe1_awg1331"
    SE8_MXFE1_AWG2222 = "se8_mxfe1_awg2222"
    SE8_MXFE1_AWG3113 = "se8_mxfe1_awg3113"


class Quel1RuntimeOption(Enum):
    ALLOW_DUAL_MODULUS_NCO = 1
