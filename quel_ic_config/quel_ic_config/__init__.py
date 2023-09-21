from typing import Union

from quel_ic_config.ad5328 import Ad5328ConfigHelper, Ad5328RegNames, Ad5328Regs
from quel_ic_config.ad9082_v106 import Ad9082JesdParam, ChipTemperatures, NcoFtw
from quel_ic_config.adrf6780 import Adrf6780ConfigHelper, Adrf6780LoSideband, Adrf6780RegNames, Adrf6780Regs
from quel_ic_config.generic_gpio import GenericGpioConfigHelper, GenericGpioRegNames, GenericGpioRegs
from quel_ic_config.lmx2594 import Lmx2594ConfigHelper, Lmx2594RegNames, Lmx2594Regs
from quel_ic_config.quel1_config_subsystem import ExstickgeProxyQuel1, Quel1ConfigSubsystem
from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemRoot
from quel_ic_config.quel_config_common import QUEL1_BOXTYPE_ALIAS, Quel1BoxType, Quel1ConfigOption
from quel_ic_config.quel_ic import (
    Ad5328,
    Ad9082V106,
    Adrf6780,
    GenericGpio,
    Lmx2594,
    QubeRfSwitchArray,
    Quel1TypeARfSwitchArray,
    Quel1TypeBRfSwitchArray,
)
from quel_ic_config.rfswitcharray import (
    AbstractRfSwitchArrayMixin,
    QubeRfSwitchRegs,
    QubeSwitchRegNames,
    Quel1SwitchRegNames,
    Quel1TypeARfSwitchRegs,
    Quel1TypeBRfSwitchRegs,
    RfSwitchArrayConfigHelper,
)
from quel_ic_config.thermistor import Quel1NormalThermistor, Quel1PathSelectorThermistor

Quel1AnyConfigSubsystem = Union[Quel1ConfigSubsystem]

__version__ = "0.4.9"

__all__ = (
    "ExstickgeProxyQuel1",
    "Ad5328",
    "Ad5328ConfigHelper",
    "Ad5328Regs",
    "Ad5328RegNames",
    "Ad9082V106",
    "Ad9082JesdParam",
    "Adrf6780",
    "Adrf6780ConfigHelper",
    "Adrf6780Regs",
    "Adrf6780RegNames",
    "Adrf6780LoSideband",
    "GenericGpio",
    "GenericGpioConfigHelper",
    "GenericGpioRegs",
    "GenericGpioRegNames",
    "Lmx2594",
    "Lmx2594ConfigHelper",
    "Lmx2594Regs",
    "Lmx2594RegNames",
    "NcoFtw",
    "ChipTemperatures",
    "Quel1AnyConfigSubsystem",
    "Quel1ConfigOption",
    "Quel1ConfigSubsystemRoot",
    "Quel1NormalThermistor",
    "Quel1PathSelectorThermistor",
    "Quel1TypeARfSwitchArray",
    "Quel1TypeBRfSwitchArray",
    "Quel1TypeARfSwitchRegs",
    "Quel1TypeBRfSwitchRegs",
    "Quel1SwitchRegNames",
    "Quel1BoxType",
    "Quel1ConfigSubsystem",
    "QubeRfSwitchRegs",
    "QubeSwitchRegNames",
    "QubeRfSwitchArray",
    "QUEL1_BOXTYPE_ALIAS",
    "RfSwitchArrayConfigHelper",
    "AbstractRfSwitchArrayMixin",
)
