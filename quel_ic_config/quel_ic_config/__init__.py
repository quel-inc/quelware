from typing import Union

from quel_ic_config.ad5328 import Ad5328ConfigHelper, Ad5328RegNames, Ad5328Regs
from quel_ic_config.ad9082_v106 import Ad9082JesdParam, ChipTemperatures, NcoFtw
from quel_ic_config.adrf6780 import Adrf6780ConfigHelper, Adrf6780LoSideband, Adrf6780RegNames, Adrf6780Regs
from quel_ic_config.generic_gpio import GenericGpioConfigHelper, GenericGpioRegNames, GenericGpioRegs
from quel_ic_config.lmx2594 import Lmx2594ConfigHelper, Lmx2594RegNames, Lmx2594Regs
from quel_ic_config.quel1_config_subsystem import (
    ExstickgeProxyQuel1,
    QubeConfigSubsystem,
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1ConfigSubsystem,
    Quel1NecConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
)
from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemRoot
from quel_ic_config.quel1se_proto8_config_subsystem import ExstickgeProxyQuel1SeProto8, Quel1SeProto8ConfigSubsystem
from quel_ic_config.quel1se_proto11_config_subsystem import ExstickgeProxyQuel1SeProto11, Quel1SeProto11ConfigSubsystem
from quel_ic_config.quel1se_proto_adda_config_subsystem import (
    ExstickgeProxyQuel1SeProtoAdda,
    Quel1SeProtoAddaConfigSubsystem,
)
from quel_ic_config.quel_config_common import QUEL1_BOXTYPE_ALIAS, Quel1BoxType, Quel1ConfigOption, Quel1Feature
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
from quel_ic_config.thermistor import (
    Quel1NormalThermistor,
    Quel1PathSelectorThermistor,
    Quel1SeProtoExternalThermistor,
    Quel1SeProtoThermistor,
)

Quel1AnyConfigSubsystem = Union[
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
    Quel1NecConfigSubsystem,
    Quel1SeProto11ConfigSubsystem,
    Quel1SeProto8ConfigSubsystem,
    Quel1SeProtoAddaConfigSubsystem,
]

__version__ = "0.7.4"

__all__ = (
    "ExstickgeProxyQuel1",
    "ExstickgeProxyQuel1SeProtoAdda",
    "ExstickgeProxyQuel1SeProto8",
    "ExstickgeProxyQuel1SeProto11",
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
    "Quel1Feature",
    "QubeConfigSubsystem",
    "Quel1ConfigSubsystem",
    "QubeOuTypeAConfigSubsystem",
    "QubeOuTypeBConfigSubsystem",
    "Quel1TypeAConfigSubsystem",
    "Quel1TypeBConfigSubsystem",
    "Quel1NecConfigSubsystem",
    "Quel1SeProtoAddaConfigSubsystem",
    "Quel1SeProto8ConfigSubsystem",
    "Quel1SeProto11ConfigSubsystem",
    "Quel1SeProtoThermistor",
    "Quel1SeProtoExternalThermistor",
    "QubeRfSwitchRegs",
    "QubeSwitchRegNames",
    "QubeRfSwitchArray",
    "QUEL1_BOXTYPE_ALIAS",
    "RfSwitchArrayConfigHelper",
    "AbstractRfSwitchArrayMixin",
)
