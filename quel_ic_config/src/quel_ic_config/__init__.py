import importlib.metadata

from quel_ic_config.ad5328 import Ad5328ConfigHelper, Ad5328RegNames, Ad5328Regs
from quel_ic_config.ad9082_v106 import Ad9082JesdParam, ChipTemperatures, NcoFtw
from quel_ic_config.adrf6780 import Adrf6780ConfigHelper, Adrf6780LoSideband, Adrf6780RegNames, Adrf6780Regs
from quel_ic_config.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config.e7workaround import (
    CaptureModule,
    CaptureUnit,
    E7FwType,
    E7LibBranch,
    detect_branch_of_library,
    resolve_hw_type,
)
from quel_ic_config.exstickge_coap_client import Quel1seBoard, get_exstickge_server_info
from quel_ic_config.generic_gpio import GenericGpioConfigHelper, GenericGpioRegNames, GenericGpioRegs
from quel_ic_config.linkupper import LinkupFpgaMxfe, LinkupStatistic, LinkupStatus
from quel_ic_config.lmx2594 import Lmx2594ConfigHelper, Lmx2594RegNames, Lmx2594Regs
from quel_ic_config.mixerboard_gpio import MixerboardGpioConfigHelper, MixerboardGpioRegNames, MixerboardGpioRegs
from quel_ic_config.pathselectorboard_gpio import (
    PathselectorboardGpioConfigHelper,
    PathselectorboardGpioRegNames,
    PathselectorboardGpioRegs,
)
from quel_ic_config.powerboard_pwm import PowerboardPwmConfigHelper, PowerboardPwmRegs, PowerboardPwmRegsName
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_box import Quel1Box, Quel1PortType
from quel_ic_config.quel1_box_intrinsic import Quel1BoxIntrinsic, Quel1LineType
from quel_ic_config.quel1_box_with_raw_wss import Quel1BoxWithRawWss
from quel_ic_config.quel1_config_subsystem import (
    ExstickgeSockClientQuel1,
    QubeConfigSubsystem,
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1ConfigSubsystem,
    Quel1NecConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
)
from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemRoot
from quel_ic_config.quel1_config_subsystem_tempctrl import Quel1seTempctrlState
from quel_ic_config.quel1_wave_subsystem import CaptureResults, CaptureReturnCode, Quel1WaveSubsystem
from quel_ic_config.quel1se_adda_config_subsystem import ExstickgeCoapClientAdda, Quel1seAddaConfigSubsystem
from quel_ic_config.quel1se_fujitsu11_config_subsystem import (
    ExstickgeCoapClientQuel1seFujitsu11,
    Quel1seFujitsu11TypeAConfigSubsystem,
    Quel1seFujitsu11TypeADebugConfigSubsystem,
    Quel1seFujitsu11TypeBConfigSubsystem,
    Quel1seFujitsu11TypeBDebugConfigSubsystem,
)
from quel_ic_config.quel1se_riken8_config_subsystem import (
    ExstickgeCoapClientQuel1seRiken8,
    ExstickgeCoapClientQuel1seRiken8Dev1,
    ExstickgeCoapClientQuel1seRiken8Dev2,
    Quel1seRiken8ConfigSubsystem,
    Quel1seRiken8DebugConfigSubsystem,
)
from quel_ic_config.quel_config_common import QUEL1_BOXTYPE_ALIAS, Quel1BoxType, Quel1ConfigOption, Quel1Feature
from quel_ic_config.quel_ic import (
    Ad5328,
    Ad9082V106,
    Adrf6780,
    GenericGpio,
    Lmx2594,
    MixerboardGpio,
    PathselectorboardGpio,
    PowerboardPwm,
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
    Quel1seExternalThermistor,
    Quel1seOnboardThermistor,
)

__version__ = importlib.metadata.version("quel_ic_config")

__all__ = (
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
    "ExstickgeCoapClientAdda",
    "ExstickgeCoapClientQuel1seRiken8Dev1",
    "ExstickgeCoapClientQuel1seRiken8Dev2",
    "ExstickgeCoapClientQuel1seRiken8",
    "ExstickgeCoapClientQuel1seFujitsu11",
    "ExstickgeSockClientQuel1",
    "GenericGpio",
    "GenericGpioConfigHelper",
    "GenericGpioRegs",
    "GenericGpioRegNames",
    "Lmx2594",
    "Lmx2594ConfigHelper",
    "Lmx2594Regs",
    "Lmx2594RegNames",
    "MixerboardGpioRegNames",
    "MixerboardGpioRegs",
    "MixerboardGpioConfigHelper",
    "MixerboardGpio",
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
    "Quel1seAddaConfigSubsystem",
    "Quel1seOnboardThermistor",
    "Quel1seExternalThermistor",
    "Quel1seRiken8ConfigSubsystem",
    "Quel1seRiken8DebugConfigSubsystem",
    "Quel1seFujitsu11TypeAConfigSubsystem",
    "Quel1seFujitsu11TypeADebugConfigSubsystem",
    "Quel1seFujitsu11TypeBConfigSubsystem",
    "Quel1seFujitsu11TypeBDebugConfigSubsystem",
    "QubeRfSwitchRegs",
    "QubeSwitchRegNames",
    "QubeRfSwitchArray",
    "QUEL1_BOXTYPE_ALIAS",
    "RfSwitchArrayConfigHelper",
    "AbstractRfSwitchArrayMixin",
    "Quel1seBoard",
    "Quel1seTempctrlState",
    "PathselectorboardGpioConfigHelper",
    "PathselectorboardGpioRegNames",
    "PathselectorboardGpioRegs",
    "PathselectorboardGpio",
    "PowerboardPwmConfigHelper",
    "PowerboardPwmRegsName",
    "PowerboardPwmRegs",
    "PowerboardPwm",
    "Quel1Box",
    "Quel1BoxWithRawWss",
    "Quel1PortType",
    "Quel1LineType",
    "Quel1BoxIntrinsic",
    "get_exstickge_server_info",
    "Quel1WaveSubsystem",
    "E7FwType",
    "E7LibBranch",
    "detect_branch_of_library",
    "resolve_hw_type",
    "CaptureUnit",
    "CaptureModule",
    "CaptureReturnCode",
    "CaptureResults",
    "Quel1E7ResourceMapper",
    "LinkupStatistic",
    "LinkupStatus",
    "LinkupFpgaMxfe",
)
