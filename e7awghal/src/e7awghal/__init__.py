import importlib.metadata

from e7awghal.abstract_cap import AbstractCapCtrl, AbstractCapParam, AbstractCapSection, AbstractCapUnit
from e7awghal.awgctrl import AwgCtrl
from e7awghal.awgunit import AwgUnit, AwgUnitCtrlReg, AwgUnitErrorReg, AwgUnitStatusReg
from e7awghal.capctrl import CapCtrlBase, CapCtrlClassic, CapCtrlFeedback, CapCtrlSimpleMulti, CapCtrlStandard
from e7awghal.capdata import CapIqDataReader
from e7awghal.capparam import CapParam, CapParamSimple, CapSection
from e7awghal.capunit import CapUnit, CapUnitSimplified, CapUnitSwitchable
from e7awghal.clockcounterctrl import ClockcounterCtrl
from e7awghal.clockmaster_hal import ClockmasterAu200Hal
from e7awghal.clockmasterctrl import ClockmasterCtrl
from e7awghal.common_defs import E7awgCaptureDataError, E7awgHardwareError, E7awgMemoryError
from e7awghal.e7awg_packet import BoxLockDelegationError
from e7awghal.fwtype import E7FwAuxAttr, E7FwLifeStage, E7FwType
from e7awghal.quel1au50_hal import (
    AbstractQuel1Au50Hal,
    Quel1Au50SimplemultiClassicHal,
    Quel1Au50SimplemultiStandardHal,
    create_quel1au50hal,
)
from e7awghal.simplemulti import SimplemultiAwgTriggers, SimplemultiSequencer
from e7awghal.versionchecker import Quel1Au50HalVersionChecker
from e7awghal.wavedata import AwgParam, WaveChunk

__version__ = importlib.metadata.version("quel_ic_config")

__all__ = (
    "E7awgHardwareError",
    "E7awgMemoryError",
    "E7awgCaptureDataError",
    "E7FwType",
    "E7FwAuxAttr",
    "E7FwLifeStage",
    "WaveChunk",
    "AwgParam",
    "CapSection",
    "CapParam",
    "CapParamSimple",
    "CapIqDataReader",
    "AwgCtrl",
    "AwgUnitCtrlReg",
    "AwgUnitStatusReg",
    "AwgUnitErrorReg",
    "AwgUnit",
    "AbstractCapCtrl",
    "AbstractCapUnit",
    "AbstractCapParam",
    "AbstractCapSection",
    "CapCtrlBase",
    "CapCtrlSimpleMulti",
    "CapCtrlClassic",
    "CapCtrlStandard",
    "CapCtrlFeedback",
    "CapUnit",
    "CapUnitSwitchable",
    "CapUnitSimplified",
    "ClockcounterCtrl",
    "AbstractQuel1Au50Hal",
    "Quel1Au50SimplemultiClassicHal",
    "Quel1Au50SimplemultiStandardHal",
    "Quel1Au50HalVersionChecker",
    "create_quel1au50hal",
    "SimplemultiAwgTriggers",
    "SimplemultiSequencer",
    "ClockmasterCtrl",
    "ClockmasterAu200Hal",
    "BoxLockDelegationError",
)
