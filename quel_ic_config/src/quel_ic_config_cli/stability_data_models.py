import logging
from abc import ABCMeta
from typing import ClassVar, Final, Optional, Type, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from quel_ic_config import (
    Quel1BoxType,
    Quel1NormalThermistor,
    Quel1PathSelectorThermistor,
    Quel1seFujitsu11TypeAConfigSubsystem,
    Quel1seFujitsu11TypeBConfigSubsystem,
    Quel1seRiken8ConfigSubsystem,
    Quel1Thermistor,
)

logger = logging.getLogger(__name__)

QUEL1_THERMISTORS: dict[int, Quel1Thermistor] = {
    0: Quel1NormalThermistor("adrf6780_0_0"),
    1: Quel1NormalThermistor("adrf6780_0_1"),
    2: Quel1NormalThermistor("adrf6780_0_2"),
    3: Quel1NormalThermistor("adrf6780_0_3"),
    4: Quel1NormalThermistor("adrf6780_1_0"),
    5: Quel1NormalThermistor("adrf6780_1_1"),
    6: Quel1NormalThermistor("adrf6780_1_2"),
    7: Quel1NormalThermistor("adrf6780_1_3"),
    8: Quel1NormalThermistor("ad9082_0"),
    9: Quel1NormalThermistor("ad9082_1"),
    10: Quel1NormalThermistor("lmx2594_0_0"),
    11: Quel1NormalThermistor("lmx2594_0_1"),
    12: Quel1NormalThermistor("lmx2594_0_2"),
    13: Quel1NormalThermistor("lmx2594_0_3"),
    14: Quel1NormalThermistor("lmx2594_1_0"),
    15: Quel1NormalThermistor("lmx2594_1_1"),
    16: Quel1NormalThermistor("lmx2594_1_2"),
    17: Quel1NormalThermistor("lmx2594_1_3"),
    18: Quel1NormalThermistor("lmx2594_12g_0"),
    19: Quel1NormalThermistor("lmx2594_12g_1"),
    20: Quel1NormalThermistor("adclk950"),
    21: Quel1NormalThermistor("adclk925"),
    22: Quel1NormalThermistor("rx_0_0"),
    23: Quel1NormalThermistor("rx_0_1"),
    24: Quel1NormalThermistor("rx_1_0"),
    25: Quel1NormalThermistor("rx_1_1"),
    26: Quel1PathSelectorThermistor("ps_0"),
    27: Quel1PathSelectorThermistor("ps_1"),
}

QUEL1_ACTUATORS: dict[str, tuple[str, int]] = {
    "adrf6780_0_0": ("peltier", 0),
    "adrf6780_0_1": ("peltier", 1),
    "adrf6780_0_2": ("peltier", 2),
    "adrf6780_0_3": ("peltier", 3),
    "adrf6780_1_0": ("peltier", 4),
    "adrf6780_1_1": ("peltier", 5),
    "adrf6780_1_2": ("peltier", 6),
    "adrf6780_1_3": ("peltier", 7),
    "ad9082_0": ("peltier", 8),
    "ad9082_1": ("peltier", 9),
    "lmx2594_0_0": ("peltier", 10),
    "lmx2594_0_1": ("peltier", 11),
    "lmx2594_0_2": ("peltier", 12),
    "lmx2594_0_3": ("peltier", 13),
    "lmx2594_1_0": ("peltier", 14),
    "lmx2594_1_1": ("peltier", 15),
    "lmx2594_1_2": ("peltier", 16),
    "lmx2594_1_3": ("peltier", 17),
    "lmx2594_12g_0": ("peltier", 18),
    "lmx2594_12g_1": ("peltier", 19),
    "adclk950": ("peltier", 20),
    "adclk925": ("peltier", 21),
    "rx_0_0": ("peltier", 22),  # Notes: inverted peltier
    "rx_0_1": ("peltier", 23),  # Notes: inverted peltier
    "rx_1_0": ("peltier", 24),  # Notes: inverted peltier
    "rx_1_1": ("peltier", 25),  # Notes: inverted peltier
    "ps_0": ("heater", 26),
    "ps_1": ("heater", 27),
}

_MXFE_THERMISTORS: Final[set[str]] = {"mxfe0_max", "mxfe0_min", "mxfe1_max", "mxfe1_min"}


class QuelDataModel(BaseModel, extra="forbid", metaclass=ABCMeta):
    time_from_start: float = Field(ge=0.0)


class WaveStatistics(QuelDataModel):
    output_port: str = Field()
    input_port: str = Field()
    power_mean: float = Field(ge=0, le=32767)
    power_std: float = Field(ge=0, le=32767)
    angle_mean: float = Field(
        ge=-180.0, lt=188.2
    )  # (pi + (pi-3.0)) * 180/pi = 188.11 see WaveStabilityMeasurement.phase_stat in wave_stability_check.py
    angle_std: float = Field(ge=0, le=180.0)  # max standard deviation in the range [-pi,pi] is (pi-(-pi))/2 = pi
    angle_deltamax: float = Field(ge=0, le=360.0)


class TempCtrlStatus(QuelDataModel):
    location_name: str = Field()
    temperature: float = Field(ge=0, lt=120)
    # Notes: for thermistors that doesn't have its proximal thermal actuator, actuator_type should be None.
    actuator_type: Optional[str] = Field(default=None)
    # Notes: for locations of which actuator_type is None, actuator_val should be None, either.
    actuator_val: Optional[float] = Field(default=None)

    _LOCATION_NAMES: ClassVar[set[str]] = set()
    _VALID_ACTUATOR_TYPES: ClassVar[set[Union[str, None]]] = set()
    _CORRESPONDING_BOXTYPES: ClassVar[set[Quel1BoxType]] = set()

    @field_validator("location_name")
    def validate_location(cls, val):
        if val not in cls._LOCATION_NAMES:
            raise ValueError(f"no such thermistor or actuator location {val}")
        return val

    @field_validator("actuator_type")
    def validate_actuator_type(cls, val):
        if val not in cls._VALID_ACTUATOR_TYPES:
            raise ValueError(
                f"actuator type must be either of {', '.join([str(at) for at in cls._VALID_ACTUATOR_TYPES])}"
            )
        return val

    @classmethod
    def is_corresponding_boxtype(cls, boxtype: Quel1BoxType) -> bool:
        return boxtype in cls._CORRESPONDING_BOXTYPES


class Quel1TempCtrlStatus(TempCtrlStatus):
    _LOCATION_NAMES: ClassVar[set[str]] = set(QUEL1_ACTUATORS.keys())
    _VALID_ACTUATOR_TYPES: ClassVar[set[Union[str, None]]] = {"heater", "peltier"}

    # Notes: be sure that QuBE doesn't have CMOD-35T with uart_monitor firmware.
    _CORRESPONDING_BOXTYPES: ClassVar[set[Quel1BoxType]] = {
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuEL1_TypeB,
        Quel1BoxType.QuEL1_NEC,
        Quel1BoxType.QuEL1_NTT,
    }

    @model_validator(mode="after")
    @classmethod
    def validate_actuator_val(cls, model):
        if model.actuator_type == "peltier":
            if model.actuator_val is None or not (1 <= model.actuator_val <= 200):
                raise ValueError("actuator value for peltiers must be within 1-200")
        elif model.actuator_type == "heater":
            if model.actuator_val is None or not (1 <= model.actuator_val <= 300):
                raise ValueError("actuator value for heaters must be within 1-300")
        else:
            # Notes: Notes: Never happens
            raise AssertionError("never happens")

        return model


class Quel1seTempCtrlStatus(TempCtrlStatus):
    _VALID_ACTUATOR_TYPES: ClassVar[set[Union[str, None]]] = {"fan", "heater", None}

    @field_validator("actuator_val")
    def validate_actuator_val(cls, val):
        if val and not (0.0 <= val <= 1.0):
            raise ValueError("actuator value must be in the range btwn 0.0 and 1.0")
        return val


class Quel1seRiken8TempCtrlStatus(Quel1seTempCtrlStatus):
    _LOCATION_NAMES: ClassVar[set[str]] = {
        th.name for th in Quel1seRiken8ConfigSubsystem._THERMISTORS.values()
    } | _MXFE_THERMISTORS
    _CORRESPONDING_BOXTYPES: ClassVar[set[Quel1BoxType]] = {Quel1BoxType.QuEL1SE_RIKEN8, Quel1BoxType.QuEL1SE_RIKEN8DBG}


class Quel1seFujitsu11TypeATempCtrlStatus(Quel1seTempCtrlStatus):
    _LOCATION_NAMES: ClassVar[set[str]] = {
        thermistor.name for thermistor in Quel1seFujitsu11TypeAConfigSubsystem._THERMISTORS.values()
    } | _MXFE_THERMISTORS
    _CORRESPONDING_BOXTYPES: ClassVar[set[Quel1BoxType]] = {
        Quel1BoxType.QuEL1SE_FUJITSU11_TypeA,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA,
    }


class Quel1seFujitsu11TypeBTempCtrlStatus(Quel1seTempCtrlStatus):
    _LOCATION_NAMES: ClassVar[set[str]] = {
        thermistor.name for thermistor in Quel1seFujitsu11TypeBConfigSubsystem._THERMISTORS.values()
    } | _MXFE_THERMISTORS
    _CORRESPONDING_BOXTYPES: ClassVar[set[Quel1BoxType]] = {
        Quel1BoxType.QuEL1SE_FUJITSU11_TypeB,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeB,
    }


def _get_all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in _get_all_subclasses(c)])


def get_tempctrlstatus_from_boxtype(boxtype: Quel1BoxType) -> Type[TempCtrlStatus]:
    for tcscls in _get_all_subclasses(TempCtrlStatus):
        if tcscls.is_corresponding_boxtype(boxtype):
            return tcscls
    raise ValueError(f"unsupported boxtype: {Quel1BoxType.tostr(boxtype)}")
