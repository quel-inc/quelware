import collections.abc
import json
import logging
import time
from abc import abstractmethod
from enum import Enum, IntEnum
from typing import Any, Collection, Dict, Final, List, Mapping, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError, field_validator
from typing_extensions import Self

import adi_ad9081_v106 as ad9081
from adi_ad9081_v106 import ChipTemperatures, CmsError, Device
from adi_ad9081_v106 import NcoFtw as _NcoFtw
from quel_ic_config.abstract_ic import AbstractIcMixin
from quel_ic_config.ad9082_nco import AbstractNcoFtw

logger = logging.getLogger(__name__)


def update_mapping_recursive(d: Dict[str, Any], u: Mapping[str, Any]):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_mapping_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class NcoFtw(AbstractNcoFtw):
    @classmethod
    def from_ftw(cls, ftw: _NcoFtw) -> Self:
        enable_fraction = ftw.modulus_a != 0
        x = cls._decode_u48_as_s48(ftw.ftw)
        if enable_fraction:
            b = cls._decode_u48_as_s48(ftw.delta_b)
            a = ftw.modulus_a
        else:
            b = 0
            a = 1
        return cls(ftw=x, delta_b=b, modulus_a=a, enable_fraction=enable_fraction)

    def to_ftw(self) -> _NcoFtw:
        obj = _NcoFtw()
        obj.ftw = self._encode_s48_as_u48(self.ftw)
        if self.enable_fraction:
            obj.delta_b = self._encode_s48_as_u48(self.delta_b)
            obj.modulus_a = self.modulus_a
        else:
            obj.delta_b = 0
            obj.modulus_a = 0
        return obj


class NoExtraBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrozenRootModel(RootModel):
    model_config = ConfigDict(frozen=True)


class FrozenSequenceRootModel(FrozenRootModel):
    root: Tuple[Any, ...] = ()

    def __len__(self):
        return len(self.root)

    def __getitem__(self, key):
        return self.root[key]

    def __iter__(self):
        return iter(self.root)


class Ad9082JesdParam(NoExtraBaseModel):
    l: int
    f: int
    m: int
    s: int
    hd: int
    k: int
    n: int
    np: int
    cf: int
    cs: int
    did: int
    bid: int
    lid0: int
    subclass: int
    scr: int
    duallink: int
    jesdv: int
    mode_id: int
    mode_c2r_en: int
    mode_s_sel: int

    def as_cpptype(self) -> ad9081.AdiCmsJesdParam:
        d = ad9081.AdiCmsJesdParam()
        d.l = self.l  # noqa: E741
        d.f = self.f
        d.m = self.m
        d.s = self.s
        d.hd = self.hd
        d.k = self.k
        d.n = self.n
        d.np = self.np
        d.cf = self.cf
        d.cs = self.cs
        d.did = self.did
        d.bid = self.bid
        d.lid0 = self.lid0
        d.subclass = self.subclass
        d.scr = self.scr
        d.duallink = self.duallink
        d.jesdv = self.jesdv
        d.mode_id = self.mode_id
        d.mode_c2r_en = self.mode_c2r_en
        d.mode_s_sel = self.mode_s_sel
        return d


class Ad9082SpiPinConfigEnum(str, Enum):
    SPI_SDO = "SDO"
    SPI_SDIO = "SDIO"

    def as_cpptype(self) -> ad9081.CmsSpiSdoConfig:
        return _Ad9082SpiPinConfigEnum_cpptype_map[self]


_Ad9082SpiPinConfigEnum_cpptype_map: Dict[str, ad9081.CmsSpiSdoConfig] = {
    Ad9082SpiPinConfigEnum.SPI_SDO: ad9081.SPI_SDO,
    Ad9082SpiPinConfigEnum.SPI_SDIO: ad9081.SPI_SDIO,
}


class Ad9082SpiMsbConfigEnum(str, Enum):
    SPI_MSB_FIRST = "FIRST"
    SPI_MSB_LAST = "LAST"

    def as_cpptype(self) -> ad9081.CmsSpiMsbConfig:
        return _Ad9082SpiMsbConfigEnum_cpptype_map[self]


_Ad9082SpiMsbConfigEnum_cpptype_map: Dict[str, ad9081.CmsSpiMsbConfig] = {
    Ad9082SpiMsbConfigEnum.SPI_MSB_FIRST: ad9081.SPI_MSB_FIRST,
    Ad9082SpiMsbConfigEnum.SPI_MSB_LAST: ad9081.SPI_MSB_LAST,
}


class Ad9082SpiAddrNextConfigEnum(str, Enum):
    SPI_ADDR_INC = "INC"
    SPI_ADDR_DEC = "DEC"

    def as_cpptype(self) -> ad9081.CmsSpiAddrInc:
        return _Ad9082SpiAddrNextConfigEnum_cpptype_map[self]


_Ad9082SpiAddrNextConfigEnum_cpptype_map: Dict[str, ad9081.CmsSpiAddrInc] = {
    Ad9082SpiAddrNextConfigEnum.SPI_ADDR_INC: ad9081.SPI_ADDR_INC_AUTO,
    Ad9082SpiAddrNextConfigEnum.SPI_ADDR_DEC: ad9081.SPI_ADDR_DEC_AUTO,
}


class Ad9082SpiConfig(NoExtraBaseModel):
    pin: Ad9082SpiPinConfigEnum
    msb: Ad9082SpiMsbConfigEnum
    addr_next: Ad9082SpiAddrNextConfigEnum


class Ad9082ClockConfig(NoExtraBaseModel):
    ref: int
    dac: int
    adc: int


class _Ad9082SerSwingConfigEnum(IntEnum):
    SWING500 = 500
    SWING750 = 750
    SWING850 = 850
    SWING1000 = 1000

    def as_cpptype(self) -> ad9081.SerSwing:
        return _Ad9082SerSwingConfigEnum_cpptype_map[self]


_Ad9082SerSwingConfigEnum_cpptype_map: Dict[int, ad9081.SerSwing] = {
    _Ad9082SerSwingConfigEnum.SWING500: ad9081.SER_SWING_500,
    _Ad9082SerSwingConfigEnum.SWING750: ad9081.SER_SWING_750,
    _Ad9082SerSwingConfigEnum.SWING850: ad9081.SER_SWING_850,
    _Ad9082SerSwingConfigEnum.SWING1000: ad9081.SER_SWING_1000,
}


class _Ad9082SerPreEmpConfigEnum(IntEnum):
    PREEMP0 = 0
    PREEMP3 = 3
    PREEMP6 = 6

    def as_cpptype(self) -> ad9081.SerPreEmp:
        return _Ad9082SerPreEmpConfigEnum_cpptype_map[self]


_Ad9082SerPreEmpConfigEnum_cpptype_map: Dict[int, ad9081.SerPreEmp] = {
    _Ad9082SerPreEmpConfigEnum.PREEMP0: ad9081.SER_PRE_EMP_0DB,
    _Ad9082SerPreEmpConfigEnum.PREEMP3: ad9081.SER_PRE_EMP_3DB,
    _Ad9082SerPreEmpConfigEnum.PREEMP6: ad9081.SER_PRE_EMP_6DB,
}


class _Ad9082SerPostEmpConfigEnum(IntEnum):
    POSTEMP0 = 0
    POSTEMP3 = 3
    POSTEMP6 = 6
    POSTEMP9 = 9
    POSTEMP12 = 12

    def as_cpptype(self) -> ad9081.SerPostEmp:
        return _Ad9082SerPostEmpConfigEnum_cpptype_map[self]


_Ad9082SerPostEmpConfigEnum_cpptype_map: Dict[int, ad9081.SerPostEmp] = {
    _Ad9082SerPostEmpConfigEnum.POSTEMP0: ad9081.SER_POST_EMP_0DB,
    _Ad9082SerPostEmpConfigEnum.POSTEMP3: ad9081.SER_POST_EMP_3DB,
    _Ad9082SerPostEmpConfigEnum.POSTEMP6: ad9081.SER_POST_EMP_6DB,
    _Ad9082SerPostEmpConfigEnum.POSTEMP9: ad9081.SER_POST_EMP_9DB,
    _Ad9082SerPostEmpConfigEnum.POSTEMP12: ad9081.SER_POST_EMP_12DB,
}


class _Ad9082LaneConfigEnum(IntEnum):
    LANE0 = 0
    LANE1 = 1
    LANE2 = 2
    LANE3 = 3
    LANE4 = 4
    LANE5 = 5
    LANE6 = 6
    LANE7 = 7


class _Ad9082LaneMappingConfig(FrozenSequenceRootModel):
    root: Tuple[
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
    ]

    def as_cpptype(self, d: Union[None, NDArray]) -> NDArray:
        if d is None:
            d = np.zeros(8, np.uint8)
        for i in range(8):
            d[i] = self.root[i]
        return d


class _Ad9082LaneMask(FrozenSequenceRootModel):
    root: Tuple[bool, bool, bool, bool, bool, bool, bool, bool]

    def as_cpptype(self) -> int:
        u = 0
        for i, v in enumerate(self.root):
            u |= int(v) << i
        return u


class _Ad9082LaneCtleFilterConfig(FrozenSequenceRootModel):
    root: Tuple[int, int, int, int, int, int, int, int]

    def as_cpptype(self, d: Union[None, NDArray]) -> NDArray:
        if d is None:
            d = np.zeros(8, np.uint8)
        for i in range(8):
            d[i] = self.root[i]
        return d


class Ad9082DesConfig(NoExtraBaseModel):
    boost: _Ad9082LaneMask
    invert: _Ad9082LaneMask
    ctle_filter: _Ad9082LaneCtleFilterConfig
    lane_mappings: Tuple[_Ad9082LaneMappingConfig, _Ad9082LaneMappingConfig]


class Ad9082SerConfig(NoExtraBaseModel):
    invert: _Ad9082LaneMask
    swing: Tuple[
        _Ad9082SerSwingConfigEnum,
        _Ad9082SerSwingConfigEnum,
        _Ad9082SerSwingConfigEnum,
        _Ad9082SerSwingConfigEnum,
        _Ad9082SerSwingConfigEnum,
        _Ad9082SerSwingConfigEnum,
        _Ad9082SerSwingConfigEnum,
        _Ad9082SerSwingConfigEnum,
    ]
    pre_emp: Tuple[
        _Ad9082SerPreEmpConfigEnum,
        _Ad9082SerPreEmpConfigEnum,
        _Ad9082SerPreEmpConfigEnum,
        _Ad9082SerPreEmpConfigEnum,
        _Ad9082SerPreEmpConfigEnum,
        _Ad9082SerPreEmpConfigEnum,
        _Ad9082SerPreEmpConfigEnum,
        _Ad9082SerPreEmpConfigEnum,
    ]
    post_emp: Tuple[
        _Ad9082SerPostEmpConfigEnum,
        _Ad9082SerPostEmpConfigEnum,
        _Ad9082SerPostEmpConfigEnum,
        _Ad9082SerPostEmpConfigEnum,
        _Ad9082SerPostEmpConfigEnum,
        _Ad9082SerPostEmpConfigEnum,
        _Ad9082SerPostEmpConfigEnum,
        _Ad9082SerPostEmpConfigEnum,
    ]
    lane_mappings: Tuple[_Ad9082LaneMappingConfig, _Ad9082LaneMappingConfig]


class Ad9082SerdesConfig(NoExtraBaseModel):
    ser: Ad9082SerConfig
    des: Ad9082DesConfig


class _Ad9082FducRateConfigEnum(IntEnum):
    FDUC1 = 1
    FDUC2 = 2
    FDUC3 = 3
    FDUC4 = 4
    FDUC6 = 6
    FDUC8 = 8


class _Ad9082CducRateConfigEnum(IntEnum):
    CDUC12 = 12
    CDUC8 = 8
    CDUC6 = 6
    CDUC4 = 4
    CDUC2 = 2
    CDUC1 = 1


class Ad9082ChannelAssignConfig(NoExtraBaseModel):
    dac0: List[_Ad9082LaneConfigEnum]
    dac1: List[_Ad9082LaneConfigEnum]
    dac2: List[_Ad9082LaneConfigEnum]
    dac3: List[_Ad9082LaneConfigEnum]

    @staticmethod
    def _as_cpptype_sub(chl: Sequence[int]) -> int:
        v: int = 0
        for ch in chl:
            v |= 1 << ch
        return v

    def as_cpptype(self) -> List[int]:
        return [
            self._as_cpptype_sub(self.dac0),
            self._as_cpptype_sub(self.dac1),
            self._as_cpptype_sub(self.dac2),
            self._as_cpptype_sub(self.dac3),
        ]

    @staticmethod
    def dac_order(idx: int, v: List[int]) -> List[int]:
        u = sorted(v, reverse=True)
        if v != u:
            logger.warning(f"param.dac.channel_assign[{idx}] is not sorted in reversed order, fix it")
        return u

    @field_validator("dac0", mode="before")
    def dac0_order(cls, v: List[int]) -> List[int]:
        return cls.dac_order(0, v)

    @field_validator("dac1", mode="before")
    def dac1_order(cls, v: List[int]) -> List[int]:
        return cls.dac_order(1, v)

    @field_validator("dac2", mode="before")
    def dac2_order(cls, v: List[int]) -> List[int]:
        return cls.dac_order(2, v)

    @field_validator("dac3", mode="before")
    def dac3_order(cls, v: List[int]) -> List[int]:
        return cls.dac_order(3, v)


class Ad9082InterpolationRateConfig(NoExtraBaseModel):
    channel: _Ad9082FducRateConfigEnum
    main: _Ad9082CducRateConfigEnum


class Ad9082ShiftFreqConfig(NoExtraBaseModel):
    channel: Tuple[int, int, int, int, int, int, int, int]
    main: Tuple[int, int, int, int]


class Ad9082DacConfig(NoExtraBaseModel):
    jesd204: Ad9082JesdParam
    lane_xbar: Tuple[
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
    ]
    channel_assign: Ad9082ChannelAssignConfig
    interpolation_rate: Ad9082InterpolationRateConfig
    shift_freq: Ad9082ShiftFreqConfig
    fullscale_current: Tuple[int, int, int, int]


class _Ad9082VirtualConverterConfigEnum(IntEnum):
    VC0 = 0
    VC1 = 1
    VC2 = 2
    VC3 = 3
    VC4 = 4
    VC5 = 5
    VC6 = 6
    VC7 = 7
    VC8 = 8
    VC9 = 9
    VC10 = 10
    VC11 = 11
    VC12 = 12
    VC13 = 13
    VC14 = 14
    VC15 = 15


class _Ad9082ConvSel(FrozenSequenceRootModel):
    root: Tuple[
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
        _Ad9082VirtualConverterConfigEnum,
    ]

    def as_cpptype(self) -> ad9081.JtxConvSel:
        d = ad9081.JtxConvSel()
        d.virtual_converter0_index = self[0]
        d.virtual_converter1_index = self[1]
        d.virtual_converter2_index = self[2]
        d.virtual_converter3_index = self[3]
        d.virtual_converter4_index = self[4]
        d.virtual_converter5_index = self[5]
        d.virtual_converter6_index = self[6]
        d.virtual_converter7_index = self[7]
        d.virtual_converter8_index = self[8]
        d.virtual_converter9_index = self[9]
        d.virtual_convertera_index = self[10]
        d.virtual_converterb_index = self[11]
        d.virtual_converterc_index = self[12]
        d.virtual_converterd_index = self[13]
        d.virtual_convertere_index = self[14]
        d.virtual_converterf_index = self[15]
        return d


class _Ad9082FddcRateConfigEnum(IntEnum):
    FDDC1 = 1
    FDDC2 = 2
    FDDC3 = 3
    FDDC4 = 4
    FDDC6 = 6
    FDDC8 = 8
    FDDC12 = 12
    FDDC16 = 16
    FDDC24 = 24

    def as_cpptype(self) -> ad9081.AdcFineDdcDcm:
        return _Ad9082FddcRateConfigEnum_cpptype_map[self]


_Ad9082FddcRateConfigEnum_cpptype_map: Dict[int, ad9081.AdcFineDdcDcm] = {
    _Ad9082FddcRateConfigEnum.FDDC1: ad9081.ADC_FDDC_DCM_1,
    _Ad9082FddcRateConfigEnum.FDDC2: ad9081.ADC_FDDC_DCM_2,
    _Ad9082FddcRateConfigEnum.FDDC3: ad9081.ADC_FDDC_DCM_3,
    _Ad9082FddcRateConfigEnum.FDDC6: ad9081.ADC_FDDC_DCM_4,
    _Ad9082FddcRateConfigEnum.FDDC8: ad9081.ADC_FDDC_DCM_6,
    _Ad9082FddcRateConfigEnum.FDDC12: ad9081.ADC_FDDC_DCM_8,
    _Ad9082FddcRateConfigEnum.FDDC16: ad9081.ADC_FDDC_DCM_12,
    _Ad9082FddcRateConfigEnum.FDDC24: ad9081.ADC_FDDC_DCM_24,
}


class _Ad9082DecimationRateChannelConfig(FrozenSequenceRootModel):
    root: Tuple[
        _Ad9082FddcRateConfigEnum,
        _Ad9082FddcRateConfigEnum,
        _Ad9082FddcRateConfigEnum,
        _Ad9082FddcRateConfigEnum,
        _Ad9082FddcRateConfigEnum,
        _Ad9082FddcRateConfigEnum,
        _Ad9082FddcRateConfigEnum,
        _Ad9082FddcRateConfigEnum,
    ]

    def as_cpptype(self) -> List[int]:
        return [int(x.as_cpptype()) for x in self.root]


class _Ad9082CddcRateConfigEnum(IntEnum):
    CDDC1 = 1
    CDDC2 = 2
    CDDC3 = 3
    CDDC4 = 4
    CDDC6 = 6
    CDDC8 = 8
    CDDC9 = 9
    CDDC12 = 12
    CDDC16 = 16
    CDDC18 = 18
    CDDC24 = 24
    CDDC36 = 36

    def as_cpptype(self) -> ad9081.AdcCoarseDdcDcm:
        return _Ad9082CddcRateConfigEnum_cpptype_map[self]


_Ad9082CddcRateConfigEnum_cpptype_map: Dict[int, ad9081.AdcCoarseDdcDcm] = {
    _Ad9082CddcRateConfigEnum.CDDC1: ad9081.ADC_CDDC_DCM_1,
    _Ad9082CddcRateConfigEnum.CDDC2: ad9081.ADC_CDDC_DCM_2,
    _Ad9082CddcRateConfigEnum.CDDC3: ad9081.ADC_CDDC_DCM_3,
    _Ad9082CddcRateConfigEnum.CDDC4: ad9081.ADC_CDDC_DCM_4,
    _Ad9082CddcRateConfigEnum.CDDC6: ad9081.ADC_CDDC_DCM_6,
    _Ad9082CddcRateConfigEnum.CDDC8: ad9081.ADC_CDDC_DCM_8,
    _Ad9082CddcRateConfigEnum.CDDC9: ad9081.ADC_CDDC_DCM_9,
    _Ad9082CddcRateConfigEnum.CDDC12: ad9081.ADC_CDDC_DCM_12,
    _Ad9082CddcRateConfigEnum.CDDC16: ad9081.ADC_CDDC_DCM_16,
    _Ad9082CddcRateConfigEnum.CDDC18: ad9081.ADC_CDDC_DCM_18,
    _Ad9082CddcRateConfigEnum.CDDC24: ad9081.ADC_CDDC_DCM_24,
    _Ad9082CddcRateConfigEnum.CDDC36: ad9081.ADC_CDDC_DCM_36,
}


class _Ad9082DecimationRateMainConfig(FrozenSequenceRootModel):
    root: Tuple[
        _Ad9082CddcRateConfigEnum, _Ad9082CddcRateConfigEnum, _Ad9082CddcRateConfigEnum, _Ad9082CddcRateConfigEnum
    ]

    def as_cpptype(self) -> List[int]:
        return [int(x.as_cpptype()) for x in self.root]


class Ad9082DecimationRateConfig(NoExtraBaseModel):
    channel: _Ad9082DecimationRateChannelConfig
    main: _Ad9082DecimationRateMainConfig


class Ad9082ComplexToRealEnableConfig(NoExtraBaseModel):
    channel: Tuple[bool, bool, bool, bool, bool, bool, bool, bool]
    main: Tuple[bool, bool, bool, bool]


class Ad9082AdcConfig(NoExtraBaseModel):
    jesd204: Tuple[Ad9082JesdParam, Ad9082JesdParam]
    lane_xbar: Tuple[
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
        _Ad9082LaneConfigEnum,
    ]
    converter_mappings: Tuple[_Ad9082ConvSel, _Ad9082ConvSel]
    decimation_rate: Ad9082DecimationRateConfig
    shift_freq: Ad9082ShiftFreqConfig
    c2r_enable: Ad9082ComplexToRealEnableConfig


class Ad9082Config(NoExtraBaseModel):
    spi: Ad9082SpiConfig
    clock: Ad9082ClockConfig
    serdes: Ad9082SerdesConfig
    dac: Ad9082DacConfig
    adc: Ad9082AdcConfig


class Ad9082V106Mixin(AbstractIcMixin):
    """wrapping adi_ad9081_v106 APIs to make them Pythonic"""

    WORKAROUND_FREQ_DISCOUNT_RATE = 0.97

    def __init__(self, name: str, param_in: Union[str, Dict[str, Any], Ad9082Config]):
        super().__init__(name)
        # Notes: self.param is self.loaded into self.device at initialization().
        self.param: Ad9082Config = self.load_settings(param_in)
        self.device: Final[Device] = Device()
        self.device.callback_set(self._read_reg_cb, self._write_reg_cb, self._delay_us_cb, self._log_write_cb)
        # Notes: caches
        self._interp_cache: Union[Tuple[int, int], None] = None
        self._fduc_map_cache: Union[Tuple[Tuple[int, ...], ...], None] = None

    def load_settings(self, param_in: Union[str, Dict[str, Any], Ad9082Config]):
        if isinstance(param_in, str):
            param: Ad9082Config = Ad9082Config.model_validate(json.loads(param_in))
        elif isinstance(param_in, dict):
            param = Ad9082Config.model_validate(param_in)
        elif isinstance(param_in, Ad9082Config):
            param = param_in
        else:
            raise AssertionError
        return param

    def update_settings(self, param_update: Dict[str, Any]):
        # Notes: this method is provided for development purpose. should not be used in nominal operations.
        p = self.param.model_dump()
        update_mapping_recursive(p, param_update)
        self.param = self.load_settings(p)

    def __del__(self):
        self.device.callback_unset()

    def read_reg(self, addr: int) -> int:
        return self.hal_reg_get(addr)

    def write_reg(self, addr: int, data: int) -> None:
        self.hal_reg_set(addr, data)

    def dump_regs(self) -> Dict[int, int]:
        raise NotImplementedError

    def _read_u48(self, addr) -> int:
        ftw_ftw = bytearray(6)
        for i in range(6):
            ftw_ftw[i] = self.read_reg(addr + i)
        return int.from_bytes(ftw_ftw, byteorder="little", signed=False)

    @abstractmethod
    def _read_reg_cb(self, address: int) -> Tuple[bool, int]:
        pass

    @abstractmethod
    def _write_reg_cb(self, address: int, value: int) -> Tuple[bool]:
        pass

    def _delay_us_cb(self, us: int) -> Tuple[bool]:
        logger.debug(f"delay {us}us")
        time.sleep(us * 1e-6)
        return (True,)

    def _log_write_cb(self, level: int, msg: str) -> Tuple[bool]:
        if level & 0x0001:
            logger.warning(msg)  # error-level log will be generated at the higher layer
        elif level & 0x0002:
            logger.warning(msg)
        elif level & 0x0004:
            logger.info(msg)
        else:
            logger.debug(msg)
        return (True,)

    def device_reset(self) -> None:
        logger.info(f"reset {self.name}")
        rc = ad9081.device_reset(self.device, ad9081.SOFT_RESET)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

    def device_init(self) -> None:
        logger.info(f"init {self.name}")
        rc = ad9081.device_init(self.device)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

    def hal_reg_set(self, addr: int, data: int) -> None:
        addrdata = ad9081.AddrData(addr, data)
        rc = ad9081.hal_reg_set(self.device, addrdata)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

    def hal_reg_get(self, addr: int) -> int:
        addrdata = ad9081.AddrData(addr)
        rc = ad9081.hal_reg_get(self.device, addrdata)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)
        return addrdata.data & 0xFF

    def device_chip_id_get(self) -> ad9081.CmsChipId:
        chip_id = ad9081.CmsChipId()
        rc = ad9081.device_chip_id_get(self.device, chip_id)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)
        return chip_id

    def device_clk_config_set(self, dac_clk_hz: int, adc_clk_hz: int, dev_ref_clk_hz: int) -> None:
        logger.info(
            f"clock_config {self.name}: dac_clk = {dac_clk_hz}Hz, "
            f"adc_clk = {adc_clk_hz}Hz, ref_clk = {dev_ref_clk_hz}Hz"
        )
        rc = ad9081.device_clk_config_set(self.device, dac_clk_hz, adc_clk_hz, dev_ref_clk_hz)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

    # Notes: it is possible to refer to the contents of param_spi, param_d, and so on without taking it as an argument.
    #        On the other hand, this relative access of structured data increases the robustness of the code against
    #        the possible modifications of the data structure (e.g., version update of ADI library.)
    def _set_spi_settings(self, param_spi: Ad9082SpiConfig) -> None:
        spi_pin = param_spi.pin.as_cpptype()
        spi_msb = param_spi.msb.as_cpptype()
        spi_addr_inc = param_spi.addr_next.as_cpptype()
        logger.debug(f"spi_pin: {spi_pin}, spi_msb: {spi_msb}, spi_addr_inc: {spi_addr_inc}")
        self.device.spi_conf_set(spi_pin, spi_msb, spi_addr_inc)

    def _set_des_settings(self, param_d: Ad9082DesConfig) -> None:
        self.device.serdes_info.des_settings.boost_mask = param_d.boost.as_cpptype()
        self.device.serdes_info.des_settings.invert_mask = param_d.invert.as_cpptype()
        param_d.ctle_filter.as_cpptype(self.device.serdes_info.des_settings.ctle_filter)
        param_d.lane_mappings[0].as_cpptype(self.device.serdes_info.des_settings.lane_mapping0)
        param_d.lane_mappings[1].as_cpptype(self.device.serdes_info.des_settings.lane_mapping1)
        logger.debug(f"ctle_filter: {self.device.serdes_info.des_settings.ctle_filter}")

    def _set_ser_settings(self, param_s: Ad9082SerConfig) -> None:
        self.device.serdes_info.ser_settings.invert_mask = param_s.invert.as_cpptype()
        for i in range(8):
            lane_settings = self.device.serdes_info.ser_settings.lane_settings[i]
            lane_settings[ad9081.SWING_SETTING] = param_s.swing[i].as_cpptype()
            lane_settings[ad9081.PRE_EMP_SETTING] = param_s.pre_emp[i].as_cpptype()
            lane_settings[ad9081.POST_EMP_SETTING] = param_s.post_emp[i].as_cpptype()
        param_s.lane_mappings[0].as_cpptype(self.device.serdes_info.ser_settings.lane_mapping0)
        param_s.lane_mappings[1].as_cpptype(self.device.serdes_info.ser_settings.lane_mapping1)

    def _set_serdes_settings(self, param_sd: Ad9082SerdesConfig):
        self._set_des_settings(param_sd.des)
        self._set_ser_settings(param_sd.ser)

    def initialize(
        self,
        reset: bool = False,
        link_init: bool = False,
        use_204b: bool = False,
        use_bg_cal: bool = True,
        wait_after_device_init: float = 0.1,
    ) -> None:
        if reset:
            self.device_reset()

        self._set_spi_settings(self.param.spi)  # values are set to the device object.
        self._set_serdes_settings(self.param.serdes)  # values are set to a device object.
        if link_init:
            self.device_init()  # the values set above is applied to the device here.
            time.sleep(wait_after_device_init)

        if use_204b and link_init:
            # Notes: "FREQ_DISCOUNT" was deprecated. TODO: will remove it soon.
            dev_ref_clk_hz = int(self.param.clock.ref * self.WORKAROUND_FREQ_DISCOUNT_RATE)
            dac_clk_hz = int(self.param.clock.dac * self.WORKAROUND_FREQ_DISCOUNT_RATE)
            adc_clk_hz = int(self.param.clock.adc * self.WORKAROUND_FREQ_DISCOUNT_RATE)
        else:
            dev_ref_clk_hz = int(self.param.clock.ref)
            dac_clk_hz = int(self.param.clock.dac)
            adc_clk_hz = int(self.param.clock.adc)

        if link_init:
            self.device_clk_config_set(dac_clk_hz, adc_clk_hz, dev_ref_clk_hz)
            self._startup_tx(self.param.dac, use_204b, use_bg_cal)
            self._startup_rx(self.param.adc, use_204b)
            self._establish_link()
        else:
            self.device.clk_conf_set(dac_clk_hz, adc_clk_hz, dev_ref_clk_hz)
            self.device.dev_info.dev_rev = self.device_chip_id_get().dev_revision

        if use_204b and link_init:
            # Notes: "FREQ_DISCOUNT" was deprecated. TODO: will remove it soon.
            #       We know this is a completely wrong way. the DISCOUNT_RATE is chosen very carefully to avoid
            #       catastrophy. For the reason of its necessity, ask the senior guys of QuEL.
            self.device.dev_info.dev_freq_hz = self.param.clock.ref
            self.device.dev_info.dac_freq_hz = self.param.clock.dac
            self.device.dev_info.adc_freq_hz = self.param.clock.adc

    def _startup_tx(self, param_tx: Ad9082DacConfig, use_204b: bool, use_bg_cal: bool) -> None:
        logger.info("starting-up DACs")

        # Notes: clear it before the modification of its corresponding registers.
        self._interp_cache = None
        self._fduc_map_cache = None

        if use_204b:
            main_freq: Tuple[int, int, int, int] = cast(
                Tuple[int, int, int, int],
                tuple([x * self.WORKAROUND_FREQ_DISCOUNT_RATE for x in param_tx.shift_freq.main]),
            )
            channel_freq: Tuple[int, int, int, int, int, int, int, int] = cast(
                Tuple[int, int, int, int, int, int, int, int],
                tuple([x * self.WORKAROUND_FREQ_DISCOUNT_RATE for x in param_tx.shift_freq.channel]),
            )
            # Notes: putting main_freq and channel_freq into Ad9082ShiftFreqConfig for validation.
            shift_freq: Ad9082ShiftFreqConfig = Ad9082ShiftFreqConfig(
                main=main_freq,
                channel=channel_freq,
            )
        else:
            shift_freq = param_tx.shift_freq

        rc = ad9081.device_startup_tx(
            self.device,
            param_tx.interpolation_rate.main,
            param_tx.interpolation_rate.channel,
            param_tx.channel_assign.as_cpptype(),
            list(shift_freq.main),
            list(shift_freq.channel),
            param_tx.jesd204.as_cpptype(),
        )
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

        for i in range(8):
            rc = ad9081.jesd_rx_lane_xbar_set(self.device, ad9081.LINK_0, i, param_tx.lane_xbar[i])
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"{CmsError(rc).name} at configuring {i}-th lane")

        # TODO(XXX): this should be moved to establish_link()
        # clearing PHY_PD (phy powerdown) of all the lanes
        self.hal_reg_set(0x0401, 0x00)

        # TODO(XXX): this should be moved to establish_link()
        if not use_204b:
            logger.info("calibrating JESD204C rx link")
            if use_bg_cal:
                logger.info("activating background calibration")
            rc = ad9081.jesd_rx_calibrate_204c(
                self.device, 1, self.device.serdes_info.des_settings.boost_mask, 1 if use_bg_cal else 0
            )
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"{CmsError(rc).name}")

        # TODO(XXX): this should be moved to establish_link()
        logger.info("enabling JESD204C rx link")
        rc = ad9081.jesd_rx_link_enable_set(self.device, ad9081.LINK_0, 1)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(f"{CmsError(rc).name}")

        for i in range(4):
            self.set_fullscale_current(1 << i, param_tx.fullscale_current[i])

    def _startup_rx(self, param_rx: Ad9082AdcConfig, use_204b: bool) -> None:
        logger.info("starting up ADCs")

        if use_204b:
            main_freq: Tuple[int, int, int, int] = cast(
                Tuple[int, int, int, int],
                tuple([x * self.WORKAROUND_FREQ_DISCOUNT_RATE for x in param_rx.shift_freq.main]),
            )
            channel_freq: Tuple[int, int, int, int, int, int, int, int] = cast(
                Tuple[int, int, int, int, int, int, int, int],
                tuple([x * self.WORKAROUND_FREQ_DISCOUNT_RATE for x in param_rx.shift_freq.channel]),
            )
            # Notes: putting main_freq and channel_freq into Ad9082ShiftFreqConfig for validation.
            shift_freq: Ad9082ShiftFreqConfig = Ad9082ShiftFreqConfig(
                main=main_freq,
                channel=channel_freq,
            )
        else:
            shift_freq = param_rx.shift_freq

        rc = ad9081.device_startup_rx(
            self.device,
            int(ad9081.ADC_CDDC_ALL),
            int(ad9081.ADC_FDDC_ALL),
            list(shift_freq.main),
            list(shift_freq.channel),
            param_rx.decimation_rate.main.as_cpptype(),
            param_rx.decimation_rate.channel.as_cpptype(),
            list(param_rx.c2r_enable.main),
            list(param_rx.c2r_enable.channel),
            [x.as_cpptype() for x in param_rx.jesd204],
            [x.as_cpptype() for x in param_rx.converter_mappings],
        )
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

        rc = ad9081.jesd_tx_lanes_xbar_set(self.device, ad9081.LINK_0, [int(x) for x in param_rx.lane_xbar])
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

    def _establish_link(self) -> None:
        lid = [0, 0, 0, 0, 0, 0, 0, 0]
        rc = ad9081.jesd_tx_lids_cfg_set(self.device, ad9081.LINK_0, lid)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

        rc = ad9081.jesd_tx_link_enable_set(self.device, ad9081.LINK_0, 1)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(CmsError(rc).name)

        # clearing CRC IRQ status
        self.write_reg(0x05BB, 0x00)
        self.write_reg(0x05BB, 0x01)

    def get_link_status(self) -> Tuple[int, int]:
        link_status: int = self.read_reg(0x055E)
        crc_flag: int = self.read_reg(0x05BB)
        return link_status, crc_flag

    def dump_jesd_status(self) -> None:
        vals: Dict[int, int] = {}
        for addr in range(0x670, 0x678):
            val = self.hal_reg_get(addr)
            vals[addr] = val
        for addr in (0x702, 0x728, 0x0CA):
            val = self.hal_reg_get(addr)
            vals[addr] = val

        for addr, val in vals.items():
            logger.info(f"*({addr:04x}) = {val:02x}")

    def is_bgcal_enabled(self) -> bool:
        return self.hal_reg_get(0x21C1) & 0x08 == 0x08

    def get_crc_error_counts(self) -> List[int]:
        return [self.hal_reg_get(0x0584 + lane_idx) for lane_idx in range(8)]

    def calc_dac_cnco_ftw(self, shift_hz: float, fractional_mode=False) -> NcoFtw:
        obj = NcoFtw.from_frequency(shift_hz, self.device.dev_info.dac_freq_hz)
        if not fractional_mode:
            obj.round()
        return obj

    def calc_dac_fnco_ftw(self, shift_hz: float, fractional_mode=False) -> NcoFtw:
        obj = NcoFtw.from_frequency(shift_hz, self.device.dev_info.dac_freq_hz // self.get_main_interpolation_rate())
        if not fractional_mode:
            obj.round()
        return obj

    def calc_dac_cnco_freq(self, ftw: NcoFtw) -> float:
        return ftw.to_frequency(self.device.dev_info.dac_freq_hz)

    def calc_dac_fnco_freq(self, ftw: NcoFtw) -> float:
        return ftw.to_frequency(self.device.dev_info.dac_freq_hz // self.get_main_interpolation_rate())

    def set_dac_cnco(self, dacs: Collection[int], ftw: NcoFtw) -> None:
        # dacs is actually cducs.
        dac_mask: ad9081.DacSelect = ad9081.DAC_NONE
        for i in dacs:
            if not 0 <= i <= 3:
                raise ValueError("invalid index of dac: {i}")
            dac_mask |= getattr(ad9081, f"DAC_{i:d}")

        if dac_mask != ad9081.DAC_NONE:
            rc = ad9081.dac_duc_nco_enable_set(self.device, int(dac_mask), int(ad9081.DAC_CH_NONE), 1)
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"ad9081.dac_duc_nco_enable_set() failed with error code: {rc}")
            rc = ad9081.dac_duc_nco_ftw_set(self.device, int(dac_mask), int(ad9081.DAC_CH_NONE), ftw.to_ftw())
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"ad9081.dac_duc_nco_ftw_set() failed with error code: {rc}")

    def get_dac_cnco(self, dac: int) -> NcoFtw:
        # dac is actually cduc
        if not 0 <= dac <= 3:
            raise ValueError(f"invalid index of dac: {dac}")
        dac_mask: ad9081.DacSelect = getattr(ad9081, f"DAC_{dac:d}")
        rc = ad9081.dac_select_set(self.device, int(dac_mask))
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(f"ad9081.dac_duc_nco_enable_set() failed with error code: {rc}")
        ftw = _NcoFtw()
        enable_fractional: bool = (self.hal_reg_get(0x01C9) & 0x04) == 0x04
        ftw.ftw = self._read_u48(0x1CB)
        if enable_fractional:
            ftw.modulus_a = self._read_u48(0x1D3)
            ftw.delta_b = self._read_u48(0x1D9)
        else:
            ftw.modulus_a = 0
            ftw.delta_b = 0

        try:
            r = NcoFtw.from_ftw(ftw)
        except ValidationError:
            logger.error(f"invalid FTW is detected for DAC CNCO-#{dac}, ignoring it")
            r = NcoFtw.from_frequency(0, self.device.dev_info.dac_freq_hz)
        return r

    def set_dac_fnco(self, channels: Collection[int], ftw: NcoFtw) -> None:
        # channel is actually fduc
        ch_mask: ad9081.DacChannelSelect = ad9081.DAC_CH_NONE
        for i in channels:
            if not 0 <= i <= 7:
                raise ValueError(f"invalid channel: {i}")
            ch_mask |= getattr(ad9081, f"DAC_CH_{i:d}")

        if ch_mask != ad9081.DAC_CH_NONE:
            rc = ad9081.dac_duc_nco_enable_set(self.device, int(ad9081.DAC_NONE), int(ch_mask), 1)
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"ad9081.dac_duc_nco_enable_set() failed with error code: {rc}")
            rc = ad9081.dac_duc_nco_ftw_set(self.device, int(ad9081.DAC_NONE), int(ch_mask), ftw.to_ftw())
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"ad9081.dac_duc_nco_ftw_set() failed with error code: {rc}")

    def get_dac_fnco(self, channel: int) -> NcoFtw:
        # channel is actually fduc
        if not 0 <= channel <= 7:
            raise ValueError(f"invalid index of channel: {channel}")
        ch_mask: ad9081.DacChannelSelect = getattr(ad9081, f"DAC_CH_{channel:d}")

        rc = ad9081.dac_chan_select_set(self.device, int(ch_mask))
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(f"ad9081.dac_duc_nco_enable_set() failed with error code: {rc}")

        ftw = _NcoFtw()

        enable_fractional: bool = (self.hal_reg_get(0x01A0) & 0x04) == 0x04
        ftw.ftw = self._read_u48(0x1A2)
        if enable_fractional:
            ftw.modulus_a = self._read_u48(0x1AA)
            ftw.delta_b = self._read_u48(0x1B0)
        else:
            ftw.modulus_a = 0
            ftw.delta_b = 0

        try:
            r = NcoFtw.from_ftw(ftw)
        except ValidationError:
            logger.error(f"invalid FTW is detected for DAC FNCO-#{channel}, ignoring it")
            r = NcoFtw.from_frequency(0, self.device.dev_info.dac_freq_hz)  # Notes: interpolation rate is omitted.
        return r

    def calc_adc_cnco_ftw(self, shift_hz: float, fractional_mode=False) -> NcoFtw:
        obj = NcoFtw.from_frequency(shift_hz, self.device.dev_info.adc_freq_hz)
        if not fractional_mode:
            obj.round()
        return obj

    def _check_congruency_of_main_decimation_rate(self) -> None:
        dr0 = self.param.adc.decimation_rate.main[0]
        for dr in self.param.adc.decimation_rate.main:
            if dr != dr0:
                raise RuntimeError("it is not supported to have different decimation rate among ADCs")

    def calc_adc_fnco_ftw(self, shift_hz: float, fractional_mode=False) -> NcoFtw:
        # TODO: take index of ADC in the case that different decimation rates are used among ADCs.
        # TODO: double shift value when c2r is enabled (low priority).
        self._check_congruency_of_main_decimation_rate()
        # Notes: be aware that rounding error may be induced here.
        obj = NcoFtw.from_frequency(
            shift_hz, self.device.dev_info.adc_freq_hz // self.param.adc.decimation_rate.main[0]
        )
        if not fractional_mode:
            obj.round()
        return obj

    def calc_adc_cnco_freq(self, ftw: NcoFtw) -> float:
        return ftw.to_frequency(self.device.dev_info.adc_freq_hz)

    def calc_adc_fnco_freq(self, ftw: NcoFtw) -> float:
        # TODO: make it possible to eliminate the following check
        self._check_congruency_of_main_decimation_rate()
        return ftw.to_frequency(self.device.dev_info.adc_freq_hz // self.param.adc.decimation_rate.main[0])

    def set_adc_cnco(self, adcs: Collection[int], ftw: NcoFtw) -> None:
        adc_mask: ad9081.AdcCoarseDdcSelect = ad9081.ADC_CDDC_NONE
        for i in adcs:
            if not 0 <= i <= 3:
                raise ValueError(f"invalid index of adc: {i}")
            adc_mask |= getattr(ad9081, f"ADC_CDDC_{i:d}")

        if adc_mask != ad9081.ADC_CDDC_NONE:
            # TODO: clarify how NCO_ZIF and NCO_VIF works.
            rc = ad9081.adc_ddc_coarse_nco_mode_set(
                self.device,
                int(adc_mask),
                ad9081.ADC_NCO_ZIF if ftw.is_zero() else ad9081.ADC_NCO_VIF,
            )
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"ad9081.adc_ddc_coarse_nco_mode_set() failed with error code: {rc}")
            rc = ad9081.adc_ddc_coarse_nco_ftw_set(self.device, int(adc_mask), ftw.to_ftw())
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"ad9081.adc_ddc_nco_ftw_set() failed with error code: {rc}")

    def get_adc_cnco(self, adc: int) -> NcoFtw:
        # Note: adc is actually cddc
        if not 0 <= adc <= 3:
            raise ValueError(f"invalid index of adc: {adc}")
        adc_mask: ad9081.AdcCoarseDdcSelect = getattr(ad9081, f"ADC_CDDC_{adc:d}")

        ftw = _NcoFtw()
        rc = ad9081.adc_ddc_coarse_nco_ftw_get(self.device, int(adc_mask), ftw)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(f"ad9081.adc_ddc_coarse_nco_ftw_get() failed with error code: {rc}")

        try:
            r = NcoFtw.from_ftw(ftw)
        except ValidationError:
            logger.error(f"invalid FTW is detected ADC CNCO-#{adc}, ignoring it")
            r = NcoFtw.from_frequency(0, self.device.dev_info.adc_freq_hz)
        return r

    def set_adc_fnco(self, channels: Collection[int], ftw: NcoFtw) -> None:
        ch_mask: ad9081.AdcFineDdcSelect = ad9081.ADC_FDDC_NONE
        for i in channels:
            if not 0 <= i <= 7:
                raise ValueError(f"invalid index of adc channel: {i}")
            ch_mask |= getattr(ad9081, f"ADC_FDDC_{i:d}")

        if ch_mask != ad9081.ADC_FDDC_NONE:
            # TODO: clarify how NCO_ZIF and NCO_VIF works.
            rc = ad9081.adc_ddc_fine_nco_mode_set(
                self.device,
                int(ch_mask),
                ad9081.ADC_NCO_ZIF if ftw.is_zero() else ad9081.ADC_NCO_VIF,
            )
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"ad9081.adc_ddc_nco_mode_set() failed with error code: {rc}")
            rc = ad9081.adc_ddc_fine_nco_ftw_set(self.device, int(ch_mask), ftw.to_ftw())
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"ad9081.adc_ddc_fine_nco_set() failed with error code: {rc}")

    def get_adc_fnco(self, channel: int) -> NcoFtw:
        # Note: channel is actually fddc
        if not 0 <= channel <= 7:
            raise ValueError(f"invalid index of adc channel: {channel}")
        ch_mask: ad9081.AdcFineDdcSelect = getattr(ad9081, f"ADC_FDDC_{channel:d}")

        ftw = _NcoFtw()
        rc = ad9081.adc_ddc_fine_nco_ftw_get(self.device, int(ch_mask), ftw)
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(f"ad9081.adc_ddc_fine_nco_ftw_get() failed with error code: {rc}")

        try:
            r = NcoFtw.from_ftw(ftw)
        except ValidationError:
            logger.error(f"invalid FTW is detected for ADC FNCO-#{channel}, ignoring it")
            r = NcoFtw.from_frequency(0, self.device.dev_info.adc_freq_hz)  # Notes: decimation rate is omitted.
        return r

    def is_equivalent_dac_cnco(self, freq0: float, freq1: float, fractional_mode: bool = False) -> bool:
        ftw0 = self.calc_dac_cnco_ftw(freq0, fractional_mode)
        ftw1 = self.calc_dac_cnco_ftw(freq1, fractional_mode)
        return ftw0 == ftw1

    def is_equivalent_dac_fnco(self, freq0: float, freq1: float, fractional_mode: bool = False) -> bool:
        ftw0 = self.calc_dac_fnco_ftw(freq0, fractional_mode)
        ftw1 = self.calc_dac_fnco_ftw(freq1, fractional_mode)
        return ftw0 == ftw1

    def is_equivalent_adc_cnco(self, freq0: float, freq1: float, fractional_mode: bool = False) -> bool:
        ftw0 = self.calc_adc_cnco_ftw(freq0, fractional_mode)
        ftw1 = self.calc_adc_cnco_ftw(freq1, fractional_mode)
        return ftw0 == ftw1

    def is_equivalent_adc_fnco(self, freq0: float, freq1: float, fractional_mode: bool = False) -> bool:
        ftw0 = self.calc_adc_fnco_ftw(freq0, fractional_mode)
        ftw1 = self.calc_adc_fnco_ftw(freq1, fractional_mode)
        return ftw0 == ftw1

    def set_fullscale_current(self, dacs: int, current: int) -> None:
        if dacs == 0 or (dacs & 0x0F) != dacs:
            raise ValueError(f"wrong specifier of DACs {dacs:x}")

        # Notes: this implementation is kept for the historical reason.
        #        40527uA is the historically used fsc, it is called as 40520 by mistake for a while...
        if not ((7000 <= current <= 40000) or (current in {40520, 40527})):
            raise ValueError(f"invalid current {current}uA")

        if current in (40520, 40527):
            # Notes: adi_ad9081_dac_fsc_set() API pose the maximum limit of fsc as 40000uA.
            logger.info("setting fullscale current to 40527uA, that is the conventional value for QuEL-1")
            self.hal_reg_set(0x001B, dacs)
            self.hal_reg_set(0x0117, 0xA0)
            self.hal_reg_set(0x0118, 0xFF)
        else:
            rc = ad9081.dac_fsc_set(self.device, dacs, current)
            if rc != CmsError.API_CMS_ERROR_OK:
                raise RuntimeError(f"{CmsError(rc).name}")

    def decode_fullscale_current(self, r117: int, r118: int) -> int:
        fscmin = (r117 >> 4) & 0x0F
        fscctrl_1_0 = r117 & 0x03
        fscctrl_9_2 = r118 & 0xFF
        return round((fscmin / 16 + ((fscctrl_9_2 << 2) + fscctrl_1_0) / 1024) * 25000)

    def get_fullscale_current(self, dac: int) -> int:
        if not 0 <= dac <= 3:
            raise ValueError(f"invalid index of dac: {dac}")
        dac_mask: ad9081.DacSelect = getattr(ad9081, f"DAC_{dac:d}")
        rc = ad9081.dac_select_set(self.device, int(dac_mask))
        if rc != CmsError.API_CMS_ERROR_OK:
            raise RuntimeError(f"ad9081.dac_duc_nco_enable_set() failed with error code: {rc}")

        r117 = self.hal_reg_get(0x117)
        r118 = self.hal_reg_get(0x118)
        return self.decode_fullscale_current(r117, r118)

    def is_equal_fullscale_current(self, cur0: int, cur1: int):
        # Notes: rounding error is at most 13 (= ceil(25000 / 1024 / 2)).
        # Notes: one of the given values must be calculated from the actual register values.
        return abs(cur0 - cur1) <= 13

    def _dac_xbar_get(self) -> tuple[tuple[int, ...], ...]:
        r: list[tuple[int, ...]] = []
        _, chnl_intrp = self.dac_interpolation_get()
        for i in range(4):
            ad9081.dac_select_set(self.device, (1 << i))
            if chnl_intrp > 1:
                channels_list: list[int] = []
                channels = self.hal_reg_get(0x01BA)
                for j in reversed(range(8)):  # Notes: the order is important!
                    if channels & (0x01 << j) != 0:
                        channels_list.append(j)
                r.append(tuple(channels_list))
            else:
                raise NotImplementedError("channel interpolation rate is assumed to be more than 1")
        return tuple(r)

    def get_fduc_of_dac(self, dac: int) -> Tuple[int, ...]:
        if not 0 <= dac <= 3:
            raise ValueError(f"invalid index of dac: {dac}")

        if self._fduc_map_cache is None:
            self._fduc_map_cache = self._dac_xbar_get()
        return self._fduc_map_cache[dac]

    def get_virtual_adc_select(self) -> List[int]:
        # Notes: 16 comes from the value of JESD M parameter. (see p.68 of UG-1578 rev.A)
        convsel = []
        for i in range(16):
            v = self.read_reg(0x600 + i)
            if v & 0x80:
                convsel.append(-1)
            else:
                convsel.append(v)
        return convsel

    def dac_interpolation_get(self) -> tuple[int, int]:
        # Notes: (main, channel)
        if self._interp_cache is None:
            intrp_mode = self.hal_reg_get(0x01FF)
            self._interp_cache = (int((intrp_mode >> 4) & 0x0F), int(intrp_mode & 0x0F))
        return self._interp_cache

    def get_main_interpolation_rate(self) -> int:
        return self.dac_interpolation_get()[0]

    def get_channel_interpolation_rate(self) -> int:
        return self.dac_interpolation_get()[1]

    def get_temperatures(self) -> Tuple[int, int]:
        temperatures = ChipTemperatures()
        ad9081.device_get_temperature(self.device, temperatures)
        return temperatures.temp_max, temperatures.temp_min
