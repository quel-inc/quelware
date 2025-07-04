import logging
from collections.abc import Sequence
from typing import Annotated, Literal, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, PlainValidator, ValidationInfo, model_validator

from e7awghal.capdata import CapIqParser
from e7awghal.classification import ClassificationParam

logger = logging.getLogger(__name__)


class CapSection(BaseModel, validate_assignment=True):
    name: str = Field(default="")
    num_capture_word: int = Field(ge=0x0000_0001, le=0xFFFF_FFFF)  # no default value
    num_blank_word: int = Field(ge=0x0000_0001, le=0xFFFF_FFFF, default=0x0000_0001)


def sum_range_validation(v: Sequence[int], info: ValidationInfo) -> tuple[int, int]:
    if not (isinstance(v, Sequence) and len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], int)):
        raise ValueError("sum range must be a pair of integer")

    b, e = v
    if not (0 <= b <= 0xFFFFFFFF) or not (0 <= e <= 0xFFFFFFFF) or (b > e):
        raise ValueError("invalid sum range: ({b}, {e})")

    return b, e


SumRange = Annotated[
    tuple[int, int],
    PlainValidator(sum_range_validation),
]


def cfir_coeff_validation(
    v: Union[
        npt.NDArray[np.complex128], npt.NDArray[np.complex64], npt.NDArray[np.float64], npt.NDArray[np.float32], None
    ],
    info: ValidationInfo,
) -> npt.NDArray[np.complex64]:
    if v is None:
        v = np.zeros(16, dtype=np.complex64)
        v[15] = 1.0 + 0.0j
    else:
        if not (
            isinstance(v, np.ndarray)
            and v.dtype in (np.complex128, np.complex64, np.float64, np.float32)
            and v.ndim == 1
        ):
            raise ValueError("coefficients of complex FIR filter must be one-dimensional ndarray of complex")
        v = v.astype(np.complex64)  # Notes: copying it even when the given v is np.complex64.

    if v.shape[0] != 16:
        raise ValueError("the length of coefficients of complex FIR filter must be 16")

    # Notes: range check is postponed by the model validator, since it depends on complexfir_exponent_offset field.

    # Notes: prohibiting elementwise modification since there is no efficient way of validation.
    v.flags.writeable = False
    return cast(npt.NDArray[np.complex64], v)


CfirCoeff = Annotated[
    npt.NDArray[np.complex64],
    PlainValidator(cfir_coeff_validation),
]


def rfir_coeff_validation(
    v: Union[npt.NDArray[np.float64], npt.NDArray[np.float32], None], info: ValidationInfo
) -> npt.NDArray[np.float32]:
    if v is None:
        v = np.zeros(8, dtype=np.float32)
        v[7] = 1.0
    else:
        if v.dtype in (np.complex64, np.complex128):
            # Notes: to raise exception instead of warning irrespective of user's settings.
            raise ValueError("coefficients of real FIR filters must be one-dimensional ndarray of float")
        v = v.astype(dtype=np.float32)

    if not (isinstance(v, np.ndarray) and v.dtype in (np.float64, np.float32) and v.ndim == 1):
        raise ValueError("coefficients of real FIR filters must be one-dimensional ndarray of float")

    if v.shape[0] != 8:
        raise ValueError("the length of coefficients of real FIR filter must be 8")

    if v.dtype == np.float64:
        v = v.astype(dtype=np.float32)

    # Notes: range check is postponed by the model validator, since it depends on complexfir_exponent_offset field.

    # Notes: prohibiting elementwise modification since there is no efficient way of validation.
    v.flags.writeable = False
    return cast(npt.NDArray[np.float32], v)


RfirCoeff = Annotated[
    npt.NDArray[np.float32],
    PlainValidator(rfir_coeff_validation),
]


def window_coeff_validation(
    v: Union[npt.NDArray[np.complex128], npt.NDArray[np.float64], None], info: ValidationInfo
) -> npt.NDArray[np.complex128]:
    if v is None:
        v = np.array([1.0 + 0.0j], dtype=np.complex128)
    else:
        if not (isinstance(v, np.ndarray) and v.dtype in (np.complex128, np.float64) and v.ndim == 1):
            raise ValueError("coefficients of window function must be one-dimensional ndarray of complex128")
        v = v.astype(np.complex128)

    if not (1 <= v.shape[0] <= 2048):
        raise ValueError("the length of coefficients of window function must be betweeb 1 and 2048")

    vr = v.real
    vi = v.imag
    if (vr < -2.0).any() or (2.0 <= vr).any():
        raise ValueError("any coefficients of window function must be more than or equal to -2.0 and less than 2.0")
    if (vi < -2.0).any() or (2.0 <= vi).any():
        raise ValueError("any coefficients of window function must be more than or equal to -2.0 and less than 2.0")
    # Notes: prohibiting elementwise modification since there is no efficient way of validation.
    v.flags.writeable = False
    return v


WindowCoeff = Annotated[
    npt.NDArray[np.complex128],
    PlainValidator(window_coeff_validation),
]


class _BaseCapParam(BaseModel, validate_assignment=True, extra="forbid"):
    num_wait_word: int = Field(ge=0x0000_0000, le=0xFFFF_FFFE, default=0x0000_0000, multiple_of=16)
    num_repeat: int = Field(ge=0x0000_0001, le=0xFFFF_FFFF, default=0x0000_0001)
    sections: list[CapSection] = Field(default=[])

    complexfir_enable: bool
    complexfir_exponent_offset: int
    complexfir_coeff: CfirCoeff
    decimation_enable: bool
    realfirs_enable: bool
    realfirs_exponent_offset: int
    realfirs_real_coeff: RfirCoeff
    realfirs_imag_coeff: RfirCoeff
    window_enable: bool
    window_coeff: WindowCoeff
    sum_enable: bool
    sum_range: SumRange
    integration_enable: bool
    classification_enable: bool
    classification_param: ClassificationParam

    @property
    def num_section(self) -> int:
        return len(self.sections)

    def _calc_datasize_in_sample(self) -> tuple[int, list[int], int]:
        num_repeat = 1 if self.integration_enable else self.num_repeat
        section_sizes_sample = [
            (1 if self.sum_enable else s.num_capture_word * (1 if self.decimation_enable else 4)) for s in self.sections
        ]
        # if self.classification_enable:
        #     section_sizes_sample = [np.ceil(s/32) for s in section_sizes_sample]
        total_size_sample = num_repeat * sum(section_sizes_sample)
        return num_repeat, section_sizes_sample, total_size_sample

    def get_datasize_in_sample(self) -> int:
        _, _, total_size = self._calc_datasize_in_sample()
        return total_size

    def total_exponent_offset(self) -> int:
        s = 0
        if self.complexfir_enable:
            s += self.complexfir_exponent_offset
        if self.realfirs_enable:
            s += self.realfirs_exponent_offset
        return s

    def get_parser(self) -> CapIqParser:
        num_repeat, section_sizes_in_sample, total_size_in_sample = self._calc_datasize_in_sample()
        sns: list[str] = [s.name for s in self.sections]
        # Notes: invalidate the list of names if names of the sections are duplicated.
        section_names: Union[tuple[str, ...], None] = tuple(sns) if len(set(sns)) == self.num_section else None
        return CapIqParser(
            total_size_in_sample=total_size_in_sample,
            num_repeat=num_repeat,
            section_sizes_in_sample=tuple(section_sizes_in_sample),
            section_names=section_names,
            total_exponent_offset=self.total_exponent_offset(),
            classification_enable=self.classification_enable,
        )


class CapParamSimple(_BaseCapParam):
    complexfir_enable: Literal[False] = Field(default=False)
    complexfir_exponent_offset: Literal[14] = Field(default=14)
    complexfir_coeff: CfirCoeff = Field(default=None, validate_default=True)
    decimation_enable: Literal[False] = Field(default=False)
    realfirs_enable: Literal[False] = Field(default=False)
    realfirs_exponent_offset: Literal[14] = Field(default=14)
    realfirs_real_coeff: RfirCoeff = Field(default=None, validate_default=True)
    realfirs_imag_coeff: RfirCoeff = Field(default=None, validate_default=True)
    window_enable: Literal[False] = Field(default=False)
    window_coeff: WindowCoeff = Field(default=None, validate_default=True)
    sum_enable: Literal[False] = Field(default=False, frozen=True)
    sum_range: SumRange = Field(default=(0x0000_0000, 0xFFFF_FFFF), validate_default=True, frozen=True)
    integration_enable: Literal[False] = Field(default=False)
    classification_enable: Literal[False] = Field(default=False)
    classification_param: ClassificationParam = Field(default=ClassificationParam(), frozen=True)


class CapParam(_BaseCapParam):
    complexfir_enable: bool = Field(default=False)
    complexfir_exponent_offset: int = Field(default=14, ge=0, le=15)
    complexfir_coeff: CfirCoeff = Field(default=None, validate_default=True)
    decimation_enable: bool = Field(default=False)
    realfirs_enable: bool = Field(default=False)
    realfirs_exponent_offset: int = Field(default=14, ge=0, le=15)
    realfirs_real_coeff: RfirCoeff = Field(default=None, validate_default=True)
    realfirs_imag_coeff: RfirCoeff = Field(default=None, validate_default=True)
    window_enable: bool = Field(default=False)
    window_coeff: WindowCoeff = Field(default=None, validate_default=True)
    sum_enable: bool = Field(default=False)
    sum_range: SumRange = Field(default=(0x0000_0000, 0xFFFF_FFFF), validate_default=True)
    integration_enable: bool = Field(default=False)
    classification_enable: bool = Field(default=False)
    classification_param: ClassificationParam = Field(default=ClassificationParam())

    @model_validator(mode="after")
    def check_fir(self) -> "CapParam":
        # Notes: 'after' validator is fine since it allows to be invalid intermediately when exception is ignored.
        #        invalid parameter is rejected by the other validation mechanisms which is activated just before it is
        #        written to the actual hardware registers.
        cvmax = 1 << (15 - self.complexfir_exponent_offset)
        if (
            (self.complexfir_coeff.real < -cvmax).any()
            or (cvmax <= self.complexfir_coeff.real).any()
            or (self.complexfir_coeff.imag < -cvmax).any()
            or (cvmax <= self.complexfir_coeff.imag).any()
        ):
            raise ValueError("some coefficients of complex FIR are out of range")

        rvmax = 1 << (15 - self.realfirs_exponent_offset)
        if (self.realfirs_real_coeff < -rvmax).any() or (rvmax <= self.realfirs_real_coeff).any():
            raise ValueError("some coefficients of real I-FIR are out of range")
        if (self.realfirs_imag_coeff < -rvmax).any() or (rvmax <= self.realfirs_imag_coeff).any():
            raise ValueError("some coefficients of real Q-FIR are out of range")

        return self
