from enum import Enum


class E7FwType(Enum):
    AUTO_DETECT = 0
    SIMPLEMULTI_CLASSIC = 1  # deprecated
    FEEDBACK_VERYEARLY = 2  # deprecated
    NEC_EARLY = 3
    SIMPLEMULTI_STANDARD = 4
    FEEDBACK_EARLY = 5
    SIMPLEMULTI_WIDE = 6


class E7FwAuxAttr(Enum):
    UNKNOWN_VERSION = 65536
    NO_SYSREF_LATCH = 63337
    BROKEN_AWG_RESET = 65538
    IRREGULAR_ADC_FNCO = 65539
    DSP_v0a = 65541  # the original DSP IP with overflow bug
    DSP_v0b = 65542  # th original DSP IP with the bug-fix of the overflow issue
    DSP_v1a = 65543  # a new implementation of the DSP IP by e-trees


class E7FwLifeStage(Enum):
    UNKNOWN = 0
    EXPERIMENTAL = 1
    SUPPORTING = 2
    TO_DEPRECATE = 3
    DEPRECATED = 4
    SUSPENDED = 5
