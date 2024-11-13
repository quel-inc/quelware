from typing import Final

SAMPLING_FREQ: Final[float] = 500.0e6
DECIMATION_RATE: Final[int] = 4
_CFIR_NTAPS: Final[int] = 16
_RFIRS_NTAPS: Final[int] = 8

_DEFAULT_TIMEOUT: Final[float] = 3.0  # [s]
_DEFAULT_TIMEOUT_FOR_CAPTURE_RESERVE: Final[float] = 0.0  # no timeout
_DEFAULT_POLLING_PERIOD: Final[float] = 0.025  # [s]

_AWG_MINIMUM_ALIGN: Final[int] = 32  # [bytes]
_AWG_CHUNK_SIZE_UNIT_IN_SAMPLE: Final[int] = 64  # [samples]
_CAP_MINIMUM_ALIGN: Final[int] = 512  # [bytes]


class E7awgHardwareError(Exception):
    pass


class E7awgMemoryError(Exception):
    pass


class E7awgCaptureDataError(Exception):
    pass
