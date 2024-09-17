# default parameters
_DEFAULT_TIMEOUT = 3.0  # [s]
_DEFAULT_TIMEOUT_FOR_CAPTURE_RESERVE = 0.0  # no timeout
_DEFAULT_POLLING_PERIOD = 0.025  # [s]

_AWG_MINIMUM_ALIGN = 32  # [bytes]
_AWG_CHUNK_SIZE_UNIT_IN_SAMPLE = 64  # [samples]
_CAP_MINIMUM_ALIGN = 512  # [bytes]


class E7awgHardwareError(Exception):
    pass


class E7awgMemoryError(Exception):
    pass


class E7awgCaptureDataError(Exception):
    pass
