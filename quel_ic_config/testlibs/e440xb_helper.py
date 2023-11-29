import logging

import numpy as np
from quel_inst_tool import (  # noqa: F401
    E440xb,
    E440xbTraceMode,
    E4405b,
    E4407b,
    ExpectedSpectrumPeaks,
    InstDevManager,
    MeasuredSpectrumPeak,
)

logger = logging.getLogger(__name__)


DEFAULT_FREQ_CENTER = 9e9
DEFAULT_FREQ_SPAN = 5e9
DEFAULT_SWEEP_POINTS = 4001
DEFAULT_RESOLUTION_BANDWIDTH = 1e5


def init_e440xb(
    spa_type: str = "E4405B",
    freq_center: float = DEFAULT_FREQ_CENTER,
    freq_span: float = DEFAULT_FREQ_SPAN,
    sweep_points: int = DEFAULT_SWEEP_POINTS,
    resolution_bandwidth: float = DEFAULT_RESOLUTION_BANDWIDTH,
) -> E440xb:
    im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so", blacklist=["GPIB0::6::INSTR"])
    im.scan()
    dev = im.lookup(prod_id=spa_type)
    if dev is None:
        raise RuntimeError(f"no spectrum analyzer '{spa_type}' is detected")

    if spa_type == "E4405B":
        e440xb: E440xb = E4405b(dev)
    elif spa_type == "E4407B":
        e440xb = E4407b(dev)
    else:
        raise ValueError("invalid spectrum analyzer type, it must be either E4405B or E4407B")

    e440xb.reset()
    e440xb.display_enable = False
    e440xb.trace_mode = E440xbTraceMode.WRITE
    e440xb.freq_range_set(freq_center, freq_span)
    e440xb.sweep_points = sweep_points
    e440xb.resolution_bandwidth = resolution_bandwidth
    # e4405b.resolution_bandwidth = 5e4   # floor noise < -70dBm, but spurious peaks higher than -70dBm exist
    # e4405b.resolution_bandwidth = 1e5  # floor noise < -65dBm
    # e4405b.resolution_bandwidth = 1e6   # floor noise < -55dBm
    return e440xb


def measure_floor_noise(e4405b: E440xb, n_iter: int = 5) -> float:
    # Checkout
    t0 = e4405b.trace_get()
    fln = t0[:, 1]
    for i in range(n_iter - 1):
        fln = np.maximum(fln, e4405b.trace_get()[:, 1])

    mfln = fln.max()
    logger.info(f"maximum floor noise = {mfln:.1f}dBm")
    return mfln
