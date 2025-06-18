import logging
import socket
from typing import Sequence

import numpy as np
from quel_inst_tool import (
    E440xb,
    E440xbTraceMode,
    E4405b,
    E4407b,
    InstDevManager,
    Ms2xxxx,
    Ms2xxxxTraceMode,
    Ms2090a,
    Ms2720t,
    SpectrumAnalyzer,
)

logger = logging.getLogger(__name__)

# to cover 6.5GHz -- 11.5GHz
E440XB_DEFAULT_FREQ_CENTER = 9e9
E440XB_DEFAULT_FREQ_SPAN = 5e9
E440XB_DEFAULT_SWEEP_POINTS = 4001
E440XB_DEFAULT_RESOLUTION_BANDWIDTH = 1e5


# to cover 6GHz -- 24GHz
MS2XXXX_DEFAULT_FREQ_CENTER = 15e9
MS2XXXX_DEFAULT_FREQ_SPAN = 18e9
MS2XXXX_DEFAULT_SWEEPPOINTS = 2001  # Nots: only for ms2090a
MS2XXXX_DEFAULT_RESOLUTION_BANDWIDTH = 1e5


def init_e440xb(
    spa_type: str = "E4405B",
    freq_center: float = E440XB_DEFAULT_FREQ_CENTER,
    freq_span: float = E440XB_DEFAULT_FREQ_SPAN,
    sweep_points: int = E440XB_DEFAULT_SWEEP_POINTS,
    resolution_bandwidth: float = E440XB_DEFAULT_RESOLUTION_BANDWIDTH,
    blacklist: Sequence[str] = (),
) -> SpectrumAnalyzer:
    im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so", blacklist=blacklist)
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


def init_ms2xxxx(
    spa_name: str,
    freq_center: float = MS2XXXX_DEFAULT_FREQ_CENTER,
    freq_span: float = MS2XXXX_DEFAULT_FREQ_SPAN,
    sweep_points: int = MS2XXXX_DEFAULT_SWEEPPOINTS,
    resolution_bandwidth: float = MS2XXXX_DEFAULT_RESOLUTION_BANDWIDTH,
) -> SpectrumAnalyzer:
    im = InstDevManager(ivi="@py")
    try:
        ipaddr_spa = socket.gethostbyname(spa_name)
    except socket.gaierror:
        raise RuntimeError(f"spectrum analyzer '{spa_name}' not found on the network.")

    if spa_name.startswith("ms2090"):
        spa: Ms2xxxx = Ms2090a(im.get_inst_device(Ms2090a.get_visa_name(ipaddr=ipaddr_spa)))
    else:
        spa = Ms2720t(im.get_inst_device(Ms2720t.get_visa_name(ipaddr=ipaddr_spa)))

    spa.reset()
    if isinstance(spa, Ms2090a):
        spa.sweep_points = sweep_points
    spa.continuous_sweep = False
    spa.average_enable = False
    spa.trace_mode = Ms2xxxxTraceMode.NORM
    spa.freq_range_set(freq_center, freq_span)
    spa.resolution_bandwidth = resolution_bandwidth
    return spa


def measure_floor_noise(spa: SpectrumAnalyzer, n_iter: int = 5) -> float:
    # Checkout
    t0 = spa.trace_get()
    fln = t0[:, 1]
    for i in range(n_iter - 1):
        fln = np.maximum(fln, spa.trace_get()[:, 1])

    mfln = fln.max()
    logger.info(f"maximum floor noise = {mfln:.1f}dBm")
    return mfln
