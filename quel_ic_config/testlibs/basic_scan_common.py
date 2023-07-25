import argparse
import logging
import time
from pathlib import Path
from typing import Collection, Tuple, Union

import e7awgsw
import numpy as np
from quel_inst_tool import E4405b, ExpectedSpectrumPeaks, InstDevManager, MeasuredSpectrumPeak

from quel_ic_config import Quel1BoxType, Quel1ConfigObjects, QuelConfigOption
from testlibs.updated_linkup_phase0 import Quel1ConfigSubSystemReferencePhase0, Quel1ConfigSubSystemReferencePhase0Pre
from testlibs.updated_linkup_phase1 import Quel1ConfigSubSystemReferencePhase1
from testlibs.updated_linkup_phase2 import Quel1WaveGen
from testlibs.updated_linkup_phase3 import Quel1WaveCap

logger = logging.getLogger(__name__)

MAX_CAPTURE_RETRY = 5

DEFAULT_FREQ_CENTER = 9e9
DEFAULT_FREQ_SPAN = 5e9
DEFAULT_SWEEP_POINTS = 4001
DEFAULT_RESOLUTION_BANDWIDTH = 1e5


def init_box(
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: str,
    mxfe: str,
    config_root: str,
    config_options: Union[Collection[str], None] = None,
) -> Tuple[bool, bool, Quel1WaveGen, Quel1WaveGen, Quel1WaveCap, Quel1WaveCap]:
    """create QuEL testing objects and initialize ICs
    :param ipaddr_wss: IP address of the wave generation subsystem of the target box
    :param ipaddr_sss: IP address of the sequencer subsystem of the target box
    :param ipaddr_css: IP address of the configuration subsystem of the target box
    :param boxtype: type of the target box
    :param mxfe: target mxfes of the target box
    :param config_root: root path of config setting files to read
    :param config_options: a collection of config options
    :return: QuEL testing objects
    """
    _boxtype = Quel1BoxType.fromstr(boxtype)
    _config_options = {QuelConfigOption(o) for o in config_options} if config_options is not None else set()
    _config_root = Path(config_root)
    qco = Quel1ConfigObjects(ipaddr_css, _boxtype, _config_root, _config_options)

    p0_pre = Quel1ConfigSubSystemReferencePhase0Pre(qco)
    p0_g0 = Quel1ConfigSubSystemReferencePhase0(qco, 0)
    p0_g1 = Quel1ConfigSubSystemReferencePhase0(qco, 1)
    p1_g0 = Quel1ConfigSubSystemReferencePhase1(ipaddr_wss, qco, 0)
    p1_g1 = Quel1ConfigSubSystemReferencePhase1(ipaddr_wss, qco, 1)
    p2_g0 = Quel1WaveGen(ipaddr_wss, qco, 0)
    p2_g1 = Quel1WaveGen(ipaddr_wss, qco, 1)
    p3_g0 = Quel1WaveCap(ipaddr_wss, qco, 0)
    p3_g1 = Quel1WaveCap(ipaddr_wss, qco, 1)

    linkup_ok = [False, False]
    p0_pre.run()

    if mxfe in ("0", "both"):
        p0_g0.run()
        for i in range(MAX_CAPTURE_RETRY):
            try:
                judge, _ = p1_g0.run()
                if judge:
                    linkup_ok[0] = True
                break
            except e7awgsw.exception.CaptureUnitTimeoutError as e:
                logger.warning(e)
                time.sleep(1)
        else:
            raise RuntimeError("too many capture unit failure")
        p2_g0.force_stop()

    if mxfe in ("1", "both"):
        p0_g1.run()
        for i in range(MAX_CAPTURE_RETRY):
            try:
                judge, _ = p1_g1.run()
                if judge:
                    linkup_ok[1] = True
                break
            except e7awgsw.exception.CaptureUnitTimeoutError as e:
                logger.warning(e)
                time.sleep(1)
        else:
            raise RuntimeError("too many capture unit failure")
        p2_g1.force_stop()

    if ipaddr_sss is not None:
        # TODO: write scheduler object creation here.
        pass

    return linkup_ok[0], linkup_ok[1], p2_g0, p2_g1, p3_g0, p3_g1


def init_e4405b(
    freq_center=DEFAULT_FREQ_CENTER,
    freq_span=DEFAULT_FREQ_SPAN,
    sweep_points=DEFAULT_SWEEP_POINTS,
    resolution_bandwidth=DEFAULT_RESOLUTION_BANDWIDTH,
):
    im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so", blacklist=["GPIB0::6::INSTR"])
    e4405b = E4405b(im.lookup(prod_id="E4405B"))
    e4405b.reset()
    e4405b.display_enable = False
    e4405b.trace_mode = "WRIT"
    e4405b.freq_center = freq_center
    e4405b.freq_span = freq_span
    e4405b.sweep_points = sweep_points
    e4405b.resolution_bandwidth = resolution_bandwidth
    # e4405b.resolution_bandwidth = 5e4   # floor noise < -70dBm, but spurious peaks higher than -70dBm exist
    # e4405b.resolution_bandwidth = 1e5  # floor noise < -65dBm
    # e4405b.resolution_bandwidth = 1e6   # floor noise < -55dBm
    return e4405b


def measure_floor_noise(e4405b: E4405b, n_iter: int = 5) -> float:
    # Checkout
    t0 = e4405b.trace_get()
    fln = t0[:, 1]
    for i in range(n_iter - 1):
        fln = np.maximum(fln, e4405b.trace_get()[:, 1])

    mfln = fln.max()
    logger.info(f"maximum floor noise = {mfln:.1f}dBm")
    return mfln


def add_common_arguments(p: argparse.ArgumentParser, postfix: str = "") -> None:
    p.add_argument(
        f"--ipaddr_wss{postfix}",
        type=str,
        required=True,
        help="IP address of the wave generation/capture subsystem of the target box",
    )
    p.add_argument(
        f"--ipaddr_sss{postfix}",
        type=str,
        required=True,
        help="IP address of the wave sequencer subsystem of the target box",
    )
    p.add_argument(
        f"--ipaddr_css{postfix}",
        type=str,
        required=True,
        help="IP address of the configuration subsystem of the target box",
    )
    p.add_argument(
        f"--boxtype{postfix}",
        type=str,
        choices=["quel1-a", "quel1-b", "qube-a", "qube-b"],
        required=True,
        help="a unit type of the target box",
    )
    p.add_argument(f"--mxfe{postfix}", choices=("0", "1", "both"), required=True, help="MxFEs under test")
    parser.add_argument(
        "--config_root",
        type=Path,
        default="settings",
        help="path to configuration file root",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = argparse.ArgumentParser(
        description="check the basic functionalities about wave generation of QuEL-1 with a spectrum analyzer"
    )
    add_common_arguments(parser)
    args = parser.parse_args()

    # Init: QuEL mxfe
    linkup_g0, linkup_g1, css_p2_g0, css_p2_g1, css_p3_g0, css_p3_g1 = init_box(
        args.ipaddr_wss, args.ipaddr_sss, args.ipaddr_css, args.boxtype, args.mxfe, args.config_root
    )

    # Init: Spectrum Analyzer
    e4405b_ = init_e4405b()
    max_floor_noise = measure_floor_noise(e4405b_, 5)
    assert max_floor_noise < -62.0

    # Measurement Example
    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (8991e6, -20)])
    assert e0.validate_with_measurement_condition(e4405b_.max_freq_error_get())

    if linkup_g0:
        css_p2_g0.run(2, 0, cnco_mhz=2000, fnco_mhz=13)
        css_p2_g0.run(3, 0, cnco_mhz=3000, fnco_mhz=9)
        m0 = MeasuredSpectrumPeak.from_spectrumanalyzer(e4405b_, -60.0)
        j0, s0, w0 = e0.match(m0)
        assert all(j0) and len(s0) == 0 and len(w0) == 0
    else:
        logger.error("linkup of mxfe0 fails, no test is conducted")
