import logging
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from testlibs.basic_scan_common import init_box
from testlibs.updated_linkup_phase2 import Quel1WaveGen
from testlibs.updated_linkup_phase3 import Quel1WaveCap

logger = logging.getLogger(__name__)


def make_boxpool(
    settings: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Tuple[bool, bool, Quel1WaveGen, Quel1WaveGen, Quel1WaveCap, Quel1WaveCap]]:
    boxpool: Dict[str, Tuple[bool, bool, Quel1WaveGen, Quel1WaveGen, Quel1WaveCap, Quel1WaveCap]] = {}
    for k, v in settings.items():
        if k.startswith("BOX"):
            boxpool[k] = init_box(**v)
    return boxpool


def _retrieve_sender(boxpool, conf: Mapping[str, Any]) -> Tuple[Quel1WaveGen, int, int]:
    sender_box = boxpool[conf["box"]]
    if sender_box[conf["mxfe"]]:
        return sender_box[2 + conf["mxfe"]], conf["dac"], conf["vatt"]
    else:
        raise RuntimeError("sender is not available due to failed initialization")


def _retrieve_capturer(boxpool, conf: Mapping[str, Any]) -> Quel1WaveCap:
    capturer_box = boxpool[conf["box"]]
    if capturer_box[conf["mxfe"]]:
        return capturer_box[4 + conf["mxfe"]]
    else:
        raise RuntimeError("capturer is not available due to failed initialization")


def init_wgs(
    senders: Sequence[str],
    settings: Mapping[str, Mapping[str, Any]],
    boxpool: Dict[str, Tuple[bool, bool, Quel1WaveGen, Quel1WaveGen, Quel1WaveCap, Quel1WaveCap]],
) -> Dict[str, Tuple[Quel1WaveGen, int]]:
    common = settings["COMMON"]

    if len(set(senders)) != len(senders):
        raise ValueError("duplicated sender in the given list of senders")

    wgs: Dict[str, Tuple[Quel1WaveGen, int]] = {}
    for sender in senders:
        wg, dac_idx, vatt = _retrieve_sender(boxpool, settings[sender])
        wg.run(
            line=dac_idx,
            awg_idx=0,
            sideband=common["sideband"],
            lo_mhz=common["lo_mhz"],
            cnco_mhz=common["cnco_mhz"],
            vatt=vatt,
            d=False,
        )
        wgs[sender] = (wg, dac_idx)

    return wgs


def init_capture(
    settings: Mapping[str, Mapping[str, Any]],
    boxpool: Dict[str, Tuple[bool, bool, Quel1WaveGen, Quel1WaveGen, Quel1WaveCap, Quel1WaveCap]],
) -> Tuple[Quel1WaveCap, npt.NDArray[np.complexfloating]]:
    cp = _retrieve_capturer(boxpool, settings["CAPTURER"])

    # confirm the capture data is silent when no signal
    common = settings["COMMON"]
    cp.run("r", enable_internal_loop=False, lo_mhz=common["lo_mhz"], cnco_mhz=common["cnco_mhz"])
    c_silent = cp.complete()
    if c_silent is None:
        raise RuntimeError("failed to capture data")

    return cp, c_silent[0]


def init_units(
    senders: Sequence[str], settings: Mapping[str, Mapping[str, Any]]
) -> Tuple[Dict[str, Tuple[Quel1WaveGen, int]], Tuple[Quel1WaveCap, npt.NDArray[np.complexfloating]]]:
    boxpool = make_boxpool(settings)
    wgs = init_wgs(senders, settings, boxpool)
    cp = init_capture(settings, boxpool)
    return wgs, cp


def calc_angle(iq) -> Tuple[float, float]:
    angle = np.angle(iq)
    min_angle = min(angle)
    max_angle = max(angle)
    if max_angle - min_angle > 6.0:
        angle = (angle + 2 * np.pi) % np.pi

    avg = np.mean(angle) * 180.0 / np.pi
    sd = np.sqrt(np.var(angle)) * 180.0 / np.pi
    return avg, sd


def plot_iqs(iq_dict) -> None:
    n_plot = len(iq_dict)
    fig = plt.figure()

    m = 0
    for _, iq in iq_dict.items():
        m = max(m, np.max(abs(np.real(iq))))
        m = max(m, np.max(abs(np.imag(iq))))

    idx = 0
    for title, iq in iq_dict.items():
        ax = fig.add_subplot(n_plot, 1, idx + 1)
        ax.plot(np.real(iq))
        ax.plot(np.imag(iq))
        ax.text(0.05, 0.1, f"{title}", transform=ax.transAxes)
        ax.set_ylim([-m * 1.1, m * 1.1])
        idx += 1
    plt.show()
