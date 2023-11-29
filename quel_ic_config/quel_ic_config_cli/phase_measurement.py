import json
import logging
import sys
import time
from argparse import ArgumentParser
from typing import Any, Collection, Dict, Final, List, Union, cast

import numpy as np
import numpy.typing as npt
from quel_cmod_scripting import QuelCmod
from quel_pyxsdb import get_jtagterminal_port

from quel_ic_config import Quel1ConfigSubsystem, Quel1NormalThermistor, Quel1PathSelectorThermistor
from quel_ic_config_utils import (
    CaptureReturnCode,
    E7HwType,
    Quel1E7ResourceMapper,
    Quel1WaveSubsystem,
    add_common_arguments,
    complete_ipaddrs,
)

DEFAULT_XSDB_PORT_FOR_CMOD: Final[int] = 36335
DEFAULT_HWSVR_PORT_FOR_CMOD: Final[int] = 6121
DEFAULT_NUM_SAMPLES_IN_EPOCH: Final[int] = 50000


logger = logging.getLogger()


# notes 10kHz ~ 50000samples at 500MSps
def phase_stat(iq: npt.NDArray[np.complex64], num_samples=DEFAULT_NUM_SAMPLES_IN_EPOCH) -> Dict[str, float]:
    if len(iq) < num_samples:
        logger.info(f"processing {len(iq)} samples")
    else:
        iq = iq[:num_samples]

    pwr = np.abs(iq)
    angle: npt.NDArray[np.float64] = np.angle(iq)
    # Notes: angle changes from 3.14 --> -3.14 suddenly.
    if max(angle) >= 3.0 and min(angle) < -3.0:
        angle = (angle + 2 * np.pi) % (2 * np.pi)

    pwr_mean = np.mean(pwr)
    pwr_std = np.sqrt(np.var(pwr))
    agl_mean = float(np.mean(angle)) * 180.0 / np.pi
    agl_std = np.sqrt(float(np.var(angle))) * 180.0 / np.pi
    agl_deltamax = (np.max(angle) - np.min(angle)) * 180.0 / np.pi
    return {
        "pwr_mean": np.floor(pwr_mean * 1000 + 0.5) / 1000.0,
        "pwr_std": np.floor(pwr_std * 1000 + 0.5) / 1000.0,
        "agl_mean": np.floor(agl_mean * 1000 + 0.5) / 1000.0,
        "agl_std": np.floor(agl_std * 1000 + 0.5) / 1000.0,
        "agl_deltamax": np.floor(agl_deltamax * 1000 + 0.5) / 1000.0,
    }


class Quel1TemperatureMeasurement:
    def __init__(self, cmod: QuelCmod):
        self._cmod = cmod
        self._ver = cmod.ver()
        self._th0 = Quel1NormalThermistor("0--21")
        self._th1 = Quel1PathSelectorThermistor("22--27")

    def measure(self) -> List[float]:
        for _ in range(3):
            t0 = self._cmod.thall()
            if t0 is None:
                continue
            return self._convert(t0)
        else:
            raise RuntimeError("failed to acquire temperature repeatedly")

    def plstat(self) -> List[int]:
        # TODO: workaround 'cast' should be removed in a correct manner.
        pls: npt.NDArray[np.int32] = cast(npt.NDArray[np.int32], self._cmod.plstat())
        return [int(x) for x in pls]

    def _convert(self, adc: npt.NDArray[np.int32]) -> List[float]:
        r = []
        for i in range(0, 26):
            r.append(self._th0.convert(adc[i]))
        for i in range(26, 28):
            r.append(self._th1.convert(adc[i]))
        return [np.floor(t * 100 + 0.5) / 100.0 for t in r]


#
# Loop
#
def loop(
    mxfe_to_use: Collection[int],
    active_rlines: Dict[int, str],
    wss: Quel1WaveSubsystem,
    rmap: Quel1E7ResourceMapper,
    tm: Union[Quel1TemperatureMeasurement, None],
):
    if tm is not None:
        logger.info(json.dumps({"temp": tm.measure()}))
        logger.info(json.dumps({"peltier": tm.plstat()}))

    capmods = {g: rmap.get_capture_module_of_rline(g, active_rlines[g]) for g in mxfe_to_use}

    for i in range(4):
        for g in mxfe_to_use:
            if active_rlines[g] == "r" and i >= 1:
                continue
            awg = rmap.get_awg_of_channel(g, i, 0)
            for _ in range(3):
                thunk = wss.simple_capture_start(
                    capmod=capmods[g], capunits={0}, num_words=16384 + 512, delay=0, triggering_awg=awg
                )
                wss.simple_cw_gen(awg=awg, amplitude=32767, num_repeats=(1, 1024 + 32))
                status, iqs = thunk.result()
                if status is CaptureReturnCode.SUCCESS:
                    break
            else:
                raise RuntimeError(f"failed to capture signal repeatedly (#{g})")
            stat: Dict[str, Any] = {"mxfe": g, "line": i}
            iq = iqs[0]
            stat.update(phase_stat(iq[1024:-1024]))
            logger.info(json.dumps(stat))


def cli_body():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    logging.getLogger("quel_ic_config_utils.quel1_wave_subsystem").setLevel(logging.WARNING)

    settings = {
        "2022": {"lo_mult": 85, "cnco": 1500_000_000, "sb": "U"},  # 10GHz
        "ntt": {"lo_mult": 85, "cnco": 1500_000_000, "sb": "L"},  # 7GHz
        "2023pseudo": {"lo_mult": 105, "cnco": 2500_000_000, "sb": "L"},  # 8GHz
    }

    parser = ArgumentParser(description="a phase fluctuation measurement tool")
    add_common_arguments(
        parser, use_mxfe=True, allow_implicit_mxfe=True, use_config_root=False, use_config_options=False
    )
    parser.add_argument(
        "--cmod_host",
        type=str,
        default="localhost",
        help="ip address of host PC of Cmod USB",
    )
    parser.add_argument(
        "--xsdb_port",
        type=int,
        default=DEFAULT_XSDB_PORT_FOR_CMOD,
        help="port of xsdb managing Cmod USB",
    )
    parser.add_argument(
        "--hwsvr_port",
        type=int,
        default=DEFAULT_HWSVR_PORT_FOR_CMOD,
        help="port of hw_server managing Cmod USB",
    )
    parser.add_argument("--cmod_jtag", type=str, default="", help="jtag id of the Cmod USB adapter")
    parser.add_argument("--duration", type=int, default=30, help="measurement duration in second")
    parser.add_argument(
        "--frequency",
        choices=set(settings.keys()),
        default="2022",
        help="frequency for test signal, 10GHz for 2022, 7GHz for ntt, and 8GHz for 2023",
    )
    args = parser.parse_args()
    complete_ipaddrs(args)

    mxfe_to_use = args.mxfe
    #
    # Thermal
    #
    if args.cmod_jtag != "":
        cmod_port = get_jtagterminal_port(
            adapter_id=args.cmod_jtag, host=args.cmod_host, xsdb_port=args.xsdb_port, hwsvr_port=args.hwsvr_port
        )
        cmod = QuelCmod(host=args.cmod_host, port=cmod_port)
        tm: Union[Quel1TemperatureMeasurement, None] = Quel1TemperatureMeasurement(cmod)
    else:
        tm = None

    #
    # Config (1/2)
    #
    css = Quel1ConfigSubsystem(
        css_addr=str(args.ipaddr_css),
        boxtype=args.boxtype,
    )
    for mxfe in mxfe_to_use:
        if not css.configure_mxfe(mxfe):
            raise RuntimeError(f"phase stability test is not ready for group-{mxfe}")

    #
    # Wave
    #
    wss = Quel1WaveSubsystem(str(args.ipaddr_wss), E7HwType.SIMPLE_MULTI_CLASSIC)

    #
    # Targets select
    #
    rmap = Quel1E7ResourceMapper(css, wss)
    active_rlines: Dict[int, str] = {}
    for g in mxfe_to_use:
        active_rline = rmap.get_active_rlines_of_group(g)
        assert len(active_rline) == 1
        active_rlines[g] = tuple(active_rline)[0]
        logger.info(f"group-{g} is configured to watch {'read' if active_rline == 'r' else 'monitor'} path")

    #
    # Config (2/2)
    #
    for j in mxfe_to_use:
        setting = settings[args.frequency]
        for i in range(4):
            css.set_lo_multiplier(j, i, cast(int, setting["lo_mult"]))
            css.set_dac_cnco(j, i, cast(int, setting["cnco"]))
            css.set_dac_fnco(j, i, 0, 0)
            css.set_vatt(j, i, 0xC00)
            css.set_sideband(j, i, cast(str, setting["sb"]))

        if active_rlines[j] == "r":
            css.set_adc_cnco(j, "r", cast(int, setting["cnco"]))
            css.set_adc_fnco(j, "r", 0, freq_in_hz=0)
            css.activate_read_loop(j)
        elif active_rlines[j] == "m":
            css.set_adc_cnco(j, "m", cast(int, setting["cnco"]))
            css.set_adc_fnco(j, "m", 0, freq_in_hz=0)
            css.activate_monitor_loop(j)
        else:
            raise AssertionError

    #
    # Measurement!
    #
    t_complete = time.perf_counter() + int(args.duration)
    while time.perf_counter() < t_complete:
        loop(mxfe_to_use, active_rlines, wss, rmap, tm)


def cli_main():
    try:
        cli_body()
        sys.exit(0)
    except Exception as e:
        logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    cli_body()
