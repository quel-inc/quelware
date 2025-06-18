import json
import logging
import os.path as osp
import sys
import time
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Final, Optional, Set, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import polars as pl
from e7awghal import AwgParam, CapParam, CapSection, WaveChunk
from quel_cmod_scripting import QuelCmod
from quel_pyxsdb import get_jtagterminal_port

from quel_ic_config import Quel1Box, Quel1BoxType, Quel1PortType, Quel1seAnyConfigSubsystem, Quel1Thermistor
from quel_ic_config_cli.stability_data_models import (
    QUEL1_ACTUATORS,
    QUEL1_THERMISTORS,
    QuelDataModel,
    TempCtrlStatus,
    WaveStatistics,
    get_tempctrlstatus_from_boxtype,
)
from quel_ic_config_utils import add_common_arguments, add_common_workaround_arguments, complete_ipaddrs

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH: Final[Path] = Path(osp.dirname(__file__)) / "settings"


class StabilityMeasurement(metaclass=ABCMeta):
    @abstractmethod
    def get_stability_info(self, time_from_start: float) -> list[QuelDataModel]: ...


# Notes: I guess StabilityMeasurement is too abstract, keep it before its necessity is clarified.
class TemperatureMeasurement(StabilityMeasurement):
    def __init__(self, boxtype: Quel1BoxType):
        self._tcscls: Type[TempCtrlStatus] = get_tempctrlstatus_from_boxtype(boxtype)


class Quel1TemperatureMeasurement(TemperatureMeasurement):
    DEFAULT_XSDB_PORT_FOR_CMOD: Final[int] = 36335
    DEFAULT_HWSVR_PORT_FOR_CMOD: Final[int] = 6121
    DEFAULT_PERIOD: Final[float] = 10.0

    _THERMISTORS: dict[int, Quel1Thermistor] = QUEL1_THERMISTORS
    _ACTUATORS: dict[str, tuple[str, int]] = QUEL1_ACTUATORS

    def __init__(self, cmod: QuelCmod, boxtype: Quel1BoxType):
        super().__init__(boxtype)
        self._cmod = cmod
        self._ver = cmod.ver()

    def _measure(self) -> dict[str, float]:
        for _ in range(3):
            t0 = self._cmod.thall()
            if t0 is None:
                continue
            return self._convert(t0)
        else:
            raise RuntimeError("failed to acquire temperature repeatedly")

    def _plstat(self) -> npt.NDArray[np.int32]:
        p = self._cmod.plstat()
        if p is None:
            raise RuntimeError("failed to acquire peltier drive values")
        return p

    def _convert(self, adc: npt.NDArray[np.int32]) -> dict[str, float]:
        return {th.name: round(th.convert(adc[idx]), 2) for idx, th in self._THERMISTORS.items()}

    def _make_tempctrl_stat(
        self, time_from_start: float, temperatures: dict[str, float], actuators: npt.NDArray[np.int32]
    ) -> list[QuelDataModel]:
        tempctrl_dicts: list[QuelDataModel] = []
        for name, temp in temperatures.items():
            a_type, a_idx = self._ACTUATORS.get(name, (None, None))
            tempctrl_model = self._tcscls(
                time_from_start=time_from_start,
                location_name=name,
                temperature=temp,
                actuator_type=a_type,
                actuator_val=float(actuators[a_idx]) if a_idx is not None else None,
            )
            tempctrl_dicts.append(tempctrl_model)

        return tempctrl_dicts

    def get_stability_info(self, time_from_start: float) -> list[QuelDataModel]:
        t0 = time.perf_counter()
        temps = self._measure()
        plstat = self._plstat()
        while time.perf_counter() - t0 < self.DEFAULT_PERIOD:
            time.sleep(0.25)
        return self._make_tempctrl_stat(time_from_start, temps, plstat)


class Quel1seTemperatureMeasurement(TemperatureMeasurement):
    def __init__(self, css: Quel1seAnyConfigSubsystem):
        super().__init__(css.boxtype)
        self._css: Quel1seAnyConfigSubsystem = css

    def _make_tempctrl_stat(
        self,
        time_from_start: float,
        temperatures: dict[str, float],
        actuators: dict[str, dict[str, float]],
    ) -> list[QuelDataModel]:
        tempmodels: list[QuelDataModel] = []

        for loc_name, temp in temperatures.items():
            if loc_name in actuators["fan"]:
                tm = self._tcscls(
                    time_from_start=time_from_start,
                    location_name=loc_name,
                    temperature=temp,
                    actuator_type="fan",
                    actuator_val=actuators["fan"][loc_name],
                )
            elif loc_name in actuators["heater"]:
                tm = self._tcscls(
                    time_from_start=time_from_start,
                    location_name=loc_name,
                    temperature=temp,
                    actuator_type="heater",
                    actuator_val=actuators["heater"][loc_name],
                )
            else:
                tm = self._tcscls(
                    time_from_start=time_from_start,
                    location_name=loc_name,
                    temperature=temp,
                    actuator_type=None,
                    actuator_val=None,
                )
            tempmodels.append(tm)

        return tempmodels

    def get_stability_info(self, time_from_start: float) -> list[QuelDataModel]:
        temperatures: dict[str, float] = self._css.get_tempctrl_temperature().result()
        for mxfe_idx in range(2):
            mxfe_temp_max, mxfe_temp_min = self._css.get_mxfe_temperature_range(mxfe_idx)
            temperatures[f"mxfe{mxfe_idx}_max"] = mxfe_temp_max
            temperatures[f"mxfe{mxfe_idx}_min"] = mxfe_temp_min
        actuators = self._css.get_tempctrl_actuator_output()
        return self._make_tempctrl_stat(time_from_start, temperatures, actuators)


class WaveStabilityMeasurement(StabilityMeasurement):
    _DEFAULT_NUM_SAMPLES_IN_EPOCH: Final[int] = 50000

    def __init__(
        self,
        box: Quel1Box,
        target_output_ports: Set[Quel1PortType],
    ):
        self._box = box
        self._wvcls: Type[WaveStatistics] = WaveStatistics
        self._inout_combinations = self._valid_inout_combinations(target_output_ports)
        self._init_awg_and_cap()

    def _is_matched_inout_frequency(self, in_port_dump: Dict[str, Any], out_port_dump: Dict[str, Any]) -> bool:
        # up-convert output frequency with LO and NCO frequency for channel 0
        if_freq = out_port_dump["channels"][0]["fnco_freq"] + out_port_dump["cnco_freq"]
        if "lo_freq" in out_port_dump.keys():
            # analog mixing
            out_freq = (
                if_freq + out_port_dump["lo_freq"]
                if out_port_dump["sideband"] == "U"
                else out_port_dump["lo_freq"] - if_freq
            )
        else:
            out_freq = if_freq

        # analog down-conversion with LO
        analog_dc_freq = abs(out_freq - in_port_dump["lo_freq"])
        # after digital down-conversion, check if the signal is DC
        if analog_dc_freq - in_port_dump["runits"][0]["fnco_freq"] - in_port_dump["cnco_freq"] == 0.0:
            return True

        return False

    def _valid_inout_combinations(
        self, target_output_ports: Set[Quel1PortType]
    ) -> Set[Tuple[Quel1PortType, Quel1PortType]]:
        boxdump = self._box.dump_box()["ports"]
        valid_combinations: Set[Tuple[Quel1PortType, Quel1PortType]] = set()
        for input_port in self._box.get_input_ports():
            for output_port in self._box.get_loopbacks_of_port(input_port):
                if output_port not in target_output_ports:
                    continue
                if self._is_matched_inout_frequency(boxdump[input_port], boxdump[output_port]):
                    valid_combinations.add((input_port, output_port))
        logger.info(f"valid input-output port combinations: {valid_combinations}")
        return valid_combinations

    def _init_awg_and_cap(self):
        cw_iq = np.zeros(64, dtype=np.complex64)
        cw_iq[:] = 32767.0 + 0.0j
        ap = AwgParam(num_wait_word=0, num_repeat=1024 + 32)
        ap.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=1))
        for port in self._box.get_output_ports():
            if port in {ports[1] for ports in self._inout_combinations}:
                self._box.register_wavedata(port, 0, "cw32767", cw_iq)
                self._box.config_channel(port, 0, awg_param=ap)

        cp = CapParam(num_repeat=1)
        cp.sections.append(CapSection(name="s0", num_capture_word=(16384 + 512)))
        for port in {ports[0] for ports in self._inout_combinations}:
            self._box.config_runit(port, 0, capture_param=cp)

    def phase_stat(
        self,
        time_from_start: float,
        input_port: Quel1PortType,
        output_port: Quel1PortType,
        iq: npt.NDArray[np.complex64],
    ) -> WaveStatistics:
        num_samples = self._DEFAULT_NUM_SAMPLES_IN_EPOCH
        if len(iq) < num_samples:
            logger.info(f"processing {len(iq)} samples")
        else:
            iq = iq[:num_samples]

        pwr: npt.NDArray[np.float64] = np.abs(iq)
        angle: npt.NDArray[np.float64] = np.angle(iq)
        # Notes: angle changes from pi --> -pi suddenly.
        if max(angle) >= 3.0 and min(angle) < -3.0:
            angle = (angle + 2 * np.pi) % (2 * np.pi)  # the max value is (-3.0 + 2*pi) % (2*pi) = pi + (pi-3.0) > pi

        pwr_mean: float = float(np.mean(pwr))
        pwr_std: float = float(np.sqrt(np.var(pwr)))
        agl_mean: float = float(np.mean(angle)) * 180.0 / np.pi
        agl_std: float = float(np.sqrt(np.var(angle))) * 180.0 / np.pi
        agl_deltamax: float = float((np.max(angle) - np.min(angle))) * 180.0 / np.pi
        model = self._wvcls(
            time_from_start=time_from_start,
            input_port=str(input_port),
            output_port=str(output_port),
            power_mean=round(pwr_mean, 3),
            power_std=round(pwr_std, 3),
            angle_mean=round(agl_mean, 3),
            angle_std=round(agl_std, 3),
            angle_deltamax=round(agl_deltamax, 3),
        )
        return model

    def get_stability_info(self, time_from_start: float) -> list[QuelDataModel]:
        wavestability_list: list[QuelDataModel] = []
        for input_port, output_port in self._inout_combinations:
            c_task, g_task = self._box.start_capture_by_awg_trigger(
                {(input_port, 0)},
                {(output_port, 0)},
            )
            assert g_task.result() is None
            rdrs0 = c_task.result()
            data0 = rdrs0[input_port, 0].as_wave_dict()
            iq = data0["s0"][0]
            model = self.phase_stat(time_from_start, input_port, output_port, iq[1024:-1024])
            wavestability_list.append(model)
        return wavestability_list


class DataHandler:
    def __init__(
        self,
        name: str,
        label: str,
        measurement: StabilityMeasurement,
        start_time: float,
        out_dir_path: Path,
        interval: Optional[float] = None,
    ):
        self._name: str = name
        self._label: str = label
        self._measurement: StabilityMeasurement = measurement
        self._header_flag: bool = True
        self._start_time: float = start_time
        self._interval: Union[float, None] = interval
        self._filepath: Path = self.make_csv_path(out_dir_path)

    def make_csv_path(self, out_dir_path: Path) -> Path:
        out_subdir_path = out_dir_path / Path(self._name)
        out_subdir_path.mkdir(parents=True, exist_ok=True)
        if not out_subdir_path.exists():
            raise RuntimeError(f"failed to create a directory '{str(out_subdir_path)}")
        return out_subdir_path / Path(f"{self._name}_{self._label}.csv")

    def get_and_write_data(self):
        time_from_start: float = time.perf_counter() - self._start_time
        model_list = self._measurement.get_stability_info(time_from_start)
        model_dumps = [model.model_dump() for model in model_list]
        for m in model_dumps:
            logger.info(json.dumps(m))
        df_stability = pl.DataFrame(model_dumps)
        with open(self._filepath, mode="ab") as f:
            df_stability.write_csv(f, include_header=self._header_flag)
        if self._header_flag:
            self._header_flag = False
        if isinstance(self._measurement, Quel1TemperatureMeasurement) and (self._interval is not None):
            time.sleep(self._interval)


def parse_target_ports(optstr: str) -> set[Quel1PortType]:
    tokens = [t.strip() for t in optstr.split(",")]
    ctx: list[str] = []
    outports = set()
    for cur in tokens:
        if len(cur) == 0:
            raise ValueError
        if cur.startswith("("):
            if len(ctx) != 0:
                raise ValueError
            else:
                ctx.append(cur)
        elif cur.endswith(")"):
            if len(ctx) != 1:
                raise ValueError
            else:
                outport: Quel1PortType = (int(ctx[0][1:]), int(cur[:-1]))
                ctx.clear()
                if outport in outports:
                    raise ValueError
                else:
                    outports.add(outport)
        else:
            if len(ctx) != 0:
                raise ValueError
            else:
                outport = int(cur)
                if outport in outports:
                    raise ValueError
                else:
                    outports.add(outport)
    return outports


def load_configs(box: Quel1Box, json_path: Optional[Path] = None) -> None:
    boxtype = Quel1BoxType.fromstr(box.boxtype)
    if boxtype in {Quel1BoxType.QuEL1SE_RIKEN8, Quel1BoxType.QuEL1SE_RIKEN8DBG}:
        json_name: str = "wave_stability_quel1se-riken8.json"
    elif boxtype in {Quel1BoxType.QuEL1SE_FUJITSU11_TypeA, Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA}:
        json_name = "wave_stability_quel1-a.json"
    elif boxtype in {Quel1BoxType.QuEL1SE_FUJITSU11_TypeB, Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeB}:
        json_name = "wave_stability_quel1-b.json"
    elif boxtype == Quel1BoxType.QuEL1_TypeA:
        json_name = "wave_stability_quel1-a.json"
    elif boxtype == Quel1BoxType.QuEL1_TypeB:
        json_name = "wave_stability_quel1-b.json"
    else:
        raise ValueError(f"{boxtype} is not supported yet")

    json_path = json_path or _DEFAULT_CONFIG_PATH / json_name
    if json_path.exists():
        box.config_box_from_jsonfile(json_path)
    else:
        raise ValueError(f"'{json_path}' does not exist.")


def dump_configs(
    box: Quel1Box,
    out_dir_path: Path,
    title: str,
):
    out_dir_path_configs = out_dir_path / Path("configs")
    out_dir_path_configs.mkdir(parents=True, exist_ok=True)
    if not out_dir_path_configs.exists():
        raise RuntimeError(f"failed to create a directory '{str(out_dir_path_configs)}'")

    json_file_path = out_dir_path_configs / Path(f"configs_{title}.json")
    box.dump_box_to_jsonfile(json_file_path)


def cli_args() -> Namespace:
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    logging.getLogger("quel_ic_config_utils.quel1_wave_subsystem").setLevel(logging.WARNING)

    parser = ArgumentParser(description="a phase fluctuation measurement tool")
    add_common_arguments(parser, use_config_root=False, use_config_options=False)
    add_common_workaround_arguments(parser, use_ignore_crc_error_of_mxfe=True)
    parser.add_argument(
        "--cmod_host",
        type=str,
        default="localhost",
        help="ip address of host PC of Cmod USB",
    )
    parser.add_argument(
        "--xsdb_port",
        type=int,
        default=Quel1TemperatureMeasurement.DEFAULT_XSDB_PORT_FOR_CMOD,
        help="port of xsdb managing Cmod USB",
    )
    parser.add_argument(
        "--hwsvr_port",
        type=int,
        default=Quel1TemperatureMeasurement.DEFAULT_HWSVR_PORT_FOR_CMOD,
        help="port of hw_server managing Cmod USB",
    )
    parser.add_argument("--cmod_jtag", type=str, default="", help="jtag id of the Cmod USB adapter")
    parser.add_argument("--conf", type=Path, default=None, help="json file for configuration")
    parser.add_argument("--duration", type=int, default=30, help="measurement duration in second")
    parser.add_argument("--outdir", type=Path, required=True, help="output file directory")
    parser.add_argument("--skip_configs", action="store_true", default=False, help="skip configurations")
    parser.add_argument("--boxname", type=str, default=None, help="custom box name")
    parser.add_argument(
        "--outports",
        type=parse_target_ports,
        default=None,
        help="target output ports with ',' seperation. ex. '1,(1,1),6'. ",
    )
    args = parser.parse_args()

    complete_ipaddrs(args)

    return args


def cli_init(args: Namespace):
    box: Quel1Box = Quel1Box.create(
        ipaddr_wss=str(args.ipaddr_wss),
        ipaddr_sss=str(args.ipaddr_sss),
        ipaddr_css=str(args.ipaddr_css),
        boxtype=args.boxtype,
    )
    status = box.reconnect()

    target_output_ports: set[Quel1PortType] = args.outports or box.get_output_ports()
    required_mxfe: set[int] = set()
    for outport in target_output_ports:
        # TODO: this part is tricky, consider better ways...
        gr, ln = box._convert_output_port(outport)
        mxfe_idx, _ = box.css.get_dac_idx(gr, ln)
        required_mxfe.add(mxfe_idx)

    for mxfe_idx in required_mxfe:
        if not status[mxfe_idx]:
            raise RuntimeError(f"mxfe-#{mxfe_idx} of box {str(args.ipaddr_wss)} is not linked up properly")

    if args.boxtype.is_quel1se():
        if isinstance(box.css, Quel1seAnyConfigSubsystem):
            tmeas: Union[TemperatureMeasurement, None] = Quel1seTemperatureMeasurement(box.css)
        else:
            raise AssertionError("never happens")
    else:
        if args.cmod_jtag != "":
            cmod_port = get_jtagterminal_port(
                adapter_id=args.cmod_jtag, host=args.cmod_host, xsdb_port=args.xsdb_port, hwsvr_port=args.hwsvr_port
            )
            cmod = QuelCmod(host=args.cmod_host, port=cmod_port)
            tmeas = Quel1TemperatureMeasurement(cmod, args.boxtype)
        else:
            tmeas = None
            logger.warning("ONLY PHASE MEASUREMENT will be conducted since jtag adapter id is not specified")

    if args.skip_configs is False:
        load_configs(box, args.conf)
    pmeas = WaveStabilityMeasurement(box, target_output_ports)
    return box, tmeas, pmeas


def cli_body(args: Namespace, box: Quel1Box, tmeas: Optional[StabilityMeasurement], pmeas: StabilityMeasurement):
    output_file_label = f"{str(args.ipaddr_wss).replace('.','_')}_{time.strftime('%Y%m%d%H%M%S')}"
    if args.boxname is not None:
        output_file_label = args.boxname + "_" + output_file_label
    dump_configs(box, args.outdir, output_file_label)
    start_time = time.perf_counter()
    data_handlers: list[DataHandler] = []
    data_handlers.append(DataHandler("wavestability", output_file_label, pmeas, start_time, args.outdir))
    if tmeas is not None:
        data_handlers.append(DataHandler("tempctrl", output_file_label, tmeas, start_time, args.outdir))

    executor = ThreadPoolExecutor(max_workers=len(data_handlers))
    futures: list[Future[None]] = []
    for data_handler in data_handlers:
        futures.append(executor.submit(data_handler.get_and_write_data))

    while (time.perf_counter() - start_time) < args.duration:
        for idx, future in enumerate(futures):
            if future.done():
                future.result()
                futures[idx] = executor.submit(data_handlers[idx].get_and_write_data)
        time.sleep(0.01)

    for future in futures:
        future.result()


def cli_main():
    box: Union[Quel1Box, None] = None
    try:
        args = cli_args()
        box, tmeas, pmeas = cli_init(args)
        if box is None:
            raise AssertionError("never happens")
        cli_body(args, box, tmeas, pmeas)
    except Exception as e:
        logger.error(e)
        sys.exit(1)
    finally:
        if box:
            del box

    sys.exit(0)


if __name__ == "__main__":
    args0 = cli_args()
    box0, tmeas0, pmeas0 = cli_init(args0)
    cli_body(args0, box0, tmeas0, pmeas0)
