import csv
import glob
import logging
import os
import re
import sys
import time
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Final, List, Literal, Optional, Set, Tuple, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons
from pandas import Series

from quel_ic_config import QUEL1_BOXTYPE_ALIAS, Quel1BoxType
from quel_ic_config_cli.stability_data_models import QuelDataModel, WaveStatistics, get_tempctrlstatus_from_boxtype
from quel_ic_config_utils.common_arguments import parse_boxtype

logger = logging.getLogger()
UPDATE_INTERVAL: Final[int] = 20000


class AbstractGraph(metaclass=ABCMeta):
    WARN_TIME_THRESHOLD: Final[float] = 60.0

    def __init__(
        self,
        csv_path: Path,
        time_window: Tuple[Optional[float], Optional[float]],
        boxtype: Optional[Quel1BoxType] = None,
    ):
        self._time_window: Tuple[Optional[float], Optional[float]] = time_window
        self._latest_time = 0.0
        self._file = open(csv_path, "r")
        self._csv_reader = csv.DictReader(self._file)
        self._schema = self._get_schema(boxtype)

    def _get_schema(self, boxtype: Optional[Quel1BoxType]) -> Type[QuelDataModel]:
        if isinstance(self, WaveGraph):
            return WaveStatistics
        elif isinstance(self, TempCtrlGraph):
            if boxtype is not None:
                return get_tempctrlstatus_from_boxtype(boxtype)
            else:
                raise ValueError("boxtype is required to get tempctrl schema")
        else:
            raise ValueError(f"{type(self)} is not a recognized subclass of AbstractGraph")

    def _get_new_lines(self) -> Optional[pd.DataFrame]:
        try:
            new_lines: List[QuelDataModel] = []
            for row in self._csv_reader:
                # Convert empty strings to None
                processed_row = {
                    key: (None if value == "" else value) for key, value in row.items() if key != "thermistor_id"
                }
                new_lines.append(self._schema.model_validate(processed_row))
            if new_lines:
                self._latest_time = time.time()  # Store the current time when data comes
                return pd.DataFrame([quelmodel.model_dump() for quelmodel in new_lines])
            else:
                elapsed_time = time.time() - self._latest_time
                if elapsed_time >= self.WARN_TIME_THRESHOLD:
                    logger.warning(f"No data received for {int(elapsed_time)} seconds since the last valid data.")
                return None

        except OSError as e:
            logger.error(f"An unexpected error occurred while accessing the file: {e}")
            self._file.close()
            sys.exit(1)

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def update_graph(self, frame: int):  # need argument "frame" for FuncAnimation
        pass

    @property
    @abstractmethod
    def fig(self) -> Figure:
        pass

    def _update_axes(self, axs, fig):
        for ax in axs:
            ax.relim(visible_only=True)
            ax.set_autoscalex_on(True)
            ax.autoscale_view()
            xmin, xmax = ax.get_xlim()
            # when _time_window[i] is None, set autoscaled values
            xmin = self._time_window[0] if self._time_window[0] is not None else xmin
            xmax = self._time_window[1] if self._time_window[1] is not None else xmax
            ax.set_xlim(xmin, xmax)
        fig.tight_layout()
        fig.canvas.draw_idle()


class WaveGraph(AbstractGraph):
    _COLUMNS_HEADER: List[str] = [
        "power_mean",
        "power_std",
        "angle_mean",
        "angle_std",
        "angle_deltamax",
    ]
    _COLUMNS_UNIT: List[str] = [
        "ADC Counts",
        "ADC Counts",
        "DEGREE",
        "DEGREE",
        "DEGREE",
    ]

    def __init__(self, csv_path: Path, time_window: Tuple[Optional[float], Optional[float]]):
        super().__init__(csv_path, time_window)
        self._fig, self._axs = plt.subplots(2, 3, figsize=(10, 7))
        fig_manager = self._fig.canvas.manager
        if fig_manager is not None:
            fig_manager.set_window_title("Waveform Stability")
        self._checkbuttons: Optional[CheckButtons] = None
        self._lines: Dict[str, List[Line2D]] = {}
        self._port_combination_labels: Set[str] = set()
        self._angle_mean_offsets: Dict[Tuple[str, str], float] = {}
        self._initialize()

    @property
    def fig(self) -> Figure:
        return self._fig

    def _set_graph_label(self):
        for idx, ax in enumerate(self._axs.flat):
            if idx < len(self._COLUMNS_HEADER):
                ax.set_ylabel(f"{self._COLUMNS_HEADER[idx]} ({self._COLUMNS_UNIT[idx]})")
                ax.set_xlabel("Time From Start (s)")
                ax.grid(axis="both")
            else:
                ax.axis("off")

    def _check_func(self, label: Optional[str]):
        if label is None:
            raise AssertionError("never happan")
        for line in self._lines[label]:
            line.set_visible(not line.get_visible())
        self._update_axes(self._axs.flat, self._fig)

    def _create_checkbuttons(self):
        labels = [label for label in self._lines.keys()]
        line_colors = []
        for label in labels:
            line = self._lines[label][0]
            line_colors.append(line.get_color())
        self._checkbuttons = CheckButtons(
            self._axs.flat[-1],
            labels=labels,
            actives=[True] * len(labels),
            label_props={"color": line_colors},
            frame_props={"edgecolor": line_colors},
            check_props={"facecolor": line_colors},
        )
        if self._checkbuttons is not None:
            self._checkbuttons.on_clicked(self._check_func)

    def _draw_lines(
        self,
        port_combination: Tuple[int, int],
        group: pd.DataFrame,
        option: Literal["initialize", "update"],
    ):
        label = f"Output {port_combination[0]} - Input {port_combination[1]}"
        if option == "initialize":
            self._port_combination_labels.add(label)
            self._lines[label] = []
            for idx, ax in enumerate(self._axs.flat):
                if idx < len(self._COLUMNS_HEADER):
                    (line,) = ax.plot(group["time_from_start"], group[self._COLUMNS_HEADER[idx]], label=label)
                    self._lines[label].append(line)
        elif option == "update":
            for idx, line in enumerate(self._lines[label]):
                x_data, y_data = line.get_data()
                x_data = np.concatenate([x_data, group["time_from_start"].values])
                y_data = np.concatenate([y_data, group[self._COLUMNS_HEADER[idx]].values])
                line.set_data(x_data, y_data)

    def _phase_offset_subtract(self, group: pd.DataFrame, offset: float) -> pd.DataFrame:
        group["angle_mean"] = group["angle_mean"] - offset

        # if offset corrected data is very large, it is regarded as a wrap-around
        # phase variations within +- 60 degrees are correctly displayed.
        group["angle_mean"] = group["angle_mean"].apply(
            lambda x: x - 360.0 if x > 60.0 else (x + 360.0 if x < -60.0 else x)
        )
        return group

    def _process_data(self, data_df: pd.DataFrame, option: Literal["initialize", "update"]):
        ports = data_df.groupby(["output_port", "input_port"])
        for port_combination, group in ports:
            if option == "initialize":
                self._angle_mean_offsets[port_combination] = group["angle_mean"].iloc[0]
            group = self._phase_offset_subtract(group, self._angle_mean_offsets[port_combination])
            self._draw_lines(port_combination, group, option)

    def _initialize(self):
        new_data_df: Optional[pd.DataFrame] = self._get_new_lines()
        if new_data_df is None:
            raise ValueError("no data found. the csv for wavestability may be empty")
        self._set_graph_label()
        self._process_data(new_data_df, "initialize")
        self._create_checkbuttons()
        self._update_axes(self._axs.flat, self._fig)

    def update_graph(self, frame: int):
        new_data_df: Optional[pd.DataFrame] = self._get_new_lines()
        if new_data_df is None:
            return
        self._process_data(new_data_df, "update")
        self._update_axes(self._axs.flat, self._fig)

    def print_wave_stability(self):
        start_time, end_time = (self._time_window[0] or 0.0, self._time_window[1] or 1e10)
        for label in self._port_combination_labels:
            print(f"\n{label}")
            print("-" * len(label))
            title_width = 15
            num_width = 10
            print(f"{'Title':<{title_width}} | {'Std':>{num_width}} | {'Mean':>{num_width}} | {'Ratio':>{num_width}}")
            print(f"{'-'*title_width}-+-{'-'*num_width}-+-{'-'*num_width}-+-{'-'*num_width}")
            for idx, line in enumerate(self._lines[label]):
                x_data, y_data = line.get_data()
                title = self._COLUMNS_HEADER[idx]
                x_data = np.array(x_data, dtype=float)
                y_data = np.array(y_data, dtype=float)
                mask = (x_data > start_time) & (x_data < end_time)
                std = np.std(y_data[mask])
                mean = np.mean(y_data[mask])
                if "angle_mean" in title:
                    ratio_str: str = "N/A"
                else:
                    ratio_str = f"{std/mean:.6f}"
                print(
                    f"{title:<{title_width}} | {std:>{num_width}.6f} |"
                    f" {mean:>{num_width}.6f} | {ratio_str:>{num_width}}"
                )
            print(f"{'-'*title_width}---{'-'*num_width}-+-{'-'*num_width}---{'-'*num_width}")


class TempCtrlGraph(AbstractGraph):
    _LOCATION_TYPE: Dict[str, int] = {}
    _CORRESPONDING_BOXTYPES: Set[Quel1BoxType] = set()

    def __init__(self, csv_path: Path, time_window: Tuple[Optional[float], Optional[float]], boxtype: Quel1BoxType):
        super().__init__(csv_path, time_window, boxtype)
        self._lines: Dict[str, Dict[str, List[Line2D]]] = {"temperature": {}, "actuator_val": {}}
        self._time_window: Tuple[Optional[float], Optional[float]] = time_window
        t_fig, t_axs = plt.subplots(2, 4, figsize=(12, 7))
        a_fig, a_axs = plt.subplots(2, 4, figsize=(12, 7))
        self._figs: Dict[str, Figure] = {"temperature": t_fig, "actuator_val": a_fig}
        self._axs: Dict[str, Tuple[Axes, ...]] = {"temperature": t_axs.flat, "actuator_val": a_axs.flat}
        for key, fig in self._figs.items():
            fig_manager = fig.canvas.manager
            if fig_manager is None:
                continue
            if key == "temperature":
                fig_manager.set_window_title("Temperatures")
            else:
                fig_manager.set_window_title("Actuator Values")
        self._initialize()

    @classmethod
    def is_corresponding_boxtype(cls, boxtype: Quel1BoxType) -> bool:
        return boxtype in cls._CORRESPONDING_BOXTYPES

    @property
    def fig(self) -> Figure:
        return self._figs["temperature"]

    def _set_graph_label(self):
        for key, axs in self._axs.items():
            for loc, idx in self._LOCATION_TYPE.items():
                axs[idx].set_title(loc)
                if key == "temperature":
                    axs[idx].set_ylabel(r"Temperature [$^\circ$C]")
                else:
                    axs[idx].set_ylabel("Actuator Strength")
                axs[idx].set_xlabel("Time From Start (s)")
                axs[idx].grid(axis="both")
                if axs[idx].get_legend_handles_labels()[0]:  # Check if there are any labels
                    axs[idx].legend(loc="upper left", fontsize=8)

    def _get_location_type(self, location: str) -> str:
        for location_type in self._LOCATION_TYPE.keys():
            if location_type not in location:
                continue
            else:
                return location_type
        else:
            raise ValueError(f"can not find location type for location: {location}")

    def _draw_lines(
        self,
        ax_key: Literal["temperature", "actuator_val"],
        location: str,
        data: Tuple["Series[float]", "Series[float]"],
        option: Literal["initialize", "update"],
    ):
        loc = location.replace("_", "")
        if option == "initialize":
            location_type = self._get_location_type(location)
            self._lines[ax_key][loc] = []
            label = loc.replace(location_type, "")
            (line,) = self._axs[ax_key][self._LOCATION_TYPE[location_type]].plot(data[0], data[1], label=label)
            self._lines[ax_key][loc].append(line)
        elif option == "update":
            loc = location.replace("_", "")
            old_x, old_y = self._lines[ax_key][loc][0].get_data()
            new_x = np.concatenate([old_x, data[0]])
            new_y = np.concatenate([old_y, data[1]])
            self._lines[ax_key][loc][0].set_data(new_x, new_y)

    def _process_data(self, data_df: pd.DataFrame, option: Literal["initialize", "update"]):
        for location, group in data_df.groupby("location_name"):
            assert isinstance(location, str)
            location_type = self._get_location_type(location)

            # there are too many thermistors in path selector boards.
            # only thermistors whose corresponding actuators exist are displayed
            if location_type in ("ps0", "ps1") and pd.isna(group["actuator_val"].values[0]):
                continue

            self._draw_lines("temperature", location, (group["time_from_start"], group["temperature"]), option)

            if pd.isna(group["actuator_val"].values[0]):
                continue
            self._draw_lines("actuator_val", location, (group["time_from_start"], group["actuator_val"]), option)

    def _initialize(self):
        new_data_df: Optional[pd.DataFrame] = self._get_new_lines()
        if new_data_df is None:
            raise ValueError("csv for tempctrl may be empty.")
        self._process_data(new_data_df, "initialize")
        self._set_graph_label()
        for key in ["temperature", "actuator_val"]:
            self._update_axes(self._axs[key], self._figs[key])

    def update_graph(self, frame: int):
        new_data_df: Optional[pd.DataFrame] = self._get_new_lines()
        if new_data_df is None:
            return
        self._process_data(new_data_df, "update")
        self._set_graph_label()
        for key in ["temperature", "actuator_val"]:
            self._update_axes(self._axs[key], self._figs[key])


class TempCtrlGraphQuel1(TempCtrlGraph):
    _LOCATION_TYPE: Dict[str, int] = {
        "ad9082": 0,
        "lmx2594": 1,
        "adrf6780": 2,
        "panel": 3,
        "rx": 4,
        "adclk": 5,
        "ps": 6,
    }
    _CORRESPONDING_BOXTYPES: Set[Quel1BoxType] = {
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuEL1_TypeB,
        Quel1BoxType.QuEL1_NEC,
        Quel1BoxType.QuEL1_NTT,
    }


class TempCtrlGraphQuel1seRiken8(TempCtrlGraph):
    _LOCATION_TYPE: Dict[str, int] = {
        "mxfe": 0,
        "lmx2594": 1,
        "adrf6780": 2,
        "panel": 3,
        "hmc8193": 4,
        "amp": 4,
        "ps0": 5,
        "ps1": 6,
    }
    _CORRESPONDING_BOXTYPES: Set[Quel1BoxType] = {
        Quel1BoxType.QuEL1SE_RIKEN8,
        Quel1BoxType.QuEL1SE_RIKEN8DBG,
    }


class TempCtrlGraphQuel1seFujitsu11(TempCtrlGraph):
    _LOCATION_TYPE: Dict[str, int] = {
        "mxfe": 0,
        "lmx2594": 1,
        "adrf6780": 2,
        "panel": 3,
        "rx": 4,
        "hmc8193": 5,
        "ps0": 6,
        "ps1": 7,
    }
    _CORRESPONDING_BOXTYPES: Set[Quel1BoxType] = {
        Quel1BoxType.QuEL1SE_FUJITSU11_TypeA,
        Quel1BoxType.QuEL1SE_FUJITSU11_TypeB,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeB,
    }


def validate_filename(filename):
    pattern = r"^wavestability_.*_\d{14}\.csv$"
    if not re.match(pattern, os.path.basename(filename)):
        raise ValueError(f"Invalid filename format: {filename}")
    return filename


def tempctrl_csv_path(wave_csv_file_str: str) -> Optional[Path]:

    # temperature file path is data_dir/tempctrl/tempctrl_*.csv
    # while wavestability file path is data_dir/wavestability/wavestability_*.csv
    temp_csv_file_path: Path = Path(wave_csv_file_str.replace("wavestability", "tempctrl"))
    if temp_csv_file_path.exists():
        return temp_csv_file_path

    # in case of specifying the relative path in the directory in data_dir/wavestability/
    temp_csv_file_path = Path("../tempctrl") / temp_csv_file_path
    if temp_csv_file_path.exists():
        return temp_csv_file_path

    logger.warning(f"Temperature control data file not found: {temp_csv_file_path}")
    return None


def get_csv_path(
    args: Namespace,
) -> Tuple[Path, Optional[Path]]:
    # Cases are now validated:
    # Case 1. args.wave_csv is specified; others are not (handled at the beginning)
    # Case 2. args.data_dir and args.ipaddr_wss are specified; others are not
    # Case 3. args.data_dir and args.boxname are specified; others are not

    # DATA LOCATION (OUTDIR is specified directory in quel1_stability_check command),
    # -wave data csv        : OUTDIR/wavestability
    # -temperature data csv :  OUTDIR/tempctrl

    # In Case 1, specify the wave stability CSV file directly (OUTDIR/wavestability/xxxx.csv),
    # and the temperature data CSV file is found automatically.
    # In Case 2 and Case 3, specify only OUTDIR and either the IP address or box name.
    # due to many candidates in this cases, the newest one is selected

    if args.wave_csv is not None:
        if args.data_dir is not None or args.ipaddr_wss is not None or args.boxname is not None:
            raise ValueError(
                "When --wave_csv is specified, None of --data_dir, --ipaddr_wss and --boxname should not be used."
            )
        if not args.wave_csv.exists():
            raise FileNotFoundError(f"Wave stability CSV file not found: {args.wave_csv}")
        return args.wave_csv, tempctrl_csv_path(str(args.wave_csv))

    if args.data_dir is None:
        raise ValueError("Data directory must be specified with --data_dir option when --wave_csv is not provided.")

    if args.ipaddr_wss is not None and args.boxname is None:
        staging_no = int(args.ipaddr_wss.split(".")[-1])
        # possible filename format
        # 1. only ipaddr is specified in data taking
        # 2. boxname is also specified with ipaddr in data taking
        # 3. only ipaddr is specified in data taking (only 4th octet used for file name) This is old file name !
        patterns = [
            f"wavestability_{args.ipaddr_wss.strip().replace('.', '_')}_*.csv",  # 1
            f"wavestability_*_{args.ipaddr_wss.strip().replace('.', '_')}_*.csv",  # 2
            f"wavestability_{staging_no:03}_*.csv",  # 3
        ]
        files = [f for pattern in patterns for f in glob.glob(os.path.join(args.data_dir, "wavestability", pattern))]
    elif args.ipaddr_wss is None and args.boxname is not None:
        files = glob.glob(os.path.join(args.data_dir, "wavestability", f"wavestability_{args.boxname}_*.csv"))
    else:
        raise ValueError("Either --ipaddr or --boxname must be specified when --wave_csv_file is not provided.")

    wave_csv_file_str = max(
        filter(lambda f: validate_filename(f), files),
        key=lambda f: os.path.basename(f).split("_")[-1].split(".")[0],
    )

    return Path(wave_csv_file_str), tempctrl_csv_path(wave_csv_file_str)


def get_tempctrlgraph(
    csv_path: Path, time_window: Tuple[Optional[float], Optional[float]], boxtype: Quel1BoxType
) -> TempCtrlGraph:
    for tcgr in TempCtrlGraph.__subclasses__():
        if tcgr.is_corresponding_boxtype(boxtype):
            return tcgr(csv_path, time_window, boxtype)
    else:
        raise ValueError(f"{args.boxtype} is not supported yet")


def cli_args() -> Namespace:
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = ArgumentParser(description="a phase fluctuation measurement tool")
    parser.add_argument(
        "--boxtype",
        type=parse_boxtype,
        required=True,
        help=f"a type of the target box: either of "
        f"{', '.join([t for t in QUEL1_BOXTYPE_ALIAS if not t.startswith('x_')])}",
    )
    parser.add_argument(
        "--ipaddr_wss",
        type=str,
        default=None,
        help="ipaddr_wss specified when stability data was taken",
    )
    parser.add_argument(
        "--boxname",
        type=str,
        help="box name specified when stability data was taken",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="directory for stability data",
    )
    parser.add_argument(
        "--wave_csv",
        type=Path,
        default=None,
        help="csv file path for wave stability data",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=None,
        help="start time of the plots",
    )
    parser.add_argument(
        "--end_time",
        type=float,
        default=None,
        help="end time of the plots",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=False,
        help="specify for not updating",
    )
    return parser.parse_args()


def cli_body(args: Namespace):
    mpl.use("Gtk3Agg")

    wave_csv_path, temp_csv_path = get_csv_path(args)

    graphs: List[AbstractGraph] = [WaveGraph(wave_csv_path, (args.start_time, args.end_time))]
    if temp_csv_path is not None:
        graphs.append(get_tempctrlgraph(temp_csv_path, (args.start_time, args.end_time), args.boxtype))

    animations = []  # FuncAnimation object must be kept until plt.show()
    if not args.freeze:
        # if args.freeze is False, all graphs will be continuously updated
        # as long as DAQ is on going.
        for graph in graphs:
            animations.append(
                FuncAnimation(  # noqa: F841
                    graph.fig,
                    graph.update_graph,
                    interval=UPDATE_INTERVAL,
                    cache_frame_data=False,
                )
            )
    else:
        # if args.freeze is True,
        # statistical summary about waveform stabilities is displayed.
        for graph in graphs:
            if isinstance(graph, WaveGraph):
                graph.print_wave_stability()

    plt.show()


def cli_main():
    try:
        args = cli_args()
        cli_body(args)
        sys.exit(0)
    except Exception as e:
        logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    args = cli_args()
    cli_body(args)
