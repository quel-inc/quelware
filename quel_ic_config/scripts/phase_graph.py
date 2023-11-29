import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class MultiBoxPhaseLogLoader:
    _DATA_START_AT = 4
    _CLOCK_FREQ = 125_000_000

    def __init__(self, logpath: Path, load_now: bool = True):
        self._logpath = logpath
        self._logdata: Dict[int, List[Dict[str, Any]]] = {}
        self._timerange: Dict[int, List[float]] = {}
        if load_now:
            self.load()

    def load(self):
        with open(self._logpath) as f:
            for logline in f:
                sline = logline.strip().split()
                try:
                    data1 = json.loads(" ".join(sline[self._DATA_START_AT :]))
                except json.decoder.JSONDecodeError:
                    continue

                line_idx = data1["line"]
                if line_idx not in self._logdata:
                    self._logdata[line_idx] = []
                    self._timerange[line_idx] = [data1["time"], -1]
                self._logdata[line_idx].append(data1)
                self._timerange[line_idx][1] = data1["time"]

    def lines(self):
        return self._logdata.keys()

    def extract(self, line: int, start_at: float = 0, duration: float = -1.0) -> npt.NDArray[np.float32]:
        if duration < 0:
            duration = (self._timerange[line][1] - self._timerange[line][0]) / self._CLOCK_FREQ

        phases = []
        for data1 in self._logdata[line]:
            t = data1["time"]
            ofst = self._timerange[line][0]
            if ofst + start_at * self._CLOCK_FREQ <= t <= ofst + (start_at + duration) * self._CLOCK_FREQ:
                phases.append(data1["agl_mean"])
        u = np.array(phases, dtype=np.float32)
        logger.info(f"line: {line}, mean: {np.mean(u)}, std:{np.std(u)}")
        return u

    def dump_tsv(self, line: int, outpath: Path, start_at: float = 0, duration: float = -1.0) -> None:
        if duration < 0:
            duration = (self._timerange[line][1] - self._timerange[line][0]) / self._CLOCK_FREQ

        with open(outpath, "w") as outfile:
            for data1 in self._logdata[line]:
                mod = dict(data1)
                mod["time"] = (mod["time"] - self._timerange[line][0]) / self._CLOCK_FREQ
                mod["time"] -= start_at
                t = mod["time"]
                if 0 <= t <= duration:
                    if outfile is not None:
                        outfile.write(" ".join([str(v) for k, v in mod.items() if k not in {"line"}]))
                        outfile.write("\n")


if __name__ == "__main__":
    from argparse import ArgumentParser

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = ArgumentParser(description="")
    parser.add_argument("--logfile", type=Path, required=True, help="path of the logfile to analyze")
    parser.add_argument("--start_at", type=float, default=0.0, help="start of the target part of the log in second")
    parser.add_argument("--duration", type=float, default=-1.0, help="duration of the target part of the log in second")
    parser.add_argument("--out_postfix", type=str, default="", help="prefix of generated output file")
    args = parser.parse_args()

    loader = MultiBoxPhaseLogLoader(args.logfile)
    for i in loader.lines():
        loader.extract(i, start_at=args.start_at, duration=args.duration)
        if len(args.out_postfix) == 0:
            loader.dump_tsv(i, Path(f"line{i}.tsv"), start_at=args.start_at, duration=args.duration)
        else:
            loader.dump_tsv(i, Path(f"line{i}_{args.out_postfix}.tsv"), start_at=args.start_at, duration=args.duration)
