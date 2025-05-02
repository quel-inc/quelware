import logging
import re
from typing import Dict, Final, Set, Union

import numpy as np
import numpy.typing as npt

from quel_cmod_scripting.quel_cmod_abstract import QuelCmodAbstract

logger = logging.getLogger(__name__)


class QuelCmod(QuelCmodAbstract):
    CORRESPONDING_VARIANTS: Set[str] = {"quel1", "fancontrol", "switchcontrol"}
    MINIMUM_VERSION: str = "0.0.1"

    NUM_THERMISTOR: Final[int] = 28
    NUM_PELTIER: Final[int] = 28

    def __init__(self, host: str, port: int):
        super().__init__(host, port)

    def thall(self) -> Union[None, npt.NDArray[np.int32]]:
        rep = self.execute("thall")
        return self.thall_decode(rep)

    def thall_in_json(self) -> Union[None, Dict[str, int]]:
        rep = self.execute("thall")
        return self.thall_decode_json(rep)

    def thall_decode(self, rep: str) -> Union[None, npt.NDArray[np.int32]]:
        converted = np.zeros(self.NUM_THERMISTOR, dtype=np.int32)
        lines = rep.strip().splitlines()
        if len(lines) != self.NUM_THERMISTOR:
            logger.error(
                f"unexpected number of received lines ({len(lines)} != {self.NUM_THERMISTOR})"
            )
            return None
        for idx, line in enumerate(lines):
            key, vals = line.split()
            if key != f"th[{idx:d}]:":
                logger.error("unexpected received data")
                return None
            val: int = int(vals)
            converted[idx] = val
        return converted

    def thall_decode_json(self, rep: str) -> Union[None, Dict[str, int]]:
        converted = self.thall_decode(rep)
        if converted is not None:
            converted2: Dict[str, int] = {}
            for i in range(self.NUM_THERMISTOR):
                converted2[f"th{i:02d}"] = int(
                    converted[i]
                )  # numpy.int32 is not JSON serializable
            return converted2
        return None

    def pl(self, idx: int, val: int) -> bool:
        rep = self.execute(f"pl! {idx:d} {val:d}")
        if len(rep) > 0:
            logger.error(f"{rep}")
            return False
        else:
            return True

    def plstat(self) -> Union[None, npt.NDArray[np.int32]]:
        rep = self.execute("plstat")
        return self.plstat_decode(rep)

    def plstat_in_json(self) -> Union[None, Dict[str, int]]:
        rep = self.execute("plstat")
        return self.plstat_decode_json(rep)

    def plstat_decode(self, rep: str) -> Union[None, npt.NDArray[np.int32]]:
        converted = np.zeros(self.NUM_PELTIER, dtype=np.int32)
        lines = rep.strip().splitlines()
        if len(lines) != self.NUM_PELTIER:
            logger.error("unexpected number of received lines")
            return None
        for idx, line in enumerate(lines):
            key_vals_arrow = line.split()  # need to ignore "<--" in the output
            key = key_vals_arrow[0]
            vals = key_vals_arrow[1]
            if key != f"pl[{idx:d}]:":
                logger.error("unexpected received data")
                return None
            val: int = int(vals)
            converted[idx] = val
        return converted

    def plstat_decode_json(self, rep: str) -> Union[None, Dict[str, int]]:
        converted = self.plstat_decode(rep)
        if converted is not None:
            converted2: Dict[str, int] = {}
            for i in range(self.NUM_THERMISTOR):
                converted2[f"pl{i:02d}"] = int(
                    converted[i]
                )  # numpy.int32 is not JSON serializable
            return converted2
        return None


class FanControlCmod(QuelCmod):
    CORRESPONDING_VARIANTS: Set[str] = {"fancontrol", "switchcontrol"}
    MINIMUM_VERSION: str = "0.0.1"

    FAN_PERIOD = 40000
    FAN_PATTERN = r"fan\[(?P<idx>\d+)\] = (?P<highperiod>\d+)/" + str(FAN_PERIOD)
    SWITCH_PATTERN = r"switch\[(?P<idx>\d+)\] = (?P<state>\w+)"

    def __init__(self, host: str, port: int):
        super().__init__(host, port)

    def fan_get(self) -> Union[npt.NDArray[np.int32], None]:
        rep = self.execute("fan")
        return self.fan_decode(rep)

    def fan_get_in_json(self) -> Union[Dict[str, int], None]:
        rep = self.execute("fan")
        data = self.fan_decode(rep)
        if data is not None:
            r: Dict[str, int] = {
                "fan0": data[0],
                "fan1": data[1],
            }
            return r
        return None

    def fan_decode(self, rep: str) -> Union[npt.NDArray[np.int32], None]:
        if self.variant != "fancontrol" and self.variant != "switchcontrol":
            raise RuntimeError("not supported by firmware")
        lines = rep.strip().split("\r\n")
        r = np.zeros(2, dtype=int)
        for line in lines:
            p = re.match(self.FAN_PATTERN, line)
            if p is not None:
                r[int(p.group("idx"))] = int(p.group("highperiod"))
            else:
                rep_oneline = rep.strip().replace("\r\n", "_")
                logger.error(f"unexpected output for 'fan' command: '{rep_oneline}'")
                return None
        return r

    def fan_set(self, idx: int, high_period: int) -> bool:
        if self.variant != "fancontrol" and self.variant != "switchcontrol":
            raise RuntimeError("not supported by firmware")
        if not (0 <= idx <= 1):
            raise ValueError("invalid index of fan: {idx}")
        if not (50 <= high_period <= 39950):
            raise ValueError(f"invalid range of high period: {high_period}")

        r = self.execute(f"fan! {idx} {high_period}")
        if len(r) > 0:
            logger.error(f"error occurred at fan! command: {r.strip()} ")
            return False
        return True


class SwitchControlCmod(FanControlCmod):
    CORRESPONDING_VARIANTS: Set[str] = {"switchcontrol"}
    MINIMUM_VERSION: str = "0.0.1"

    SWITCH_PATTERN = r"switch\[(?P<idx>\d+)\] = (?P<state>\w+)"

    def __init__(self, host: str, port: int):
        super().__init__(host, port)

    def switch_decode(self, rep: str) -> Union[npt.NDArray[np.bool_], None]:
        if self.variant != "switchcontrol":
            raise RuntimeError("not supported by firmware")
        lines = rep.strip().split("\r\n")
        r = np.zeros(4, dtype=bool)
        for line in lines:
            p = re.match(self.SWITCH_PATTERN, line)
            if p is not None:
                r[int(p.group("idx"))] = True if p.group("state") == "ON" else False
            else:
                rep_oneline = rep.strip().replace("\r\n", "_")
                logger.error(f"unexpected output for 'switch' command: '{rep_oneline}'")
                return None
        return r

    def switch_get(self) -> Union[npt.NDArray[np.bool_], None]:
        rep = self.execute("switch")
        return self.switch_decode(rep)

    def switch_get_in_json(self) -> Union[Dict[str, bool], None]:
        rep = self.execute("switch")
        data = self.switch_decode(rep)
        if data is not None:
            r: Dict[str, bool] = {
                "switch0": data[0],
                "switch1": data[1],
                "switch2": data[2],
                "switch3": data[3],
            }
            return r
        return None

    def switch_set(self, idx: int, ctrl: bool) -> bool:
        if self.variant != "switchcontrol":
            raise RuntimeError("not supported by firmware")
        if not (0 <= idx <= 3):
            raise ValueError("invalid index of switch: {idx}")

        r = self.execute(f"switch! {idx} {int(ctrl)}")
        if len(r) > 0:
            logger.error(f"error occurred at switch! command: {r.strip()} ")
            return False
        return True

    def all_switches_set(self, state: int) -> bool:
        if self.variant != "switchcontrol":
            raise RuntimeError("not supported by firmware")
        if not (0 <= state <= 15):
            raise ValueError("invalid index of fan: {idx}")

        r = self.execute(f"switch_all! {hex(state)}")
        if len(r) > 0:
            logger.error(f"error occurred at switch_all! command: {r.strip()} ")
            return False
        return True


if __name__ == "__main__":
    import pprint
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print(f"usage for test: python {sys.argv[0]} port")
    else:
        obj = SwitchControlCmod("localhost", int(sys.argv[1]))
        pprint.pprint(obj.thall())
        pprint.pprint(obj.thall_in_json())
        pprint.pprint(obj.pl(0, 0))
        if obj.variant == "fancontrol":
            pprint.pprint(obj.fan_get())
        if obj.variant == "switchcontrol":
            pprint.pprint(obj.switch_get())
