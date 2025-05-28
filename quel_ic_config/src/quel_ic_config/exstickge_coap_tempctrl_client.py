from enum import Enum
from typing import Dict, List, Sequence, Tuple, Union

import aiocoap
import aiocoap.error

from quel_ic_config.exstickge_coap_client import Quel1seBoard, _ExstickgeCoapClientBase


class Quel1seTempctrlState(str, Enum):
    INIT = "init"
    PRERUN = "prerun"
    RUN = "run"
    DRYRUN = "dryrun"
    IDLE = "idle"

    @classmethod
    def fromstr(self, label: str) -> "Quel1seTempctrlState":
        return Quel1seTempctrlState(label)

    def tostr(self) -> str:
        return str(self.value)


class _ExstickgeCoapClientQuel1seTempctrlBase(_ExstickgeCoapClientBase):
    _AVAILABLE_BOARDS: Tuple[Quel1seBoard, ...]
    _TEMPCTRL_AD7490_NAME: Tuple[str, ...]

    def read_tempctrl_loop_count(self) -> int:
        uri = f"coap://{self._target[0]}/tempctrl/loop/count"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16)

    def read_tempctrl_loop_interval(self) -> float:
        uri = f"coap://{self._target[0]}/tempctrl/loop/interval"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16) / 1000.0  # Notes: ms --> s

    def read_tempctrl_state(self) -> Quel1seTempctrlState:
        uri = f"coap://{self._target[0]}/tempctrl/state"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return Quel1seTempctrlState(res.payload.decode())

    def write_tempctrl_state(self, new_state: Quel1seTempctrlState) -> None:
        if not isinstance(new_state, Quel1seTempctrlState):
            raise TypeError(f"invalid tempctrl state: '{new_state}'")
        uri = f"coap://{self._target[0]}/tempctrl/state"
        res = self._core.request_and_wait(code=aiocoap.PUT, payload=new_state.value.encode(), uri=uri)
        self._coap_return_check(res, uri)

    def read_tempctrl_state_count(self) -> int:
        uri = f"coap://{self._target[0]}/tempctrl/state/count"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16)

    def write_tempctrl_state_count(self, new_count: int) -> None:
        if not 1 <= new_count <= 0xFFFFFFFF:
            raise ValueError("invalid count value: {new_count}")
        uri = f"coap://{self._target[0]}/tempctrl/state/count"
        res = self._core.request_and_wait(code=aiocoap.PUT, payload=f"{new_count:08x}".encode(), uri=uri)
        self._coap_return_check(res, uri)
        return

    def read_tempctrl_skip_count(self) -> int:
        uri = f"coap://{self._target[0]}/tempctrl/skip/count"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16)

    def read_tempctrl_error_count(self) -> int:
        uri = f"coap://{self._target[0]}/tempctrl/error/count"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16)

    def read_tempctrl_timeout_count(self) -> int:
        uri = f"coap://{self._target[0]}/tempctrl/timeout/count"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16)

    def read_tempctrl_broken_count(self) -> int:
        uri = f"coap://{self._target[0]}/tempctrl/broken/count"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16)

    def read_tempctrl_mismatch_count(self) -> int:
        uri = f"coap://{self._target[0]}/tempctrl/mismatch/count"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16)

    def read_tempctrl_mismatch_max(self) -> int:
        uri = f"coap://{self._target[0]}/tempctrl/mismatch/max"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        return int(res.payload.decode(), 16)

    def read_tempctrl_rawdata_ad7490(self, ad7490_idx: int) -> Tuple[int, List[int]]:
        ad7490_id = self._TEMPCTRL_AD7490_NAME[ad7490_idx]
        uri = f"coap://{self._target[0]}/tempctrl/rawdata/ad7490/{ad7490_id}"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        payload = res.payload.decode()
        separated = payload.split(",")
        if len(separated) != 19:
            raise RuntimeError(f"unexpected payload '{payload}' from '{uri}'")

        timestamp = int(separated[0], 16)
        validity = int(separated[1], 16)
        num_channel = int(separated[2], 16)
        if validity != 1:
            raise RuntimeError(f"failed data acquisition from '{uri}'")

        data = []
        for i in range(num_channel):
            v = int(separated[3 + i], 16)
            if (v >> 12) & 0x0F != i:
                raise RuntimeError(f"broken readings detected at {i}-th data from '{uri}'")
            data.append(v & 0xFFF)

        return timestamp, data

    def _read_tempctrl_rawdata_pwrpwm(self, label: str) -> Dict[str, List[int]]:
        if label not in {"current", "next"}:
            raise ValueError(f"unexpected label: {label}")

        uri = f"coap://{self._target[0]}/tempctrl/rawdata/pwrpwm/{label}"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        payload = res.payload.decode()
        separated = payload.split(",")
        if len(separated) != 42:
            raise RuntimeError(f"unexpected payload '{payload}' from '{uri}'")
        return {
            "fan": [int(x, 16) for x in separated[0:2]],
            "heater": [int(x, 16) for x in separated[2:42]],
        }

    def read_tempctrl_rawdata_pwrpwm_current(self) -> Dict[str, List[int]]:
        return self._read_tempctrl_rawdata_pwrpwm("current")

    def read_tempctrl_rawdata_pwrpwm_next(self) -> Dict[str, List[int]]:
        return self._read_tempctrl_rawdata_pwrpwm("next")

    def write_tempctrl_rawdata_pwrpwm_next(self, fan: Sequence[int], heater: Sequence[int]) -> None:
        if len(fan) != 2:
            raise ValueError("number of fan must be 2")
        if len(heater) != 40:
            raise ValueError("number of heater must be 40")

        for i, x in enumerate(fan):
            if not (isinstance(x, int) and 0 <= x <= 1000):
                raise ValueError(f"invalid data for fan[{i}]: '{x}'")
        for i, x in enumerate(heater):
            if not (isinstance(x, int) and 0 <= x <= 1000):
                raise ValueError(f"invalid data for heater[{i}]: '{x}'")

        uri = f"coap://{self._target[0]}/tempctrl/rawdata/pwrpwm/next"
        converted = [f"{x:04x}" for x in fan]
        converted.extend([f"{x:04x}" for x in heater])
        payload = ",".join(converted)
        res = self._core.request_and_wait(code=aiocoap.PUT, payload=payload.encode(), uri=uri)
        self._coap_return_check(res, uri)

    def read_tempctrl_feedback_setpoint(self) -> Dict[str, List[float]]:
        uri = f"coap://{self._target[0]}/tempctrl/feedback/setpoint"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)

        payload = res.payload.decode()
        separated = payload.split(",")
        if len(separated) != 42:
            raise RuntimeError(f"unexpected payload '{payload}' from '{uri}'")

        return {
            "fan": [int(v, 16) / 256.0 for v in separated[0:2]],
            "heater": [int(v, 16) / 256.0 for v in separated[2:42]],
        }

    def write_tempctrl_feedback_setpoint(self, fan: Sequence[float], heater: Sequence[float]) -> None:
        if len(fan) != 2:
            raise ValueError("number of fan must be 2")
        if len(heater) != 40:
            raise ValueError("number of heater must be 40")

        for i, x in enumerate(fan):
            if not (isinstance(x, float) and 0.0 <= x <= 256.0):
                raise ValueError(f"invalid data for fan[{i}]: '{x}'")
        for i, x in enumerate(heater):
            if not (isinstance(x, float) and 0.0 <= x <= 256.0):
                raise ValueError(f"invalid data for heater[{i}]: '{x}'")

        uri = f"coap://{self._target[0]}/tempctrl/feedback/setpoint"
        converted = [f"{round(x * 256.0):04x}" for x in fan]
        converted.extend([f"{round(x * 256.0):04x}" for x in heater])
        payload = ",".join(converted)
        res = self._core.request_and_wait(code=aiocoap.PUT, payload=payload.encode(), uri=uri)
        self._coap_return_check(res, uri)

    def read_tempctrl_feedback_coeffcient(self) -> Dict[str, Dict[str, float]]:
        retval: Dict[str, Dict[str, float]] = {}

        for atype in ("fan", "heater"):
            uri = f"coap://{self._target[0]}/tempctrl/feedback/coeff/{atype}"
            res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
            self._coap_return_check(res, uri)

            payload = res.payload.decode()
            separated = payload.split(",")
            if len(separated) != 2:
                raise RuntimeError(f"unexpected payload '{payload}' from '{uri}'")
            retval[atype] = {"Kp": int(separated[0], 16) / 16777216.0, "Ki": int(separated[1], 16) / 16777216.0}

        return retval

    def _write_tempctrl_feedback_coefficient_sub(self, atype: str, coeff: Dict[str, float]) -> None:
        if "Kp" not in coeff:
            raise ValueError(f"No 'Kp' exists in the given coefficient for {atype}")
        if "Ki" not in coeff:
            raise ValueError(f"No 'Ki' exists in the given coefficient for {atype}")

        for k in ("Kp", "Ki"):
            if not (isinstance(coeff[k], float) and 0.0 <= coeff[k] < 1.0):
                raise ValueError(f"invalid value {coeff[k]} for coefficient '{k}' of {atype}")

        uri = f"coap://{self._target[0]}/tempctrl/feedback/coeff/{atype}"
        converted = [f"{round(coeff[k] * 16777216.0):06x}" for k in ("Kp", "Ki")]
        payload = ",".join(converted)
        res = self._core.request_and_wait(code=aiocoap.PUT, payload=payload.encode(), uri=uri)
        self._coap_return_check(res, uri)

    def write_tempctrl_feedback_coeffcient(
        self, fan: Union[Dict[str, float], None], heater: Union[Dict[str, float], None]
    ) -> None:
        if fan is not None:
            self._write_tempctrl_feedback_coefficient_sub("fan", fan)
        if heater is not None:
            self._write_tempctrl_feedback_coefficient_sub("heater", heater)

    def read_tempctrl_feedback_integral(self) -> Dict[str, List[float]]:
        uri = f"coap://{self._target[0]}/tempctrl/feedback/integral"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)

        payload = res.payload.decode()
        separated = payload.split(",")
        if len(separated) != 42:
            raise RuntimeError(f"unexpected payload '{payload}' from '{uri}'")

        return {
            "fan": [int(v, 16) / 16777216.0 for v in separated[0:2]],
            "heater": [int(v, 16) / 16777216.0 for v in separated[2:42]],
        }

    def read_tempctrl_test_ad7490(self, ad7490_idx: int) -> Tuple[int, int, int, List[int]]:
        ad7490_id = self._TEMPCTRL_AD7490_NAME[ad7490_idx]
        uri = f"coap://{self._target[0]}/tempctrl/test/rawdata/ad7490/{ad7490_id}"
        res = self._core.request_and_wait(code=aiocoap.GET, uri=uri)
        self._coap_return_check(res, uri)
        payload = res.payload.decode()
        separated = payload.split(",")
        if len(separated) != 19:
            raise RuntimeError(f"unexpected payload '{payload}' from '{uri}'")

        timestamp = int(separated[0], 16)
        validity = int(separated[1], 16)
        num_channel = int(separated[2], 16)  # Notes: must be 16, actually.
        data = [int(separated[3 + i], 16) for i in range(num_channel)]
        return timestamp, validity, num_channel, data
