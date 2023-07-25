import json
import logging
import urllib.error
import urllib.request
from base64 import b64decode
from typing import Dict, Final, Tuple, Union
from urllib.parse import urlencode

import numpy as np
import numpy.typing as npt

from quel_inst_tool import E4405bReadableParams, E4405bWritableParams

logger = logging.getLogger(__name__)

DEFAULT_PEAK_MINIMUM_POWER: Final[float] = -60.0


class E4405bClient:
    def __init__(self, server_host: str, port: int):
        self._server_host = server_host
        self._port = port

    def reset(self) -> Tuple[int, bool]:
        try:
            with urllib.request.urlopen(f"http://{self._server_host:s}:{self._port:d}/reset") as response:
                body = json.loads(response.read())
                status = response.getcode()
        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            return status, "details" in body and body["details"] == "ok"
        else:
            return status, False

    def average_clear(self) -> Tuple[int, bool]:
        try:
            with urllib.request.urlopen(f"http://{self._server_host:s}:{self._port:d}/average_clear") as response:
                body = json.loads(response.read())
                status = response.getcode()
        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            return status, "details" in body and body["details"] == "ok"
        else:
            return status, False

    def param_get(self) -> Tuple[int, Union[E4405bReadableParams, None]]:
        status: int = 0
        try:
            with urllib.request.urlopen(f"http://{self._server_host:s}:{self._port:d}/param") as response:
                body = json.loads(response.read())
                status = response.getcode()
        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            print(body)
            return status, E4405bReadableParams(**body)
        else:
            return status, None

    def param_set(self, param: E4405bWritableParams) -> Tuple[int, Union[E4405bWritableParams, None]]:
        status: int = 0
        try:
            headers = {
                "Content-Type": "application/json",
            }
            req = urllib.request.Request(
                f"http://{self._server_host:s}:{self._port:d}/param", param.json().encode(), headers
            )
            with urllib.request.urlopen(req) as response:
                body = json.loads(response.read())
                status = response.getcode()
        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            return status, E4405bWritableParams(**body)
        else:
            return status, None

    def trace_get(
        self,
        trace: bool = True,
        peak: bool = False,
        minimum_power: float = DEFAULT_PEAK_MINIMUM_POWER,
        meta: bool = False,
    ) -> Tuple[
        int, Union[npt.NDArray[np.float_], None], Union[npt.NDArray[np.float_], None], Union[E4405bReadableParams, None]
    ]:
        query: Dict[str, Union[bool, float]] = {
            "trace": trace,
            "peak": peak,
            "minimum_power": minimum_power,
            "meta": meta,
        }

        status: int = 0
        trace_data: Union[npt.NDArray[np.float_], None] = None
        peak_data: Union[npt.NDArray[np.float_], None] = None
        meta_data: Union[E4405bReadableParams, None] = None

        logger.info(f"http://{self._server_host:s}:{self._port:d}/trace?{urlencode(query)}")

        try:
            with urllib.request.urlopen(
                f"http://{self._server_host:s}:{self._port:d}/trace?{urlencode(query)}"
            ) as response:
                body = json.loads(response.read())
                status = response.getcode()
                if body["trace"] is not None:
                    trace_data = np.frombuffer(b64decode(body["trace"]))
                    trace_data = trace_data.reshape((trace_data.shape[0] // 2, 2))

                if body["peak"] is not None:
                    peak_data = np.frombuffer(b64decode(body["peak"]))

                if body["meta"] is not None:
                    meta_data = E4405bReadableParams(**json.loads(body["meta"]))
        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            return status, trace_data, peak_data, meta_data
        else:
            return status, None, None, None
