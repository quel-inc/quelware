import json
import logging
import urllib.error
import urllib.request
from base64 import b64decode
from typing import Dict, Final, Tuple, Union
from urllib.parse import urlencode

import numpy as np
import numpy.typing as npt

from quel_inst_tool import E440xbReadableParams, E440xbWritableParams

logger = logging.getLogger(__name__)

DEFAULT_PEAK_MINIMUM_POWER: Final[float] = -60.0
URLOPEN_TIMEOUT = 5


class E440xbClient:
    def __init__(self, server_host: str, port: int):
        self._server_host = server_host
        self._port = port

    def reset(self) -> Tuple[int, bool]:
        status: int = 0
        try:
            with urllib.request.urlopen(
                f"http://{self._server_host:s}:{self._port:d}/reset", timeout=URLOPEN_TIMEOUT
            ) as response:
                body = json.loads(response.read())
                status = response.getcode()

        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            return status, "details" in body and body["details"] == "ok"
        else:
            return status, False

    def average_clear(self) -> Tuple[int, bool]:
        status: int = 0
        try:
            with urllib.request.urlopen(
                f"http://{self._server_host:s}:{self._port:d}/average_clear", timeout=URLOPEN_TIMEOUT
            ) as response:
                body = json.loads(response.read())
                status = response.getcode()
        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            return status, "details" in body and body["details"] == "ok"
        else:
            return status, False

    def param_get(self) -> Tuple[int, Union[E440xbReadableParams, None]]:
        status: int = 0
        try:
            with urllib.request.urlopen(
                f"http://{self._server_host:s}:{self._port:d}/param", timeout=URLOPEN_TIMEOUT
            ) as response:
                body = json.loads(response.read())
                status = response.getcode()
        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            print(body)
            return status, E440xbReadableParams(**body)
        else:
            return status, None

    def param_set(self, param: E440xbWritableParams) -> Tuple[int, Union[E440xbReadableParams, Dict, None]]:
        status: int = 0
        try:
            headers = {
                "Content-Type": "application/json",
            }
            req = urllib.request.Request(
                f"http://{self._server_host:s}:{self._port:d}/param", param.json().encode(), headers
            )
            with urllib.request.urlopen(req, timeout=URLOPEN_TIMEOUT) as response:
                body = json.loads(response.read())
                status = response.getcode()

        except urllib.error.HTTPError as e:
            logger.error(e.reason)
            status = e.code
            if status == 400:
                body = json.loads(e.read())

        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            return status, E440xbReadableParams(**body)
        elif status == 400:
            return status, body
        else:
            return status, None

    def trace_get(
        self,
        trace: bool = True,
        peak: bool = False,
        minimum_power: float = DEFAULT_PEAK_MINIMUM_POWER,
        meta: bool = False,
    ) -> Tuple[
        int, Union[npt.NDArray[np.float_], None], Union[npt.NDArray[np.float_], None], Union[E440xbReadableParams, None]
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
        meta_data: Union[E440xbReadableParams, None] = None

        logger.info(f"http://{self._server_host:s}:{self._port:d}/trace?{urlencode(query)}")

        try:
            with urllib.request.urlopen(
                f"http://{self._server_host:s}:{self._port:d}/trace?{urlencode(query)}", timeout=URLOPEN_TIMEOUT
            ) as response:
                body = json.loads(response.read())
                status = response.getcode()
                if body["trace"] is not None:
                    trace_data = np.frombuffer(b64decode(body["trace"]))
                    trace_data = trace_data.reshape((trace_data.shape[0] // 2, 2))
                if body["peak"] is not None:
                    peak_data = np.frombuffer(b64decode(body["peak"]))

                if body["meta"] is not None:
                    meta_data = E440xbReadableParams(**json.loads(body["meta"]))
        except urllib.error.URLError as e:
            logger.error(e.reason)

        if status == 200:
            return status, trace_data, peak_data, meta_data
        else:
            return status, None, None, None
