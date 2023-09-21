import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Final, Generator, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quel_inst_tool import E440xbClient, E440xbReadableParams, E440xbTraceMode, E440xbWritableParams

logger = logging.getLogger()


DEVICE_CONFIG: Final[Dict[str, Dict[str, Any]]] = {
    "E4405B": {
        "host": "localhost",
        "port": 13389,
        "env": "E4405b.env",
    },
    "E4407B": {
        "host": "localhost",
        "port": 13390,
        "env": "E4407b.env",
    },
}

NONEXISTENT_SERVER: Final[Tuple[str, int]] = ("192.168.254.253", 49999)


@pytest.fixture(scope="module")
def outsubdir(e440xb_name, outdir) -> Path:
    dirname = "client_" + e440xb_name
    os.makedirs(outdir / dirname)
    return outdir / dirname


@pytest.fixture(scope="module")
def e440xb_remote(e440xb_name) -> Generator[E440xbClient, E440xbClient, None]:
    host = DEVICE_CONFIG[e440xb_name]["host"]
    port = DEVICE_CONFIG[e440xb_name]["port"]
    env = DEVICE_CONFIG[e440xb_name]["env"]
    cmd: str = f"exec uvicorn quel_inst_tool.e440xb_remote_server:app --port {port} --env-file {env}"
    logger.info(f"starting remote server: {cmd}")
    p_e440xb = subprocess.Popen(
        cmd,
        shell=True,
        preexec_fn=os.setsid,
        stderr=subprocess.PIPE,
    )
    assert p_e440xb.stderr is not None

    logger.info("waiting for starting up the server")
    while True:
        line = p_e440xb.stderr.readline()  # Notes: omit implementing timeout.
        logger.info(f"server: {line.decode()}")
        if len(line) == 0:
            pytest.fail("failed to start the server")
        if line.startswith(b"INFO:     Uvicorn running"):
            break
    logger.info("the server is ready")
    e440xb_remote = E440xbClient(host, port)
    yield e440xb_remote
    p_e440xb.terminate()
    p_e440xb.wait()
    del p_e440xb
    logger.info("the server is terminated")


class TestClient:
    @pytest.mark.parametrize(
        ("freq_center", "freq_span", "valid_parameter", "out_of_range"),
        [
            (9e9, 5e9, True, {}),
            (9e9, 10e9, True, {"E4405B"}),
            (9e9, None, False, {}),
            (23e9, 8e9, True, {"E4405B", "E4407B"}),
            (None, 10e9, False, {}),
        ],
    )
    def test_param_set_get(self, e440xb_remote, e440xb_name, freq_center, freq_span, valid_parameter, out_of_range):
        param0 = E440xbWritableParams(
            trace_mode=E440xbTraceMode.WRITE,
            freq_center=freq_center,
            freq_span=freq_span,
            sweep_points=801,
            resolution_bandwidth=1e5,
            average_enable=False,
        )
        target = e440xb_remote
        status1, param1 = target.param_set(param0)
        if valid_parameter and (e440xb_name not in out_of_range):
            assert status1 == 200

            assert isinstance(param1, E440xbReadableParams)

            assert param1 is not None
            assert param1.trace_mode == param0.trace_mode
            assert param1.freq_center == param0.freq_center
            assert param1.freq_span == param0.freq_span
            assert param1.sweep_points == param0.sweep_points
            assert param1.resolution_bandwidth == param0.resolution_bandwidth
            assert param1.average_enable == param0.average_enable
        else:
            assert status1 == 400

        status2, param2 = target.param_get()
        assert status2 == 200

        if valid_parameter and (e440xb_name not in out_of_range):
            assert isinstance(param2, E440xbReadableParams)

            assert param2 is not None
            assert param2.trace_mode == param0.trace_mode
            assert param2.freq_center == param0.freq_center
            assert param2.freq_span == param0.freq_span
            assert param2.sweep_points == param0.sweep_points
            assert param2.resolution_bandwidth == param0.resolution_bandwidth
            assert param2.average_enable == param0.average_enable

    def test_average_clear(self, e440xb_remote):
        target = e440xb_remote
        assert target.average_clear() == (200, True)

    def _gen_spectrum_image(self, trace, filepath):
        plt.cla()
        plt.plot(trace[:, 0], trace[:, 1])
        plt.savefig(filepath)

    def test_trace_typical(self, e440xb_remote, outsubdir):
        target = e440xb_remote
        assert target.reset() == (200, True)
        param0 = E440xbWritableParams(
            trace_mode=E440xbTraceMode.WRITE,
            freq_center=9e9,
            freq_span=5e9,
            sweep_points=4001,
            resolution_bandwidth=1e5,
            average_enable=False,
        )
        status1, _ = target.param_set(param0)
        assert status1 == 200
        status2, trace2, peak2, meta2 = target.trace_get(trace=True, peak=True, minimum_power=-30.0, meta=True)
        assert status2 == 200
        assert isinstance(meta2, E440xbReadableParams)
        assert meta2.trace_mode == param0.trace_mode
        assert meta2.freq_center == param0.freq_center
        assert meta2.freq_span == param0.freq_span
        assert meta2.sweep_points == param0.sweep_points
        assert meta2.resolution_bandwidth == param0.resolution_bandwidth
        assert meta2.average_enable == param0.average_enable
        assert isinstance(trace2, np.ndarray)
        assert trace2.shape == (param0.sweep_points, 2)
        self._gen_spectrum_image(trace2, outsubdir / "typical")

    def test_trace_maxhold(self, e440xb_remote, outsubdir):
        target = e440xb_remote
        assert target.reset() == (200, True)
        param0 = E440xbWritableParams(
            trace_mode=E440xbTraceMode.MAXHOLD,
            freq_center=8e9,
            freq_span=8e9,
            sweep_points=4001,
            resolution_bandwidth=1e5,
            average_enable=False,
        )
        status1, _ = target.param_set(param0)
        assert status1 == 200
        time.sleep(5.0)

        status2, trace2, peak2, meta2 = target.trace_get(trace=True, peak=True, minimum_power=-30.0, meta=True)
        assert status2 == 200
        assert isinstance(meta2, E440xbReadableParams)
        assert meta2.trace_mode == param0.trace_mode
        assert meta2.freq_center == param0.freq_center
        assert meta2.freq_span == param0.freq_span
        assert meta2.sweep_points == param0.sweep_points
        assert meta2.resolution_bandwidth == param0.resolution_bandwidth
        assert meta2.average_enable == param0.average_enable
        assert isinstance(trace2, np.ndarray)
        assert trace2.shape == (param0.sweep_points, 2)
        self._gen_spectrum_image(trace2, outsubdir / "maxhold")

    def test_trace_minold(self, e440xb_remote, outsubdir):
        target = e440xb_remote
        assert target.reset() == (200, True)

        param0 = E440xbWritableParams(
            trace_mode=E440xbTraceMode.MINHOLD,
            freq_center=8e9,
            freq_span=1e9,
            sweep_points=2001,
            resolution_bandwidth=1e5,
            average_enable=False,
        )
        status1, _ = target.param_set(param0)
        assert status1 == 200
        time.sleep(5.0)

        status2, trace2, peak2, meta2 = target.trace_get(trace=True, peak=True, minimum_power=-30.0, meta=True)
        assert status2 == 200
        assert isinstance(meta2, E440xbReadableParams)
        assert meta2.trace_mode == param0.trace_mode
        assert meta2.freq_center == param0.freq_center
        assert meta2.freq_span == param0.freq_span
        assert meta2.sweep_points == param0.sweep_points
        assert meta2.resolution_bandwidth == param0.resolution_bandwidth
        assert meta2.average_enable == param0.average_enable
        assert isinstance(trace2, np.ndarray)
        assert trace2.shape == (param0.sweep_points, 2)
        self._gen_spectrum_image(trace2, outsubdir / "minhold")

    def test_nonexistent_server(self, e440xb_name):
        host, port = NONEXISTENT_SERVER
        target = E440xbClient(host, port)
        assert target.reset() == (0, False)
        assert target.average_clear() == (0, False)
        assert target.param_get() == (0, None)
        param0 = E440xbWritableParams(
            trace_mode=E440xbTraceMode.WRITE,
            freq_center=8e9,
            freq_span=1e9,
            sweep_points=2001,
            resolution_bandwidth=1e5,
            average_enable=False,
        )
        assert target.param_set(param0) == (0, None)
        assert target.trace_get() == (0, None, None, None)
