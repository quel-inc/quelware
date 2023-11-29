import json
import os
from base64 import b64decode
from pathlib import Path
from typing import Dict, Generator, Union
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
import pytest
from fastapi.testclient import TestClient

from quel_inst_tool import E440xbReadableParams, E440xbTraceMode, E440xbWritableParams
from quel_inst_tool.e440xb_remote_server import app


@pytest.fixture(scope="module")
def client(e440xb_name) -> Generator[TestClient, TestClient, None]:
    os.environ["SPECTRUM_ANALYZER_NAME"] = e440xb_name
    with TestClient(app) as client:
        yield client
    del os.environ["SPECTRUM_ANALYZER_NAME"]


@pytest.fixture(scope="module")
def outsubdir(e440xb_name, outdir) -> Path:
    dirname = "server_" + e440xb_name
    os.makedirs(outdir / dirname)
    return outdir / dirname


class TestServers:
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
    def test_post_params(self, client, e440xb_name, freq_center, freq_span, valid_parameter, out_of_range):
        param0 = E440xbWritableParams(
            trace_mode=E440xbTraceMode.WRITE,
            freq_center=freq_center,
            freq_span=freq_span,
            sweep_points=801,
            resolution_bandwidth=1e5,
            average_enable=False,
        )
        response = client.post("/param", headers={"Content-Type": "application/json"}, json=param0.model_dump())
        if valid_parameter and (e440xb_name not in out_of_range):
            assert response.status_code == 200
            body = response.json()
            rprms = E440xbReadableParams(**body)
            assert rprms.trace_mode == param0.trace_mode
            assert rprms.freq_center == param0.freq_center
            assert rprms.freq_span == param0.freq_span
            assert rprms.sweep_points == param0.sweep_points
            assert rprms.resolution_bandwidth == param0.resolution_bandwidth
        else:
            assert response.status_code == 400

    @pytest.mark.parametrize(
        ("freq_center", "freq_scan", "out_of_range"),
        [
            (9e9, 5e9, {}),
            (9e9, 10e9, {"E4405B"}),
        ],
    )
    def test_trace_typical(self, client, e440xb_name, freq_center, freq_scan, out_of_range, outsubdir):
        param0 = E440xbWritableParams(
            trace_mode=E440xbTraceMode.WRITE,
            freq_center=freq_center,
            freq_span=freq_scan,
            sweep_points=801,
            resolution_bandwidth=1e5,
            average_enable=False,
        )
        query: Dict[str, Union[bool, float]] = {
            "trace": True,
            "peak": True,
            "minimum_power": -30,
            "meta": True,
        }

        response = client.post("/param", headers={"Content-Type": "application/json"}, json=param0.model_dump())
        if e440xb_name not in out_of_range:
            assert response.status_code == 200
            response = client.get(f"/trace?{urlencode(query)}")
            assert response.status_code == 200

            body = response.json()
            assert body["meta"] is not None
            assert body["trace"] is not None

            meta = E440xbReadableParams(**json.loads(body["meta"]))
            assert meta.trace_mode == param0.trace_mode
            assert meta.freq_center == param0.freq_center
            assert meta.freq_span == param0.freq_span
            assert meta.sweep_points == param0.sweep_points
            assert meta.resolution_bandwidth == param0.resolution_bandwidth
            assert meta.average_enable == param0.average_enable

            trace = np.frombuffer(b64decode(body["trace"]))
            trace = trace.reshape((trace.shape[0] // 2, 2))
            assert trace.shape == (param0.sweep_points, 2)
            self._gen_spectrum_image(trace, outsubdir / "typical_server")
        else:
            assert response.status_code == 400

    def _gen_spectrum_image(self, trace, filepath):
        plt.cla()
        plt.plot(trace[:, 0], trace[:, 1])
        plt.savefig(filepath)
