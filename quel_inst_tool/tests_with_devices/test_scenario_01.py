import os
import shutil
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from quel_inst_tool import E4405bClient, E4405bTraceMode, E4405bWritableParams

DEVICE_SETTINGS = {
    "e4405b_host": "172.30.2.203",
    "e4405b_port": 14405,
}


OUTPUT_SETTINGS = {
    "e4405b_test_output": "./artifacts/e4405b_test",
}


@pytest.fixture(scope="session", params=(DEVICE_SETTINGS,))
def e4405b_remote(request) -> E4405bClient:
    settings = request.param
    client = E4405bClient(settings["e4405b_host"], settings["e4405b_port"])
    assert client.reset() == (200, True)
    return client


@pytest.fixture(scope="session", params=(OUTPUT_SETTINGS,))
def outdir(request):
    mpl.use("Qt5Agg")
    dirname = request.param["e4405b_test_output"]
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    os.makedirs(dirname)
    return Path(dirname)


def test_param_set(e4405b_remote):
    param0 = E4405bWritableParams(
        trace_mode=E4405bTraceMode.WRITE,
        freq_center=9e9,
        freq_span=5e9,
        sweep_points=801,
        resolution_bandwidth=1e5,
        average_enable=False,
    )
    status1, param1 = e4405b_remote.param_set(param0)
    assert status1 == 200
    assert param1 is not None
    assert param1.trace_mode == param0.trace_mode
    assert param1.freq_center == param0.freq_center
    assert param1.freq_span == param0.freq_span
    assert param1.sweep_points == param0.sweep_points
    assert param1.resolution_bandwidth == param0.resolution_bandwidth
    assert param1.average_enable == param0.average_enable


def _gen_spectrum_image(trace, filepath):
    plt.cla()
    plt.plot(trace[:, 0], trace[:, 1])
    plt.savefig(filepath)


def test_trace_typical(e4405b_remote, outdir):
    e4405b_remote.reset()
    param0 = E4405bWritableParams(
        trace_mode=E4405bTraceMode.WRITE,
        freq_center=9e9,
        freq_span=5e9,
        sweep_points=4001,
        resolution_bandwidth=1e5,
        average_enable=False,
    )
    status1, _ = e4405b_remote.param_set(param0)
    assert status1 == 200

    status2, trace2, peak2, meta2 = e4405b_remote.trace_get(trace=True, peak=True, minimum_power=-30.0, meta=True)
    assert status2 == 200
    assert meta2.trace_mode == param0.trace_mode
    assert meta2.freq_center == param0.freq_center
    assert meta2.freq_span == param0.freq_span
    assert meta2.sweep_points == param0.sweep_points
    assert meta2.resolution_bandwidth == param0.resolution_bandwidth
    assert meta2.average_enable == param0.average_enable

    assert trace2.shape == (param0.sweep_points, 2)
    _gen_spectrum_image(trace2, outdir / "typical")


def test_trace_maxhold(e4405b_remote, outdir):
    e4405b_remote.reset()
    param0 = E4405bWritableParams(
        trace_mode=E4405bTraceMode.MAXHOLD,
        freq_center=8e9,
        freq_span=8e9,
        sweep_points=4001,
        resolution_bandwidth=1e5,
        average_enable=False,
    )
    status1, _ = e4405b_remote.param_set(param0)
    assert status1 == 200
    time.sleep(5.0)

    status2, trace2, peak2, meta2 = e4405b_remote.trace_get(trace=True, peak=True, minimum_power=-30.0, meta=True)
    assert status2 == 200
    assert meta2.trace_mode == param0.trace_mode
    assert meta2.freq_center == param0.freq_center
    assert meta2.freq_span == param0.freq_span
    assert meta2.sweep_points == param0.sweep_points
    assert meta2.resolution_bandwidth == param0.resolution_bandwidth
    assert meta2.average_enable == param0.average_enable

    assert trace2.shape == (param0.sweep_points, 2)
    _gen_spectrum_image(trace2, outdir / "maxhold")


def test_trace_minold(e4405b_remote, outdir):
    e4405b_remote.reset()
    param0 = E4405bWritableParams(
        trace_mode=E4405bTraceMode.MINHOLD,
        freq_center=8e9,
        freq_span=1e9,
        sweep_points=2001,
        resolution_bandwidth=1e5,
        average_enable=False,
    )
    status1, _ = e4405b_remote.param_set(param0)
    assert status1 == 200
    time.sleep(5.0)

    status2, trace2, peak2, meta2 = e4405b_remote.trace_get(trace=True, peak=True, minimum_power=-30.0, meta=True)
    assert status2 == 200
    assert meta2.trace_mode == param0.trace_mode
    assert meta2.freq_center == param0.freq_center
    assert meta2.freq_span == param0.freq_span
    assert meta2.sweep_points == param0.sweep_points
    assert meta2.resolution_bandwidth == param0.resolution_bandwidth
    assert meta2.average_enable == param0.average_enable

    assert trace2.shape == (param0.sweep_points, 2)
    _gen_spectrum_image(trace2, outdir / "minhold")
