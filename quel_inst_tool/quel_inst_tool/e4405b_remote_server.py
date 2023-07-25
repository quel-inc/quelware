import base64
import io
import logging
from typing import Any, Dict, Final

import matplotlib.pyplot as plt
from fastapi import FastAPI, Response

from quel_inst_tool.e4405b import E4405b
from quel_inst_tool.e4405b_model import E4405bReadableParams, E4405bWritableParams
from quel_inst_tool.spectrum_analyzer import InstDevManager

logger = logging.getLogger(__name__)

im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so", blacklist=["GPIB0::6::INSTR"])
e4405b = E4405b(im.lookup(prod_id="E4405B"))

DEFAULT_PEAK_MINIMUM_POWER: Final[float] = -60.0

app = FastAPI()


@app.get("/")
async def root():
    return {"greeting": "Hello world"}


@app.get("/param")
async def param_get() -> E4405bReadableParams:
    return E4405bReadableParams.from_e4405b(e4405b)


@app.post("/param")
async def param_set(param: E4405bWritableParams) -> E4405bWritableParams:
    return param.update_e4405b(e4405b)


@app.get("/trace/raw")
async def trace_raw_get():
    fd0 = e4405b.trace_get()
    return Response(content=fd0.tobytes(), media_type="application/octet_stream")


@app.get("/trace/png")
async def trace_png_get():
    fd0 = e4405b.trace_get()

    plt.cla()
    plt.plot(fd0[:, 0], fd0[:, 1])
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/trace")
async def trace_get(
    trace: bool = True, peak: bool = False, minimum_power: float = DEFAULT_PEAK_MINIMUM_POWER, meta: bool = False
):
    retval: Dict[str, Any] = {
        "trace": None,
        "peak": None,
        "meta": None,
    }

    fd0, p0 = e4405b.trace_and_peak_get(minimum_power=minimum_power)
    if trace:
        retval["trace"] = base64.b64encode(fd0.tobytes())
    if peak:
        retval["peak"] = base64.b64encode(p0.tobytes())
    if meta:
        retval["meta"] = E4405bReadableParams.from_e4405b(e4405b).json()
    return retval


@app.get("/reset")
async def reset_get():
    e4405b.reset()
    return {"details": "ok"}


@app.get("/average_clear")
async def average_clear_get():
    e4405b.average_clear()
    return {"details": "ok"}
