import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Final, Union

import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, Response

from quel_inst_tool.e440xb import E440xbReadableParams, E440xbWritableParams
from quel_inst_tool.e4405b import E4405b
from quel_inst_tool.e4407b import E4407b
from quel_inst_tool.spectrum_analyzer import InstDevManager

logger = logging.getLogger("uvicorn")

DEFAULT_PEAK_MINIMUM_POWER: Final[float] = -60.0

e440xb: Union[E4405b, E4407b]


@asynccontextmanager
async def lifespan(application: FastAPI):
    global e440xb
    saname = os.getenv("SPECTRUM_ANALYZER_NAME", default="E4405B")
    im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so")
    if saname == "E4407B":
        e440xb = E4407b(im.lookup(prod_id="E4407B"))
    else:
        e440xb = E4405b(im.lookup(prod_id="E4405B"))
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"greeting": "Hello world"}


@app.get("/param")
async def param_get() -> E440xbReadableParams:
    global e440xb
    return E440xbReadableParams.from_e440xb(e440xb)


@app.post("/param")
async def param_set(param: E440xbWritableParams) -> E440xbReadableParams:
    global e440xb
    if param.freq_center is not None or param.freq_span is not None:
        if (
            param.freq_center is None
            or param.freq_span is None
            or not e440xb.freq_range_check(param.freq_center, param.freq_span)
        ):
            raise HTTPException(status_code=400, detail="invalid frequency range")

    param.update_device_parameter(e440xb)
    time.sleep(1)
    return e440xb.check_prms(param)


@app.get("/trace/raw")
async def trace_raw_get():
    global e440xb
    fd0 = e440xb.trace_get()
    return Response(content=fd0.tobytes(), media_type="application/octet_stream")


@app.get("/trace/png")
async def trace_png_get():
    global e440xb
    fd0 = e440xb.trace_get()
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
    global e440xb
    retval: Dict[str, Any] = {
        "trace": None,
        "peak": None,
        "meta": None,
    }
    fd0, p0 = e440xb.trace_and_peak_get(minimum_power=minimum_power)
    if trace:
        retval["trace"] = base64.b64encode(fd0.tobytes())
    if peak:
        retval["peak"] = base64.b64encode(p0.tobytes())
    if meta:
        retval["meta"] = E440xbReadableParams.from_e440xb(e440xb).json()
    return retval


@app.get("/reset")
async def reset_get():
    global e440xb
    e440xb.reset()
    return {"details": "ok"}


@app.get("/average_clear")
async def average_clear_get():
    global e440xb
    e440xb.average_clear()
    return {"details": "ok"}
