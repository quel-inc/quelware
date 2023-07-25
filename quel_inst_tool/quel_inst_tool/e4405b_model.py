import logging
from enum import Enum
from typing import Union

from pydantic import BaseModel, Extra

from quel_inst_tool.e4405b import E4405b

logger = logging.getLogger(__name__)


class E4405bTraceMode(str, Enum):
    WRITE = "WRIT"
    MAXHOLD = "MAXH"
    MINHOLD = "MINH"
    VIEW = "VIEW"
    BLANK = "BLAN"


class E4405bParams(BaseModel):
    class Config:
        extra = Extra.forbid

    trace_mode: Union[E4405bTraceMode, None] = None
    freq_center: Union[float, None] = None
    freq_span: Union[float, None] = None
    resolution_bandwidth: Union[float, None] = None
    resolution_bandwidth_auto: Union[bool, None] = None
    sweep_points: Union[int, None] = None
    display_enable: Union[bool, None] = None
    average_enable: Union[bool, None] = None
    average_count: Union[int, None] = None
    video_bandwidth: Union[float, None] = None
    video_bandwidth_auto: Union[bool, None] = None
    video_bandwidth_ratio: Union[float, None] = None
    video_bandwidth_ratio_auto: Union[bool, None] = None
    input_attenuation: Union[float, None] = None


class E4405bReadableParams(E4405bParams):
    prod_id: Union[str, None] = None
    max_freq_error: Union[float, None] = None

    @classmethod
    def from_e4405b(cls, obj: E4405b) -> "E4405bReadableParams":
        model = E4405bReadableParams()
        # TODO: replace it with model_fields().
        for k in cls.__fields__.keys():
            setattr(model, k, getattr(obj, k))

        # TODO: validate the model before returning it.
        return model


class E4405bWritableParams(E4405bParams):
    def update_e4405b(self, obj: E4405b) -> "E4405bWritableParams":
        diff_model = E4405bWritableParams()
        # TODO: replace it with model_fields().
        for k in E4405bParams.__fields__.keys():
            v0 = getattr(self, k)
            if v0 is None:
                continue
            setattr(obj, k, v0)
            v1 = getattr(obj, k)
            if v0 != v1:
                logger.info(f"E4405B.{k} is supposed to be {v0}, but is actually set to {v1}")
            setattr(diff_model, k, v1)
        return diff_model
