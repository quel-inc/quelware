import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field


class ClassificationParam(BaseModel, validate_assignment=True, extra="forbid"):
    pivot_x: float = Field(default=0.0)
    pivot_y: float = Field(default=0.0)
    angle_main: float = Field(ge=-180.0, le=180.0, default=0.0)
    angle_sub: float = Field(ge=-180.0, le=180.0, default=-90.0)


def calc_abc(x0: float, y0: float, angle: float, total_exponent_offset: int) -> npt.NDArray[np.float32]:
    fullscale = 32767
    if (-180 <= angle < -135) or (-45.0 <= angle < 45.0) or (135 <= angle < 180):
        a = -np.tan(np.deg2rad(angle)) * fullscale
        b = fullscale
    else:
        a = -fullscale
        b = fullscale / np.tan(np.deg2rad(angle))

    if angle < -45 or 135 <= angle:
        a = -a
        b = -b

    return np.array([a, b, -(a * x0 + b * y0) * (1 << total_exponent_offset)], np.float32)
