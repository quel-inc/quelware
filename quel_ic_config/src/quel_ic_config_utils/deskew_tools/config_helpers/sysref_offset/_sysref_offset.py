import logging
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLES_PER_CLOCK = 4
_SAMPLING_RATE = 500_000_000
_SAMPLING_PERIOD_PS = int(1_000_000_000_000 / _SAMPLING_RATE)
_SYSREF_PERIOD = 2000

BoxName = str


class NameSysrefCountPair(NamedTuple):
    name: str
    sysref_count: int


def _wrap_to_range(val: int, minval: int, maxval: int):
    _length = maxval - minval
    _offset = val - minval
    return minval + _offset % _length


def propose_wait_amount_by_sysref_offset(
    name_sysref_count_pairs: Iterable[tuple[str, int]], range_warning_threshold: int = 500
) -> dict[str, int]:
    name_offset_pairs: list[tuple[BoxName, int]] = []
    for pair in name_sysref_count_pairs:
        pair = NameSysrefCountPair(*pair)
        sysref_offset = pair.sysref_count % _SYSREF_PERIOD
        name_offset_pairs.append((pair.name, sysref_offset))

    # offsets are mapped to the unit circle to find the center.
    angles: list[float] = [offset / _SYSREF_PERIOD * 2 * np.pi for _, offset in name_offset_pairs]
    x_mean = np.mean(np.cos(angles))
    y_mean = np.mean(np.sin(angles))
    angle_of_center_of_mass = np.arctan2(y_mean, x_mean)
    shifted_angles = [(theta - angle_of_center_of_mass) for theta in angles]
    shifted_counts = [
        _wrap_to_range(int(shifted_angle * _SYSREF_PERIOD / 2 / np.pi), -_SYSREF_PERIOD // 2, _SYSREF_PERIOD // 2)
        for shifted_angle in shifted_angles
    ]
    if (_range := max(shifted_counts) - min(shifted_counts)) > range_warning_threshold:
        logger.warning(f"The range of count offsets from SYSREF is too large: {_range} > {range_warning_threshold}")

    logger.info(f"shifted_counts: {shifted_counts}")
    name_to_wait = {}
    for pair, shifted in zip(name_offset_pairs, shifted_counts):
        name, _ = pair
        wait_ps = -shifted * _SAMPLES_PER_CLOCK * _SAMPLING_PERIOD_PS
        name_to_wait[name] = wait_ps

    return name_to_wait
