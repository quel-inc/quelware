import logging
import time
from argparse import ArgumentParser
from typing import List, Sequence, Union, cast

from common_args import add_common_arguments, open_cmod

from quel_cmod_scripting import Quel1SeProtoCmod

logger = logging.getLogger()


def log_temp(cmod: Quel1SeProtoCmod, duration: int, step: int):
    t0 = time.perf_counter()
    t1 = t0

    t = time.perf_counter()
    while t < t0 + duration:
        if t1 < t:
            temp = cmod.read_temp()
            logger.info(temp)
            t1 += step

        t = time.perf_counter()
        time.sleep(1)


def scan_b_heater(
    cmod: Quel1SeProtoCmod,
    heaters: List[Union[None, str]],
    duration: int,
    step: int,
    ratio: float = 0.6,
):
    prev_idx: Union[str, None] = None
    for idx in heaters:
        if prev_idx is not None:
            logger.info(f"turning OFF heater {prev_idx}")
            cmod.set_b_heater(prev_idx, 0.0)
        prev_idx = idx
        if idx is not None:
            logger.info(f"turning ON heater {idx} with ratio of {ratio}")
            cmod.set_b_heater(idx, ratio)
        log_temp(cmod, duration, step)

    if prev_idx is not None:
        logger.info(f"turning OFF heater {prev_idx}")
        cmod.set_b_heater(prev_idx, 0.0)


def scan_x_heater(
    cmod: Quel1SeProtoCmod,
    heaters: List[Union[None, str]],
    duration: int,
    step: int,
    ratio: float = 0.3,
):
    # XXX: You need to call cmod.init_heater() to activate the master switch of all the eXtended/eXternal heaters.
    prev_idx: Union[str, None] = None
    for idx in heaters:
        if prev_idx is not None:
            logger.info(f"turning OFF heater {prev_idx}")
            cmod.set_x_heater(prev_idx, 0.0)
        prev_idx = idx
        if idx is not None:
            logger.info(f"turning ON heater {idx} with ratio of {ratio}")
            cmod.set_x_heater(idx, ratio)
        log_temp(cmod, duration, step)

    if prev_idx is not None:
        logger.info(f"turning OFF heater {prev_idx}")
        cmod.set_x_heater(prev_idx, 0.0)


def scan_fan(
    cmod: Quel1SeProtoCmod,
    duration: int,
    step: int,
    fan_settings: Sequence[float],
):
    for fan_ratio in fan_settings:
        cmod.set_fan(fan_ratio)
        logger.info(f"fan ratio is set to {fan_ratio}")
        log_temp(cmod, duration, step)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )
    parser = ArgumentParser(
        description="open uart monitor at the given port or JTAG adapter id"
    )
    add_common_arguments(parser, Quel1SeProtoCmod)
    args = parser.parse_args()
    cmod: Quel1SeProtoCmod = cast(Quel1SeProtoCmod, open_cmod(args, Quel1SeProtoCmod))
