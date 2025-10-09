import concurrent.futures
import logging
import os
from collections.abc import Collection, Iterable, Iterator
from ipaddress import IPv4Address
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

import quel_ic_config as qi

logger = logging.getLogger(__name__)


class ClockMaster(BaseModel):
    ipaddr: IPv4Address


class Box(BaseModel):
    name: str
    ipaddr: IPv4Address
    boxtype: str
    ignore_crc_error_of_mxfe: set[int] = Field(default=set())
    ignore_access_failure_of_adrf6780: set[int] = Field(default=set())
    ignore_lock_failure_of_lmx2594: set[int] = Field(default=set())
    marks: set[str] = Field(default=set())


class SystemConfiguration(BaseModel):
    version: int
    clockmaster: list[ClockMaster]
    boxes: list[Box]

    @model_validator(mode="after")
    def has_unique_names(self):
        names = set()
        for b in self.boxes:
            if b.name in names:
                raise ValueError(f"All names must be unique: {b.name}")
            else:
                names.add(b.name)
        return self

    @model_validator(mode="after")
    def has_unique_ipaddrs(self):
        ipaddrs = set()
        for b in self.boxes:
            if b.ipaddr in ipaddrs:
                raise ValueError(f"All IP addresses must be unique: {b.ipaddr}")
            else:
                ipaddrs.add(b.ipaddr)
        return self

    @model_validator(mode="after")
    def has_zero_or_one_clockmaster(self):
        if len(self.clockmaster) > 1:
            raise ValueError(f"Number of clockmaster must be equal to zero or one. Found {len(self.clockmaster)}.")
        return self


def get_boxes(conf_boxes: Iterable[Box]) -> Iterator[qi.Quel1Box]:
    boxes: list[qi.Quel1Box] = []
    for b in conf_boxes:
        logger.info(f"Starting to create a box instance for {b.name}.")
        boxtype = qi.Quel1BoxType.fromstr(b.boxtype)
        box = qi.Quel1Box.create(
            ipaddr_wss=str(b.ipaddr),
            boxtype=boxtype,
            name=b.name,
            ignore_crc_error_of_mxfe=b.ignore_crc_error_of_mxfe,
            ignore_access_failure_of_adrf6780=b.ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=b.ignore_lock_failure_of_lmx2594,
        )
        logger.info(f"Finished to create a box instance for {b.name} successfully.")
        boxes.append(box)

    return (b for b in boxes)


def get_boxes_in_parallel(conf_boxes: Iterable[Box]) -> Iterator[qi.Quel1Box]:
    boxes: list[qi.Quel1Box] = []
    _conf_boxes = list(conf_boxes)
    if len(_conf_boxes) == 0:
        return iter([])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_conf_boxes)) as executer:
        future_to_conf_box = {executer.submit(get_boxes, [conf_box]): conf_box for conf_box in _conf_boxes}
        for future in concurrent.futures.as_completed(future_to_conf_box):
            try:
                boxes.extend(b for b in future.result())
            except Exception as exc:
                conf_box = future_to_conf_box[future]
                logger.warning(f"A thread to connect to {conf_box.name} generates an exception: {exc}")

    return (b for b in boxes)


def reconnect_and_get_link_status(
    box: qi.Quel1Box,
    background_noise_threshold: Optional[float] = None,
    ignore_crc_error_of_mxfe: Optional[Collection[int]] = None,
) -> dict[Any, bool]:
    logger.info(f"Starting to reconnect to {box.name}.")
    status = box.reconnect(
        background_noise_threshold=background_noise_threshold,
        ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
        ignore_invalid_linkstatus=True,
    )
    logger.info(f"Finishing to reconnect to {box.name}.")
    logger.debug(f"link status of {box.name} is {status}.")
    return status


def reconnect_and_get_link_status_in_parallel(
    boxes: Iterable[qi.Quel1Box],
    background_noise_threshold: Optional[float] = None,
    ignore_all_crc_errors=False,
) -> dict[str, dict[Any, bool]]:
    name_to_status: dict[str, dict[Any, bool]] = {}
    _boxes = list(boxes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_boxes)) as executer:
        future_to_box = {
            executer.submit(
                reconnect_and_get_link_status,
                box,
                background_noise_threshold,
                (0, 1) if ignore_all_crc_errors else None,
            ): box
            for box in _boxes
        }
        for future in concurrent.futures.as_completed(future_to_box):
            box = future_to_box[future]
            try:
                name_to_status[box.name] = future.result()
            except Exception as exc:
                logger.warning(f"A thread to reconnect to {box.name} generates an exception: {exc}")

    return name_to_status


def get_clockmaster(
    conf_clockmaster: list[ClockMaster], boxes: Collection[qi.Quel1Box]
) -> Optional[qi.QuelClockMasterV1]:
    for cm in conf_clockmaster:
        clockmaster = qi.QuelClockMasterV1(ipaddr=str(cm.ipaddr), boxes=boxes)
        if not clockmaster.check_availability():
            raise Exception("Clockmaster is not available.")
        return clockmaster
    return None


def load_default_configuration() -> SystemConfiguration:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))

    config_dir = os.path.join(xdg_config_home, "quelware")
    config_file = os.path.join(config_dir, "sysconf.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found at {config_file}.")

    try:
        with open(config_file, "r") as f:
            obj = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error reading {config_file}: {e}") from e

    return SystemConfiguration.model_validate(obj)
