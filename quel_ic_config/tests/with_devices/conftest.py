import logging
import os
import shutil
from pathlib import Path
from typing import Any, Generator

import pytest
from quel_inst_tool import SpectrumAnalyzer

from quel_ic_config.quel1_box import Quel1Box, Quel1BoxType
from testlibs.register_cw import register_cw_to_all_ports
from testlibs.spa_helper import init_e440xb, init_ms2xxxx, measure_floor_noise

logger = logging.getLogger(__name__)

if (artifacts_path := os.getenv("QUEL_TESTING_ARTIFACTS_DIR")) is None:
    artifacts_path = "./artifacts"

TEST_SETTINGS_MS2720T1 = (
    {
        "spa_type": "MS2XXXX",
        "spa_name": "ms2720t-1",
        "spa_parameters": {
            "freq_center": 5e9,
            "freq_span": 8e9,
            "resolution_bandwidth": 1e4,
        },
        "max_background_noise": -50.0,
    },
)

TEST_SETTINGS_QUEL1 = (
    {
        "label": "staging-074",
        "box_config": {
            "ipaddr_wss": "10.1.0.74",
            "ipaddr_sss": "10.2.0.74",
            "ipaddr_css": "10.5.0.74",
            "boxtype": Quel1BoxType.fromstr("quel1-a"),
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "image_path": artifacts_path,
        "relative_loss": 0,
        "linkup": False,
    },
    {
        "label": "staging-050",
        "box_config": {
            "ipaddr_wss": "10.1.0.50",
            "ipaddr_sss": "10.2.0.50",
            "ipaddr_css": "10.5.0.50",
            "boxtype": Quel1BoxType.fromstr("quel1-a"),
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "image_path": artifacts_path,
        "relative_loss": 0,
        "linkup": False,
    },
    {
        "label": "staging-060",
        "box_config": {
            "ipaddr_wss": "10.1.0.60",
            "ipaddr_sss": "10.2.0.60",
            "ipaddr_css": "10.5.0.60",
            "boxtype": Quel1BoxType.fromstr("quel1-b"),
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "ignore_crc_error_of_mxfe": (0, 1),
            "use_204b": False,
        },
        "reconnect_config": {
            "ignore_crc_error_of_mxfe": (0, 1),
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "image_path": artifacts_path,
        "relative_loss": 9,
        "linkup": False,
    },
)

TEST_SETTINGS_E4405B = (
    {
        "spa_type": "E4405B",
        "spectrum_image_path": f"{artifacts_path}/spectrum-060",
        "max_background_noise": -55.0,
        "spa_parameters": {
            "freq_center": 8.5e9,
            "freq_span": 6e9,
            "resolution_bandwidth": 3e4,
            "sweep_points": 4001,
        },
    },
)

TEST_SETTINGS_RIKEN8 = (
    {
        "label": "staging-094",
        "box_config": {
            "ipaddr_wss": "10.1.0.94",
            "ipaddr_sss": "10.2.0.94",
            "ipaddr_css": "10.5.0.94",
            "boxtype": Quel1BoxType.fromstr("quel1se-riken8"),
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "image_path": artifacts_path,
        "relative_loss": 0,
        "linkup": False,
    },
)

TEST_SETTINGS_FUJITSU11A = (
    {
        "label": "staging-157",
        "box_config": {
            "ipaddr_wss": "10.1.0.157",
            "ipaddr_sss": "10.2.0.157",
            "ipaddr_css": "10.5.0.157",
            "boxtype": Quel1BoxType.fromstr("quel1se-fujitsu11-a"),
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "spa_type": "",
        "image_path": artifacts_path,
        "relative_loss": 0,
        "linkup": False,
    },
)


TEST_SETTINGS_FUJITSU11B = (
    {
        "label": "staging-164",
        "box_config": {
            "ipaddr_wss": "10.1.0.164",
            "ipaddr_sss": "10.2.0.164",
            "ipaddr_css": "10.5.0.164",
            "boxtype": Quel1BoxType.fromstr("quel1se-fujitsu11-b"),
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "spa_type": "",
        "image_path": artifacts_path,
        "relative_loss": 0,
        "linkup": False,
    },
)


def make_topoutdir(param) -> Path:

    dirpath = Path(param["image_path"]) / param["label"]
    if os.path.exists(dirpath):
        logger.info(f"deleting the existing directory: '{dirpath}'")
        shutil.rmtree(dirpath)
    return dirpath


def make_box_fixture(param0):
    topdirpath = make_topoutdir(param0)

    box = Quel1Box.create(**param0["box_config"])
    assert box.has_lock, "no lock is available"

    if param0["linkup"]:
        linkstatus = box.relinkup(**param0["linkup_config"])
    else:
        cfg = param0["reconnect_config"] if "reconnect_config" in param0 else {}
        linkstatus = box.reconnect(**cfg)
    assert linkstatus[0]
    assert linkstatus[1]

    register_cw_to_all_ports(box)

    return box, topdirpath


def make_spa_fixture(param0):
    if param0["spa_type"] == "MS2XXXX":
        spa: SpectrumAnalyzer = init_ms2xxxx(param0["spa_name"], **param0["spa_parameters"])
    elif param0["spa_type"] == "E4405B":
        spa = init_e440xb("E4405B")
    else:
        # Notes: to be added by need.
        assert False

    if spa:
        max_noise = measure_floor_noise(spa)
        assert max_noise < param0["max_background_noise"]

    return spa


@pytest.fixture(scope="session", params=TEST_SETTINGS_MS2720T1)
def fixture_ms2720t1(
    request,
) -> Generator[SpectrumAnalyzer, None, None]:
    param0 = request.param
    spa = make_spa_fixture(param0)
    yield spa
    del spa


@pytest.fixture(scope="session", params=TEST_SETTINGS_QUEL1)
def fixtures1(
    request,
) -> Generator[tuple[Quel1Box, dict[str, Any], Path], None, None]:
    param0 = request.param

    box, topdirpath = make_box_fixture(param0)

    yield box, param0, topdirpath

    box.initialize_all_awgunits()
    box.activate_monitor_loop(0)
    box.activate_monitor_loop(1)
    del box


@pytest.fixture(scope="session", params=TEST_SETTINGS_E4405B)
def fixture_e4405b(
    request,
) -> Generator[SpectrumAnalyzer, None, None]:
    param0 = request.param
    spa = make_spa_fixture(param0)
    yield spa
    del spa


@pytest.fixture(scope="session", params=TEST_SETTINGS_RIKEN8)
def fixtures8(
    request,
) -> Generator[tuple[Quel1Box, dict[str, Any], Path], None, None]:
    param0 = request.param

    box, topdirpath = make_box_fixture(param0)

    yield box, param0, topdirpath

    box.initialize_all_awgunits()
    box.activate_monitor_loop(0)
    box.activate_monitor_loop(1)
    del box


@pytest.fixture(scope="session", params=TEST_SETTINGS_FUJITSU11A)
def fixtures11a(
    request,
) -> Generator[tuple[Quel1Box, dict[str, Any], Path], None, None]:
    param0 = request.param

    box, topdirpath = make_box_fixture(param0)

    yield box, param0, topdirpath

    box.initialize_all_awgunits()
    box.activate_monitor_loop(0)
    box.activate_monitor_loop(1)
    del box


@pytest.fixture(scope="session", params=TEST_SETTINGS_FUJITSU11B)
def fixtures11b(
    request,
) -> Generator[tuple[Quel1Box, dict[str, Any], Path], None, None]:
    param0 = request.param

    box, topdirpath = make_box_fixture(param0)

    yield box, param0, topdirpath

    box.initialize_all_awgunits()
    box.activate_monitor_loop(0)
    box.activate_monitor_loop(1)
    del box
