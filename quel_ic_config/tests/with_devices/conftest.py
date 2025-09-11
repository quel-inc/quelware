import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Generator, Union

import pytest
import yaml
from quel_inst_tool import SpectrumAnalyzer

from quel_ic_config.quel1_box import Quel1Box, Quel1BoxType
from quel_ic_config_utils import configuration
from testlibs.register_cw import register_cw_to_all_ports
from testlibs.spa_helper import init_e440xb, init_ms2xxxx, measure_floor_noise

logger = logging.getLogger(__name__)

artifacts_path = str(os.getenv("QUEL_TESTING_ARTIFACTS_DIR", "./artifacts"))

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


def make_topoutdir(param) -> Path:
    dirpath = Path(param["image_path"]) / param["label"]
    if os.path.exists(dirpath):
        logger.info(f"deleting the existing directory: '{dirpath}'")
        shutil.rmtree(dirpath)
    return dirpath


def prepare_artifact_dir(label: str) -> Path:
    dirpath = Path(artifacts_path) / label
    return dirpath


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


@pytest.fixture(scope="session", params=TEST_SETTINGS_E4405B)
def fixture_e4405b(
    request,
) -> Generator[SpectrumAnalyzer, None, None]:
    param0 = request.param
    spa = make_spa_fixture(param0)
    yield spa
    del spa


def pytest_addoption(parser):
    parser.addoption("--sysconf", action="store", help="path to sysconf file (.yaml)")


@pytest.fixture(scope="session")
def sysconf(request):
    sysconf_path = request.config.getoption("--sysconf")
    if sysconf_path:
        if not os.path.exists(sysconf_path):
            raise FileNotFoundError(f"system configuration file is not found: '{sysconf_path}'")
        with open(sysconf_path) as io:
            obj = yaml.safe_load(io)
            sysconf = configuration.SystemConfiguration.model_validate(obj)
    else:
        logger.info("Default system configuration will be used.")
        sysconf = configuration.load_default_configuration()
    return sysconf


class BoxProvider:
    def __init__(self, sysconf: configuration.SystemConfiguration):
        self._name_to_existing_box: dict[str, Quel1Box] = {}
        self._sysconf = sysconf

    def _create_and_reconnect(self, boxconf: configuration.Box):
        box = next(configuration.get_boxes((boxconf,)))
        assert box.has_lock, "no lock is available"

        status = configuration.reconnect_and_get_link_status(
            box, ignore_crc_error_of_mxfe=boxconf.ignore_crc_error_of_mxfe
        )
        if not all(status.values()):
            raise ValueError(f"Wrong link status on {box.name}: {status}")
        register_cw_to_all_ports(box)
        return box

    def get_box_from_type(self, boxtype: Union[Quel1BoxType, str], skip_if_not_found: bool = True) -> Quel1Box:
        boxconf = next(self.find_boxconf_from_type(boxtype, skip_if_not_found))
        if boxconf.name not in self._name_to_existing_box:
            box = self._create_and_reconnect(boxconf)
            self._name_to_existing_box[boxconf.name] = box
        return self._name_to_existing_box[boxconf.name]

    def find_boxconf_from_type(
        self, boxtype: Union[Quel1BoxType, str], skip_if_not_found: bool = True
    ) -> Iterator[configuration.Box]:
        if isinstance(boxtype, str):
            boxtype = Quel1BoxType.fromstr(boxtype)
        for b in self._sysconf.boxes:
            if Quel1BoxType.fromstr(b.boxtype) is boxtype:
                yield b
        if skip_if_not_found:
            pytest.skip(reason=f"box with type `{boxtype}` not found in system configuration.")
        raise ValueError(
            f"box with type `{boxtype}` not found in system configuration. Ensure the '--sysconf' option is set correctly."
        )

    def clean_up(self):
        for box in self._name_to_existing_box.values():
            box.initialize_all_awgunits()
            if box.boxtype != "quel1-nec":
                box.activate_monitor_loop(0)
                box.activate_monitor_loop(1)
        for name in [k for k in self._name_to_existing_box.keys()]:
            del self._name_to_existing_box[name]


@pytest.fixture(scope="session")
def box_provider(
    sysconf,
) -> Generator[BoxProvider, None, None]:
    box_repo = BoxProvider(sysconf)
    yield box_repo
    box_repo.clean_up()
