import copy
import json
from pathlib import Path
from typing import Any, Dict, Set

import pytest
from pydantic.v1.utils import deep_update

from quel_ic_config import QubeConfigSubsystem, Quel1BoxType, Quel1ConfigOption, Quel1Feature


def _remove_comments(settings: Dict[str, Any]) -> Dict[str, Any]:
    s1: Dict[str, Any] = {}
    for k, v in settings.items():
        if not k.startswith("#"):
            if isinstance(v, dict):
                s1[k] = _remove_comments(v)
            else:
                s1[k] = v
    return s1


def _load_settings_reference(boxtype: Quel1BoxType, config_options: Set[Quel1ConfigOption]) -> Dict[str, Any]:
    NUM_AD9082 = 2
    NUM_LMX2594 = 10
    NUM_ADRF6780 = 8
    NUM_AD5328 = 1
    NUM_GPIO = 1

    _config_path = Path("src/quel_ic_config/settings")

    # Notes: boxtype_main is defined in the class implementing loader, because it determines proxy objects to create
    #        independently of the configuration file.
    boxtype_main = "quel-1"

    # TODO: a root setting file will be selected with boxtype, finally.
    root: Dict[str, Any] = {
        "meta": {
            "root": {"version": "1.99.0"},
            "ad9082": [],
            "lmx2594": [],
            "adrf6780": [],
            "ad5328": [],
            "gpio": [],
        },
        "ad9082": [],
        "lmx2594": [],
        "adrf6780": [],
        "ad5328": [],
        "gpio": [],
    }

    # TODO: loading root file tree from "quel-1.json" to find other json files to load.
    #       (currently, files are loaded in an ad hoc manner.)
    for idx in range(NUM_AD9082):
        with open(_config_path / boxtype_main / "ad9082.json") as f:
            setting: Dict[str, Any] = json.load(f)
        with open(_config_path / boxtype_main / f"ad9082_dac_channel_assign_for_mxfe{idx:d}.json") as f:
            setting = deep_update(setting, json.load(f))
        if idx == 0:
            if Quel1ConfigOption.USE_READ_IN_MXFE0 in config_options:
                with open(_config_path / boxtype_main / "ad9082_read.json") as f:
                    setting = deep_update(setting, json.load(f))
            else:
                with open(_config_path / boxtype_main / "ad9082_monitor.json") as f:
                    setting = deep_update(setting, json.load(f))
        elif idx == 1:
            if Quel1ConfigOption.USE_READ_IN_MXFE1 in config_options:
                with open(_config_path / boxtype_main / "ad9082_read.json") as f:
                    setting = deep_update(setting, json.load(f))
            else:
                with open(_config_path / boxtype_main / "ad9082_monitor.json") as f:
                    setting = deep_update(setting, json.load(f))
        root["meta"]["ad9082"].append(setting["meta"])
        del setting["meta"]
        root["ad9082"].append(setting)

    for idx in range(NUM_LMX2594):
        if idx in {1, 6}:
            with open(_config_path / boxtype_main / "lmx2594_two_lo.json") as f:
                setting = json.load(f)
        elif idx in {0, 7}:
            if boxtype in {Quel1BoxType.QuEL1_TypeA, Quel1BoxType.QuBE_OU_TypeA, Quel1BoxType.QuBE_RIKEN_TypeA}:
                with open(_config_path / boxtype_main / "lmx2594_two_lo.json") as f:
                    setting = json.load(f)
            elif boxtype in {Quel1BoxType.QuEL1_TypeB, Quel1BoxType.QuBE_OU_TypeB, Quel1BoxType.QuBE_RIKEN_TypeB}:
                with open(_config_path / boxtype_main / "lmx2594_lo.json") as f:
                    setting = json.load(f)
            else:
                raise AssertionError
        elif idx in {2, 3, 4, 5}:
            with open(_config_path / boxtype_main / "lmx2594_lo.json") as f:
                setting = json.load(f)
        elif idx in {8}:
            if Quel1ConfigOption.REFCLK_CORRECTED_MXFE0 in config_options:
                with open(_config_path / boxtype_main / "lmx2594_refclk_corrected.json") as f:
                    setting = json.load(f)
            else:
                with open(_config_path / boxtype_main / "lmx2594_refclk.json") as f:
                    setting = json.load(f)
        elif idx in {9}:
            if Quel1ConfigOption.REFCLK_CORRECTED_MXFE1 in config_options:
                with open(_config_path / boxtype_main / "lmx2594_refclk_corrected.json") as f:
                    setting = json.load(f)
            else:
                with open(_config_path / boxtype_main / "lmx2594_refclk.json") as f:
                    setting = json.load(f)
        else:
            raise AssertionError
        root["meta"]["lmx2594"].append(setting["meta"])
        del setting["meta"]
        root["lmx2594"].append(_remove_comments(setting))

    for idx in range(NUM_ADRF6780):
        with open(_config_path / boxtype_main / "adrf6780.json") as f:
            setting = json.load(f)
        with open(_config_path / boxtype_main / "adrf6780_lsb.json") as f:
            setting = deep_update(setting, json.load(f))
        root["meta"]["adrf6780"].append(setting["meta"])
        del setting["meta"]
        root["adrf6780"].append(setting)

    for idx in range(NUM_AD5328):
        with open(_config_path / boxtype_main / "ad5328.json") as f:
            setting = json.load(f)
        root["meta"]["ad5328"].append(setting["meta"])
        del setting["meta"]
        root["ad5328"].append(setting)

    for idx in range(NUM_GPIO):
        with open(_config_path / boxtype_main / "rfswitch.json") as f:
            setting = json.load(f)
        root["meta"]["gpio"].append(setting["meta"])
        del setting["meta"]
        root["gpio"].append(setting)

    return root


@pytest.mark.parametrize(
    ("boxtype", "features", "config_options"),
    [
        (
            Quel1BoxType.QuEL1_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_READ_IN_MXFE0, Quel1ConfigOption.USE_READ_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuEL1_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_READ_IN_MXFE0, Quel1ConfigOption.REFCLK_CORRECTED_MXFE0},
        ),
        (Quel1BoxType.QuEL1_TypeA, {Quel1Feature.SINGLE_ADC}, {Quel1ConfigOption.REFCLK_CORRECTED_MXFE1}),
        (
            Quel1BoxType.QuEL1_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_MONITOR_IN_MXFE0, Quel1ConfigOption.USE_READ_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuEL1_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_READ_IN_MXFE0, Quel1ConfigOption.USE_MONITOR_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuEL1_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_MONITOR_IN_MXFE0, Quel1ConfigOption.USE_MONITOR_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuEL1_TypeB,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_MONITOR_IN_MXFE0, Quel1ConfigOption.USE_MONITOR_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuBE_OU_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_READ_IN_MXFE0, Quel1ConfigOption.USE_READ_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuBE_RIKEN_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_MONITOR_IN_MXFE0, Quel1ConfigOption.USE_READ_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuBE_RIKEN_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_READ_IN_MXFE0, Quel1ConfigOption.USE_MONITOR_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuBE_RIKEN_TypeA,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_MONITOR_IN_MXFE0, Quel1ConfigOption.USE_MONITOR_IN_MXFE1},
        ),
        (
            Quel1BoxType.QuBE_RIKEN_TypeB,
            {Quel1Feature.SINGLE_ADC},
            {Quel1ConfigOption.USE_MONITOR_IN_MXFE0, Quel1ConfigOption.USE_MONITOR_IN_MXFE1},
        ),
    ],
)
def test_config_loader(boxtype: Quel1BoxType, features: Set[Quel1Feature], config_options: Set[Quel1ConfigOption]):
    qco = QubeConfigSubsystem("241.3.5.6", boxtype, features, Path("src/quel_ic_config/settings"), config_options)

    target = copy.copy(qco._param)
    del target["meta"]

    answer = _load_settings_reference(boxtype, config_options)
    del answer["meta"]

    assert target["gpio"] == answer["gpio"]
    assert target["ad5328"] == answer["ad5328"]
    assert target["adrf6780"] == answer["adrf6780"]
    assert target["lmx2594"] == answer["lmx2594"]
    assert target["ad9082"] == answer["ad9082"]
    assert target == answer
