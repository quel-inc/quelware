import logging
import re
from typing import Collection

import pytest
from boxdump_quel1a_01 import a1, a1d, a2d

from quel_ic_config.quel1_box import Quel1Box
from quel_ic_config.quel1_box_with_raw_wss import Quel1BoxWithRawWss
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.74",
            "ipaddr_sss": "10.2.0.74",
            "ipaddr_css": "10.5.0.74",
            "boxtype": Quel1BoxType.fromstr("quel1-a"),
            "config_root": None,
            "config_options": (),
        },
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.60",
            "ipaddr_sss": "10.2.0.60",
            "ipaddr_css": "10.5.0.60",
            "boxtype": Quel1BoxType.fromstr("quel1-b"),
            "config_root": None,
            "config_options": (),
        },
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.132",
            "ipaddr_sss": "10.2.0.132",
            "ipaddr_css": "10.5.0.132",
            "boxtype": Quel1BoxType.fromstr("quel1se-riken8"),
            "config_root": None,
            "config_options": (),
        },
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.80",
            "ipaddr_sss": "10.2.0.80",
            "ipaddr_css": "10.5.0.80",
            "boxtype": Quel1BoxType.fromstr("quel1-nec"),
            "config_root": None,
            "config_options": (),
        },
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.74",
            "ipaddr_sss": "10.2.0.74",
            "ipaddr_css": "10.5.0.74",
            "boxtype": Quel1BoxType.fromstr("quel1-a"),
            "config_root": None,
            "config_options": (),
        },
        "with_raw_wss": True,
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.132",
            "ipaddr_sss": "10.2.0.132",
            "ipaddr_css": "10.5.0.132",
            "boxtype": Quel1BoxType.fromstr("quel1se-riken8"),
            "config_root": None,
            "config_options": (),
        },
        "with_raw_wss": True,
    },
)


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def box(request) -> Quel1Box:
    param0 = request.param

    # TODO: write something to modify boxtype.

    if "with_raw_wss" in param0 and param0["with_raw_wss"]:
        box = Quel1BoxWithRawWss.create(**param0["box_config"], ignore_crc_error_of_mxfe={0, 1})
    else:
        box = Quel1Box.create(**param0["box_config"], ignore_crc_error_of_mxfe={0, 1})
    box.reconnect()
    for mxfe_idx, status in box.link_status().items():
        if not status:
            raise RuntimeError(f"test is not ready for {param0['box_config']['ipaddr_wss']}:mxfe-{mxfe_idx}")
    return box


def test_config_box_intrinsic_read_and_write(box):
    config = box._dev.dump_box()["lines"]
    assert box._dev.config_validate_box(config)

    box._dev.config_box(config)


def test_config_box_intrinsic_mismatch(box):
    config = box._dev.dump_box()["lines"]
    config[(1, 0)]["cnco_freq"] = 314159
    if box.boxtype != "quel1-nec":
        config[(1, 3)]["channels"][0]["fnco_freq"] = 271828
    else:
        config[(3, 1)]["channels"][0]["fnco_freq"] = 271828
    assert not box._dev.config_validate_box(config)


def test_config_box_intrinsic_inconsistent(box):
    config = box._dev.dump_box()["lines"]
    if (0, "r") not in config:
        pytest.skip("no readout channel for this box")
    assert config[(0, 0)]["lo_freq"] == config[(0, "r")]["lo_freq"]
    config[(0, 0)]["lo_freq"] -= 1e6
    with pytest.raises(ValueError):
        box._dev.config_box(config)


def test_config_box_validation(box):
    config = box.dump_box()["ports"]
    assert box.config_validate_box(config)

    box.config_box(config)


def test_config_box_mismatch(box):
    config = box.dump_box()["ports"]
    config[1]["cnco_freq"] = 314159  # Notes: readin for QuEL-1, Readout for QuEL-1 SE.
    if (1, 1) in config:
        config[(1, 1)]["fullscale_current"] = 29979
    if box.boxtype in {"quel1se-riken8"}:  # Notes: ctrl for QuEL-1, monitor-in for QuEL-1 SE.
        config[4]["runits"][0]["fnco_freq"] = 271828
    else:
        config[4]["channels"][0]["fnco_freq"] = 271828
    assert not box.config_validate_box(config)


def test_config_box_inconsistent(box):
    config = box.dump_box()["ports"]
    if box.boxtype == "quel1-a":
        assert config[0]["lo_freq"] == config[1]["lo_freq"]
        config[0]["lo_freq"] -= 1e6
    elif box.boxtype in {"quel1se-riken8", "x-quel1se-riken8"}:
        assert config[1]["lo_freq"] == config[0]["lo_freq"]
        config[1]["lo_freq"] -= 1e6
    elif box.boxtype == "quel1-b":
        pytest.skip("no readout channel for this box")
    elif box.boxtype == "quel1-nec":
        assert config[0]["lo_freq"] == config[2]["lo_freq"]
        config[0]["lo_freq"] -= 1e6
    else:
        assert False, f"unexpected boxtype: {box.boxtype}"

    with pytest.raises(ValueError):
        box.config_box(config)


def test_invalid_port(box):
    if box.boxtype == "quel1-b":
        with pytest.raises(ValueError, match="invalid port of quel1-b: 0"):
            box.config_box({0: {"lo_freq": 12e9}})

    with pytest.raises(ValueError, match=f"invalid port of {box.boxtype}: 20"):
        box.config_box({20: {"lo_freq": 12e9}})


@pytest.mark.parametrize(
    (
        "boxtypes",
        "for_box",
        "config",
    ),
    [
        (
            {"quel1-a"},
            True,
            a1,
        ),
        (
            {"quel1-a"},
            False,
            a1d,
        ),
        (
            {"quel1-a"},
            False,
            a2d,
        ),
    ],
)
def test_config_box_basic(boxtypes: Collection[str], for_box: bool, config, box):
    if box.boxtype in boxtypes:
        if for_box:
            box.config_box(config["ports"])
        else:
            box._dev.config_box(config["lines"])
    else:
        pytest.skip("TBA.")


def test_config_rfswitch_basic(box):
    rc = box.dump_rfswitches()
    box.config_rfswitches(rc)


def test_config_rfswitch_invalid(box):
    if box.boxtype == "quel1-a":
        with pytest.raises(ValueError, match="invalid configuration of an input switch: block"):
            box.config_rfswitches({0: "block"})
        with pytest.raises(ValueError, match="invalid configuration of an output switch: loop"):
            box.config_rfswitches({1: "loop"})
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({0: "loop", 1: "pass"})
    elif box.boxtype == "quel1-b":
        with pytest.raises(ValueError, match="invalid configuration of an output switch: loop"):
            box.config_rfswitches({1: "loop"})
    elif box.boxtype == "quel1se-riken8":
        with pytest.raises(ValueError, match="invalid configuration of an input switch: block"):
            box.config_rfswitches({0: "block"})
        with pytest.raises(ValueError, match="invalid configuration of an output switch: loop"):
            box.config_rfswitches({1: "loop"})
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({0: "loop", 1: "pass"})
    elif box.boxtype == "quel1-nec":
        with pytest.raises(ValueError, match="invalid configuration of an output switch: open"):
            box.config_rfswitches({0: "open"})
        with pytest.raises(ValueError, match="invalid configuration of an input switch: pass"):
            box.config_rfswitches({2: "pass"})
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({2: "loop"})
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({0: "block"})
    else:
        assert False, f"an unexpected fixture: {box}"


def test_config_fsc(box):
    if box.boxtype == "quel1se-riken8":
        for i in range(12000, 12100):
            box.config_box({6: {"fullscale_current": i}})
        box.config_box({6: {"fullscale_current": 40520}})
        box.config_box({6: {"fullscale_current": 40527}})
    else:
        pytest.skip()


def test_config_box_abnormal(box):
    if box.boxtype == "quel1se-riken8":
        assert not box.config_validate_box({6: {"non-existent": 999}})
        assert not box.config_validate_box({0: {"fullscale_current": 12000}})
        with pytest.raises(ValueError, match="invalid port: 5"):
            box.config_validate_box({5: {"fullscale_current": 12000}})

        with pytest.raises(ValueError, match=re.escape("port-#06 is not an input port, not applicable")):
            box.config_box(
                {
                    6: {
                        "runits": {
                            1: {"fnco_freq": 0.0},
                            2: {"fnco_freq": 0.0},
                            3: {"fnco_freq": 0.0},
                            0: {"fnco_freq": 0.0},
                        }
                    }
                }
            )
        with pytest.raises(ValueError, match=re.escape("port-#00 is not an output port, not applicable")):
            box.config_box({0: {"channels": {0: {"fnco_freq": 0.0}}}})
        assert not box.config_validate_box(
            {
                6: {
                    "runits": {
                        1: {"fnco_freq": 0.0},
                        2: {"fnco_freq": 0.0},
                        3: {"fnco_freq": 0.0},
                        0: {"fnco_freq": 0.0},
                    }
                }
            }
        )
        assert not box.config_validate_box({0: {"channels": {0: {"fnco_freq": 0.0}}}})

        with pytest.raises(ValueError, match=re.escape("invalid runit:2 for port-#04")):
            box.config_box({4: {"runits": {2: {"fnco_freq": 0.0}}}})
        with pytest.raises(ValueError, match=re.escape("invalid runit:2 for port-#04")):
            box.config_validate_box({4: {"runits": {2: {"fnco_freq": 0.0}}}})
        assert not box.config_validate_box({4: {"runits": {0: {"non-existent": 0.0}}}})

        with pytest.raises(ValueError, match=re.escape("an invalid combination of port-#08, and channel:1")):
            box.config_box({8: {"channels": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}}}})
        with pytest.raises(ValueError, match=re.escape("an invalid combination of port-#08, and channel:1")):
            box.config_validate_box(
                {8: {"channels": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}}}}
            )
        assert not box.config_validate_box({8: {"channels": {0: {"non-existent": 0.0}}}})

        with pytest.raises(ValueError, match="port-#04 is not an output port"):
            box.config_box({0: {"cnco_locked_with": 4}})
        with pytest.raises(ValueError, match="no cnco_locked_with is available for the output port-#06"):
            box.config_box({6: {"cnco_locked_with": 7}})

        with pytest.raises(ValueError, match=re.escape("no variable attenuator is available for port-#06")):
            box.config_box({6: {"vatt": 0xC00}})
        with pytest.raises(ValueError, match=re.escape("no variable attenuator is available for port-#01, subport-#1")):
            box.config_box({(1, 1): {"vatt": 0xC00}})
    else:
        pytest.skip()


def test_config_port_abnormal(box):
    if box.boxtype == "quel1se-riken8":
        with pytest.raises(ValueError, match="no DAC is available for the input port-#00"):
            box.config_port(0, fullscale_current=10000)
        with pytest.raises(ValueError, match=re.escape("no variable attenuator is available for port-#03")):
            box.config_port(3, vatt=0xC00)
        with pytest.raises(ValueError, match=re.escape("no variable attenuator is available for port-#01, subport-#1")):
            box.config_port(1, subport=1, vatt=0xC00)
        with pytest.raises(ValueError, match=re.escape("invalid subport-#2 of port-#01")):
            box.config_port(1, subport=2, vatt=0xC00)
    else:
        pytest.skip()


def test_config_rfswitches_abnormal(box):
    if box.boxtype == "quel1se-riken8":
        with pytest.raises(ValueError, match="invalid port of quel1se-riken8: non-existent"):
            box.config_rfswitches({"non-existent": "invalid"})  # type: ignore
        with pytest.raises(ValueError, match=re.escape("invalid configuration of an input switch: invalid")):
            box.config_rfswitches({0: "invalid"})
        with pytest.raises(ValueError, match=re.escape("invalid configuration of an input switch: pass")):
            box.config_rfswitches({0: "pass"})
        with pytest.raises(ValueError, match=re.escape("invalid configuration of an output switch: loop")):
            box.config_rfswitches({1: "loop"})
        with pytest.raises(ValueError, match=re.escape("invalid configuration of an output switch: loop")):
            box.config_rfswitches({2: "loop"})
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({0: "loop", 1: "pass"})
    else:
        pytest.skip()
