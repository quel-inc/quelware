import logging
from typing import Collection

import pytest
from boxdump_quel1a_01 import a1, a1d, a1df, a1f, a2d, a2df

from quel_ic_config.quel1_box import Quel1Box
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
        },
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.60",
            "ipaddr_sss": "10.2.0.60",
            "ipaddr_css": "10.5.0.60",
            "boxtype": Quel1BoxType.fromstr("quel1-b"),
        },
    },
    # {
    #     "box_config": {
    #         "ipaddr_wss": "10.1.0.80",
    #         "ipaddr_sss": "10.2.0.80",
    #         "ipaddr_css": "10.5.0.80",
    #         "boxtype": Quel1BoxType.fromstr("quel1-nec"),
    #         "config_root": None,
    #         "config_options": (),
    #     },
    # },
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def box(request) -> Quel1Box:
    param0 = request.param

    # TODO: write something to modify boxtype.

    box = Quel1Box.create(**param0["box_config"], ignore_crc_error_of_mxfe={0, 1})
    box.reconnect()
    for mxfe_idx, status in box.link_status().items():
        if not status:
            raise RuntimeError(
                f"test is not ready for {param0['box_config']['ipaddr_wss']}:mxfe-{mxfe_idx} "
                f"due to invalid link status (= {status})"
            )
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
    config[1]["cnco_freq"] = 314159  # Notes: Readin for QuEL-1,
    if (1, 1) in config:
        config[(1, 1)]["fullscale_current"] = 29979
    config[4]["channels"][0]["fnco_freq"] = 271828
    assert not box.config_validate_box(config)


def test_config_box_inconsistent(box):
    config = box.dump_box()["ports"]
    if box.boxtype == "quel1-a":
        assert config[0]["lo_freq"] == config[1]["lo_freq"]
        config[0]["lo_freq"] -= 1e6
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
        with pytest.raises(ValueError, match="invalid port of quel1-b: #00"):
            box.config_box({0: {"lo_freq": 12e9}})

    with pytest.raises(ValueError, match=f"invalid port of {box.boxtype}: #20"):
        box.config_box({20: {"lo_freq": 12e9}})


@pytest.mark.parametrize(
    (
        "boxtypes",
        "for_box",
        "dual_modulus_nco",
        "config",
    ),
    [
        (
            {"quel1-a"},
            True,
            False,
            a1,
        ),
        (
            {"quel1-a"},
            False,
            False,
            a1d,
        ),
        (
            {"quel1-a"},
            False,
            False,
            a2d,
        ),
        (
            {"quel1-a"},
            True,
            True,
            a1f,
        ),
        (
            {"quel1-a"},
            False,
            True,
            a1df,
        ),
        (
            {"quel1-a"},
            False,
            True,
            a2df,
        ),
    ],
)
def test_config_box_basic(boxtypes: Collection[str], for_box: bool, dual_modulus_nco: bool, config, box):
    if box.boxtype in boxtypes:
        box.css.allow_dual_modulus_nco = dual_modulus_nco
        if for_box:
            box.config_box(config["ports"])
        else:
            box._dev.config_box(config["lines"])
        box.css.allow_dual_modulus_nco = True
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
