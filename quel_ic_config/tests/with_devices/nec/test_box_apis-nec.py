import logging

import pytest

from quel_ic_config.quel1_box import Quel1Box
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

# Notes: to be merged into test cases in 'common' directory.

TEST_SETTINGS = (
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
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def box(request) -> Quel1Box:
    param0 = request.param

    # TODO: write something to modify boxtype.

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
    pass


def test_invalid_port(box):
    if box.boxtype == "quel1-b":
        with pytest.raises(ValueError, match="invalid port of quel1-b: 0"):
            box.config_box({0: {"lo_freq": 12e9}})

    with pytest.raises(ValueError, match=f"invalid port of {box.boxtype}: 20"):
        box.config_box({20: {"lo_freq": 12e9}})


def test_config_box_basic(box):
    a4 = {
        "mxfes": {
            0: {"channel_interporation_rate": 4, "main_interporation_rate": 6},
            1: {"channel_interporation_rate": 4, "main_interporation_rate": 6},
        },
        "ports": {
            0: {
                "channels": {0: {"fnco_freq": 0.0}},
                "cnco_freq": 1500000000.0,
                "direction": "out",
                "fullscale_current": 40527,
                "lo_freq": 12000000000,
                "sideband": "L",
            },
            1: {
                "channels": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}},
                "cnco_freq": 1500000000.0,
                "direction": "out",
                "fullscale_current": 40527,
                "lo_freq": 12000000000,
                "sideband": "L",
            },
            2: {
                "cnco_freq": 1500000000.0,
                "direction": "in",
                "lo_freq": 12000000000,
                "runits": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}, 3: {"fnco_freq": 0.0}},
            },
            3: {
                "channels": {0: {"fnco_freq": 0.0}},
                "cnco_freq": 1500000000.0,
                "direction": "out",
                "fullscale_current": 40527,
                "lo_freq": 12000000000,
                "sideband": "L",
            },
            4: {
                "channels": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}},
                "cnco_freq": 1500000000.0,
                "direction": "out",
                "fullscale_current": 40527,
                "lo_freq": 12000000000,
                "sideband": "L",
            },
            5: {
                "cnco_freq": 1500000000.0,
                "direction": "in",
                "lo_freq": 12000000000,
                "runits": {0: {"fnco_freq": 0.0}},
            },
            6: {
                "channels": {0: {"fnco_freq": 0.0}},
                "cnco_freq": 1500000000.0,
                "direction": "out",
                "fullscale_current": 40527,
                "lo_freq": 12000000000,
                "sideband": "L",
            },
            7: {
                "channels": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}},
                "cnco_freq": 1500000000.0,
                "direction": "out",
                "fullscale_current": 40527,
                "lo_freq": 12000000000,
                "sideband": "L",
            },
            8: {
                "cnco_freq": 1500000000.0,
                "direction": "in",
                "lo_freq": 12000000000,
                "runits": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}, 3: {"fnco_freq": 0.0}},
            },
            9: {
                "channels": {0: {"fnco_freq": 0.0}},
                "cnco_freq": 1500000000.0,
                "direction": "out",
                "fullscale_current": 40527,
                "lo_freq": 12000000000,
                "sideband": "L",
            },
            10: {
                "channels": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}},
                "cnco_freq": 1500000000.0,
                "direction": "out",
                "fullscale_current": 40527,
                "lo_freq": 12000000000,
                "sideband": "L",
            },
            11: {
                "cnco_freq": 1500000000.0,
                "direction": "in",
                "lo_freq": 12000000000,
                "runits": {0: {"fnco_freq": 0.0}},
            },
        },
    }

    if box.boxtype == "quel1-nec":
        box.config_box(a4["ports"])
        # box._dev.config_box(a4d["lines"])
    else:
        pytest.skip("TBA.")


def test_config_rfswitch_basic(box):
    rc = box.dump_rfswitches()
    box.config_rfswitches(rc)


@pytest.mark.parametrize(
    ("boxtypes", "config", "msg"),
    [
        (
            {"quel1-nec"},
            {0: "loop"},
            "invalid configuration of an output switch: loop",
        ),
        (
            {"quel1-nec"},
            {2: "block"},
            "invalid configuration of an input switch: block",
        ),
    ],
)
def test_config_rfswitch_invalid(boxtypes, config, msg, box):
    if box.boxtype in boxtypes:
        with pytest.raises(ValueError, match=msg):
            box.config_rfswitches(config)
    else:
        assert False, f"an unexpected fixture: {box}"


@pytest.mark.parametrize(
    (
        "boxtypes",
        "config",
    ),
    [
        (
            {"quel1-nec"},
            {0: "block"},
        ),
        (
            {"quel1-nec"},
            {2: "loop"},
        ),
    ],
)
def test_config_rfswitch_unrealizable(boxtypes, config, box):
    if box.boxtype in boxtypes:
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches(config)
    else:
        assert False, f"an unexpected fixture: {box}"
