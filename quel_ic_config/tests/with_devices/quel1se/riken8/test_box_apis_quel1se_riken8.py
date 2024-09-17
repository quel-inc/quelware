import json
import logging
import os.path as osp
import re
import tempfile
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def test_just_run_apis(fixtures8):
    box, params, dpath = fixtures8

    box.initialize_all_awgunits()

    box.get_names_of_wavedata(1, 0)
    box.get_names_of_wavedata((1, 0), 0)
    box.get_names_of_wavedata((1, 1), 0)
    with pytest.raises(ValueError, match=re.escape("invalid channel-#1 of port-#01")):
        box.get_names_of_wavedata(1, 1)
    with pytest.raises(ValueError, match=re.escape("invalid port of quel1se-riken8: #00-1")):
        box.get_names_of_wavedata((0, 1), 1)
    with pytest.raises(ValueError, match="port-#00 is not an output port"):
        box.get_names_of_wavedata(0, 0)
    with pytest.raises(ValueError, match="invalid port of quel1se-riken8: #05"):
        box.get_names_of_wavedata(5, 0)
    with pytest.raises(TypeError):
        box.get_names_of_wavedata(0)  # type: ignore
    with pytest.raises(TypeError):
        box.get_names_of_wavedata("wrong type")  # type: ignore

    # Notes: monitor-out port is not included.  TODO: reconsider this design to improve error readability of error.
    for p_sp in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10):
        p, sp = box._decode_port(p_sp)
        assert p == p_sp
        assert sp == 0
        ps = box.get_ports_sharing_physical_port(p_sp)
        if p_sp != 1:
            assert len(ps) == 1
            assert list(ps)[0] == p_sp
        else:
            assert ps == {1, (1, 1)}

    p, sp = box._decode_port((1, 1))
    assert p == 1
    assert sp == 1
    ps = box.get_ports_sharing_physical_port((1, 1))
    assert ps == {1, (1, 1)}

    for g in (0, 1):
        box.deactivate_monitor_loop(g)
        assert not box.is_loopedback_monitor(g)
    for p, conf in box.dump_rfswitches().items():
        if p in {4, 10}:
            assert conf == "open"

    for g in (0, 1):
        box.activate_monitor_loop(g)
        assert box.is_loopedback_monitor(g)
    for p, conf in box.dump_rfswitches().items():
        if p in {4, 10}:
            assert conf == "loop"

    box.pass_all_output_ports()
    for p, conf in box.dump_rfswitches().items():
        if p in {2, 3, 6, 7, 8, 9}:  # Notes: 1 is excluded since it is a copy of port-#0
            assert conf == "pass"
        elif p == 0:
            assert conf == "open"
    for p in (1, 2, 3, 6, 7, 8, 9):
        assert box.dump_rfswitch(p) == "pass"

    box.block_all_output_ports()
    for p, conf in box.dump_rfswitches().items():
        if p in {2, 3, 6, 7, 8, 9}:  # Notes: 1 is excluded since it is a copy of port-#0
            assert conf == "block"
        elif p == 0:
            assert conf == "loop"
    for p in (1, 2, 3, 6, 7, 8, 9):
        assert box.dump_rfswitch(p) == "block"


def test_config_fsc(fixtures8):
    box, params, dpath = fixtures8

    if box.boxtype == "quel1se-riken8":
        for i in range(12000, 12100):
            box.config_box({6: {"fullscale_current": i}})
        box.config_box({6: {"fullscale_current": 40520}})
        box.config_box({6: {"fullscale_current": 40527}})
    else:
        pytest.skip()


@pytest.mark.parametrize(
    ("conffilename", "ports_only"),
    [("outports.json", True), ("full.json", False), ("mxfes.json", False)],
)
def test_config_box_json(conffilename, ports_only, fixtures8):
    box, params, dpath = fixtures8

    if box.boxtype == "quel1se-riken8":
        conf0path = Path(osp.dirname(__file__)) / "settings" / conffilename
        box.config_box_from_jsonfile(conf0path)
        with tempfile.TemporaryDirectory() as dirpath:
            conf1path = Path(dirpath) / "reproduced.json"
            box.dump_box_to_jsonfile(conf1path)
            with open(conf0path) as f:
                conf0 = json.load(f)
            with open(conf1path) as g:
                conf1 = json.load(g)

        if ports_only:
            conf1 = conf1["ports"]
            for pidx, pconf0 in conf0.items():
                pconf1 = conf1[pidx]
                del pconf1["direction"]
                assert pconf0 == pconf1
        else:
            for k, v in conf0.items():
                if k == "mxfes":
                    assert conf1[k] == v
                elif k == "ports":
                    for pidx, pconf0 in v.items():
                        pconf1 = conf1["ports"][pidx]
                        del pconf1["direction"]
                        assert pconf0 == pconf1
                else:
                    assert False, f"unexpected key (='{k}') in the test data {conf0path}"
    else:
        pytest.skip()


@pytest.mark.parametrize(
    ("conffilename", "msg"),
    [
        ("mxfes_wrong.json", "the provided settings looks to be inconsistent"),
        ("full_wrong_1.json", "invalid key (= 'hoge) is detected in the box configration data"),
        ("full_wrong_2.json", "invalid key (= 'hoge) is detected in the box configration data"),
        ("ports_wrong_1.json", "unexpected name of port: 'hoge'"),
    ],
)
def test_config_box_json_wrong(conffilename, msg, fixtures8):
    box, params, dpath = fixtures8

    if box.boxtype == "quel1se-riken8":
        conf0path = Path(osp.dirname(__file__)) / "settings" / conffilename
        with pytest.raises(ValueError, match=re.escape(msg)):
            box.config_box_from_jsonfile(conf0path)
    else:
        pytest.skip()


def test_config_box_abnormal(fixtures8):
    box, params, dpath = fixtures8

    if box.boxtype == "quel1se-riken8":
        assert not box.config_validate_box({6: {"non-existent": 999}})
        assert not box.config_validate_box({0: {"fullscale_current": 12000}})
        with pytest.raises(ValueError, match="invalid port of quel1se-riken8: #05"):
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
        with pytest.raises(ValueError, match=re.escape("no variable attenuator is available for port-#01-1")):
            box.config_box({(1, 1): {"vatt": 0xC00}})
    else:
        pytest.skip()


def test_config_port(fixtures8):
    box, params, dpath = fixtures8

    box.config_port((1, 0), fullscale_current=10000)
    assert box.dump_port((1, 0))["fullscale_current"] == 10010  # due to the quantization
    assert box.dump_port(1)["fullscale_current"] == 10010  # due to the quantization

    box.config_port(1, fullscale_current=20000)
    assert box.dump_port((1, 0))["fullscale_current"] == 19995  # due to the quantization
    assert box.dump_port(1)["fullscale_current"] == 19995  # due to the quantization


def test_config_port_abnormal(fixtures8):
    box, params, dpath = fixtures8

    if box.boxtype == "quel1se-riken8":
        with pytest.raises(ValueError, match="no DAC is available for the input port-#00"):
            box.config_port(0, fullscale_current=10000)
        with pytest.raises(ValueError, match=re.escape("no variable attenuator is available for port-#03")):
            box.config_port(3, vatt=0xC00)
        with pytest.raises(ValueError, match=re.escape("no variable attenuator is available for port-#01-1")):
            box.config_port((1, 1), vatt=0xC00)
        with pytest.raises(ValueError, match=re.escape("invalid port of quel1se-riken8: #01-2")):
            box.config_port((1, 2), vatt=0xC00)
        with pytest.raises(ValueError, match=re.escape("malformed port: 'hoge'")):
            box.config_port("hoge", vatt=0xC00)  # type: ignore
    else:
        pytest.skip()


def test_config_rfswitches_abnormal(fixtures8):
    box, params, dpath = fixtures8

    if box.boxtype == "quel1se-riken8":
        with pytest.raises(ValueError, match="malformed port: 'non-existent'"):
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
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({1: "pass", (1, 1): "block"})
    else:
        pytest.skip()


def test_config_box_mismatch(fixtures8):
    box, params, dpath = fixtures8

    config = box.dump_box()["ports"]
    config[1]["cnco_freq"] = 314159  # Notes: Readout for QuEL-1 SE.
    if (1, 1) in config:
        config[(1, 1)]["fullscale_current"] = 29979
    config[4]["runits"][0]["fnco_freq"] = 271828
    assert not box.config_validate_box(config)


def test_config_box_inconsistent(fixtures8):
    box, params, dpath = fixtures8

    config = box.dump_box()["ports"]
    if box.boxtype in {"quel1se-riken8", "x-quel1se-riken8"}:
        assert config[1]["lo_freq"] == config[0]["lo_freq"]
        config[1]["lo_freq"] -= 1e6
    else:
        assert False, f"unexpected boxtype: {box.boxtype}"

    with pytest.raises(ValueError):
        box.config_box(config)


def test_invalid_port(fixtures8):
    box, params, dpath = fixtures8

    with pytest.raises(ValueError, match=f"invalid port of {box.boxtype}: #20"):
        box.config_box({20: {"lo_freq": 12e9}})


def test_config_rfswitch_basic(fixtures8):
    box, params, dpath = fixtures8

    rc = box.dump_rfswitches()
    box.config_rfswitches(rc)


def test_config_rfswitch_invalid(fixtures8):
    box, params, dpath = fixtures8

    if box.boxtype == "quel1se-riken8":
        with pytest.raises(ValueError, match="invalid configuration of an input switch: block"):
            box.config_rfswitches({0: "block"})
        with pytest.raises(ValueError, match="invalid configuration of an output switch: loop"):
            box.config_rfswitches({1: "loop"})
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({0: "loop", 1: "pass"})
    else:
        assert False, f"an unexpected fixture: {box}"
