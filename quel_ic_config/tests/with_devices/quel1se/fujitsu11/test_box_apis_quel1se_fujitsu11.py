import json
import logging
import os.path as osp
import re
import tempfile
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def test_just_run_apis_typea(fixtures11a):
    _test_just_run_apis(fixtures11a)


def test_just_run_apis_typeb(fixtures11b):
    _test_just_run_apis(fixtures11b)


def _test_just_run_apis(fixtures):
    box, param, topdirpath = fixtures

    box.initialize_all_awgunits()

    box.get_names_of_wavedata(1, 0)
    box.get_names_of_wavedata((1, 0), 0)
    with pytest.raises(ValueError, match=r"invalid channel-#1 of port-#01"):
        box.get_names_of_wavedata(1, 1)
    with pytest.raises(ValueError, match=r"invalid port of (x-)?quel1se-fujitsu11-[ab]: #00-1"):
        box.get_names_of_wavedata((0, 1), 1)

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        with pytest.raises(ValueError, match="port-#00 is not an output port"):
            box.get_names_of_wavedata(0, 0)
    elif box.boxtype in {"quel1se-fujitsu11-b", "x-quel1se-fujitsu11-b"}:
        with pytest.raises(ValueError, match=r"invalid port of (x-)?quel1se-fujitsu11-b: #00"):
            box.get_names_of_wavedata(0, 0)
    else:
        assert False

    with pytest.raises(ValueError, match=r"invalid port of (x-)?quel1se-fujitsu11-[ab]: #06"):
        box.get_names_of_wavedata(6, 0)
    with pytest.raises(TypeError):
        box.get_names_of_wavedata(0)  # type: ignore
    with pytest.raises(TypeError):
        box.get_names_of_wavedata("wrong type")  # type: ignore

    # Notes: monitor-out port is not included.  TODO: reconsider this design to improve error readability of error.
    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        for p_sp in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12):
            p, sp = box._decode_port(p_sp)
            assert p == p_sp
            assert sp == 0
    elif box.boxtype in {"quel1se-fujitsu11-b", "x-quel1se-fujitsu11-b"}:
        for p_sp in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12):
            p, sp = box._decode_port(p_sp)
            assert p == p_sp
            assert sp == 0
    else:
        assert False

    with pytest.raises(ValueError, match=r"invalid port of (x-)?quel1se-fujitsu11-[ab]: #01-1"):
        p, sp = box._decode_port((1, 1))

    for g in (0, 1):
        box.deactivate_monitor_loop(g)
        assert not box.is_loopedback_monitor(g)
    for p, conf in box.dump_rfswitches().items():
        if p in {5, 12}:
            assert conf == "open"

    for g in (0, 1):
        box.activate_monitor_loop(g)
        assert box.is_loopedback_monitor(g)
    for p, conf in box.dump_rfswitches().items():
        if p in {5, 12}:
            assert conf == "loop"

    box.pass_all_output_ports()
    for p, conf in box.dump_rfswitches().items():
        if p in {2, 3, 4, 9, 10, 11}:  # Notes: #01 and #08 is excluded since it is a copy of #00 and #07, respectively.
            assert conf == "pass"
        elif p in {0, 7}:
            assert conf == "open"
    for p in (2, 3, 4, 9, 10, 11):
        assert box.dump_rfswitch(p) == "pass"

    box.block_all_output_ports()
    for p, conf in box.dump_rfswitches().items():
        if p in {2, 3, 4, 9, 10, 11}:  # Notes: 1 is excluded since it is a copy of port-#0
            assert conf == "block"
        elif p in {0, 7}:
            assert conf == "loop"
    for p in (2, 3, 4, 9, 10, 11):
        assert box.dump_rfswitch(p) == "block"


def test_config_fsc_typea(fixtures11a):
    _test_config_fsc(fixtures11a)


def test_config_fsc_typeb(fixtures11b):
    _test_config_fsc(fixtures11b)


def _test_config_fsc(fixtures):
    box, param, topdirpath = fixtures

    for i in range(12000, 12100):
        box.config_box({8: {"fullscale_current": i}})
    box.config_box({8: {"fullscale_current": 40520}})
    box.config_box({8: {"fullscale_current": 40527}})


@pytest.mark.parametrize(
    ("conffilename", "ports_only"),
    [
        ("outports.json", True),
        # ("full.json", False),
        # ("mxfes.json", False)
    ],
)
def test_config_box_json_typea(conffilename, ports_only, fixtures11a):
    _test_config_box_json(conffilename, ports_only, fixtures11a)


def _test_config_box_json(conffilename, ports_only, fixtures):
    box, param, topdirpath = fixtures

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
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
    elif box.boxtype == "quel1se-fujitsu11-b":
        pytest.skip("not implemented yet")
    else:
        assert False, f"an unexpected fixture: {box}"


"""
@pytest.mark.parametrize(
    ("conffilename", "msg"),
    [
        ("mxfes_wrong.json", "the provided settings looks to be inconsistent"),
        ("full_wrong_1.json", "invalid key (= 'hoge) is detected in the box configration data"),
        ("full_wrong_2.json", "invalid key (= 'hoge) is detected in the box configration data"),
        ("ports_wrong_1.json", "unexpected name of port: 'hoge'"),
    ],
)
def test_config_box_json_wrong(conffilename, msg, box):
    if box.boxtype == "quel1se-fujitsu11-a":
        conf0path = Path(osp.dirname(__file__)) / "settings" / conffilename
        with pytest.raises(ValueError, match=re.escape(msg)):
            box.config_box_from_jsonfile(conf0path)
    else:
        pytest.skip()
"""


def test_config_box_abnormal_typea(fixtures11a):
    _test_config_box_abnormal(fixtures11a)


def test_config_box_abnormal_typeb(fixtures11b):
    _test_config_box_abnormal(fixtures11b)


def _test_config_box_abnormal(fixtures):
    box, param, topdirpath = fixtures

    assert not box.config_validate_box({8: {"non-existent": 999}})

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        assert not box.config_validate_box({0: {"fullscale_current": 12000}})
    elif box.boxtype in {"quel1se-fujitsu11-b", "x-quel1se-fujitsu11-b"}:
        with pytest.raises(ValueError, match=r"invalid port of (x-)?quel1se-fujitsu11-b: #00"):
            _ = box.config_validate_box({0: {"fullscale_current": 12000}})
    else:
        assert False

    with pytest.raises(ValueError, match=r"invalid port of (x-)?quel1se-fujitsu11-[ab]: #06"):
        box.config_validate_box({6: {"fullscale_current": 12000}})

    with pytest.raises(ValueError, match=re.escape("port-#08 is not an input port, not applicable")):
        box.config_box(
            {
                8: {
                    "runits": {
                        1: {"fnco_freq": 0.0},
                        2: {"fnco_freq": 0.0},
                        3: {"fnco_freq": 0.0},
                        0: {"fnco_freq": 0.0},
                    }
                }
            }
        )

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        with pytest.raises(ValueError, match=r"port-#00 is not an output port, not applicable"):
            box.config_box({0: {"channels": {0: {"fnco_freq": 0.0}}}})

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        assert not box.config_validate_box(
            {
                7: {
                    "runits": {
                        1: {"fnco_freq": 1.0},
                        2: {"fnco_freq": 2.0},
                        3: {"fnco_freq": -1.0},
                        0: {"fnco_freq": -2.0},
                    }
                }
            }
        )
        assert not box.config_validate_box({0: {"channels": {0: {"fnco_freq": 0.0}}}})
    elif box.boxtype in {"quel1se-fujitsu11-b", "x-quel1se-fujitsu11-b"}:
        with pytest.raises(ValueError, match=r"invalid port of (x-)?quel1se-fujitsu11-b: #07"):
            _ = box.config_validate_box({7: {"channels": {0: {"fnco_freq": 0.0}}}})
    else:
        assert False

    with pytest.raises(ValueError, match=r"invalid runit:2 for port-#05"):
        box.config_box({5: {"runits": {2: {"fnco_freq": 0.0}}}})
    with pytest.raises(ValueError, match=r"invalid runit:2 for port-#05"):
        box.config_validate_box({5: {"runits": {2: {"fnco_freq": 0.0}}}})
    assert not box.config_validate_box({5: {"runits": {0: {"non-existent": 0.0}}}})

    with pytest.raises(ValueError, match=r"an invalid combination of port-#08, and channel:1"):
        box.config_box({8: {"channels": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}}}})
    assert not box.config_validate_box({9: {"channels": {0: {"non-existent": 0.0}}}})
    with pytest.raises(ValueError, match=r"an invalid combination of port-#10, and channel:1"):
        box.config_validate_box(
            {10: {"channels": {0: {"fnco_freq": 0.0}, 1: {"fnco_freq": 0.0}, 2: {"fnco_freq": 0.0}}}}
        )

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        with pytest.raises(ValueError, match=r"port-#05 is not an output port"):
            box.config_box({0: {"cnco_locked_with": 5}})

    with pytest.raises(ValueError, match=r"no cnco_locked_with is available for the output port-#08"):
        box.config_box({8: {"cnco_locked_with": 9}})

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        with pytest.raises(ValueError, match=r"no configurable mixer is available for the input port-#07"):
            box.config_box({7: {"vatt": 0xC00}})


def test_config_port_typea(fixtures11a):
    _test_config_port(fixtures11a)


def test_config_port_typeb(fixtures11b):
    _test_config_port(fixtures11b)


def _test_config_port(fixtures):
    box, param, topdirpath = fixtures

    box.config_port((1, 0), fullscale_current=10000)
    assert box.dump_port((1, 0))["fullscale_current"] == 10010  # due to the quantization
    assert box.dump_port(1)["fullscale_current"] == 10010  # due to the quantization

    box.config_port(1, fullscale_current=20000)
    assert box.dump_port((1, 0))["fullscale_current"] == 19995  # due to the quantization
    assert box.dump_port(1)["fullscale_current"] == 19995  # due to the quantization


def test_config_port_abnormal_typea(fixtures11a):
    _test_config_port_abnormal(fixtures11a)


def test_config_port_abnormal_typeb(fixtures11b):
    _test_config_port_abnormal(fixtures11b)


def _test_config_port_abnormal(fixtures):
    box, param, topdirpath = fixtures

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        with pytest.raises(ValueError, match="no DAC is available for the input port-#00"):
            box.config_port(0, fullscale_current=10000)

    with pytest.raises(ValueError, match=r"invalid port of (x-)?quel1se-fujitsu11-[ab]: #01-1"):
        box.config_port((1, 1), vatt=0xC00)
    with pytest.raises(ValueError, match=re.escape("malformed port: 'hoge'")):
        box.config_port("hoge", vatt=0xC00)  # type: ignore


def test_config_rfswitches_abnormal_typea(fixtures11a):
    _test_config_rfswitches_abnormal(fixtures11a)


def test_config_rfswitches_abnormal_typeb(fixtures11b):
    _test_config_rfswitches_abnormal(fixtures11b)


def _test_config_rfswitches_abnormal(fixtures):
    box, param, topdirpath = fixtures

    with pytest.raises(ValueError, match="malformed port: 'non-existent'"):
        box.config_rfswitches({"non-existent": "invalid"})  # type: ignore

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        with pytest.raises(ValueError, match=re.escape("invalid configuration of an input switch: invalid")):
            box.config_rfswitches({0: "invalid"})
        with pytest.raises(ValueError, match=re.escape("invalid configuration of an input switch: pass")):
            box.config_rfswitches({0: "pass"})
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({0: "loop", 1: "pass"})

    with pytest.raises(ValueError, match=re.escape("invalid configuration of an output switch: loop")):
        box.config_rfswitches({1: "loop"})
    with pytest.raises(ValueError, match=re.escape("invalid configuration of an output switch: loop")):
        box.config_rfswitches({2: "loop"})


def test_config_box_mismatch_typea(fixtures11a):
    _test_config_box_mismatch(fixtures11a)


def test_config_box_mismatch_typeb(fixtures11b):
    _test_config_box_mismatch(fixtures11b)


def _test_config_box_mismatch(fixtures):
    box, param, topdirpath = fixtures

    config = box.dump_box()["ports"]
    config[1]["cnco_freq"] = 314159  # Notes: Readout for QuEL-1 SE.
    config[5]["runits"][0]["fnco_freq"] = 271828
    assert not box.config_validate_box(config)


def test_config_box_inconsistent_typea(fixtures11a):
    _test_config_box_inconsistent(fixtures11a)


def test_config_box_inconsistent_typeb(fixtures11b):
    _test_config_box_inconsistent(fixtures11b)


def _test_config_box_inconsistent(fixtures):
    box, param, topdirpath = fixtures

    config = box.dump_box()["ports"]
    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        assert config[1]["lo_freq"] == config[0]["lo_freq"]
        config[1]["lo_freq"] -= 1e6
    elif box.boxtype in {"quel1se-fujitsu11-b", "x-quel1se-fujitsu11-b"}:
        assert config[3]["lo_freq"] == config[5]["lo_freq"]
        config[3]["lo_freq"] -= 1e6
    else:
        assert False

    with pytest.raises(ValueError):
        box.config_box(config)


def test_invalid_port_typea(fixtures11a):
    _test_invalid_port(fixtures11a)


def test_invalid_port_typeb(fixtures11b):
    _test_invalid_port(fixtures11b)


def _test_invalid_port(fixtures):
    box, param, topdirpath = fixtures

    with pytest.raises(ValueError, match=f"invalid port of {box.boxtype}: #20"):
        box.config_box({20: {"lo_freq": 12e9}})


def test_config_rfswitch_basic_typea(fixtures11a):
    _test_config_rfswitch_basic(fixtures11a)


def test_config_rfswitch_basic_typeb(fixtures11b):
    _test_config_rfswitch_basic(fixtures11b)


def _test_config_rfswitch_basic(fixtures):
    box, param, topdirpath = fixtures

    rc = box.dump_rfswitches()
    box.config_rfswitches(rc)


def test_config_rfswitch_invalid_typea(fixtures11a):
    _test_config_rfswitch_invalid(fixtures11a)


def test_config_rfswitch_invalid_typeb(fixtures11b):
    _test_config_rfswitch_invalid(fixtures11b)


def _test_config_rfswitch_invalid(fixtures):
    box, param, topdirpath = fixtures

    if box.boxtype in {"quel1se-fujitsu11-a", "x-quel1se-fujitsu11-a"}:
        with pytest.raises(ValueError, match="invalid configuration of an input switch: block"):
            box.config_rfswitches({0: "block"})
        with pytest.raises(ValueError, match="the specified configuration of rf switches is not realizable"):
            box.config_rfswitches({0: "loop", 1: "pass"})

    with pytest.raises(ValueError, match="invalid configuration of an output switch: loop"):
        box.config_rfswitches({1: "loop"})
