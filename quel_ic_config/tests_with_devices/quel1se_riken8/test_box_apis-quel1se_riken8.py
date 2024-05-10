import logging
import re

import pytest

from quel_ic_config.quel1_box import Quel1Box, Quel1BoxType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.132",
            "ipaddr_sss": "10.2.0.132",
            "ipaddr_css": "10.5.0.132",
            "boxtype": Quel1BoxType.fromstr("quel1se-riken8"),
            "config_root": None,
            "config_options": [],
        },
        "linkup_config": {
            "mxfes_to_linkup": (0, 1),
            "use_204b": False,
        },
        "port_availability": {
            "unavailable": [],
            "via_monitor_out": [],
        },
        "linkup": False,
    },
)


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def box(request):
    param0 = request.param

    box = Quel1Box.create(**param0["box_config"])
    if request.param["linkup"]:
        linkstatus = box.relinkup(**param0["linkup_config"])
    else:
        linkstatus = box.reconnect()
    assert linkstatus[0]
    assert linkstatus[1]

    yield box

    box.easy_stop_all()
    box.activate_monitor_loop(0)
    box.activate_monitor_loop(1)
    box.css.terminate()


def test_just_run_apis(box):
    box.initialize_all_awgs()

    box.prepare_for_emission({(1, 0)})
    with pytest.raises(ValueError, match="invalid channel-#1 of subport-#0 of port-#01"):
        box.prepare_for_emission({(1, 1)})
    with pytest.raises(ValueError, match="port-#00 is not an output port"):
        box.prepare_for_emission({(0, 0)})
    with pytest.raises(ValueError, match="invalid port of quel1se-riken8: 5"):
        box.prepare_for_emission({(5, 0)})
    with pytest.raises(TypeError):
        box.prepare_for_emission(0)  # type: ignore
    with pytest.raises(ValueError):
        box.prepare_for_emission("wrong type")  # type: ignore

    # Notes: monitor-out port is not included.  TODO: reconsider this design to improve error readability of error.
    for p_sp in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10):
        p, sp = box.decode_port(p_sp)
        assert p == p_sp
        assert sp == 0

    p, sp = box.decode_port((1, 1))
    assert p == 1
    assert sp == 1

    box.easy_stop_all()
    with pytest.raises(ValueError, match="port-#00 is not an output port"):
        box.easy_stop(port=0)
    box.easy_stop(port=1)
    box.easy_stop(port=1, subport=1)
    with pytest.raises(ValueError, match="invalid subport-#2 of port-#01"):
        box.easy_stop(port=1, subport=2)
    box.easy_stop(port=1, subport=1, channel=0)
    with pytest.raises(ValueError, match=re.escape("invalid channel-#1 of subport-#1 of port-#01")):
        box.easy_stop(port=1, subport=1, channel=1)
    box.easy_stop(port=2)
    box.easy_stop(port=3)
    with pytest.raises(ValueError, match="port-#04 is not an output port"):
        box.easy_stop(port=4)
    with pytest.raises(ValueError, match="invalid port: 5"):
        box.easy_stop(port=5)
    box.easy_stop(port=6)

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
