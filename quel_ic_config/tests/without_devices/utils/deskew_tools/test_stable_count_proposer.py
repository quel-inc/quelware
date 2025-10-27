from quel_ic_config_utils import deskew_tools
from quel_ic_config_utils.deskew_tools._stable_count_proposer import _round_up_to_grid


def test_round_to_grid():
    assert _round_up_to_grid(10.0, 5) == 10
    assert _round_up_to_grid(12.0, 5) == 15
    assert _round_up_to_grid(13.0, 5) == 15
    assert _round_up_to_grid(99.0, 10) == 100


def test_from_deskew_configuration():
    deskew_conf = deskew_tools.DeskewConfiguration(
        boxes=[
            deskew_tools.Box(name="box1", ports=[], timecounter_offset=-1),
            deskew_tools.Box(name="box2", ports=[], timecounter_offset=2),
        ]
    )
    proposer = deskew_tools.StableCountProposer.from_deskew_configuration(deskew_conf)

    assert proposer.get_offset("box1") == -1
    assert proposer.get_offset("box2") == 2


def test_propose_trigger_counts():
    proposer = deskew_tools.StableCountProposer()
    proposer.set_offset("box1", 1)
    proposer.set_offset("box2", -2)

    current_count = 100000
    delay_sec = 0.20
    target_names = ["box1", "box2", "box3"]

    counts = proposer.propose_trigger_counts(current_count, target_names, delay_sec)

    assert counts["box1"] % proposer._trigger_grid_step == 1
    assert counts["box2"] % proposer._trigger_grid_step == proposer._trigger_grid_step - 2
    assert counts["box3"] % proposer._trigger_grid_step == 0

    refcount_expected = counts["box1"] - 1
    assert counts["box2"] - (-2) == refcount_expected
    assert counts["box3"] - 0 == refcount_expected
