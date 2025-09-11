import logging

import pytest

from quel_ic_config.quel1_config_subsystem import QubeConfigSubsystem
from quel_ic_config.quel_config_common import Quel1BoxType
from tests.with_devices.conftest import BoxProvider

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


@pytest.mark.parametrize("boxtype", ["quel1-a", "quel1-b"])
def test_lo_mult(boxtype, box_provider: BoxProvider):
    box = box_provider.get_box_from_type(boxtype)
    css = box.css
    if not isinstance(css, QubeConfigSubsystem):
        assert False

    mult = {
        0: 85,
        1: 87,
        2: 90,
        3: 93,
    }

    for group in css.get_all_groups():
        for line in css.get_all_lines_of_group(group):
            convergence = css.set_lo_multiplier(group, line, mult[line])
            assert convergence
            assert css.get_lo_multiplier(group, line) == mult[line]


@pytest.mark.parametrize("boxtype", ["quel1-a", "quel1-b"])
def test_divider(boxtype, box_provider: BoxProvider):
    box = box_provider.get_box_from_type(boxtype)
    css = box.css
    if not isinstance(css, QubeConfigSubsystem):
        assert False

    if css._boxtype in {Quel1BoxType.QuEL1_TypeA}:
        assert css.lmx2594[0].get_divider_ratio() == (1, 1)
        assert css.lmx2594[1].get_divider_ratio() == (1, 1)
        assert css.lmx2594[2].get_divider_ratio() == (1, 0)
        assert css.lmx2594[3].get_divider_ratio() == (1, 0)
        assert css.lmx2594[4].get_divider_ratio() == (1, 0)
        assert css.lmx2594[5].get_divider_ratio() == (1, 0)
        assert css.lmx2594[6].get_divider_ratio() == (1, 1)
        assert css.lmx2594[7].get_divider_ratio() == (1, 1)
    elif css._boxtype in {Quel1BoxType.QuEL1_TypeB}:
        assert css.lmx2594[0].get_divider_ratio() == (1, 0)
        assert css.lmx2594[1].get_divider_ratio() == (1, 1)
        assert css.lmx2594[2].get_divider_ratio() == (1, 0)
        assert css.lmx2594[3].get_divider_ratio() == (1, 0)
        assert css.lmx2594[4].get_divider_ratio() == (1, 0)
        assert css.lmx2594[5].get_divider_ratio() == (1, 0)
        assert css.lmx2594[6].get_divider_ratio() == (1, 1)
        assert css.lmx2594[7].get_divider_ratio() == (1, 0)
    else:
        assert False

    for group in css.get_all_groups():
        for line in css.get_all_lines_of_group(group):
            assert css.get_divider_ratio(group, line) == 1

        for line in css.get_all_lines_of_group(group):
            css.set_divider_ratio(group, line, 2 ** (line + 1))

        for line in css.get_all_lines_of_group(group):
            assert css.get_divider_ratio(group, line) == 2 ** (line + 1)

    if css._boxtype in {Quel1BoxType.QuEL1_TypeA}:
        assert css.lmx2594[0].get_divider_ratio() == (2, 2)
        assert css.lmx2594[1].get_divider_ratio() == (4, 4)
        assert css.lmx2594[2].get_divider_ratio() == (8, 0)
        assert css.lmx2594[3].get_divider_ratio() == (16, 0)
        assert css.lmx2594[4].get_divider_ratio() == (16, 0)
        assert css.lmx2594[5].get_divider_ratio() == (8, 0)
        assert css.lmx2594[6].get_divider_ratio() == (4, 4)
        assert css.lmx2594[7].get_divider_ratio() == (2, 2)
    elif css._boxtype in {Quel1BoxType.QuEL1_TypeB}:
        assert css.lmx2594[0].get_divider_ratio() == (2, 0)
        assert css.lmx2594[1].get_divider_ratio() == (4, 4)
        assert css.lmx2594[2].get_divider_ratio() == (8, 0)
        assert css.lmx2594[3].get_divider_ratio() == (16, 0)
        assert css.lmx2594[4].get_divider_ratio() == (16, 0)
        assert css.lmx2594[5].get_divider_ratio() == (8, 0)
        assert css.lmx2594[6].get_divider_ratio() == (4, 4)
        assert css.lmx2594[7].get_divider_ratio() == (2, 0)
    else:
        assert False

    for group in css.get_all_groups():
        for line in css.get_all_lines_of_group(group):
            css.set_divider_ratio(group, line, 1)

    if css._boxtype in {Quel1BoxType.QuEL1_TypeA}:
        assert css.lmx2594[0].get_divider_ratio() == (1, 1)
        assert css.lmx2594[1].get_divider_ratio() == (1, 1)
        assert css.lmx2594[2].get_divider_ratio() == (1, 0)
        assert css.lmx2594[3].get_divider_ratio() == (1, 0)
        assert css.lmx2594[4].get_divider_ratio() == (1, 0)
        assert css.lmx2594[5].get_divider_ratio() == (1, 0)
        assert css.lmx2594[6].get_divider_ratio() == (1, 1)
        assert css.lmx2594[7].get_divider_ratio() == (1, 1)
    elif css._boxtype in {Quel1BoxType.QuEL1_TypeB}:
        assert css.lmx2594[0].get_divider_ratio() == (1, 0)
        assert css.lmx2594[1].get_divider_ratio() == (1, 1)
        assert css.lmx2594[2].get_divider_ratio() == (1, 0)
        assert css.lmx2594[3].get_divider_ratio() == (1, 0)
        assert css.lmx2594[4].get_divider_ratio() == (1, 0)
        assert css.lmx2594[5].get_divider_ratio() == (1, 0)
        assert css.lmx2594[6].get_divider_ratio() == (1, 1)
        assert css.lmx2594[7].get_divider_ratio() == (1, 0)
    else:
        assert False
