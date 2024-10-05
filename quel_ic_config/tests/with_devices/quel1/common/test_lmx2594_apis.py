import logging

import pytest

from quel_ic_config.quel1_config_subsystem import QubeConfigSubsystem
from quel_ic_config.quel_config_common import Quel1BoxType
from quel_ic_config_utils.init_helper_for_prebox import init_box_with_reconnect

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


# Notes: leave CNCO_1500MHz setting and test cases as comment for someday.
TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.74",
            "ipaddr_sss": "10.2.0.74",
            "ipaddr_css": "10.5.0.74",
            "boxtype": Quel1BoxType.fromstr("quel1-a"),
            # "mxfes_to_linkup": {0, 1},
            # "config_root": None,
            # "config_options": [
            #     Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE0,
            #     Quel1ConfigOption.DAC_CNCO_1500MHz_MXFE0,
            #     Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE1,
            #     Quel1ConfigOption.DAC_CNCO_2000MHz_MXFE1,
            # ],
        },
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.60",
            "ipaddr_sss": "10.2.0.60",
            "ipaddr_css": "10.5.0.60",
            "boxtype": Quel1BoxType.fromstr("quel1-b"),
            # "mxfes_to_linkup": {0, 1},
            # "config_root": None,
            # "config_options": [
            #     Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE0,
            #     Quel1ConfigOption.DAC_CNCO_1500MHz_MXFE0,
            #     Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE1,
            #     Quel1ConfigOption.DAC_CNCO_2000MHz_MXFE1,
            # ],
        },
    },
)


@pytest.fixture(scope="module", params=TEST_SETTINGS)
def fixtures(request) -> QubeConfigSubsystem:
    param0 = request.param

    # TODO: write something to modify boxtype.

    # linkstat, css, _, _, _ = init_box_with_linkup(**param0["box_config"], ignore_crc_error_of_mxfe={0, 1})
    linkstat, css, _, _, _ = init_box_with_reconnect(**param0["box_config"], ignore_crc_error_of_mxfe={0, 1})
    if isinstance(css, QubeConfigSubsystem):
        for mxfe in css.get_all_groups():
            if not linkstat[mxfe]:
                raise RuntimeError(
                    f"test is not ready for {param0['box_config']['ipaddr_wss']}:group-{mxfe} "
                    f"due to the invalid link status (= {linkstat[mxfe]})"
                )
        return css
    else:
        raise AssertionError


def test_lo_mult(fixtures):
    css: QubeConfigSubsystem = fixtures
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


def test_divider(fixtures):
    css: QubeConfigSubsystem = fixtures

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
