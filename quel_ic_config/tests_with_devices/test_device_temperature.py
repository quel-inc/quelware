import logging
from pathlib import Path

import pytest

from quel_ic_config import Quel1ConfigSubsystem
from testlibs.basic_scan_common import Quel1BoxType, Quel1ConfigOption, init_box

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.42",
            "ipaddr_sss": "10.2.0.42",
            "ipaddr_css": "10.5.0.42",
            "boxtype": Quel1BoxType.fromstr("quel1-a"),
            "config_root": Path("settings"),
            "config_options": [
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE0,
                Quel1ConfigOption.DAC_CNCO_1500MHz_IN_MXFE0,
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE1,
                Quel1ConfigOption.DAC_CNCO_2000MHz_IN_MXFE1,
            ],
            "mxfe_combination": "both",
        },
    },
)


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def fixtures(request) -> Quel1ConfigSubsystem:
    param0 = request.param

    linkup0, linkup1, css, _, _, _, _, _, _, _ = init_box(**param0["box_config"])
    assert linkup0
    assert linkup1
    if isinstance(css, Quel1ConfigSubsystem):
        return css
    else:
        raise AssertionError


def test_all_temperatures(
    fixtures,
):
    group0 = 0
    temp_max_0, temp_min_0 = fixtures.get_ad9082_temperatures(group0)

    assert temp_min_0 > 10
    assert temp_min_0 < 120
    assert temp_max_0 > 10
    assert temp_max_0 < 120
    assert temp_min_0 <= temp_max_0

    group1 = 1
    temp_max_1, temp_min_1 = fixtures.get_ad9082_temperatures(group1)
    assert temp_min_1 > 10
    assert temp_min_1 < 120
    assert temp_max_1 > 10
    assert temp_max_1 < 120
    assert temp_min_1 <= temp_max_1
