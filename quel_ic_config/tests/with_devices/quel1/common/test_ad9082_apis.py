import logging

import pytest

from quel_ic_config.quel1_box import Quel1BoxType, Quel1ConfigOption
from quel_ic_config.quel1_config_subsystem import QubeConfigSubsystem
from quel_ic_config_utils.init_helper_for_prebox import init_box_with_linkup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


TEST_SETTINGS = (
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.74",
            "ipaddr_sss": "10.2.0.74",
            "ipaddr_css": "10.5.0.74",
            "boxtype": Quel1BoxType.fromstr("quel1-a"),
            "mxfes_to_linkup": {0, 1},
            "config_root": None,
            "config_options": [
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE0,
                Quel1ConfigOption.DAC_CNCO_1500MHz_MXFE0,
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE1,
                Quel1ConfigOption.DAC_CNCO_2000MHz_MXFE1,
            ],
        },
    },
    {
        "box_config": {
            "ipaddr_wss": "10.1.0.60",
            "ipaddr_sss": "10.2.0.60",
            "ipaddr_css": "10.5.0.60",
            "boxtype": Quel1BoxType.fromstr("quel1-b"),
            "mxfes_to_linkup": {0, 1},
            "config_root": None,
            "config_options": [
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE0,
                Quel1ConfigOption.DAC_CNCO_1500MHz_MXFE0,
                Quel1ConfigOption.REFCLK_12GHz_FOR_MXFE1,
                Quel1ConfigOption.DAC_CNCO_2000MHz_MXFE1,
            ],
        },
    },
)


@pytest.fixture(scope="session", params=TEST_SETTINGS)
def fixtures(request) -> QubeConfigSubsystem:
    param0 = request.param

    # TODO: write something to modify boxtype.

    linkstat, css, _, _, _ = init_box_with_linkup(**param0["box_config"], ignore_crc_error_of_mxfe={0, 1})
    if isinstance(css, QubeConfigSubsystem):
        for mxfe in css.get_all_groups():
            if not linkstat[mxfe]:
                raise RuntimeError(f"test is not ready for group-{mxfe}")
        return css
    else:
        raise AssertionError


def test_all_temperatures(fixtures):
    css = fixtures
    for group in range(2):
        temp_max, temp_min = css.get_ad9082_temperatures(group)

        assert 10 < temp_min < 120
        assert 10 < temp_max < 120
        assert temp_min <= temp_max


def _is_near_enough(x, y):
    return abs(x - y) < 2.15e-5 * 2


@pytest.mark.parametrize(
    ("mxfe", "fractional_mode"),
    [
        (0, True),
        (0, False),
        (1, True),
        (1, False),
    ],
)
def test_nco_set_get(mxfe, fractional_mode, fixtures):
    css = fixtures
    ad9082 = css.ad9082[mxfe]

    dac_cnco_ftws = [ad9082.calc_dac_cnco_ftw(int(100e6 * i), fractional_mode=fractional_mode) for i in range(4)]
    dac_fnco_ftws = [ad9082.calc_dac_fnco_ftw(int(50e6 * i - 200e6), fractional_mode=fractional_mode) for i in range(8)]
    adc_cnco_ftws = [
        ad9082.calc_adc_cnco_ftw(
            int(
                100e6 * i,
            ),
            fractional_mode=fractional_mode,
        )
        for i in range(4)
    ]
    adc_fnco_ftws = [ad9082.calc_adc_fnco_ftw(int(50e6 * i - 200e6), fractional_mode=fractional_mode) for i in range(8)]

    logger.info(f"dac_cnco_ftws {fractional_mode} {[ftw.to_frequency(12e9) for ftw in dac_cnco_ftws]}")
    logger.info(f"dac_fnco_ftws {fractional_mode} {[ftw.to_frequency(2e9) for ftw in dac_fnco_ftws]}")
    logger.info(f"adc_cnco_ftws {fractional_mode} {[ftw.to_frequency(6e9) for ftw in adc_cnco_ftws]}")
    logger.info(f"adc_fnco_ftws {fractional_mode} {[ftw.to_frequency(1e9) for ftw in adc_fnco_ftws]}")

    for i in range(4):
        ad9082.set_dac_cnco([i], dac_cnco_ftws[i])
        ad9082.set_adc_cnco([i], adc_cnco_ftws[i])

    for i in range(8):
        ad9082.set_dac_fnco([i], dac_fnco_ftws[i])
        ad9082.set_adc_fnco([i], adc_fnco_ftws[i])

    for i in range(4):
        ftw = ad9082.get_dac_cnco(i)
        assert ftw == dac_cnco_ftws[i]

    for i in range(4):
        ftw = ad9082.get_adc_cnco(i)
        assert ftw == adc_cnco_ftws[i]

    for i in range(8):
        ftw = ad9082.get_dac_fnco(i)
        assert ftw == dac_fnco_ftws[i]

    for i in range(8):
        ftw = ad9082.get_adc_fnco(i)
        assert ftw == adc_fnco_ftws[i]


@pytest.mark.parametrize(
    ("mxfe", "fsc"),
    [
        (0, -1),
        (0, 0),
        (0, 3000),
        (0, 7000),
        (0, 20000),
        (0, 40000),
        (0, 40001),
        (0, 40520),
        (0, 40527),
        (0, 90000),
        (1, -1),
        (1, 0),
        (1, 3000),
        (1, 7000),
        (1, 20000),
        (1, 40000),
        (1, 40001),
        (1, 40520),
        (1, 40527),
        (1, 90000),
    ],
)
def test_fsc_get(mxfe, fsc, fixtures):
    css = fixtures
    ad9082 = css.ad9082[mxfe]

    if not (7000 <= fsc <= 40000) and fsc not in {40520, 40527}:
        for i in range(4):
            with pytest.raises(ValueError):
                ad9082.set_fullscale_current(1 << i, fsc)
    else:
        for i in range(4):
            ad9082.set_fullscale_current(1 << i, fsc)

        for i in range(4):
            fsc_r = ad9082.get_fullscale_current(i)
            logger.info(f"set_fsc = {fsc}uA, get_fsc = {fsc_r}uA")
            # Notes: LSB of fsc is corresponding to (1024 / 25) = 40.96 uA.
            #        Usually the error should be less than the half of it.
            assert abs(fsc_r - fsc) < 41


@pytest.mark.parametrize(
    ("mxfe",),
    [
        (0,),
        (1,),
    ],
)
def test_get_interpolation_rate(mxfe, fixtures):
    css = fixtures
    ad9082 = css.ad9082[mxfe]

    if mxfe == 0:
        assert ad9082.get_main_interpolation_rate() == 8
        assert ad9082.get_channel_interpolation_rate() == 3
    elif mxfe == 1:
        assert ad9082.get_main_interpolation_rate() == 6
        assert ad9082.get_channel_interpolation_rate() == 4
    else:
        raise AssertionError
