import logging

import pytest

from quel_ic_config import Quel1BoxType, Quel1Feature, Quel1SeProto8ConfigSubsystem

logger = logging.getLogger(__name__)


def test_parameter_validation():
    css = Quel1SeProto8ConfigSubsystem(
        css_addr="10.254.253.252",
        boxtype=Quel1BoxType.QuEL1SE_Proto8,
        features={Quel1Feature.SINGLE_ADC},
    )

    bad_group = (-1, 2, 1.5, "r", None)
    bad_line = {
        0: (-1, 4, 1.5, "r", None),
        1: (-1, 4, 1.5, "r", None),
    }
    line_with_no_lo = {
        0: (1, 3),
        1: (0, 1, 2, 3, "r"),
    }
    bad_channel = {
        (0, 0): (-1, 1, 1.5, "r", None),
        (0, 1): (-1, 1, 1.5, "r", None),
        (0, 2): (-1, 3, 1.5, "r", None),
        (0, 3): (-1, 3, 1.5, "r", None),
        (1, 0): (-1, 1, 1.5, "r", None),
        (1, 1): (-1, 1, 1.5, "r", None),
        (1, 2): (-1, 3, 1.5, "r", None),
        (1, 3): (-1, 3, 1.5, "r", None),
    }
    bad_rchannel = {
        (0, "r"): (-1, 1, 1.5, "x", None),
        (0, "m"): (-1, 1, 1.5, "x", None),
        (1, "r"): (-1, 1, 1.5, "x", None),
        (1, "m"): (-1, 1, 1.5, "x", None),
    }
    bad_rline = {
        0: ("x", -1, 0, 1, None),
        1: ("x", "r", -1, 0, 1, None),
    }
    bad_line_rline = {
        0: ("x", -1, 4, None),
        1: ("x", "r", -1, 4, None),
    }

    for g in bad_group:
        logger.info(f"g={g}")
        with pytest.raises(ValueError):
            css.configure_mxfe(g)  # type: ignore

        with pytest.raises(ValueError):
            css.get_link_status(g)  # type: ignore

        with pytest.raises(ValueError):
            css.get_ad9082_temperatures(g)  # type: ignore

    for g in (0, 1):
        for tl in bad_line[g]:
            logger.info(f"g={g}, tl={tl}")
            with pytest.raises(ValueError):
                css.dump_line(g, tl)  # type: ignore

            with pytest.raises(ValueError):
                css.set_dac_cnco(g, tl)  # type: ignore

            with pytest.raises(ValueError):
                css.get_dac_cnco(g, tl)  # type: ignore

            with pytest.raises(ValueError):
                css.set_sideband(g, tl, "L")  # type: ignore

            with pytest.raises(ValueError):
                css.get_sideband(g, tl)  # type: ignore

            with pytest.raises(ValueError):
                css.set_vatt(g, tl, 0)  # type: ignore

    for g in (0, 1):
        for trl in bad_line_rline[g]:
            logger.info(f"g={g}, trl={trl}")
            with pytest.raises(ValueError):
                css.set_lo_multiplier(g, trl, 100)  # type: ignore

            with pytest.raises(ValueError):
                css.get_lo_multiplier(g, trl)  # type: ignore

            with pytest.raises(ValueError):
                css.set_sideband(g, trl, "L")  # type: ignore

            with pytest.raises(ValueError):
                css.get_sideband(g, trl)  # type: ignore

            with pytest.raises(ValueError):
                css.set_vatt(g, trl, 0xA00)  # type: ignore

    for g in (0, 1):
        for trl in line_with_no_lo[g]:
            logger.info(f"g={g}, trl={trl}")
            with pytest.raises(ValueError):
                css.set_lo_multiplier(g, trl, 120)  # type: ignore

            with pytest.raises(ValueError):
                css.get_lo_multiplier(g, trl)  # type: ignore

            with pytest.raises(ValueError):
                css.set_sideband(g, trl, "L")  # type: ignore

            with pytest.raises(ValueError):
                css.get_sideband(g, trl)  # type: ignore

            with pytest.raises(ValueError):
                css.set_vatt(g, trl, 0xA00)  # type: ignore

    for g in (0, 1):
        for rl in bad_rline[g]:
            logger.info(f"g={g}, rl={rl}")
            with pytest.raises(ValueError):
                css.dump_rline(g, rl)  # type: ignore

            with pytest.raises(ValueError):
                css.set_adc_cnco(g, rl)  # type: ignore

            with pytest.raises(ValueError):
                css.get_adc_cnco(g, rl)  # type: ignore

    for g in (0, 1):
        for tl in (0, 1, 2, 3):
            for c in bad_channel[g, tl]:
                logger.info(f"g={g}, tl={tl}, c={c}")
                with pytest.raises(ValueError):
                    css.dump_channel(g, tl, c)  # type: ignore

                with pytest.raises(ValueError):
                    css.set_dac_fnco(g, tl, c)  # type: ignore

                with pytest.raises(ValueError):
                    css.get_dac_fnco(g, tl, c)  # type: ignore

    for g in (0, 1):
        for rl in ("r", "m"):
            for rc in bad_rchannel[g, rl]:
                logger.info(f"g={g}, tl={rl}, rc={rc}")
                with pytest.raises(ValueError):
                    css.dump_rchannel(g, rl, rc)  # type: ignore

                with pytest.raises(ValueError):
                    css.set_adc_fnco(g, rl, rc)  # type: ignore

                with pytest.raises(ValueError):
                    css.get_adc_fnco(g, rl, rc)  # type: ignore
