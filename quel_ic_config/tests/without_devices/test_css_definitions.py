import logging

import pytest

from quel_ic_config.quel1_config_subsystem import (
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1NecConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
)
from quel_ic_config.quel1se_adda_config_subsystem import Quel1seAddaConfigSubsystem
from quel_ic_config.quel1se_fujitsu11_config_subsystem import (
    Quel1seFujitsu11TypeAConfigSubsystem,
    Quel1seFujitsu11TypeADebugConfigSubsystem,
    Quel1seFujitsu11TypeBConfigSubsystem,
    Quel1seFujitsu11TypeBDebugConfigSubsystem,
)
from quel_ic_config.quel1se_riken8_config_subsystem import (
    Quel1seRiken8ConfigSubsystem,
    Quel1seRiken8DebugConfigSubsystem,
)

logger = logging.getLogger()


@pytest.mark.parametrize(
    ("csscls",),
    [
        (QubeOuTypeAConfigSubsystem,),
        (QubeOuTypeBConfigSubsystem,),
        (Quel1TypeAConfigSubsystem,),
        (Quel1TypeBConfigSubsystem,),
        (Quel1NecConfigSubsystem,),
        (Quel1seAddaConfigSubsystem,),
        (Quel1seRiken8ConfigSubsystem,),
        (Quel1seRiken8DebugConfigSubsystem,),
        (Quel1seFujitsu11TypeAConfigSubsystem,),
        (Quel1seFujitsu11TypeBConfigSubsystem,),
        (Quel1seFujitsu11TypeADebugConfigSubsystem,),
        (Quel1seFujitsu11TypeBDebugConfigSubsystem,),
    ],
)
def test_group_and_mxfe_definitions(csscls):
    for (group, line), (mxfe_idx, dac_idx) in csscls._DAC_IDX.items():
        assert group in csscls._GROUPS
        assert mxfe_idx in csscls._MXFE_IDXS
        assert 0 <= dac_idx <= 3

    for (group, rline), (mxfe_idx, adc_idx) in csscls._ADC_IDX.items():
        assert group in csscls._GROUPS
        assert mxfe_idx in csscls._MXFE_IDXS
        assert 0 <= adc_idx <= 3
