from typing import Union

from quel_ic_config.quel1_config_subsystem import (
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1NecConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
)
from quel_ic_config.quel1se_adda_config_subsystem import Quel1seAddaConfigSubsystem
from quel_ic_config.quel1se_fujitsu11_config_subsystem import Quel1seFujitsu11DebugConfigSubsystem
from quel_ic_config.quel1se_proto8_config_subsystem import Quel1seProto8ConfigSubsystem
from quel_ic_config.quel1se_proto11_config_subsystem import Quel1seProto11ConfigSubsystem
from quel_ic_config.quel1se_proto_adda_config_subsystem import Quel1seProtoAddaConfigSubsystem
from quel_ic_config.quel1se_riken8_config_subsystem import (
    Quel1seRiken8ConfigSubsystem,
    Quel1seRiken8DebugConfigSubsystem,
)

# TODO: replace it with Protocol
Quel1AnyConfigSubsystem = Union[
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
    Quel1NecConfigSubsystem,
    Quel1seProto11ConfigSubsystem,
    Quel1seProto8ConfigSubsystem,
    Quel1seProtoAddaConfigSubsystem,
    Quel1seAddaConfigSubsystem,
    Quel1seRiken8ConfigSubsystem,
    Quel1seRiken8DebugConfigSubsystem,
    Quel1seFujitsu11DebugConfigSubsystem,
]

# TODO: replace it with Protocol
Quel1AnyBoxConfigSubsystem = Union[
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
    Quel1NecConfigSubsystem,
    Quel1seRiken8ConfigSubsystem,
    Quel1seRiken8DebugConfigSubsystem,
    Quel1seFujitsu11DebugConfigSubsystem,
]
