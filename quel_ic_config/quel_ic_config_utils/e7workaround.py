import logging
from enum import Enum, IntEnum
from typing import Final, Mapping, Tuple

import e7awgsw

logger = logging.getLogger(__name__)


class E7FwType(Enum):
    AUTO_DETECT = 0
    SIMPLEMULTI_CLASSIC = 1
    FEEDBACK_VERYEARLY = 2
    NEC_EARLY = 3
    SIMPLEMULTI_WIDE = 4
    FEEDBACK_EARLY = 5


class E7LibBranch(Enum):
    SIMPLEMULTI = 0
    FEEDBACK = 1


class E7FwLifeStage(Enum):
    EXPERIMENTAL = 0
    SUPPORTED = 1
    TO_DEPRECATE = 2
    DEPRECATE = 3
    UNKNOWN = 4


# Notes: the first matched line is utilized.
_VERSION_TO_HWTYPE: Final[Mapping[Tuple[str, str, str], Tuple[E7FwType, E7FwLifeStage]]] = {
    ("K:2024/01/10-1", "K:2024/01/10-2", "*"): (E7FwType.FEEDBACK_EARLY, E7FwLifeStage.EXPERIMENTAL),
    ("K:2023/12/28-1", "K:2023/12/28-2", "*"): (E7FwType.SIMPLEMULTI_CLASSIC, E7FwLifeStage.SUPPORTED),
    ("K:2023/12/16-1", "K:2023/12/16-2", "*"): (E7FwType.SIMPLEMULTI_CLASSIC, E7FwLifeStage.SUPPORTED),
    # 20231108 should be deprecated as soon as possible.
    ("K:2023/11/08-1", "K:2023/11/08-1", "*"): (E7FwType.FEEDBACK_VERYEARLY, E7FwLifeStage.TO_DEPRECATE),
    # for 20230222, 20230422
    ("K:2023/02/22-1", "K:2023/02/22-2", "*"): (E7FwType.SIMPLEMULTI_CLASSIC, E7FwLifeStage.SUPPORTED),
    # for 20221023, 20230820, and 20230929
    # - 20221023: the oldestf firmware to support. it has overflow bug in DSP module.
    # - 20230820: supporting SYSREF latch (broken version identifiers, under investigation.)
    # - 20230928: sysref timing is modified (broken version identifiers, under investigation.)
    ("K:2022/07/14-1", "K:2022/07/14-2", "*"): (E7FwType.SIMPLEMULTI_CLASSIC, E7FwLifeStage.SUPPORTED),
    # Fallback case, DO NOT REMOVE THIS LINE.
    ("*", "*", "*"): (E7FwType.SIMPLEMULTI_CLASSIC, E7FwLifeStage.UNKNOWN),
}


def resolve_hw_type(hw_versions: Tuple[str, str, str]) -> Tuple[E7FwType, E7FwLifeStage]:
    for v, (ht, ls) in _VERSION_TO_HWTYPE.items():
        for i, u in enumerate(v):
            if u != "*" and u != hw_versions[i]:
                break
        else:
            if v == ("*", "*", "*"):
                logger.warning(f"no matching rule for the version {hw_versions}")
            logger.info(f"the hardware type is resolved as {ht}")
            return ht, ls
    # Notes: this shouldn't happen since VERSION_TO_HWTYPE should have a fallthru case at the end.
    raise RuntimeError(f"unknown combination of versions: {hw_versions}")


def detect_branch_of_library() -> E7LibBranch:
    # Notes: this is just a workaround of the poor software development process.
    #        should eliminate the need of this check in a correct manner as soon as possible.
    # Notes: redundant conditionals are introduced to avoid possible accidents caused by future modification of e7awgsw.
    #        The WSS does care about the number of capture modules and units although FEEDBACK-ness of the firmware come
    #        from the availability of the new sequencer.
    num_capmod: int = len(e7awgsw.CaptureModule.all())
    has_sequencerctrl: bool = hasattr(e7awgsw, "SequencerCtrl")
    if num_capmod == 2 and not has_sequencerctrl:
        return E7LibBranch.SIMPLEMULTI
    elif num_capmod == 4 and has_sequencerctrl:
        return E7LibBranch.FEEDBACK
    else:
        raise RuntimeError(
            f"unexpected library (num_capmod = {num_capmod}, has_sequencerctrl = {has_sequencerctrl}) is detected"
        )


class CaptureUnit(IntEnum):
    """キャプチャユニットの ID"""

    U0 = 0
    U1 = 1
    U2 = 2
    U3 = 3
    U4 = 4
    U5 = 5
    U6 = 6
    U7 = 7
    U8 = 8
    U9 = 9

    @classmethod
    def all(cls):
        """全キャプチャユニットの ID をリストとして返す"""
        return [item for item in CaptureUnit]

    @classmethod
    def of(cls, val):
        if not CaptureUnit.includes(val):
            raise ValueError("Cannot convert {} to CaptureUnit".format(val))
        return CaptureUnit.all()[val]

    @classmethod
    def includes(cls, *vals):
        units = cls.all()
        return all([val in units for val in vals])


class CaptureModule(IntEnum):
    """キャプチャモジュール (複数のキャプチャユニットをまとめて保持するモジュール) の列挙型"""

    U0 = 0
    U1 = 1
    U2 = 2
    U3 = 3

    @classmethod
    def all(cls):
        """全キャプチャモジュールの ID をリストとして返す"""
        return [item for item in CaptureModule]

    @classmethod
    def of(cls, val):
        if not CaptureModule.includes(val):
            raise ValueError("Cannot convert {} to CaptureModule".format(val))
        return CaptureModule.all()[val]

    @classmethod
    def includes(cls, *vals):
        mods = cls.all()
        return all([val in mods for val in vals])

    @classmethod
    def get_units(cls, *capmod_id_list):
        """引数で指定したキャプチャモジュールが保持するキャプチャユニットの ID を取得する

        Args:
            *capmod_id_list (list of CaptureModule): キャプチャユニットを取得するキャプチャモジュール ID

        Returns:
            list of CaptureUnit: capmod_id_list に対応するキャプチャモジュールが保持するキャプチャユニットのリスト
        """
        units = []
        for capmod_id in set(capmod_id_list):
            if capmod_id == cls.U0:
                units += [CaptureUnit.U0, CaptureUnit.U1, CaptureUnit.U2, CaptureUnit.U3]
            elif capmod_id == cls.U1:
                units += [CaptureUnit.U4, CaptureUnit.U5, CaptureUnit.U6, CaptureUnit.U7]
            else:
                raise ValueError("Invalid capture module ID {}".format(capmod_id))
        return sorted(units)
