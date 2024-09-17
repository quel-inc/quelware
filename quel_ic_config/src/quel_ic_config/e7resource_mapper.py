import logging

from e7awghal.fwtype import E7FwType

logger = logging.getLogger(__name__)


class AbstractQuel1E7ResourceMapper:
    # Notes: (mxfe_idx, fduc_idx) --> awg_idx
    _AWGS_FROM_FDUC: dict[tuple[int, int], int]

    # Notes: (mxfe_idx, fddc_idx) --> cpmd_idx
    _CAPMOD_FROM_FDDC: dict[tuple[int, int], int]

    def __init__(self):
        return

    def get_awgs_of_mxfe(self, mxfe_idx: int) -> set[int]:
        return {a for (m, _), a in self._AWGS_FROM_FDUC.items() if m == mxfe_idx}

    def get_awg_from_fduc(self, mxfe_idx: int, fduc_idx: int) -> int:
        awg_idx: int = self._AWGS_FROM_FDUC.get((mxfe_idx, fduc_idx), -1)
        if awg_idx < 0:
            raise ValueError(f"no AWG is available for (mxfe_idx: {mxfe_idx}, fduc_idx: {fduc_idx})")
        return awg_idx

    def get_fduc_from_awg(self, awg_idx: int) -> tuple[int, int]:
        for mu, a in self._AWGS_FROM_FDUC.items():
            if a == awg_idx:
                return mu
        else:
            raise ValueError(f"invalid index of awg unit: {awg_idx}")

    def get_capmod_from_fddc(self, mxfe_idx: int, fddc_idx: int) -> int:
        cpmd_idx: int = self._CAPMOD_FROM_FDDC.get((mxfe_idx, fddc_idx), -1)
        if cpmd_idx < 0:
            raise ValueError(f"no AWG is available for (mxfe_idx: {mxfe_idx}, fddc_idx: {fddc_idx})")
        return cpmd_idx


class Quel1ConventionalE7ResourceMapper(AbstractQuel1E7ResourceMapper):
    # Notes: inverted order of AWG is caused by PCB pattern between FPGA and AD9082.
    #        (5th and 7th lanes are inverted for both AD9082s.)
    # TODO: should be fixed at AD9082 settings (and replace this class with Quel1CanonicalE7ResourceMapper)
    _AWGS_FROM_FDUC: dict[tuple[int, int], int] = {
        (0, 0): 15,
        (0, 1): 14,
        (0, 2): 13,
        (0, 3): 12,
        (0, 4): 11,
        (0, 5): 8,
        (0, 6): 9,
        (0, 7): 10,
        (1, 0): 7,
        (1, 1): 6,
        (1, 2): 5,
        (1, 3): 4,
        (1, 4): 3,
        (1, 5): 0,
        (1, 6): 1,
        (1, 7): 2,
    }

    # Notes: (mxfe_idx, fddc_idx) --> capmod_idx
    #        this is specific to version of e7awghw (feedback_early and simplemulti_standard)
    # TODO: the mixup between JESD204C TX lanes and FDDC should be filxed at AD9082 settings.
    #       (and replace this class with Quel1CanonicalE7ResourceMapper)
    _CAPMOD_FROM_FDDC: dict[tuple[int, int], int] = {
        (0, 5): 1,
        (0, 4): 3,
        (1, 5): 0,
        (1, 4): 2,
    }


def create_rmap_object(boxname: str, fw_type: E7FwType) -> AbstractQuel1E7ResourceMapper:
    if fw_type in {
        E7FwType.SIMPLEMULTI_CLASSIC,
        E7FwType.FEEDBACK_VERYEARLY,
    }:
        raise ValueError(f"firmware of the box '{boxname}' is deprecated (firmware type: {fw_type})")
    elif fw_type in {
        E7FwType.SIMPLEMULTI_STANDARD,
        E7FwType.NEC_EARLY,
        E7FwType.FEEDBACK_EARLY,
    }:
        return Quel1ConventionalE7ResourceMapper()
    else:
        raise ValueError(f"unexpceted firmware type: '{fw_type}'")


# Notes: this test is basically not necessary. Just for confirming deprecated settings are not applied by mistake.
def validate_configuration_integrity(
    convsel: list[int], fw_type: E7FwType, ignore_extraordinary_converter_select: bool = False
) -> None:
    if fw_type != E7FwType.SIMPLEMULTI_CLASSIC and (convsel[10] != 10 or convsel[11] != 11):
        if ignore_extraordinary_converter_select:
            logger.error("read-in and monitor-in ports are swapped unexpectedly")
        else:
            raise RuntimeError("read-in and monitor-in ports are swapped unexpectedly")
