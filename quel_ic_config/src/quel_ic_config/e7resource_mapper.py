import logging
from typing import Any, Dict, Final, List, Set, Tuple, Union

from e7awgsw import AWG

from quel_ic_config.e7workaround import CaptureModule, E7FwType
from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemAd9082Mixin
from quel_ic_config.quel1_wave_subsystem import Quel1WaveSubsystem

logger = logging.getLogger(__name__)


# TODO: consider the right place for this class.
#       guessing that it can be private in an object abstracting a whole box.
#       it has hardware information defined in RTL code for Alveo U50, and connect it with AD9082 settings.
class Quel1E7ResourceMapper:
    # Notes: (mxfe_idx, fduc_idx) --> awg_idx
    _AWGS_FROM_FDUC: Final[Dict[Tuple[int, int], int]] = {
        (0, 0): AWG.U15,
        (0, 1): AWG.U14,
        (0, 2): AWG.U13,
        (0, 3): AWG.U12,
        (0, 4): AWG.U11,
        (0, 5): AWG.U8,
        (0, 6): AWG.U9,
        (0, 7): AWG.U10,
        (1, 0): AWG.U7,
        (1, 1): AWG.U6,
        (1, 2): AWG.U5,
        (1, 3): AWG.U4,
        (1, 4): AWG.U3,
        (1, 5): AWG.U0,
        (1, 6): AWG.U1,
        (1, 7): AWG.U2,
    }

    # Notes: (mxfe_idx, adc_idx) --> capmod_idx
    # this is specific to version of e7awghw.  (simple_multi, not later than 20230720)
    _CAPTURE_MODULE_RLINE_SHARED_WITH_MLINE: Final[Dict[Tuple[int, int], int]] = {
        (0, 3): CaptureModule.U1,
        (0, 2): CaptureModule.U1,
        (1, 3): CaptureModule.U0,
        (1, 2): CaptureModule.U0,
    }

    # for feedback-early and NEC firmware, mapping should be like this (not confirmed yet)
    _CAPTURE_MODULE_RLINE_PLUS_MLINE: Final[Dict[Tuple[int, int], int]] = {
        (0, 3): CaptureModule.U1,
        (0, 2): CaptureModule.U3,
        (1, 3): CaptureModule.U0,
        (1, 2): CaptureModule.U2,
    }

    def __init__(self, css: Quel1ConfigSubsystemAd9082Mixin, wss: Quel1WaveSubsystem):
        self._css = css
        self._wss = wss

        # TODO: generate it from the current register values.
        self.dac2fduc: Final[List[List[List[int]]]] = self._parse_tx_channel_assign(self._css._param["ad9082"])

        # Notes: using this is fine, but reconsider better interface.
        # TODO: revise the inferface.
        self.dac_idx: Final[Dict[Tuple[int, int], Tuple[int, int]]] = self._css._DAC_IDX
        self.adc_idx: Final[Dict[Tuple[int, str], Tuple[int, int]]] = self._css._ADC_IDX

    @staticmethod
    def _parse_tx_channel_assign(ad9082_params: List[Dict[str, Any]]) -> List[List[List[int]]]:
        r = []
        for mxfe_idx, p in enumerate(ad9082_params):
            q: List[List[int]] = [[], [], [], []]  # AD9082 has 4 DACs
            # TODO: consider to share validation method with ad9081 wrapper.
            for dac_name, fducs in p["dac"]["channel_assign"].items():
                if len(dac_name) != 4 or not dac_name.startswith("dac") or dac_name[3] not in "0123":
                    raise ValueError("invalid settings of ad9082[{mxfe_idx}].tx.channel_assign")
                dac_idx = int(dac_name[3])
                q[dac_idx] = fducs
            r.append(q)
        return r

    def validate_configuration_integrity(self, mxfe_idx: int, ignore_extraordinary_converter_select: bool = False):
        fw_ver = self._wss.hw_type
        if mxfe_idx in self._css.get_all_mxfes():
            convsel = self._css.get_virtual_adc_select(mxfe_idx)
            if fw_ver in {E7FwType.FEEDBACK_VERYEARLY, E7FwType.FEEDBACK_EARLY} and (
                convsel[10] != 10 or convsel[11] != 11
            ):
                if ignore_extraordinary_converter_select:
                    logger.error("read-in and monitor-in ports are swapped on 4 capture module firmware")
                else:
                    raise RuntimeError("read-in and monitor-in ports are swapped on 4 capture module firmware")
        else:
            raise ValueError("invalid mxfe: {mxfe_idx}")

    def get_active_adc_of_mxfe(self, mxfe_idx) -> Set[Tuple[int, int]]:
        adcs: Set[Tuple[int, int]] = set()

        if self._wss.is_monitor_shared_with_read():
            convsel = self._css.get_virtual_adc_select(mxfe_idx)
            if convsel[10] == 10 and convsel[11] == 11:
                adcs.add((mxfe_idx, 3))
            elif convsel[10] == 8 and convsel[11] == 9:
                adcs.add((mxfe_idx, 2))
            elif convsel[10] == -1 and convsel[11] == -1:
                logger.info(f"mxfe-#{mxfe_idx} is not configured yet")
            else:
                raise RuntimeError(f"unexpected converter select: {convsel}")
        else:
            # TODO: should be add adcs based on hwtype.
            adcs.add((mxfe_idx, 3))
            adcs.add((mxfe_idx, 2))

        return adcs

    def get_active_adc(self) -> Set[Tuple[int, int]]:
        adcs: Set[Tuple[int, int]] = set()
        for mxfe_idx in self._css.get_all_mxfes():
            adcs.update(self.get_active_adc_of_mxfe(mxfe_idx))
        return adcs

    def get_active_rlines_of_group(self, group: int) -> Set[str]:
        rlines: Set[str] = set()

        active_adcs = self.get_active_adc()
        for (g, rl), adc in self.adc_idx.items():
            if adc in active_adcs and g == group:
                rlines.add(rl)
        return rlines

    def get_active_rlines_of_mxfe(self, mxfe_idx: int) -> Set[Tuple[int, str]]:
        rlines: Set[Tuple[int, str]] = set()

        adcs = self.get_active_adc_of_mxfe(mxfe_idx)
        for (group, rline), adc in self.adc_idx.items():
            if adc in adcs:
                rlines.add((group, rline))
        return rlines

    def resolve_rline(self, group: int, rline: Union[None, str]) -> str:
        active_rlines = self.get_active_rlines_of_group(group)

        if rline is None:
            num_active_rlines = len(active_rlines)
            if num_active_rlines == 1:
                rline = tuple(active_rlines)[0]
            else:
                raise ValueError(
                    f"failed to determine rline to use, {'no' if num_active_rlines == 0 else 'multiple'} "
                    f"active rlines for group {group}"
                )
        elif rline not in active_rlines:
            # TODO: reconsider to generate exception because the handling of QuBE-OU becomes rigorous.
            logger.warning(f"rline '{rline}' is not available, try to use it anyway...")

        return rline

    def get_awgs_of_group(self, group: int) -> Set[int]:
        # Notes: group must be validated at box-level
        return {v for k, v in self._AWGS_FROM_FDUC.items() if k[0] == group}

    def get_awg_of_line(self, group: int, line: int) -> Set[int]:
        # Notes: line must be validated at box-level
        mxfe_idx, dac_idx = self.dac_idx[group, line]
        fducs = self.dac2fduc[mxfe_idx][dac_idx]
        return {self._AWGS_FROM_FDUC[mxfe_idx, fduc] for fduc in fducs}

    def get_awg_of_channel(self, group: int, line: int, channel: int) -> int:
        # Notes: channel must be validated at box-level
        mxfe_idx, dac_idx = self.dac_idx[(group, line)]
        fducs = self.dac2fduc[mxfe_idx][dac_idx]
        if channel < len(fducs):
            return self._AWGS_FROM_FDUC[mxfe_idx, fducs[channel]]
        else:
            raise ValueError(f"invalid combination of group:{group}, line:{line}, channel:{channel}")

    def get_capture_module_of_adc(self, mxfe_idx: int, adc_idx: int) -> int:
        if self._wss.is_monitor_shared_with_read():
            return self._CAPTURE_MODULE_RLINE_SHARED_WITH_MLINE[mxfe_idx, adc_idx]
        else:
            return self._CAPTURE_MODULE_RLINE_PLUS_MLINE[mxfe_idx, adc_idx]

    def get_capture_module_of_rline(self, group: int, rline: str) -> int:
        # Notes: rline must be validated at box-level
        mxfe_idx, adc_idx = self.adc_idx[group, rline]
        return self.get_capture_module_of_adc(mxfe_idx, adc_idx)

    def get_capture_units_of_rline(self, group: int, rline: str) -> Set[Tuple[int, int]]:
        # Notes: rline must be validated at box-level
        capmod = self.get_capture_module_of_rline(group, rline)
        num_capunits = self._wss.get_num_capunits_of_capmod(capmod)
        return {(capmod, i) for i in range(num_capunits)}

    def get_rchannels_of_rline(self, group: int, rline: str) -> Dict[int, int]:
        # Notes: rline must be validated at box-level
        capmod = self.get_capture_module_of_rline(group, rline)
        return self._wss.get_muc_structure()[capmod]

    def get_rchannel_of_runit(self, group: int, rline: str, runit: int) -> int:
        # Notes: rline must be validated at box-level
        capunit2rchannel = self.get_rchannels_of_rline(group, rline)
        num_capunits = len(capunit2rchannel)
        if 0 <= runit < num_capunits:
            capunit = tuple(capunit2rchannel.keys())[runit]
            return capunit2rchannel[capunit]
        else:
            raise ValueError(f"invalid runit:{runit} for group:{group}, rline:{rline}")
