import logging
from typing import Any, Dict, Final, List, Set, Tuple, Union

from e7awgsw import AWG

from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemAd9082Mixin
from quel_ic_config_utils.e7workaround import CaptureModule
from quel_ic_config_utils.quel1_wave_subsystem import Quel1WaveSubsystem

logger = logging.getLogger(__name__)


# TODO: consider the right place for this class.
#       guessing that it can be private in an object abstracting a whole box.
#       it has hardware information defined in RTL code for Alveo U50, and connect it with AD9082 settings.
class Quel1E7ResourceMapper:
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

    # this is specific to version of e7awghw.  (simple_multi, not later than 20230720)
    _CAPTURE_MODULE_RLINE_SHARED_WITH_MLINE: Final[Dict[Tuple[int, str], int]] = {
        (0, "r"): CaptureModule.U1,
        (0, "m"): CaptureModule.U1,
        (1, "r"): CaptureModule.U0,
        (1, "m"): CaptureModule.U0,
    }

    # for feedback-early firmware, mapping should be like this (not confirmed yet)
    _CAPTURE_MODULE_RLINE_PLUS_MLINE: Final[Dict[Tuple[int, str], int]] = {
        (0, "r"): CaptureModule.U1,
        (0, "m"): CaptureModule.U3,
        (1, "r"): CaptureModule.U0,
        (1, "m"): CaptureModule.U2,
    }

    # for NEC-early firmware, mapping should be like this (not confirmed yet)
    # TODO: implement when to use this table
    _CAPTURE_MODULE_FOUR_RLINE: Final[Dict[Tuple[int, str], int]] = {
        (0, "r1"): CaptureModule.U1,
        (0, "r2"): CaptureModule.U3,
        (1, "r1"): CaptureModule.U0,
        (1, "r2"): CaptureModule.U2,
    }

    def __init__(self, css: Quel1ConfigSubsystemAd9082Mixin, wss: Quel1WaveSubsystem):
        self._css = css
        self._wss = wss

        # TODO: generate it from the current register values.
        self.dac2fduc: Final[List[List[List[int]]]] = self._parse_tx_channel_assign(self._css._param["ad9082"])

        # Notes: using this is fine, but reconsider better interface.
        self.dac_idx: Final[Dict[Tuple[int, int], int]] = self._css._DAC_IDX

    @staticmethod
    def _parse_tx_channel_assign(ad9082_params: List[Dict[str, Any]]) -> List[List[List[int]]]:
        r = []
        for mxfe, p in enumerate(ad9082_params):
            q: List[List[int]] = [[], [], [], []]  # AD9082 has 4 DACs
            # TODO: consider to share validation method with ad9081 wrapper.
            for dac_name, fducs in p["tx"]["channel_assign"].items():
                if len(dac_name) != 4 or not dac_name.startswith("dac") or dac_name[3] not in "0123":
                    raise ValueError("invalid settings of ad9082[{mxfe}].tx.channel_assign")
                dac_idx = int(dac_name[3])
                q[dac_idx] = fducs
            r.append(q)
        return r

    def get_active_rlines_of_group(self, group: int) -> Set[str]:
        rlines: Set[str] = set()

        if self._wss.is_monitor_shared_with_read():
            convsel = self._css.get_virtual_adc_select(group)
            # TODO: check the validity of this code based on the design of RTL and PCB. it should work at least.
            if convsel[10] == 10:
                rlines.add("r")
            elif convsel[10] == 8:
                rlines.add("m")
            else:
                raise RuntimeError(f"unexpected converter select: {convsel}")
        else:
            # TODO: should be add rlines based on hwtype.
            rlines.add("r")
            rlines.add("m")

        return rlines

    def resolve_rline(self, group: int, rline: Union[None, str]) -> str:
        active_rlines = self.get_active_rlines_of_group(group)

        if rline is None:
            if len(active_rlines) == 1:
                rline = tuple(active_rlines)[0]
            else:
                raise ValueError("failed to determine rline to use")
        elif rline not in active_rlines:
            logger.warning(f"rline '{rline}' is not available, try to use it anyway...")

        return rline

    def get_awgs_of_group(self, group: int) -> Set[int]:
        # Notes: group must be validated at box-level
        return {v for k, v in self._AWGS_FROM_FDUC.items() if k[0] == group}

    def get_awg_of_line(self, group: int, line: int) -> Set[int]:
        # Notes: line must be validated at box-level
        fducs = self.dac2fduc[group][self.dac_idx[(group, line)]]
        return {self._AWGS_FROM_FDUC[group, fduc] for fduc in fducs}

    def get_awg_of_channel(self, group: int, line: int, channel: int) -> int:
        # Notes: channel must be validated at box-level
        fducs = self.dac2fduc[group][self.dac_idx[(group, line)]]
        if channel < len(fducs):
            return self._AWGS_FROM_FDUC[group, fducs[channel]]
        else:
            raise ValueError(f"invalid combination of (group, line, channel) = ({group}, {line}, {channel})") from None

    def get_capture_modules_of_group(self, group: int) -> Set[int]:
        # Notes: group must be validated at box-level
        if self._wss.is_monitor_shared_with_read():
            table = self._CAPTURE_MODULE_RLINE_SHARED_WITH_MLINE
        else:
            table = self._CAPTURE_MODULE_RLINE_PLUS_MLINE
        capmods: Set[int] = set()
        for g1, rl1 in table:
            if g1 == group:
                capmods.add(self.get_capture_module_of_rline(g1, rl1))
        return capmods

    def get_capture_module_of_rline(self, group: int, rline: str) -> int:
        # Notes: rline must be validated at box-level
        if self._wss.is_monitor_shared_with_read():
            return self._CAPTURE_MODULE_RLINE_SHARED_WITH_MLINE[group, rline]
        else:
            return self._CAPTURE_MODULE_RLINE_PLUS_MLINE[group, rline]

    def get_capture_units_of_group(self, group: int) -> Set[Tuple[int, int]]:
        # Notes: group must be validated at box-level
        if self._wss.is_monitor_shared_with_read():
            table = self._CAPTURE_MODULE_RLINE_SHARED_WITH_MLINE
        else:
            table = self._CAPTURE_MODULE_RLINE_PLUS_MLINE
        capunits: Set[Tuple[int, int]] = set()
        for g1, rl1 in table:
            if g1 == group:
                capunits.update(self.get_capture_units_of_rline(g1, rl1))
        return capunits

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
            raise ValueError(f"invalid runit: {runit} for (group: {group}, rline: {rline})")
