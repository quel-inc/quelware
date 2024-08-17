import copy
import logging
from concurrent.futures import Future
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, Collection, Dict, Final, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from quel_clock_master import SequencerClient  # to be replaced
from quel_ic_config.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config.linkupper import LinkupFpgaMxfe
from quel_ic_config.quel1_anytype import Quel1AnyBoxConfigSubsystem, Quel1AnyConfigSubsystem
from quel_ic_config.quel1_config_subsystem import (
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1BoxType,
    Quel1ConfigOption,
    Quel1Feature,
    Quel1NecConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
)
from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemAd9082Mixin
from quel_ic_config.quel1_wave_subsystem import CaptureReturnCode, E7FwLifeStage, E7FwType, Quel1WaveSubsystem
from quel_ic_config.quel1se_adda_config_subsystem import Quel1seAddaConfigSubsystem
from quel_ic_config.quel1se_fujitsu11_config_subsystem import Quel1seFujitsu11DebugConfigSubsystem
from quel_ic_config.quel1se_proto8_config_subsystem import Quel1seProto8ConfigSubsystem
from quel_ic_config.quel1se_proto11_config_subsystem import Quel1seProto11ConfigSubsystem
from quel_ic_config.quel1se_proto_adda_config_subsystem import Quel1seProtoAddaConfigSubsystem
from quel_ic_config.quel1se_riken8_config_subsystem import (
    Quel1seRiken8ConfigSubsystem,
    Quel1seRiken8DebugConfigSubsystem,
)

logger = logging.getLogger(__name__)


def _validate_boxtype(boxtype):
    if boxtype not in {
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuBE_OU_TypeB,
        Quel1BoxType.QuBE_RIKEN_TypeB,
        Quel1BoxType.QuEL1_TypeB,
        Quel1BoxType.QuEL1_NEC,
        Quel1BoxType.QuEL1SE_ProtoAdda,
        Quel1BoxType.QuEL1SE_Proto8,
        Quel1BoxType.QuEL1SE_Proto11,
        Quel1BoxType.QuEL1SE_Adda,
        Quel1BoxType.QuEL1SE_RIKEN8,
        Quel1BoxType.QuEL1SE_RIKEN8DBG,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA,
    }:
        raise ValueError(f"unsupported boxtype: {boxtype}")


def _is_box_available_for(boxtype: Quel1BoxType) -> bool:
    return boxtype in {
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuBE_OU_TypeB,
        Quel1BoxType.QuBE_RIKEN_TypeB,
        Quel1BoxType.QuEL1_TypeB,
        Quel1BoxType.QuEL1_NEC,
        Quel1BoxType.QuEL1SE_RIKEN8,
        Quel1BoxType.QuEL1SE_RIKEN8DBG,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA,
        Quel1BoxType.QuEL1SE_Adda,
    }


def _complete_ipaddrs(ipaddr_wss: str, ipaddr_sss: Union[str, None], ipaddr_css: Union[str, None]) -> Tuple[str, str]:
    if ipaddr_sss is None:
        ipaddr_sss = str(IPv4Address(ipaddr_wss) + (1 << 16))
    if ipaddr_css is None:
        ipaddr_css = str(IPv4Address(ipaddr_wss) + (4 << 16))
    return ipaddr_sss, ipaddr_css


def _create_wss_object(ipaddr_wss: str, features: Set[Quel1Feature]) -> Quel1WaveSubsystem:
    wss: Quel1WaveSubsystem = Quel1WaveSubsystem(ipaddr_wss)
    if wss.hw_lifestage == E7FwLifeStage.TO_DEPRECATE:
        logger.warning(f"the firmware will deprecate soon, consider to update it as soon as possible: {wss.hw_version}")
    elif wss.hw_lifestage == E7FwLifeStage.EXPERIMENTAL:
        logger.warning(f"be aware that the firmware is still in an experimental stage: {wss.hw_version}")

    wss.validate_installed_e7awgsw()
    if wss.hw_type in {E7FwType.SIMPLEMULTI_CLASSIC}:
        features.add(Quel1Feature.SINGLE_ADC)
    elif wss.hw_type in {E7FwType.FEEDBACK_VERYEARLY}:
        features.add(Quel1Feature.BOTH_ADC_EARLY)
    elif wss.hw_type in {E7FwType.SIMPLEMULTI_STANDARD, E7FwType.FEEDBACK_EARLY}:
        features.add(Quel1Feature.BOTH_ADC)
    else:
        raise ValueError(f"unsupported firmware is detected: {wss.hw_type}")
    return wss


def _create_css_object(
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    features: Collection[Quel1Feature],
    config_root: Union[Path, None],
    config_options: Collection[Quel1ConfigOption],
) -> Quel1AnyConfigSubsystem:
    if boxtype in {Quel1BoxType.QuBE_RIKEN_TypeA, Quel1BoxType.QuEL1_TypeA}:
        css: Quel1AnyConfigSubsystem = Quel1TypeAConfigSubsystem(
            ipaddr_css, boxtype, features, config_root, config_options
        )
    elif boxtype in {Quel1BoxType.QuBE_RIKEN_TypeB, Quel1BoxType.QuEL1_TypeB}:
        css = Quel1TypeBConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuBE_OU_TypeA:
        css = QubeOuTypeAConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuBE_OU_TypeB:
        css = QubeOuTypeBConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1_NEC:
        css = Quel1NecConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_ProtoAdda:
        css = Quel1seProtoAddaConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_Proto8:
        css = Quel1seProto8ConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_Proto11:
        css = Quel1seProto11ConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_Adda:
        css = Quel1seAddaConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_RIKEN8:
        css = Quel1seRiken8ConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_RIKEN8DBG:
        css = Quel1seRiken8DebugConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA:
        css = Quel1seFujitsu11DebugConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    else:
        raise ValueError(f"unsupported boxtype: {boxtype}")

    if not isinstance(css, Quel1ConfigSubsystemAd9082Mixin):
        raise AssertionError("the given ConfigSubsystem Object doesn't provide AD9082 interface")

    return css


class Quel1BoxIntrinsic:
    __slots__ = (
        "_css",
        "_sss",
        "_wss",
        "_rmap",
        "_linkupper",
        "_options",
    )

    NUM_SAMPLE_IN_WAVE_BLOCK: Final[int] = 64  # TODO: should be defined in wss.
    NUM_SAMPLE_IN_WORD: Final[int] = 4  # TODO: should be defined in wss.
    DEFAULT_AMPLITUDE: Final[float] = 16383.0
    DEFAULT_NUM_WAVE_SAMPLE: Final[int] = NUM_SAMPLE_IN_WAVE_BLOCK * 1
    DEFAULT_REPEATS: Final[Tuple[int, int]] = (0xFFFFFFFF, 0xFFFFFFFF)
    DEFAULT_NUM_WAIT_SAMPLES: Final[Tuple[int, int]] = (0, 0)
    DEFAULT_NUM_CAPTURE_SAMPLE: Final[int] = 4096
    VERY_LONG_DURATION: float = 0x10000000000000000 * 128e-9
    DEFAULT_SCHEDULE_DEADLINE: Final[float] = 0.25  # [s]
    DEFAULT_SCHEDULE_WINDOW: Final[float] = 300.0  # [s]

    @classmethod
    def is_applicable_to(cls, boxtype: Quel1BoxType) -> bool:
        return _is_box_available_for(boxtype)

    @classmethod
    def create(
        cls,
        *,
        ipaddr_wss: str,
        ipaddr_sss: Union[str, None] = None,
        ipaddr_css: Union[str, None] = None,
        boxtype: Union[Quel1BoxType, str],
        config_root: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        **options: Collection[int],
    ) -> "Quel1BoxIntrinsic":
        """create QuEL intrinsic box objects
        :param ipaddr_wss: IP address of the wave generation subsystem of the target box
        :param ipaddr_sss: IP address of the sequencer subsystem of the target box (optional)
        :param ipaddr_css: IP address of the configuration subsystem of the target box (optional)
        :param boxtype: type of the target box
        :param config_root: root path of config setting files to read (optional)
        :param config_options: a collection of config options (optional)
        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored. (optional)
        :param ignore_access_failure_of_adrf6780: a list of ADRF6780 whose communication faiulre via SPI bus is
                                                  dismissed (optional)
        :param ignore_lock_failure_of_lmx2594: a list of LMX2594 whose lock failure is ignored (optional)
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed (optional)
        :return: SimpleBoxIntrinsic objects
        """
        ipaddr_sss, ipaddr_css = _complete_ipaddrs(ipaddr_wss, ipaddr_sss, ipaddr_css)
        if isinstance(boxtype, str):
            boxtype = Quel1BoxType.fromstr(boxtype)
        if not _is_box_available_for(boxtype):
            raise ValueError(f"unsupported boxtype: {boxtype}")
        if config_options is None:
            config_options = set()

        features: Set[Quel1Feature] = set()
        wss: Quel1WaveSubsystem = _create_wss_object(ipaddr_wss, features)
        sss = SequencerClient(ipaddr_sss)
        css: Quel1AnyBoxConfigSubsystem = cast(
            Quel1AnyBoxConfigSubsystem, _create_css_object(ipaddr_css, boxtype, features, config_root, config_options)
        )
        return Quel1BoxIntrinsic(css=css, sss=sss, wss=wss, rmap=None, linkupper=None, **options)

    # TODO: consider to re-locate to the right place
    def _validate_options(self, flags: Dict[str, Collection[int]]):
        for k, v in flags.items():
            if k == "ignore_crc_error_of_mxfe":
                if not all([0 <= u < self.css._NUM_IC["ad9082"] for u in v]):
                    raise ValueError(f"invalid index of mxfe is found in {k} (= {v})")
            elif k == "ignore_access_failure_of_adrf6780":
                if not all([0 <= u < self.css._NUM_IC["adrf6780"] for u in v]):
                    raise ValueError(f"invalid index of adrf6780 is found in {k} (= {v})")
            elif k == "ignore_lock_failure_of_lmx2594":
                if not all([0 <= u < self.css._NUM_IC["lmx2594"] for u in v]):
                    raise ValueError(f"invalid index of lmx2594 is found in {k} (= {v})")
            elif k == "ignore_extraordinary_converter_select_of_mxfe":
                if not all([0 <= u < self.css._NUM_IC["ad9082"] for u in v]):
                    raise ValueError(f"invalid index of ad9082 is found in {k} (= {v})")
            else:
                raise ValueError(f"invalid workaround options: {k}")

        for k in (
            "ignore_crc_error_of_mxfe",
            "ignore_access_failure_of_adrf6780",
            "ignore_lock_failure_of_lmx2594",
            "ignore_extraordinary_converter_select_of_mxfe",
        ):
            if k not in flags:
                flags[k] = set()

    def __init__(
        self,
        *,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        sss: SequencerClient,
        rmap: Union[Quel1E7ResourceMapper, None] = None,
        linkupper: Union[LinkupFpgaMxfe, None] = None,
        **options: Collection[int],
    ):
        self._css = css
        self._wss = wss
        self._sss = sss
        if rmap is None:
            rmap = Quel1E7ResourceMapper(css, wss)
        self._rmap = rmap
        if linkupper is None:
            linkupper = LinkupFpgaMxfe(css, wss, rmap)
        self._linkupper = linkupper
        if options is None:
            options = {}
        self._validate_options(options)
        self._options: Dict[str, Collection[int]] = options

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._wss._wss_addr}:{self.boxtype}>"

    @property
    def css(self) -> Quel1AnyBoxConfigSubsystem:
        return self._css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._wss

    @property
    def sss(self) -> SequencerClient:
        return self._sss

    @property
    def rmap(self) -> Quel1E7ResourceMapper:
        return self._rmap

    @property
    def linkupper(self) -> LinkupFpgaMxfe:
        return self._linkupper

    @property
    def boxtype(self) -> str:
        return self.css.boxtype.tostr()

    @property
    def options(self) -> Dict[str, Collection[int]]:
        return self._options

    def _is_quel1se(self) -> bool:
        return self.css._boxtype in {Quel1BoxType.QuEL1SE_RIKEN8, Quel1BoxType.QuEL1SE_RIKEN8DBG}

    def reconnect(
        self,
        *,
        mxfe_list: Union[Collection[int], None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
        ignore_invalid_linkstatus: bool = False,
    ) -> Dict[int, bool]:
        """establish a configuration link between a box and host.
        the target box needs to be linked-up in advance.

        :param mxfe_list: a list of MxFEs to reconnect. (optional)
        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored. (optional)
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed. (optional)
        :return: True if success
        """
        if mxfe_list is None:
            mxfe_list = self._css.get_all_mxfes()

        if ignore_crc_error_of_mxfe is None:
            ignore_crc_error_of_mxfe = self._options["ignore_crc_error_of_mxfe"]

        if ignore_extraordinary_converter_select_of_mxfe is None:
            ignore_extraordinary_converter_select_of_mxfe = self._options[
                "ignore_extraordinary_converter_select_of_mxfe"
            ]

        link_ok = {}
        for mxfe_idx in mxfe_list:
            self._rmap.validate_configuration_integrity(
                mxfe_idx,
                ignore_extraordinary_converter_select=mxfe_idx in ignore_extraordinary_converter_select_of_mxfe,
            )
            try:
                valid_link: bool = self._css.configure_mxfe(
                    mxfe_idx, ignore_crc_error=mxfe_idx in ignore_crc_error_of_mxfe
                )
                self._css.ad9082[mxfe_idx].device_chip_id_get()
            except RuntimeError:
                valid_link = False

            if not valid_link:
                if not ignore_invalid_linkstatus:
                    logger.error(f"AD9082-#{mxfe_idx} is not working. it must be linked up in advance")
            link_ok[mxfe_idx] = valid_link

        return link_ok

    def relinkup(
        self,
        *,
        mxfes_to_linkup: Union[Collection[int], None] = None,
        hard_reset: Union[bool, None] = None,
        use_204b: bool = False,
        use_bg_cal: bool = True,
        skip_init: bool = False,
        background_noise_threshold: Union[float, None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
        restart_tempctrl: bool = False,
    ) -> Dict[int, bool]:
        if mxfes_to_linkup is None:
            mxfes_to_linkup = self._css.get_all_mxfes()
        if hard_reset is None:
            hard_reset = self._is_quel1se()
        if ignore_crc_error_of_mxfe is None:
            ignore_crc_error_of_mxfe = self._options["ignore_crc_error_of_mxfe"]
        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = self._options["ignore_access_failure_of_adrf6780"]
        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = self._options["ignore_lock_failure_of_lmx2594"]
        if ignore_extraordinary_converter_select_of_mxfe is None:
            ignore_extraordinary_converter_select_of_mxfe = self._options[
                "ignore_extraordinary_converter_select_of_mxfe"
            ]

        if not skip_init:
            self.linkupper._css.configure_peripherals(
                ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
                ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            )
            self.linkupper._css.configure_all_mxfe_clocks(
                ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            )

        linkup_ok: Dict[int, bool] = {}
        for mxfe in mxfes_to_linkup:
            linkup_ok[mxfe] = self.linkupper.linkup_and_check(
                mxfe,
                hard_reset=hard_reset,
                use_204b=use_204b,
                use_bg_cal=use_bg_cal,
                background_noise_threshold=background_noise_threshold,
                ignore_crc_error=mxfe in ignore_crc_error_of_mxfe,
                ignore_extraordinary_converter_select=mxfe in ignore_extraordinary_converter_select_of_mxfe,
                restart_tempctrl=restart_tempctrl,
            )
        return linkup_ok

    def terminate(self):
        self.css.terminate()

    def link_status(self, *, ignore_crc_error_of_mxfe: Union[Collection[int], None] = None) -> Dict[int, bool]:
        if ignore_crc_error_of_mxfe is None:
            ignore_crc_error_of_mxfe = self._options["ignore_crc_error_of_mxfe"]

        link_health: Dict[int, bool] = {}
        for mxfe_idx in self.css.get_all_mxfes():
            link_health[mxfe_idx] = self.css.check_link_status(
                mxfe_idx, False, ignore_crc_error=(mxfe_idx in ignore_crc_error_of_mxfe)
            )
        return link_health

    @staticmethod
    def _calc_wave_repeats(duration_in_sec: float, num_samples: int) -> Tuple[int, int]:
        unit_len = num_samples / 500e6
        duration_in_unit = duration_in_sec / unit_len
        u = round(duration_in_unit)
        v = 1
        while (u > 0xFFFFFFFF) and (v <= 0xFFFFFFFF):
            u //= 2
            v *= 2
        if v > 0xFFFFFFFF:
            return 0xFFFFFFFF, 0xFFFFFFFF
        else:
            return u, v

    def easy_start_cw(
        self,
        group: int,
        line: int,
        channel: int,
        *,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        fnco_freq: Union[float, None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
        fullscale_current: Union[int, None] = None,
        amplitude: float = DEFAULT_AMPLITUDE,
        duration: float = VERY_LONG_DURATION,
        control_port_rfswitch: bool = True,
        control_monitor_rfswitch: bool = False,
    ) -> None:
        """an easy-to-use API to generate continuous wave from a given channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :param lo_freq: the frequency of the corresponding local oscillator in Hz. it must be multiple of 100_000_000.
        :param cnco_freq: the frequency of the corresponding CNCO in Hz.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz.
        :param vatt: the control voltage of the corresponding VATT in unit of 3.3V / 4096.
        :param sideband: the active sideband of the corresponding mixer, "U" for upper and "L" for lower.
        :param fullscale_current: full-scale current of output DAC of AD9082 in uA.
        :param amplitude: the amplitude of the sinusoidal wave to be passed to DAC.
        :param duration: the duration of wave generation in second.
        :param control_port_rfswitch: allowing the port corresponding to the line to emit the RF signal if True.
        :param control_monitor_rfswitch: allowing the monitor-out port to emit the RF signal if True.
        :return: None
        """
        self.config_line(
            group=group,
            line=line,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
        )
        self.config_channel(group=group, line=line, channel=channel, fnco_freq=fnco_freq)
        if control_port_rfswitch:
            self.open_rfswitch(group, line)
        if control_monitor_rfswitch:
            self.open_rfswitch(group, "m")
        num_repeats = self._calc_wave_repeats(duration, 64)
        logger.info(
            f"start emitting continuous wave signal from ({group}, {line}, {channel}) "
            f"for {num_repeats[0]*num_repeats[1]*128e-9} seconds"
        )
        self.load_cw_into_channel(group, line, channel, amplitude=amplitude, num_repeats=num_repeats)
        self.start_emission({(group, line, channel)})

    def easy_stop(
        self,
        group: int,
        line: int,
        channel: Union[int, None] = None,
        *,
        control_port_rfswitch: bool = True,
        control_monitor_rfswitch: bool = False,
    ) -> None:
        """stopping the wave generation on a given channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: an index of the channel in the line. stop the signal generation of all the channel of
                        the line if None.
        :param control_port_rfswitch: blocking the emission of the RF signal from the corresponding port if True.
        :param control_monitor_rfswitch: blocking the emission of the RF signal from the monitor-out port if True.
        :return: None
        """

        if channel is None:
            channels = {(group, line, ch) for ch in range(self.css.get_num_channels_of_line(group, line))}
        else:
            channels = {(group, line, channel)}
        self.stop_emission(channels)
        logger.info(f"stop emitting continuous wave signal from ({group}, {line}, {channel})")
        self.close_rfswitch(group, line)
        if control_port_rfswitch:
            self.close_rfswitch(group, line)
        if control_monitor_rfswitch:
            self.activate_monitor_loop(group)

    def easy_stop_all(self, control_port_rfswitch: bool = True) -> None:
        """stopping the signal generation on all the channels of the box.

        :param control_port_rfswitch: blocking the emission of the RF signal from the corresponding port if True.
        :return: None
        """
        for group in self._css.get_all_groups():
            for line in self._css.get_all_lines_of_group(group):
                self.easy_stop(group, line, control_port_rfswitch=control_port_rfswitch)

    def easy_capture(
        self,
        group: int,
        rline: Union[str, None] = None,
        runit: int = 0,
        *,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        fnco_freq: Union[float, None] = None,
        activate_internal_loop: Union[None, bool] = None,
        num_samples: int = DEFAULT_NUM_CAPTURE_SAMPLE,
    ) -> npt.NDArray[np.complex64]:
        """capturing the wave signal from a given receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param rline: a group-local index of a line which the channel belongs to.
        :param runit: a line-local index of the capture unit.
        :param lo_freq: the frequency of the corresponding local oscillator in Hz. it must be multiple of 100_000_000.
        :param cnco_freq: the frequency of the corresponding CNCO in Hz.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz.
        :param activate_internal_loop: activate the corresponding loop-back path if True
        :param num_samples: number of samples to capture
        :return: captured wave data in NumPy array
        """
        input_line = self._rmap.resolve_rline(group, rline)
        self.config_rline(group=group, rline=input_line, lo_freq=lo_freq, cnco_freq=cnco_freq)
        self.config_runit(group=group, rline=input_line, runit=runit, fnco_freq=fnco_freq)

        if activate_internal_loop is not None:
            if input_line == "r":
                if activate_internal_loop:
                    self.activate_read_loop(group)
                else:
                    self.deactivate_read_loop(group)
            elif input_line == "m":
                if activate_internal_loop:
                    self.activate_monitor_loop(group)
                else:
                    self.deactivate_monitor_loop(group)
            else:
                raise AssertionError

        if num_samples % 4 != 0:
            num_samples = ((num_samples + 3) // 4) * 4
            logger.warning(f"num_samples is extended to multiples of 4: {num_samples}")

        if num_samples > 0:
            capmod = self._rmap.get_capture_module_of_rline(group, input_line)
            status, iq = self._wss.simple_capture(capmod, num_words=num_samples // 4)
            if status == CaptureReturnCode.SUCCESS:
                return iq
            elif status == CaptureReturnCode.CAPTURE_TIMEOUT:
                raise RuntimeError("failed to capture due to time out")
            elif status == CaptureReturnCode.CAPTURE_ERROR:
                raise RuntimeError("failed to capture due to internal error of FPGA")
            elif status == CaptureReturnCode.BROKEN_DATA:
                raise RuntimeError("failed to capture due to broken data")
            else:
                raise AssertionError
        elif num_samples == 0:
            logger.warning("attempting to read zero samples")
            return np.zeros(0, dtype=np.complex64)
        else:
            raise ValueError(f"nagative value for num_samples (= {num_samples})")

    def load_cw_into_channel(
        self,
        group: int,
        line: int,
        channel: int,
        *,
        amplitude: float = DEFAULT_AMPLITUDE,
        num_wave_sample: int = DEFAULT_NUM_WAVE_SAMPLE,
        num_repeats: Tuple[int, int] = DEFAULT_REPEATS,
        num_wait_samples: Tuple[int, int] = DEFAULT_NUM_WAIT_SAMPLES,
    ) -> None:
        """loading continuous wave data into a channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :param amplitude: amplitude of the continuous wave, 0 -- 32767.
        :param num_wave_sample: number of samples in wave data to generate.
        :param num_repeats: number of repetitions of a unit wave data whose length is num_wave_sample.
                            given as a tuple of two integers that specifies the number of repetition as multiple of
                            the two.
        :param num_wait_samples: number of wait duration in samples. given as a tuple of two integers that specify the
                               length of wait at the start of the whole wave sequence and the length of wait between
                               each repeated motifs, respectively.
        :return: None
        """
        if num_wave_sample % self.NUM_SAMPLE_IN_WAVE_BLOCK != 0:
            raise ValueError(f"wave samples must be multiple of {self.NUM_SAMPLE_IN_WAVE_BLOCK}")
        if num_wait_samples[0] % self.NUM_SAMPLE_IN_WORD != 0:
            raise ValueError(f"global wait samples must be multiple of {self.NUM_SAMPLE_IN_WORD}")
        if num_wait_samples[1] % self.NUM_SAMPLE_IN_WORD != 0:
            raise ValueError(f"wait samples between chunks must be multiple of {self.NUM_SAMPLE_IN_WORD}")

        awg = self._rmap.get_awg_of_channel(group, line, channel)
        self._wss.set_cw(
            awg,
            amplitude=amplitude,
            num_repeats=num_repeats,
            num_wave_blocks=num_wave_sample // self.NUM_SAMPLE_IN_WAVE_BLOCK,
            num_wait_words=(
                num_wait_samples[0] // self.NUM_SAMPLE_IN_WORD,
                num_wait_samples[1] // self.NUM_SAMPLE_IN_WORD,
            ),
        )

    def load_iq_into_channel(
        self,
        group: int,
        line: int,
        channel: int,
        *,
        iq: npt.NDArray[np.complex64],
        num_repeats: Tuple[int, int] = DEFAULT_REPEATS,
        num_wait_samples: Tuple[int, int] = DEFAULT_NUM_WAIT_SAMPLES,
    ) -> None:
        """loading arbitrary wave data into a channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :param iq: complex data of the signal to generate in 500Msps. I and Q coefficients of each sample must be
                   within the range of -32768 -- 32767. its length must be a multiple of 64.
        :param num_repeats: the number of repetitions of the given wave data given as a tuple of two integers,
                            a product of the two is the number of repetitions.
        :param num_wait_samples: number of wait duration in samples. given as a tuple of two integers that specify the
                               length of wait at the start of the whole wave sequence and the length of wait between
                               each repeated motifs, respectively.
        :return: None
        """
        if len(iq) % self.NUM_SAMPLE_IN_WAVE_BLOCK != 0:
            raise ValueError(f"the length of iq data must be multiple of {self.NUM_SAMPLE_IN_WAVE_BLOCK}")
        if num_wait_samples[0] % self.NUM_SAMPLE_IN_WORD != 0:
            raise ValueError(f"wave samples must be multiple of {self.NUM_SAMPLE_IN_WORD}")
        if num_wait_samples[1] % self.NUM_SAMPLE_IN_WORD != 0:
            raise ValueError(f"wave samples must be multiple of {self.NUM_SAMPLE_IN_WORD}")

        awg = self._rmap.get_awg_of_channel(group, line, channel)
        self._wss.set_iq(
            awg,
            iq=iq,
            num_repeats=num_repeats,
            num_wait_words=(
                num_wait_samples[0] // self.NUM_SAMPLE_IN_WORD,
                num_wait_samples[1] // self.NUM_SAMPLE_IN_WORD,
            ),
        )

    def initialize_all_awgs(self) -> None:
        self.wss.initialize_all_awgs()

    def initialize_all_capunits(self) -> None:
        self.wss.initialize_all_capunits()

    def prepare_for_emission(self, channels: Collection[Tuple[int, int, int]]):
        """making preparation of signal generation of multiple channels at the same time.

        :param channels: a collection of channels to be activated. each channel is specified as a tuple of a group,
                         a line, and a channel.
        """
        awgs = {self._rmap.get_awg_of_channel(gp, ln, ch) for gp, ln, ch in channels}
        self.wss.clear_before_starting_emission(awgs)

    def start_emission(self, channels: Collection[Tuple[int, int, int]]):
        """starting signal generation of multiple channels at the same time.

        :param channels: a collection of channels to be activated. each channel is specified as a tuple of a group,
                         a line, and a channel.
        """
        awgs = {self._rmap.get_awg_of_channel(gp, ln, ch) for gp, ln, ch in channels}
        self.wss.start_emission(awgs)

    def read_current_clock(self) -> int:
        valid, now, _ = self.sss.read_clock()
        if not valid:
            raise RuntimeError(f"failed to acquire the current time count of {self._sss.ipaddress}")
        return now

    def read_current_and_latched_clock(self) -> Tuple[int, int]:
        valid, now, latched = self.sss.read_clock()
        if not valid:
            raise RuntimeError(f"failed to acquire the current time count of {self._sss.ipaddress}")
        if latched < 0:
            raise RuntimeError(
                f"failed to acquire the latched time count of {self._sss.ipaddress}, its firmware should be updated."
            )
        return now, latched

    def reserve_emission(
        self,
        channels: Collection[Tuple[int, int, int]],
        time_count: int,
        margin: float = DEFAULT_SCHEDULE_DEADLINE,
        window: float = DEFAULT_SCHEDULE_WINDOW,
        skip_validation: bool = False,
    ) -> None:
        """scheduling to start signal generation of multiple channels at the specified timing.

        :param channels: a collection of channels to be activated. each channel is specified as a tuple of a group,
                         a line, and a channel.
        :param time_count: time to start emission in terms of the time count of the synchronization subsystem.
        :param margin: reservations with less than time `margin` in second will not be accepted.
                       default value is 0.25 seconds.
        :param window: reservations out of the time `window` in second from now is rejected.
                       default value is 300 seconds.
        :param skip_validation: skip the validation of time count if True. default is False.
        """
        awg_bitmap = sum([(1 << self._rmap.get_awg_of_channel(gp, ln, ch)) for gp, ln, ch in set(channels)])
        if not skip_validation:
            now = self.read_current_clock()
            start_of_window = now + round(margin * 125_000_000)
            end_of_window = now + round(window * 125_000_000)
            if time_count < start_of_window:
                timediff = round((time_count - now) / 125_000_000, 3)
                raise ValueError(f"time_count is too close to make a reservation, ({timediff:.3f} second from now)")
            if end_of_window < time_count:
                timediff = round((time_count - now) / 125_000_000, 3)
                raise ValueError(f"time count is too far to make a reservation, ({timediff:.3f} second from now)")

        if not self._sss.add_sequencer(time_count, awg_bitmap):
            raise RuntimeError(f"failed to schedule emission at time_count (= {time_count})")

    def stop_emission(self, channels: Collection[Tuple[int, int, int]]):
        """stopping signal generation on a given channel.

        :param channels: a collection of channels to be deactivated. each channel is specified as a tuple of a group,
                         a line, and a channel.
        """
        awgs = {self._rmap.get_awg_of_channel(gp, ln, ch) for gp, ln, ch in channels}
        self._wss.stop_emission(awgs)

    def simple_capture_start(
        self,
        group: int,
        rline: Union[str, None] = None,
        runits: Union[Collection[int], None] = None,
        *,
        num_samples: int = DEFAULT_NUM_CAPTURE_SAMPLE,
        delay_samples: int = 0,
        triggering_channel: Union[Tuple[int, int, int], None] = None,
        timeout: float = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT,
    ) -> "Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]]":
        """capturing the wave signal from a given receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param rline: a group-local index of a line which the channel belongs to.
        :param runits: line-local indices of the capture units.
        :param num_samples: number of samples to capture, recommended to be multiple of 4.
        :param delay_samples: delay in sampling clocks before starting capture.
        :param triggering_channel: a channel which triggers this capture when it starts to emit a signal.
                                   it is specified by a tuple of group, line, and channel. the capture starts
                                   immediately if None.
        :param timeout: waiting time in second before capturing thread quits.
        :return: captured wave data in NumPy array
        """
        input_line = self._rmap.resolve_rline(group, rline)
        if runits is None:
            runits = {0}

        if triggering_channel is None:
            triggering_awg: Union[int, None] = None
        elif isinstance(triggering_channel, tuple) and len(triggering_channel) == 3:
            triggering_awg = self._rmap.get_awg_of_channel(*triggering_channel)
        else:
            raise ValueError(f"invalid triggering channel: {triggering_channel}")

        if num_samples % 4 != 0:
            num_samples = ((num_samples + 3) // 4) * 4
            logger.warning(f"num_samples is extended to multiples of 4: {num_samples}")
        if delay_samples % 64 != 0:
            logger.warning(
                f"the effective delay_samples will be {((delay_samples + 32) // 64) * 64} (!= {delay_samples})"
            )
        if num_samples > 0:
            return self._wss.simple_capture_start(
                capmod=self._rmap.get_capture_module_of_rline(group, input_line),
                capunits=runits,
                num_words=num_samples // 4,
                delay=delay_samples // 4,
                triggering_awg=triggering_awg,
                timeout=timeout,
            )
        else:
            raise ValueError(f"non-positive value for num_samples (= {num_samples})")

    def capture_start(
        self,
        group: int,
        rline: str,
        runits: Collection[int],
        *,
        triggering_channel: Union[Tuple[int, int, int], None] = None,
        timeout: float = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT,
    ) -> "Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]]":
        """capturing the wave signal from a given receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param rline: a group-local index of a line which the channel belongs to.
        :param runits: line-local indices of the capture units.
        :param triggering_channel: a channel which triggers this capture when it starts to emit a signal.
                                   it is specified by a tuple of group, line, and channel. the capture starts
                                   immediately if None.
        :param timeout: waiting time in second before capturing thread quits.
        :return: captured wave data in NumPy array
        """
        if triggering_channel is None:
            triggering_awg: Union[int, None] = None
        elif isinstance(triggering_channel, tuple) and len(triggering_channel) == 3:
            triggering_awg = self._rmap.get_awg_of_channel(*triggering_channel)
        else:
            raise ValueError(f"invalid triggering channel: {triggering_channel}")

        return self._wss.capture_start(
            capmod=self._rmap.get_capture_module_of_rline(group, rline),
            capunits=runits,
            triggering_awg=triggering_awg,
            timeout=timeout,
        )

    def _config_box_inner_line(
        self,
        group: int,
        line: int,
        direction: Union[str, None],
        line_conf: Dict[str, Any],
        channel_confs: Dict[int, Dict[str, Any]],
        runit_confs: Dict[int, Dict[str, Any]],
    ):
        if runit_confs != {}:
            raise ValueError(f"'runits' is not applicable to an output line: ({group}, {line})")
        for ch, channel_conf in channel_confs.items():
            self.config_channel(group, line, ch, **channel_conf)
        self.config_line(group, line, **line_conf)
        if direction is not None:
            if direction == "out":
                logger.info(f"direction of ({group}, {line}) is confirmed to be {direction}")
            else:
                raise ValueError(
                    f"given direction '{direction}' doesn't match with the actual direction of ({group}, {line})"
                )

    def _config_box_inner_rline(
        self,
        group: int,
        rline: str,
        direction: Union[str, None],
        line_conf: Dict[str, Any],
        channel_confs: Dict[int, Dict[str, Any]],
        runit_confs: Dict[int, Dict[str, Any]],
    ):
        if channel_confs != {}:
            raise ValueError(f"'channels' is not applicable to an input line: ({group}, {rline})")
        for runit, runit_conf in runit_confs.items():
            self.config_runit(group, rline, runit, **runit_conf)
        self.config_rline(group, rline, **line_conf)
        if direction is not None:
            if direction == "in":
                logger.info(f"direction of ({group}, {rline}) is confirmed to be {direction}")
            else:
                raise ValueError(
                    f"given direction '{direction}' doesn't match with the actual direction of ({group}, {rline})"
                )

    def _config_box_inner(self, group: int, line: Union[int, str], lc: Dict[str, Any]):
        line_conf = copy.deepcopy(lc)
        if "direction" in line_conf:
            direction: Union[str, None] = line_conf["direction"]
            del line_conf["direction"]
        else:
            direction = None
        if "channels" in line_conf:
            channel_confs: Dict[int, Dict[str, Any]] = line_conf["channels"]
            del line_conf["channels"]
        else:
            channel_confs = {}
        if "runits" in line_conf:
            runit_confs: Dict[int, Dict[str, Any]] = line_conf["runits"]
            del line_conf["runits"]
        else:
            runit_confs = {}

        if self.is_output_line(group, line):
            self._config_box_inner_line(group, cast(int, line), direction, line_conf, channel_confs, runit_confs)
        elif self.is_input_line(group, line):
            self._config_box_inner_rline(group, cast(str, line), direction, line_conf, channel_confs, runit_confs)
        else:
            raise ValueError(f"invalid line: '{line}'")

    def config_box(self, box_conf: Dict[Tuple[int, Union[int, str]], Dict[str, Any]], ignore_validation: bool = False):
        # Notes: configure output lines before input ones to keep "cnco_locked_with" intuitive in config_box().
        for (group, line), lc in box_conf.items():
            if self.is_output_line(group, line):
                self._config_box_inner(group, line, lc)

        for (group, line), lc in box_conf.items():
            if self.is_input_line(group, line):
                self._config_box_inner(group, line, lc)

        if not ignore_validation:
            if not self.config_validate_box(box_conf):
                raise ValueError("the provided settings looks have inconsistent settings")

    def _config_validate_channels(
        self, group: int, line: Union[int, str], confs: Dict[int, Dict[str, Any]], parent_name: str
    ) -> bool:
        if not self.is_output_line(group, line):
            logger.error(f"'channels' appears at an unexpected {parent_name}")
            return False

        line = cast(int, line)
        valid = True
        for ch, cc in confs.items():
            acc = self.css.dump_channel(group, line, ch)
            for kk in cc:
                if kk not in acc:
                    valid = False
                    logger.error(f"unexpected settings at {parent_name}:channel-{ch}, '{kk}' is unavailable")
                elif cc[kk] != acc[kk]:
                    valid = False
                    logger.error(f"unexpected settings at {parent_name}:channel-{ch}: {kk} = {acc[kk]} (!= {cc[kk]})")
        return valid

    def _config_validate_runits(
        self, group: int, rline: Union[int, str], runits_conf: Dict[int, Dict[str, Any]], parent_name: str
    ) -> bool:
        if not self.is_input_line(group, rline):
            logger.error(f"'runits' is unavailable at {parent_name}")
            return False

        rline = cast(str, rline)
        valid = True
        for runit, uc in runits_conf.items():
            auc = self._dump_runit(group, rline, runit)
            for kk in uc:
                if kk not in auc:
                    valid = False
                    logger.error(f"unexpected settings at {parent_name}:runit-{runit}, '{kk}' is unavailable")
                elif uc[kk] != auc[kk]:
                    valid = False
                    logger.error(f"unexpected settings at {parent_name}:runit-{runit}: {kk} = {auc[kk]} (!= {uc[kk]})")
        return valid

    def _config_validate_frequency(self, group: int, line: Union[int, str], k: str, freq0: float, freq1: float) -> bool:
        if self.is_output_line(group, line):
            if k == "cnco_freq":
                return self.css.is_equivalent_dac_cnco(group, cast(int, line), freq0, freq1)
            elif k == "fnco_freq":
                return self.css.is_equivalent_dac_fnco(group, cast(int, line), freq0, freq1)
        elif self.is_input_line(group, line):
            if k == "cnco_freq":
                return self.css.is_equivalent_adc_cnco(group, cast(str, line), freq0, freq1)
            elif k == "fnco_freq":
                return self.css.is_equivalent_adc_fnco(group, cast(str, line), freq0, freq1)
        raise AssertionError

    def _config_validate_fsc(self, group: int, line: int, fsc0: int, fsc1: int) -> bool:
        mxfe_idx, _ = self._css._get_dac_idx(group, line)
        return self._css.ad9082[mxfe_idx].is_equal_fullscale_current(fsc0, fsc1)

    def _config_validate_line(self, group: int, line: Union[int, str], lc: Dict[str, Any]) -> bool:
        if self.is_output_line(group, line):
            line_name: str = f"group:{group}, line:{line}"
            alc: Dict[str, Any] = self.css.dump_line(group, cast(int, line))
            ad: str = "out"
        elif self.is_input_line(group, line):
            line_name = f"group:{group}, rline:{line}"
            alc = self.css.dump_rline(group, cast(str, line))
            ad = "in"
        else:
            raise ValueError(f"invalid line: ({group}, {line})")

        valid = True
        for k in lc:
            if k == "channels":
                if not self._config_validate_channels(group, line, lc["channels"], line_name):
                    valid = False
            elif k == "runits":
                if not self._config_validate_runits(group, line, lc["runits"], line_name):
                    valid = False
            elif k == "direction":
                if lc["direction"] != ad:
                    valid = False
                    logger.error(f"unexpected settings of {line_name}:" f"direction = {ad} (!= {lc['direction']})")
            elif k in {"cnco_freq", "fnco_freq"}:
                if not self._config_validate_frequency(group, line, k, lc[k], alc[k]):
                    valid = False
                    logger.error(f"unexpected settings at {line_name}:{k} = {alc[k]} (!= {lc[k]})")
            elif k == "cnco_locked_with":
                dac_g, dac_l = lc[k]
                alf = alc["cnco_freq"]
                lf = self._css.get_dac_cnco(dac_g, dac_l)
                if lf != alf:
                    valid = False
                    logger.error(
                        f"unexpected settings at {line_name}:cnco_freq = {alf} (!= {lf}, "
                        f"that is cnco frequency of group-{dac_g}:line-{dac_l})"
                    )
            elif k == "fullscale_current":
                if not self._config_validate_fsc(group, cast(int, line), lc[k], alc[k]):
                    valid = False
                    logger.error(f"unexpected settings at {line_name}:{k} = {alc[k]} (!= {lc[k]})")
            else:
                if lc[k] != alc[k]:
                    valid = False
                    logger.error(f"unexpected settings at {line_name}:{k} = {alc[k]} (!= {lc[k]})")
        return valid

    def config_validate_box(self, box_conf: Dict[Tuple[int, Union[int, str]], Dict[str, Any]]) -> bool:
        valid: bool = True
        for (group, line), lc in box_conf.items():
            valid &= self._config_validate_line(group, line, lc)
        return valid

    def is_output_line(self, group: int, line: Union[int, str]):
        return isinstance(line, int)

    def is_input_line(self, group: int, line: Union[int, str]):
        return isinstance(line, str)

    def is_read_input_line(self, group: int, line: Union[int, str]):
        return line == "r"

    def is_monitor_input_line(self, group: int, line: Union[int, str]):
        return line == "m"

    def config_line(
        self,
        group: int,
        line: int,
        *,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
        fullscale_current: Union[int, None] = None,
        rfswitch: Union[str, None] = None,
    ) -> None:
        """configuring parameters of a given transmitter line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :param lo_freq: frequency of the corresponding local oscillator in Hz, must be multiple of 100_000_000.
        :param cnco_freq: frequency of the corresponding CNCO in Hz.
        :param vatt: controlling voltage of the corresponding VATT in unit of 3.3V / 4096. see the specification
                     sheet of the ADRF6780 for details.
        :param sideband: "U" for upper side band, "L" for lower side band.
        :param fullscale_current: full-scale current of output DAC of AD9082 in uA.
        :param rfswitch: state of RF switch, 'block' or 'pass' for output line, "loop" or "open" for input line
        :return: None
        """
        if vatt is not None:
            self._css.set_vatt(group, line, vatt)
        if sideband is not None:
            self._css.set_sideband(group, line, sideband)
        if fullscale_current is not None:
            self._css.set_fullscale_current(group, line, fullscale_current)
        if lo_freq is not None:
            if 15_000_000_000 >= lo_freq >= 7_500_000_000:
                if lo_freq % 100000000 != 0:
                    raise ValueError("lo_freq must be multiple of 100000000 if 7.5GHz <= lo_freq <= 15GHz")
                self._css.set_divider_ratio(group, line, 1)
                self._css.set_lo_multiplier(group, line, int(lo_freq) // 100000000)
            elif lo_freq >= 3_750_000_000:
                if lo_freq % 50000000 != 0:
                    raise ValueError("lo_freq must be multiple of 50000000 if 3.75GHz <= lo_freq < 7.5GHz")
                self._css.set_divider_ratio(group, line, 2)
                self._css.set_lo_multiplier(group, line, int(lo_freq) // 50000000)
            elif lo_freq >= 1_875_000_000:
                if lo_freq % 25000000 != 0:
                    raise ValueError("lo_freq must be multiple of 25000000 if 1.875GHz <= lo_freq < 3.75GHz")
                self._css.set_divider_ratio(group, line, 4)
                self._css.set_lo_multiplier(group, line, int(lo_freq) // 25000000)
            else:
                raise ValueError(f"invalid lo_freq: {lo_freq}Hz")
        if cnco_freq is not None:
            self._css.set_dac_cnco(group, line, cnco_freq)
        if rfswitch is not None:
            self.config_rfswitch(group, line, rfswitch=rfswitch)

    def config_channel(self, group: int, line: int, channel: int, *, fnco_freq: Union[float, None] = None) -> None:
        """configuring parameters of a given transmitter channel.

        :param group: a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :return: None
        """
        if fnco_freq is not None:
            self._css.set_dac_fnco(group, line, channel, fnco_freq)

    def config_rline(
        self,
        group: int,
        rline: str,
        *,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        cnco_locked_with: Union[Tuple[int, int], None] = None,
        rfswitch: Union[str, None] = None,
    ) -> None:
        """configuring parameters of a given receiver line.

        :param group: an index of a group which the line belongs to.
        :param rline: a group-local index of the line.
        :param lo_freq: frequency of the corresponding local oscillator in Hz, must be multiple of 100_000_000.
        :param cnco_freq: frequency of the corresponding CNCO in Hz.
        :param cnco_locked_with: frequency of CNCO is constrained to be identical to the specified line.
        :param rfswitch: state of RF switch, 'loop' or 'open'
        :return: None
        """
        if lo_freq is not None:
            if 15_000_000_000 >= lo_freq >= 7_500_000_000:
                if lo_freq % 100000000 != 0:
                    raise ValueError("lo_freq must be multiple of 100000000 if 7.5GHz <= lo_freq <= 15GHz")
                self._css.set_divider_ratio(group, rline, 1)
                self._css.set_lo_multiplier(group, rline, int(lo_freq) // 100000000)
            elif lo_freq >= 3_750_000_000:
                if lo_freq % 50000000 != 0:
                    raise ValueError("lo_freq must be multiple of 50000000 if 3.75GHz <= lo_freq < 7.5GHz")
                self._css.set_divider_ratio(group, rline, 2)
                self._css.set_lo_multiplier(group, rline, int(lo_freq) // 50000000)
            elif lo_freq >= 1_875_000_000:
                if lo_freq % 25000000 != 0:
                    raise ValueError("lo_freq must be multiple of 25000000 if 1.875GHz <= lo_freq < 3.75GHz")
                self._css.set_divider_ratio(group, rline, 4)
                self._css.set_lo_multiplier(group, rline, int(lo_freq) // 25000000)
            else:
                raise ValueError(f"invalid lo_freq: {lo_freq}Hz")
        if cnco_freq is not None and cnco_locked_with is not None:
            raise ValueError("it is not allowed to specify both cnco_freq and cnco_locked_with at the same time")
        if cnco_freq is not None:
            self._css.set_adc_cnco(group, rline, freq_in_hz=cnco_freq)
        if cnco_locked_with is not None:
            if isinstance(cnco_locked_with, tuple) and len(cnco_locked_with) == 2:
                dac_group, dac_line = cnco_locked_with
                self._css.set_pair_cnco(dac_group, dac_line, group, rline, self._css.get_dac_cnco(dac_group, dac_line))
            else:
                raise ValueError(f"invalid line to locked with: {cnco_locked_with}")
        if rfswitch is not None:
            self.config_rfswitch(group, rline, rfswitch=rfswitch)

    def config_runit(self, group: int, rline: str, runit: int, *, fnco_freq: Union[float, None] = None) -> None:
        """configuring parameters of a given receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param rline: a group-local index of a line which the channel belongs to.
        :param runit: a line-local index of the capture unit.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :return: None
        """
        if fnco_freq is not None:
            rchannel = self._rmap.get_rchannel_of_runit(group, rline, runit)
            self._css.set_adc_fnco(group, rline, rchannel, freq_in_hz=fnco_freq)

    def block_all_output_lines(self) -> None:
        """set RF switch of all output lines to block.

        :return:
        """
        for group in self._css.get_all_groups():
            for line in self._css.get_all_lines_of_group(group):
                self.config_rfswitch(group, line, rfswitch="block")

    def pass_all_output_lines(self):
        """set RF switch of all output lines to pass.

        :return:
        """
        for group in self._css.get_all_groups():
            for line in self._css.get_all_lines_of_group(group):
                self.config_rfswitch(group, line, rfswitch="pass")

    def config_rfswitches(self, rfswich_confs: Dict[Tuple[int, Union[int, str]], str], ignore_validation=False) -> None:
        """configuring multiple RF switches at the same time. contradictions among the given configuration is checked.

        :param rfswich_confs: a mapping between a line and the configuration of its RF switch.
        :param ignore_validation: raises exception the given configuration is not fulfilled if True (default).
        :return:
        """
        for (group, line), rc in rfswich_confs.items():
            self.config_rfswitch(group, line, rfswitch=rc)

        valid = True
        for (group, line), rc in rfswich_confs.items():
            arc = self.dump_rfswitch(group, line)
            if arc != rc:
                valid = False
                logger.warning(f"rfswitch[{group}, {line}] is finally set to {arc} (= {rc})")
        if not (ignore_validation or valid):
            raise ValueError("the specified configuration of rf switches is not realizable")

    def config_rfswitch(
        self, group: int, line: Union[int, str], *, rfswitch: str, reconfig_forcibly: bool = False
    ) -> None:
        """configuring a single RF switch of a given line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :param rfswitch: "block" or "pass" for an output line. "loop" or "open" for an input line.
        :return:
        """
        cc = self.dump_rfswitch(group, line)
        if reconfig_forcibly or (cc != rfswitch):
            if self._encode_rfswitch_conf(group, line, rfswitch):
                self.close_rfswitch(group, line)
            else:
                self.open_rfswitch(group, line)

    def open_rfswitch(self, group: int, line: Union[int, str]):
        """opening RF switch of the port corresponding to a given line, either of transmitter or receiver one.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        return None
        """
        if hasattr(self._css, "pass_line"):
            self._css.pass_line(group, line)
        else:
            logger.info("do nothing because no RF switches are available")

    def close_rfswitch(self, group: int, line: Union[int, str]):
        """closing RF switch of the port corresponding to a given line, either of transmitter or receiver one.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        return None
        """
        if hasattr(self._css, "block_line"):
            self._css.block_line(group, line)
        else:
            logger.info("do nothing because no RF switches are available")

    def _decode_rfswitch_conf(self, group: int, line: Union[int, str], block: bool) -> str:
        if self.is_output_line(group, line):
            return "block" if block else "pass"
        elif self.is_input_line(group, line):
            return "loop" if block else "open"
        else:
            raise ValueError(f"invalid group:{group}, line:{line})")

    def _encode_rfswitch_conf(self, group: int, line: Union[int, str], conf: str) -> bool:
        if self.is_output_line(group, line):
            if conf == "block":
                return True
            elif conf == "pass":
                return False
            else:
                raise ValueError(f"invalid configuration of an output switch: {conf}")
        elif self.is_input_line(group, line):
            if conf == "loop":
                return True
            elif conf == "open":
                return False
            else:
                raise ValueError(f"invalid configuration of an input switch: {conf}")
        else:
            raise ValueError(f"invalid group:{group}, line:{line}")

    def activate_monitor_loop(self, group: int) -> None:
        """enabling an internal monitor loop-back path from a monitor-out port to a monitor-in port.

        :param group: an index of a group which the monitor path belongs to.
        :return: None
        """
        if hasattr(self._css, "activate_monitor_loop"):
            self._css.activate_monitor_loop(group)
        else:
            logger.info("do nothing because no RF switches are available")

    def deactivate_monitor_loop(self, group: int) -> None:
        """disabling an internal monitor loop-back path.

        :param group: a group which the monitor path belongs to.
        :return: None
        """
        if hasattr(self._css, "deactivate_monitor_loop"):
            self._css.deactivate_monitor_loop(group)
        else:
            logger.info("do nothing because no RF switches are available")

    def is_loopedback_monitor(self, group: int) -> bool:
        """checking if an internal monitor loop-back path is activated or not.

        :param group: an index of a group which the monitor loop-back path belongs to.
        :return: True if the monitor loop-back path is activated.
        """
        if hasattr(self._css, "is_loopedback_monitor"):
            return self._css.is_loopedback_monitor(group)
        else:
            return False

    def activate_read_loop(self, group: int) -> None:
        """enabling an internal read loop-back path from read-out port to read-in port.

        :param group: an index of a group which the read path belongs to.
        :return: None
        """
        if hasattr(self._css, "activate_read_loop"):
            self._css.activate_read_loop(group)
        else:
            logger.info("do nothing because no RF switches are available")

    def deactivate_read_loop(self, group: int) -> None:
        """disabling an internal read loop-back.

        :param group: an index of a group which the read path belongs to.
        :return: None
        """
        if hasattr(self._css, "deactivate_read_loop"):
            self._css.deactivate_read_loop(group)
        else:
            logger.info("do nothing because no RF switches are available")

    def is_loopedback_read(self, group: int) -> bool:
        """checking if an internal read loop-back path is activated or not.

        :param group: an index of a group which the read loop-back path belongs to.
        :return: True if the read loop-back path is activated.
        """
        if hasattr(self._css, "is_loopedback_read"):
            return self._css.is_loopedback_read(group)
        else:
            return False

    def _dump_runit(
        self, group: int, rline: str, runit: int, rchannel_conf: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if rchannel_conf is None:
            rchannel_conf = self.css.dump_rchannel(group, rline, self.rmap.get_rchannel_of_runit(group, rline, runit))
        else:
            rchannel_conf = copy.copy(rchannel_conf)

        # Notes: currently contents of runit_settings and rchannel_setttings are identical.
        runit_conf = rchannel_conf
        return runit_conf

    def _rchannels_to_runits(
        self, group: int, rline: str, rchannels: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        runits: Dict[int, Dict[str, Any]] = {}
        for rmod, runit in self.rmap.get_capture_units_of_rline(group, rline):
            rch = self.rmap.get_rchannel_of_runit(group, rline, runit)
            runits[runit] = self._dump_runit(group, rline, runit, rchannels[rch])
        return runits

    def dump_rfswitch(self, group: int, line: Union[int, str]) -> str:
        """dumpling the current configuration of a single RF switch

        :param group: a index of group the line belongs to.
        :param line: a group-local index of the line.
        :return:  the current configuration of the switch of the line.
        """
        if hasattr(self._css, "is_blocked_line"):
            state: bool = self._css.is_blocked_line(group, line)
        else:
            # Notes: always passing or opening
            state = False
        return self._decode_rfswitch_conf(group, line, state)

    def dump_rfswitches(self, exclude_subordinate: bool = True) -> Dict[Tuple[int, Union[int, str]], str]:
        """dumpling the current configurations of all the RF switches.

        :param exclude_subordinate: avoid to have copies of the same RF switches if True (default).
        :return: a mapping between a line and the configuration of its corresponding RF switch.
        """
        retval: Dict[Tuple[int, Union[int, str]], str] = {}
        for group, line in self._css.get_all_any_lines():
            if not (
                exclude_subordinate
                and hasattr(self._css, "is_subordinate_rfswitch")
                and self._css.is_subordinate_rfswitch(group, line)
            ):
                retval[(group, line)] = self.dump_rfswitch(group, line)
        return retval

    def _dump_rline(self, group: int, rline: str) -> Dict[str, Any]:
        rl_conf = self._css.dump_rline(group, rline)
        rchannels = rl_conf["channels"]
        del rl_conf["channels"]
        rl_conf["runits"] = self._rchannels_to_runits(group, rline, rchannels)
        return rl_conf

    def dump_line(self, group: int, line: Union[int, str]) -> Dict[str, Any]:
        """dumping the current configuration of a single line.

        :return: the current configuration of the line in dictionary.
        """
        if self.is_output_line(group, line):
            retval: Dict[str, Any] = self._css.dump_line(group, cast(int, line))
        elif self.is_input_line(group, line):
            retval = self._dump_rline(group, cast(str, line))
        else:
            raise ValueError(f"invalid (group, line): ({group}, {line})")
        return retval

    def dump_box(self) -> Dict[str, Dict[Union[int, Tuple[int, Union[int, str]]], Dict[str, Any]]]:
        """dumping the current configuration of the box.

        :return: the current configuration of the box in dictionary.
        """
        retval: Dict[str, Dict[Union[int, Tuple[int, Union[int, str]]], Dict[str, Any]]] = {"mxfes": {}, "lines": {}}

        for mxfe_idx in self.css.get_all_mxfes():
            retval["mxfes"][mxfe_idx] = {
                "channel_interporation_rate": self._css.get_channel_interpolation_rate(mxfe_idx),
                "main_interporation_rate": self._css.get_main_interpolation_rate(mxfe_idx),
            }

        for group in self.css.get_all_groups():
            for line in self._css.get_all_lines_of_group(group):
                retval["lines"][(group, line)] = self._css.dump_line(group, line)
            for rline in self._css.get_all_rlines_of_group(group):
                retval["lines"][(group, rline)] = self._dump_rline(group, rline)

        return retval

    def get_channels_of_line(self, group: int, line: int) -> Set[int]:
        if self.is_output_line(group, line):
            return set(range(self.css.get_num_channels_of_line(group, line)))
        else:
            raise ValueError(f"invalid output line: ({group}, {line})")

    def get_runits_of_rline(self, group: int, rline: str) -> Set[int]:
        if self.is_input_line(group, rline):
            return set(range(len(self.rmap.get_capture_units_of_rline(group, rline))))
        else:
            raise ValueError(f"invalid input line: ({group}, {rline})")
