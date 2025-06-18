import copy
import logging
import time
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Final, Optional, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
from e7awghal import AwgParam, CapParam, E7FwType

from quel_ic_config.box_lock import BoxLockError
from quel_ic_config.e7resource_mapper import (
    AbstractQuel1E7ResourceMapper,
    create_rmap_object,
    validate_configuration_integrity,
)
from quel_ic_config.linkupper import LinkupFpgaMxfe
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_config_loader import Quel1ConfigLoader
from quel_ic_config.quel1_config_subsystem import (
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1BoxType,
    Quel1NecConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
)
from quel_ic_config.quel1_config_subsystem_common import NoRfSwitchError
from quel_ic_config.quel1_wave_subsystem import (
    AbstractCancellableTaskWrapper,
    AbstractStartAwgunitsTask,
    CapIqDataReader,
    Quel1WaveSubsystem,
    StartAwgunitsNowTask,
    StartAwgunitsTimedTask,
    StartCapunitsByTriggerTask,
    StartCapunitsNowTask,
)
from quel_ic_config.quel1se_adda_config_subsystem import Quel1seAddaConfigSubsystem, Quel2ProtoAddaConfigSubsystem
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
from quel_ic_config.quel_config_common import Quel1ConfigOption, Quel1Feature

logger = logging.getLogger(__name__)

Quel1LineType = tuple[int, Union[int, str]]


def _complete_ipaddrs(ipaddr_wss: str, ipaddr_sss: Union[str, None], ipaddr_css: Union[str, None]) -> Tuple[str, str]:
    if ipaddr_sss is None:
        ipaddr_sss = str(IPv4Address(ipaddr_wss) + (1 << 16))
    if ipaddr_css is None:
        ipaddr_css = str(IPv4Address(ipaddr_wss) + (4 << 16))
    return ipaddr_sss, ipaddr_css


def _create_css_object(
    ipaddr_css: str,
    boxtype: Quel1BoxType,
) -> Quel1AnyConfigSubsystem:
    if boxtype in {Quel1BoxType.QuBE_RIKEN_TypeA, Quel1BoxType.QuEL1_TypeA}:
        css: Quel1AnyConfigSubsystem = Quel1TypeAConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype in {Quel1BoxType.QuBE_RIKEN_TypeB, Quel1BoxType.QuEL1_TypeB}:
        css = Quel1TypeBConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuBE_OU_TypeA:
        css = QubeOuTypeAConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuBE_OU_TypeB:
        css = QubeOuTypeBConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL1_NEC:
        css = Quel1NecConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL1SE_Adda:
        css = Quel1seAddaConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL2_ProtoAdda:
        css = Quel2ProtoAddaConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL1SE_RIKEN8:
        css = Quel1seRiken8ConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL1SE_FUJITSU11_TypeA:
        css = Quel1seFujitsu11TypeAConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL1SE_FUJITSU11_TypeB:
        css = Quel1seFujitsu11TypeBConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL1SE_RIKEN8DBG:
        css = Quel1seRiken8DebugConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA:
        css = Quel1seFujitsu11TypeADebugConfigSubsystem(ipaddr_css, boxtype)
    elif boxtype == Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeB:
        css = Quel1seFujitsu11TypeBDebugConfigSubsystem(ipaddr_css, boxtype)
    else:
        raise ValueError(f"unsupported boxtype: {boxtype}")

    # TODO: should be moved to the right place.
    # Notes: for SockClients, this check is meaningless because lock acquisition is completed at the end of ctor.
    # Notes: for CoapClients, this looks required.
    for i in range(10):
        if i > 0:
            time.sleep(0.5)
        if css.has_lock:
            break
    else:
        del css
        raise BoxLockError(f"failed to acquire lock of {ipaddr_css}")

    return css  # noqa: F821 (avoiding a possible bug of pflake8)


def _dummy_auth_callback():
    # Notes: only for debug purpose. no access control will be applied.
    return True


def _create_wss_object(
    ipaddr_wss: str, ipaddr_sss: Optional[str] = None, css: Optional[Quel1AnyConfigSubsystem] = None
) -> Quel1WaveSubsystem:
    if css:
        auth_callback: Optional[Callable[[], bool]] = lambda: css.has_lock
    else:
        logger.warning(f"creating wss at {ipaddr_wss} without any access control.")
        auth_callback = _dummy_auth_callback

    return Quel1WaveSubsystem(ipaddr_wss, ipaddr_sss, auth_callback)


def create_css_wss_rmap(
    *,
    ipaddr_wss: str,
    ipaddr_sss: Union[str, None] = None,
    ipaddr_css: Union[str, None] = None,
    boxtype: Union[Quel1BoxType, str],
) -> tuple[Quel1AnyConfigSubsystem, Quel1WaveSubsystem, AbstractQuel1E7ResourceMapper]:
    ipaddr_sss, ipaddr_css = _complete_ipaddrs(ipaddr_wss, ipaddr_sss, ipaddr_css)
    if isinstance(boxtype, str):
        boxtype = Quel1BoxType.fromstr(boxtype)

    css: Quel1AnyConfigSubsystem = _create_css_object(ipaddr_css, boxtype)
    wss: Quel1WaveSubsystem = _create_wss_object(ipaddr_wss, ipaddr_sss, css)
    css.initialize()
    rmap: AbstractQuel1E7ResourceMapper = create_rmap_object(boxname=ipaddr_wss, fw_type=wss.fw_type)
    # Notes: wss will be initialized Boxi.initialize()
    return css, wss, rmap


class BoxIntrinsicStartCapunitsNowTask(
    AbstractCancellableTaskWrapper[dict[tuple[int, int], CapIqDataReader], dict[tuple[int, str, int], CapIqDataReader]]
):
    def __init__(self, task: StartCapunitsNowTask, mapping: dict[tuple[int, int], tuple[int, str, int]]):
        super().__init__(task)
        self._mapping = mapping

    def _conveter(self, orig: dict[tuple[int, int], CapIqDataReader]) -> dict[tuple[int, str, int], CapIqDataReader]:
        return {self._mapping[capunit]: reader for capunit, reader in orig.items()}


class BoxIntrinsicStartCapunitsByTriggerTask(
    AbstractCancellableTaskWrapper[dict[tuple[int, int], CapIqDataReader], dict[tuple[int, str, int], CapIqDataReader]]
):
    def __init__(self, task: StartCapunitsByTriggerTask, mapping: dict[tuple[int, int], tuple[int, str, int]]):
        super().__init__(task)
        self._mapping = mapping

    def _conveter(self, orig: dict[tuple[int, int], CapIqDataReader]) -> dict[tuple[int, str, int], CapIqDataReader]:
        return {self._mapping[capunit]: reader for capunit, reader in orig.items()}


class Quel1BoxIntrinsic:
    __slots__ = (
        "_css",
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
    def create(
        cls,
        *,
        ipaddr_wss: str,
        ipaddr_sss: Union[str, None] = None,
        ipaddr_css: Union[str, None] = None,
        boxtype: Quel1BoxType,
        skip_init: bool = False,
        **options: Collection[int],
    ) -> "Quel1BoxIntrinsic":
        """create QuEL intrinsic box objects
        :param ipaddr_wss: IP address of the wave generation subsystem of the target box
        :param ipaddr_sss: IP address of the sequencer subsystem of the target box (optional)
        :param ipaddr_css: IP address of the configuration subsystem of the target box (optional)
        :param boxtype: type of the target box
        :param config_root: root path of config setting files to read (optional)
        :param config_options: a collection of config options (optional)
        :param skip_init: skip calling box.initialization(), just for debugging.
        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored. (optional)
        :param ignore_access_failure_of_adrf6780: a list of ADRF6780 whose communication faiulre via SPI bus is
                                                  dismissed (optional)
        :param ignore_lock_failure_of_lmx2594: a list of LMX2594 whose lock failure is ignored (optional)
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed (optional)
        :return: SimpleBoxIntrinsic objects
        """
        css, wss, rmap = create_css_wss_rmap(
            ipaddr_wss=ipaddr_wss,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            boxtype=boxtype,
        )
        box = Quel1BoxIntrinsic(css=css, wss=wss, rmap=rmap, linkupper=None, **options)
        if not skip_init:
            box.initialize()
        return box

    # TODO: consider to re-locate to the right place
    def _validate_options(self, flags: Dict[str, Collection[int]]):
        num_ad9082 = self.css.get_num_ic("ad9082")
        num_adrf6780 = self.css.get_num_ic("adrf6780")
        num_lmx2594 = self.css.get_num_ic("lmx2594")

        for k, v in flags.items():
            if k == "ignore_crc_error_of_mxfe":
                if not all([0 <= u < num_ad9082 for u in v]):
                    raise ValueError(f"invalid index of mxfe is found in {k} (= {v})")
            elif k == "ignore_access_failure_of_adrf6780":
                if not all([0 <= u < num_adrf6780 for u in v]):
                    raise ValueError(f"invalid index of adrf6780 is found in {k} (= {v})")
            elif k == "ignore_lock_failure_of_lmx2594":
                if not all([0 <= u < num_lmx2594 for u in v]):
                    raise ValueError(f"invalid index of lmx2594 is found in {k} (= {v})")
            elif k == "ignore_extraordinary_converter_select_of_mxfe":
                if not all([0 <= u < num_ad9082 for u in v]):
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
        css: Quel1AnyConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: AbstractQuel1E7ResourceMapper,
        linkupper: Union[LinkupFpgaMxfe, None] = None,
        **options: Collection[int],
    ):
        self._css = css
        self._wss = wss
        self._rmap = rmap
        if linkupper is None:
            linkupper = LinkupFpgaMxfe(css, wss, rmap)
        self._linkupper = linkupper
        if options is None:
            options = {}
        self._validate_options(options)
        self._options: Dict[str, Collection[int]] = options

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._wss.ipaddr_wss}:{self.boxtype}>"

    @property
    def css(self) -> Quel1AnyConfigSubsystem:
        return self._css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._wss

    @property
    def rmap(self) -> AbstractQuel1E7ResourceMapper:
        return self._rmap

    @property
    def linkupper(self) -> LinkupFpgaMxfe:
        return self._linkupper

    @property
    def boxtype(self) -> str:
        return self.css.boxtype.tostr()

    @property
    def has_lock(self) -> bool:
        return self.css.has_lock

    @property
    def options(self) -> Dict[str, Collection[int]]:
        return self._options

    def initialize(self) -> None:
        self.wss.initialize()

    def reconnect(
        self,
        *,
        mxfe_list: Union[Collection[int], None] = None,
        skip_capture_check: bool = False,
        background_noise_threshold: Union[float, None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
        ignore_invalid_linkstatus: bool = False,
    ) -> Dict[int, bool]:
        """establish a configuration link between a box and host.
        the target box needs to be linked-up in advance.

        :param mxfe_list: a list of MxFEs to reconnect. (optional)
        :param skip_capture_check: do not check background noise of input lines if True (optional)
        :param background_noise_threshold: the largest peak of allowable noise (optional)
        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored. (optional)
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed. (optional)
        :return: True if success
        """

        # Notes: boxi.initialize() is called at BoxIntrinsic.create() unless skip_init is True.
        #        this means that all AWG units and Capture units are initialized (and should not be working).
        #        reconnect() is usually called just after boxi.create() and it is reasonable to assume that all the
        #        components of WSS are quiet.
        #        theoretically, reconnect() can be called any time in the user's code, but do not initialize WSS here
        #        to avoid unwanted destruction of configuration settings.

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
            try:
                valid_link: bool = self._css.reconnect_mxfe(
                    mxfe_idx, ignore_crc_error=mxfe_idx in ignore_crc_error_of_mxfe
                )
                if valid_link:
                    self._css.validate_chip_id(mxfe_idx)
                    validate_configuration_integrity(
                        self.css.get_virtual_adc_select(mxfe_idx),
                        self.wss.fw_type,
                        ignore_extraordinary_converter_select=mxfe_idx in ignore_extraordinary_converter_select_of_mxfe,
                    )
            except RuntimeError as e:
                logger.warning(e)
                valid_link = False

            if not valid_link:
                if not ignore_invalid_linkstatus:
                    logger.error(f"JESD204C link of AD9082-#{mxfe_idx} is not working. it must be linked up")
            link_ok[mxfe_idx] = valid_link

        # Notes: checking so-called chikuchiku problem caused by wss firmware bug.
        if not skip_capture_check and all(link_ok.values()):
            rfsw_restore = {k: "open" for k, v in self.dump_rfswitches().items() if v == "open"}
            rfsw_closed = {k: "loop" for k in rfsw_restore}
            try:
                self.config_rfswitches(rfsw_closed)
            except NoRfSwitchError:
                logger.info(
                    "background noise check may be disrupted by incoming signal "
                    "due to the unavailability of RF switches"
                )
            for mxfe_idx in mxfe_list:
                if not self.linkupper.check_all_fddcs_of_mxfe_at_reconnect(mxfe_idx, background_noise_threshold):
                    link_ok[mxfe_idx] = False
                    if not ignore_invalid_linkstatus:
                        logger.error(
                            f"JESD204C tx-link of AD9082-#{mxfe_idx} is not working properly, it must be linked up"
                        )
                    # Notes: do not break here even if here to check all the MxFEs to show information.
            self.config_rfswitches(rfsw_restore)

        return link_ok

    def get_wss_features(self) -> set[Quel1Feature]:
        features: set[Quel1Feature] = set()
        if self.wss.fw_type in {E7FwType.SIMPLEMULTI_CLASSIC}:
            features.add(Quel1Feature.SINGLE_ADC)
        elif self.wss.fw_type in {E7FwType.FEEDBACK_VERYEARLY}:
            features.add(Quel1Feature.BOTH_ADC_EARLY)
        elif self.wss.fw_type in {E7FwType.SIMPLEMULTI_STANDARD, E7FwType.FEEDBACK_EARLY}:
            features.add(Quel1Feature.BOTH_ADC)
        else:
            raise ValueError(f"unsupported firmware is detected: {self.wss.fw_type}")
        return features

    def _load_config_parameter(
        self,
        *,
        config_root: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
    ) -> Dict[str, Any]:
        config_options = config_options or set()
        loader = Quel1ConfigLoader(
            boxtype=self.css.boxtype,
            num_ic=self.css.get_num_ics(),
            config_options=config_options,
            features=self.get_wss_features(),
            config_filename=self.css.get_default_config_filename(),
            config_rootpath=config_root,
        )
        return loader.load_config()

    def hardreset_wss_units(
        self,
        mxfes: Union[Collection[int], None] = None,
        reset_awg: bool = True,
        reset_cap: bool = True,
        suppress_warning: bool = False,
    ) -> None:
        # Notes: an only way to clear hardware error flag of wss is hard-resetting the corresponding units.
        #        however, its soundness is not confirmed well yet. at least, it often destroys JESD204C link and needs
        #        the re-linkup. it is still unclear that re-linkup recovers the internal consistency again or not.
        #        be careful if you use it.
        if mxfes is None:
            mxfes = self._css.get_all_mxfes()

        if reset_awg:
            for mxfe in mxfes:
                for awg_idx in self._rmap.get_awgs_of_mxfe(mxfe):
                    self._wss._get_awgunit(awg_idx).hard_reset(suppress_warning=suppress_warning)

        if reset_cap:
            fddcs: Set[Tuple[int, int]] = set()
            for g in self._css.get_all_groups():
                for rl in self._css.get_all_rlines_of_group(g):
                    for rch in range(self._css.get_num_rchannels_of_rline(g, rl)):
                        m, d = self._css.get_fddc_idx(g, rl, rch)
                        fddcs.add((m, d))

            for mxfe, fddc in fddcs:
                if mxfe in mxfes:
                    cm_idx = self._rmap.get_capmod_from_fddc(mxfe, fddc)
                    for cu_idx in range(self._wss.num_capunit_of_capmod(cm_idx)):
                        self._wss._get_capunit((cm_idx, cu_idx)).hard_reset()

        if reset_awg or reset_cap:
            # TODO: confirm the effectiveness.
            logger.info("waiting for 15 seconds after hard-resetting wave subsystem")
            time.sleep(15)

    def relinkup(
        self,
        *,
        param: Union[Dict[str, Any], None] = None,
        config_root: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        mxfes_to_linkup: Union[Collection[int], None] = None,
        hard_reset: Union[bool, None] = None,
        use_204b: bool = False,
        use_bg_cal: bool = True,
        skip_init: bool = False,
        hard_reset_wss: bool = False,
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
            hard_reset = self.css.boxtype.is_quel1se()
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

        # Notes: user can inject a parameter object directly instead of loading it from files.
        param = param or self._load_config_parameter(config_root=config_root, config_options=config_options)

        if not skip_init:
            self.css.configure_peripherals(
                param,
                ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
                ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            )
            self.css.configure_all_mxfe_clocks(
                param,
                ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            )

        if hard_reset_wss:
            self.hardreset_wss_units(mxfes_to_linkup, suppress_warning=True)

        linkup_ok: Dict[int, bool] = {}
        for mxfe in mxfes_to_linkup:
            linkup_ok[mxfe] = self.linkupper.linkup_and_check(
                mxfe,
                param,
                hard_reset=hard_reset,
                use_204b=use_204b,
                use_bg_cal=use_bg_cal,
                background_noise_threshold=background_noise_threshold,
                ignore_crc_error=mxfe in ignore_crc_error_of_mxfe,
                ignore_extraordinary_converter_select=mxfe in ignore_extraordinary_converter_select_of_mxfe,
                restart_tempctrl=restart_tempctrl,
            )
        return linkup_ok

    def link_status(self, *, ignore_crc_error_of_mxfe: Union[Collection[int], None] = None) -> Dict[int, bool]:
        if ignore_crc_error_of_mxfe is None:
            ignore_crc_error_of_mxfe = self._options["ignore_crc_error_of_mxfe"]

        link_health: Dict[int, bool] = {}
        for mxfe_idx in self.css.get_all_mxfes():
            link_health[mxfe_idx], log_level, diag = self.css.check_link_status(
                mxfe_idx, False, ignore_crc_error=(mxfe_idx in ignore_crc_error_of_mxfe)
            )
            logger.log(log_level, diag)

        return link_health

    def _get_rchannel_from_runit(self, group: int, rline: str, runit: int) -> int:
        # TODO: implement a method to connect runit and rchannel in this (BoxIntrinsic) layer.
        if runit in self.get_runits_of_rline(group, rline):
            return 0
        else:
            raise ValueError(f"invalid runit:{runit} for group:{group}, rline:{rline}")

    def _get_awg_from_channel(self, group: int, line: int, channel: int) -> int:
        mxfe_idx, fduc_idx = self.css.get_fduc_idx(group, line, channel)
        return self.rmap.get_awg_from_fduc(mxfe_idx, fduc_idx)

    def _get_awgs_from_channels(self, channels: Collection[Tuple[int, int, int]]):
        return {self._get_awg_from_channel(gr, ln, ch) for gr, ln, ch in channels}

    def _get_capmod_from_rchannel(self, group: int, rline: str, rchannel: int) -> int:
        mxfe_idx, fddc_idx = self.css.get_fddc_idx(group, rline, rchannel)
        return self.rmap.get_capmod_from_fddc(mxfe_idx, fddc_idx)

    def _get_capunit_from_runit(self, group: int, rline: str, runit: int) -> tuple[int, int]:
        rchannel = self._get_rchannel_from_runit(group, rline, runit)
        # Notes: runit --> capunit doens't work in near future!
        return (self._get_capmod_from_rchannel(group, rline, rchannel), runit)

    def _get_capunits_from_runits(self, runits: Collection[tuple[int, str, int]]) -> set[tuple[int, int]]:
        return {self._get_capunit_from_runit(*runit) for runit in runits}

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
        return self._css.is_equal_fullscale_current(group, line, fsc0, fsc1)

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
        return (group in self.css.get_all_groups()) and isinstance(line, int)

    def is_output_channel(self, group: int, line: Union[int, str], channel: int):
        return (
            (group in self.css.get_all_groups())
            and isinstance(line, int)
            and (channel in self.get_channels_of_line(group, line))
        )

    def is_input_line(self, group: int, line: Union[int, str]):
        return (group in self.css.get_all_groups()) and isinstance(line, str)

    def is_input_runit(self, group: int, rline: Union[int, str], runit: int):
        return (
            (group in self.css.get_all_groups())
            and isinstance(rline, str)
            and (runit in self.get_runits_of_rline(group, rline))
        )

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

    def config_channel(
        self,
        group: int,
        line: int,
        channel: int,
        *,
        fnco_freq: Union[float, None] = None,
        awg_param: Union[AwgParam, None] = None,
    ) -> None:
        """configuring parameters of a given transmitter channel.

        :param group: a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :param awg_param: an object holding parameters of signal to be generated.
        :return: None
        """
        if fnco_freq is not None:
            self._css.set_dac_fnco(group, line, channel, fnco_freq)

        if awg_param is not None:
            self._wss.config_awgunit(self._get_awg_from_channel(group, line, channel), awg_param)

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

    def config_runit(
        self,
        group: int,
        rline: str,
        runit: int,
        *,
        fnco_freq: Union[float, None] = None,
        capture_param: Union[CapParam, None] = None,
    ) -> None:
        """configuring parameters of a given receiver channel.

        :param group: an index of a group which the channel belongs to.
        :param rline: a group-local index of a line which the channel belongs to.
        :param runit: a line-local index of the capture unit.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :param capture_param: an object holding capture settings.
        :return: None
        """
        if fnco_freq is not None:
            rchannel = self._get_rchannel_from_runit(group, rline, runit)
            self._css.set_adc_fnco(group, rline, rchannel, freq_in_hz=fnco_freq)

        if capture_param is not None:
            capunit = self._get_capunit_from_runit(group, rline, runit)
            self._wss.config_capunit(capunit, capture_param)

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
        self._css.pass_line(group, line)

    def close_rfswitch(self, group: int, line: Union[int, str]):
        """closing RF switch of the port corresponding to a given line, either of transmitter or receiver one.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        return None
        """
        self._css.block_line(group, line)

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
        self._css.activate_monitor_loop(group)

    def deactivate_monitor_loop(self, group: int) -> None:
        """disabling an internal monitor loop-back path.

        :param group: a group which the monitor path belongs to.
        :return: None
        """
        self._css.deactivate_monitor_loop(group)

    def is_loopedback_monitor(self, group: int) -> bool:
        """checking if an internal monitor loop-back path is activated or not.

        :param group: an index of a group which the monitor loop-back path belongs to.
        :return: True if the monitor loop-back path is activated.
        """
        return self._css.is_loopedback_monitor(group)

    def activate_read_loop(self, group: int) -> None:
        """enabling an internal read loop-back path from read-out port to read-in port.

        :param group: an index of a group which the read path belongs to.
        :return: None
        """
        self._css.activate_read_loop(group)

    def deactivate_read_loop(self, group: int) -> None:
        """disabling an internal read loop-back.

        :param group: an index of a group which the read path belongs to.
        :return: None
        """
        self._css.deactivate_read_loop(group)

    def is_loopedback_read(self, group: int) -> bool:
        """checking if an internal read loop-back path is activated or not.

        :param group: an index of a group which the read loop-back path belongs to.
        :return: True if the read loop-back path is activated.
        """
        return self._css.is_loopedback_read(group)

    def _dump_runit(
        self, group: int, rline: str, runit: int, rchannel_conf: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if rchannel_conf is None:
            rchannel_conf = self.css.dump_rchannel(group, rline, self._get_rchannel_from_runit(group, rline, runit))
        else:
            rchannel_conf = copy.copy(rchannel_conf)

        # Notes: currently contents of runit_settings and rchannel_setttings are identical.
        runit_conf = rchannel_conf
        return runit_conf

    def dump_rfswitch(self, group: int, line: Union[int, str]) -> str:
        """dumpling the current configuration of a single RF switch

        :param group: a index of group the line belongs to.
        :param line: a group-local index of the line.
        :return:  the current configuration of the switch of the line.
        """
        state: bool = self._css.is_blocked_line(group, line)
        return self._decode_rfswitch_conf(group, line, state)

    def dump_rfswitches(self, exclude_subordinate: bool = True) -> Dict[Tuple[int, Union[int, str]], str]:
        """dumpling the current configurations of all the RF switches.

        :param exclude_subordinate: avoid to have copies of the same RF switches if True (default).
        :return: a mapping between a line and the configuration of its corresponding RF switch.
        """
        retval: Dict[Tuple[int, Union[int, str]], str] = {}
        for group, line in self._css.get_all_any_lines():
            if not (exclude_subordinate and self._css.is_subordinate_rfswitch(group, line)):
                retval[(group, line)] = self.dump_rfswitch(group, line)
        return retval

    def _dump_rline(self, group: int, rline: str) -> Dict[str, Any]:
        rl_conf = self._css.dump_rline(group, rline)
        rc_conf = rl_conf["channels"]
        del rl_conf["channels"]
        ru_conf: Dict[int, Any] = {}
        for runit in self.get_runits_of_rline(group, rline):
            ru_conf[runit] = self._dump_runit(
                group, rline, runit, rc_conf[self._get_rchannel_from_runit(group, rline, runit)]
            )
        rl_conf["runits"] = ru_conf
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
        # TODO: refine the implementation later
        if self.is_input_line(group, rline):
            num_capunt = 0
            for rch in range(self.css.get_num_rchannels_of_rline(group, rline)):
                capmod = self._get_capmod_from_rchannel(group, rline, rch)
                num_capunt += self.wss.num_capunit_of_capmod(capmod)
            return set(range(num_capunt))
        else:
            raise ValueError(f"invalid input line: ({group}, {rline})")

    def get_groups(self) -> set[int]:
        return self._css.get_all_groups()

    def get_output_lines(self) -> set[tuple[int, int]]:
        ol: set[tuple[int, int]] = set()
        for g in self._css.get_all_groups():
            for ln in self._css.get_all_lines_of_group(g):
                ol.add((g, ln))
        return ol

    def get_input_rlines(self) -> set[tuple[int, str]]:
        ol: set[tuple[int, str]] = set()
        for g in self._css.get_all_groups():
            for ln in self._css.get_all_rlines_of_group(g):
                ol.add((g, ln))
        return ol

    def get_names_of_wavedata(self, group: int, line: int, channel: int) -> set[str]:
        awgunit_idx = self._get_awg_from_channel(group, line, channel)
        return self.wss.get_names_of_wavedata(awgunit_idx)

    def register_wavedata(
        self,
        group: int,
        line: int,
        channel: int,
        name: str,
        iq: npt.NDArray[np.complex64],
        allow_update: bool = True,
        **kwdargs,
    ) -> None:
        awgunit_idx = self._get_awg_from_channel(group, line, channel)
        self.wss.register_wavedata(awgunit_idx, name, iq, allow_update=allow_update, **kwdargs)

    def has_wavedata(self, group: int, line: int, channel: int, name: str) -> bool:
        awgunit_idx = self._get_awg_from_channel(group, line, channel)
        return self.wss.has_wavedata(awgunit_idx, name)

    def delete_wavedata(self, group: int, line: int, channel: int, name: str) -> None:
        awgunit_idx = self._get_awg_from_channel(group, line, channel)
        self.wss.delete_wavedata(awgunit_idx, name)

    def initialize_all_awgunits(self) -> None:
        self.wss.initialize_all_awgunits()

    def initialize_all_capunits(self) -> None:
        self.wss.initialize_all_capunits()

    def get_current_timecounter(self) -> int:
        return self.wss.get_current_timecounter()

    def get_latest_sysref_timecounter(self) -> int:
        return self.wss.get_latest_sysref_timecounter()

    def get_averaged_sysref_offset(self, num_iteration: int = 100) -> float:
        return self.wss.get_averaged_sysref_offset(num_iteration)

    def start_wavegen(
        self,
        channels: Collection[Tuple[int, int, int]],
        timecounter: Optional[int] = None,
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
        return_after_start_emission: Optional[bool] = None,
    ) -> AbstractStartAwgunitsTask:
        if len(channels) == 0:
            raise ValueError("no triggering channel are specified")
        for ch in channels:
            if not (isinstance(ch, tuple) and len(ch) == 3 and self.is_output_channel(*ch)):
                raise ValueError("invalid channel: {ch}")

        if timecounter and timecounter < 0:
            raise ValueError("negative timecounter is not allowed")
        if timeout and timeout <= 0.0:
            raise ValueError("non-positive timeout is not allowed")
        if polling_period and polling_period <= 0.0:
            raise ValueError("non-positive polling_period is not allowed")

        awgunit_idxs: Collection[int] = self._get_awgs_from_channels(channels)
        if timecounter is None:
            return self.wss.start_awgunits_now(
                awgunit_idxs,
                timeout=timeout,
                polling_period=polling_period,
                disable_timeout=disable_timeout,
                return_after_start_emission=return_after_start_emission or True,
            )
        else:
            return self.wss.start_awgunits_timed(
                awgunit_idxs,
                timecounter,
                timeout=timeout,
                polling_period=polling_period,
                disable_timeout=disable_timeout,
                return_after_start_emission=return_after_start_emission or False,
            )

    def start_capture_now(
        self,
        runits: Collection[Tuple[int, str, int]],
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ) -> BoxIntrinsicStartCapunitsNowTask:
        if len(runits) == 0:
            raise ValueError("no capture units are specified")
        for ru in runits:
            if not (isinstance(ru, tuple) and len(ru) == 3 and self.is_input_runit(*ru)):
                raise ValueError("invalid runit: {ru}")

        if timeout and timeout <= 0.0:
            raise ValueError("non-positive timeout is not allowed")
        if polling_period and polling_period <= 0.0:
            raise ValueError("non-positive polling_period is not allowed")

        mapping: dict[tuple[int, int], tuple[int, str, int]] = {}
        for ru in runits:
            cu = self._get_capunit_from_runit(*ru)
            mapping[cu] = ru

        capunit_idxs = set(mapping.keys())
        return BoxIntrinsicStartCapunitsNowTask(
            self.wss.start_capunits_now(
                capunit_idxs, timeout=timeout, polling_period=polling_period, disable_timeout=disable_timeout
            ),
            mapping,
        )

    def start_capture_by_awg_trigger(
        self,
        runits: Collection[Tuple[int, str, int]],
        channels: Collection[Tuple[int, int, int]],
        timecounter: Optional[int] = None,
        *,
        timeout_before_trigger: Optional[float] = None,
        timeout_after_trigger: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ) -> tuple[BoxIntrinsicStartCapunitsByTriggerTask, AbstractStartAwgunitsTask]:
        if len(runits) == 0:
            raise ValueError("no capture units are specified")
        for ru in runits:
            if not (isinstance(ru, tuple) and len(ru) == 3 and self.is_input_runit(*ru)):
                raise ValueError("invalid runit: {ru}")
        if len(channels) == 0:
            raise ValueError("no triggering channel are specified")
        for ch in channels:
            if not (isinstance(ch, tuple) and len(ch) == 3 and self.is_output_channel(*ch)):
                raise ValueError("invalid channel: {ch}")
        if timecounter and timecounter < 0:
            raise ValueError("negative timecounter is not allowed")
        if timeout_before_trigger and timeout_before_trigger <= 0.0:
            raise ValueError("non-positive timeout_before_trigger is not allowed")
        if timeout_after_trigger and timeout_after_trigger <= 0.0:
            raise ValueError("non-positive timeout_after_trigger is not allowed")
        if polling_period and polling_period <= 0.0:
            raise ValueError("non-positive polling_period is not allowed")

        mapping: dict[tuple[int, int], tuple[int, str, int]] = {}
        capmod_idxs: set[int] = set()
        for ru in runits:
            cu = self._get_capunit_from_runit(*ru)
            mapping[cu] = ru
            capmod_idxs.add(cu[0])

        # Notes: any channel is fine since they all will start at the same time.
        trigger_idx = self._get_awg_from_channel(*list(channels)[0])
        for capmod_idx in capmod_idxs:
            self.wss.set_triggering_awg_to_line(capmod_idx, trigger_idx)

        if timecounter:
            cur = self.wss.get_current_timecounter()
            delta = self.wss.timecounter_to_second(timecounter - cur)
            if delta < StartAwgunitsTimedTask._TRIGGER_SETTABLE_MARGIN:
                raise RuntimeError(f"cannot schedule at the past or too close timecounter (= {timecounter})")
            if timeout_before_trigger is None:
                timeout_before_trigger = delta + StartAwgunitsTimedTask._START_TIMEOUT_MARGIN
        else:
            if timeout_before_trigger is None:
                timeout_before_trigger = StartAwgunitsNowTask._START_TIMEOUT

        capunit_idxs = set(mapping.keys())
        cap_task = BoxIntrinsicStartCapunitsByTriggerTask(
            self.wss.start_capunits_by_trigger(
                capunit_idxs,
                timeout_before_trigger=timeout_before_trigger,
                timeout_after_trigger=timeout_after_trigger,
                polling_period=polling_period,
                disable_timeout=disable_timeout,
            ),
            mapping,
        )

        gen_task: AbstractStartAwgunitsTask = self.start_wavegen(channels, timecounter, polling_period=polling_period)
        return cap_task, gen_task
