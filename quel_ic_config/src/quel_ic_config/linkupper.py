import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Set, Tuple, Union

import numpy as np
from e7awghal import CapParam, CapSection, E7awgCaptureDataError

from quel_ic_config.ad9082 import LinkStatus
from quel_ic_config.e7resource_mapper import AbstractQuel1E7ResourceMapper
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_config_subsystem_tempctrl import Quel1seTempctrlState
from quel_ic_config.quel1_wave_subsystem import Quel1WaveSubsystem

logger = logging.getLogger(__name__)


class LinkupStatus(Enum):
    LINKUP_FAILURE = 1
    CRC_ERROR = 2
    RX_TIMEOUT = 3
    RX_FAILURE = 4
    EMPTY_RX_DATA = 5
    BROKEN_RX_DATA = 6
    CHIKUCHIKU = 7
    GOOD_CAPTURE = 8


@dataclass(frozen=True)
class LinkupStatistic:
    link_status: LinkStatus
    error_status: int
    fddc_idx: int
    timeout: bool
    failure: bool
    samples: int
    valid_capture: bool
    max_noise_peak: float

    def categorize(self, noise_threshold: float) -> LinkupStatus:
        if self.link_status != LinkStatus.LINK_STATUS_LOCKED:
            return LinkupStatus.LINKUP_FAILURE
        elif self.error_status != 0x01:
            return LinkupStatus.CRC_ERROR
        elif self.failure:
            return LinkupStatus.RX_FAILURE
        elif self.timeout:
            return LinkupStatus.RX_TIMEOUT
        elif not self.valid_capture:
            if self.samples == 0:
                return LinkupStatus.EMPTY_RX_DATA
            else:
                return LinkupStatus.BROKEN_RX_DATA
        elif self.max_noise_peak >= noise_threshold:
            return LinkupStatus.CHIKUCHIKU
        else:
            return LinkupStatus.GOOD_CAPTURE

    def is_success(self, noise_threshold: float) -> bool:
        return self.categorize(noise_threshold) == LinkupStatus.GOOD_CAPTURE


class LinkupFpgaMxfe:
    _LINKUP_MAX_RETRY: Final[int] = 10
    _SLEEP_BTWN_LINKUP_TRIALS: Final[float] = 0.25
    _DEFAULT_BACKGROUND_NOISE_THRESHOLD_RELINKUP: Final[float] = 256.0
    _DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT: Final[float] = 4096.0
    _STAT_HISTORY_MAX_LEN: Final[int] = 1000

    def __init__(self, css: Quel1AnyConfigSubsystem, wss: Quel1WaveSubsystem, rmap: AbstractQuel1E7ResourceMapper):
        self._css = css
        self._wss = wss
        self._rmap: AbstractQuel1E7ResourceMapper = rmap
        self._statistics: Dict[int, List[LinkupStatistic]] = {}
        self._target_capmods: Dict[int, Set[int]] = {}

    def _validate_mxfe(self, mxfe_idx: int) -> bool:
        return mxfe_idx in self._css.get_all_mxfes()

    def _get_fddcs_of_mxfe(self, mxfe_idx: int) -> Set[int]:
        fddcs: Set[int] = set()
        for g in self._css.get_all_groups():
            for rl in self._css.get_all_rlines_of_group(g):
                for rch in range(self._css.get_num_rchannels_of_rline(g, rl)):
                    m, d = self._css.get_fddc_idx(g, rl, rch)
                    if m == mxfe_idx:
                        fddcs.add(d)
        return fddcs

    def _add_linkup_statistics(
        self,
        mxfe_idx: int,
        fddc_idx: int,
        timeout: bool,
        failure: bool,
        samples: int,
        valid_capture: bool,
        max_noise_peak: float,
    ):
        link_status, crc_error_status = self._css.get_link_status(mxfe_idx)
        if mxfe_idx not in self._statistics:
            self._statistics[mxfe_idx] = []

        self._statistics[mxfe_idx].append(
            LinkupStatistic(
                link_status=link_status,
                error_status=crc_error_status,
                fddc_idx=fddc_idx,
                timeout=timeout,
                failure=failure,
                samples=samples,
                valid_capture=valid_capture,
                max_noise_peak=max_noise_peak,
            )
        )

        if len(self._statistics[mxfe_idx]) > self._STAT_HISTORY_MAX_LEN:
            self._statistics[mxfe_idx] = self._statistics[mxfe_idx][-self._STAT_HISTORY_MAX_LEN :]  # noqa: E203

    def clear_statistics(self, mxfe_idx: Union[None, int] = None):
        if mxfe_idx is None:
            self._statistics = {}
        else:
            self._validate_mxfe(mxfe_idx)
            if mxfe_idx not in self._statistics:
                logger.warning(f"no data exist for mxfe-#{mxfe_idx}, do nothing actually")
            self._statistics[mxfe_idx] = []

    @property
    def linkup_statistics(self):
        return self._statistics

    def init_wss_resources(self, mxfe_idx: int) -> None:
        self._validate_mxfe(mxfe_idx)
        self._wss.initialize_awgunits(self._rmap.get_awgs_of_mxfe(mxfe_idx))
        capunits: Set[Tuple[int, int]] = set()
        for fddc_idx in self._get_fddcs_of_mxfe(mxfe_idx):
            capunits.add((self._rmap.get_capmod_from_fddc(mxfe_idx, fddc_idx), 0))
        self._wss.initialize_capunits(capunits)

    def linkup_and_check(
        self,
        mxfe_idx: int,
        param: Dict[str, Any],
        *,
        hard_reset: bool = False,
        soft_reset: bool = True,
        use_204b: bool = False,
        use_bg_cal: bool = True,
        ignore_crc_error: bool = False,
        ignore_extraordinary_converter_select: bool = False,
        background_noise_threshold: Union[float, None] = None,
        restart_tempctrl: bool = False,
        save_dirpath: Union[Path, None] = None,
    ) -> bool:
        self._validate_mxfe(mxfe_idx)
        if background_noise_threshold is None:
            background_noise_threshold = self._DEFAULT_BACKGROUND_NOISE_THRESHOLD_RELINKUP

        judge_system: bool = False
        for i in range(self._LINKUP_MAX_RETRY):
            if i != 0:
                logger.debug(f"waiting {self._SLEEP_BTWN_LINKUP_TRIALS} seconds before retrying linkup")
                time.sleep(self._SLEEP_BTWN_LINKUP_TRIALS)

            if not self._css.configure_mxfe(
                mxfe_idx,
                param,
                hard_reset=hard_reset,
                soft_reset=soft_reset,
                use_204b=use_204b,
                use_bg_cal=use_bg_cal,
                ignore_crc_error=ignore_crc_error,
            ):
                self._add_linkup_statistics(
                    mxfe_idx,
                    fddc_idx=-1,
                    timeout=False,
                    failure=False,
                    samples=-1,
                    valid_capture=False,
                    max_noise_peak=-1,
                )
                continue

            self.init_wss_resources(mxfe_idx)

            # Notes: it is fine to check all the available adcs of the target mxfe.
            if self.check_all_fddcs_of_mxfe_at_relinkup(mxfe_idx, background_noise_threshold, save_dirpath):
                logger.info(f"successful system-level link-up of {self._css.ipaddr_css}:mxfe-#{mxfe_idx}")
                judge_system = True
                break
            else:
                # Notes: info is enough
                logger.info(f"failed system-level link-up of {self._css.ipaddr_css}:mxfe-#{mxfe_idx}")

        if judge_system:
            if self._css.tempctrl_auto_start_at_linkup:
                self.start_temperature_control(restart_tempctrl)

        return judge_system

    def check_all_fddcs_of_mxfe_at_relinkup(
        self,
        mxfe_idx: int,
        background_noise_threshold: Union[float, None] = None,
        save_dirpath: Union[Path, None] = None,
    ) -> bool:
        judge_fddcs: bool = True
        for fddc_idx in self._get_fddcs_of_mxfe(mxfe_idx):
            judge_fddcs &= self.check_fddc(
                mxfe_idx,
                fddc_idx,
                background_noise_threshold or self._DEFAULT_BACKGROUND_NOISE_THRESHOLD_RELINKUP,
                save_dirpath,
            )
        return judge_fddcs

    def check_all_fddcs_of_mxfe_at_reconnect(
        self,
        mxfe_idx: int,
        background_noise_threshold: Union[float, None] = None,
        save_dirpath: Union[Path, None] = None,
    ) -> bool:
        judge_fddcs: bool = True
        for fddc_idx in self._get_fddcs_of_mxfe(mxfe_idx):
            judge_fddcs &= self.check_fddc(
                mxfe_idx,
                fddc_idx,
                background_noise_threshold or self._DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT,
                save_dirpath,
            )
        return judge_fddcs

    def check_fddc(
        self,
        mxfe_idx: int,
        fddc_idx: int,
        background_noise_threshold: float,
        save_dirpath: Union[Path, None] = None,
    ) -> bool:
        capmod = self._rmap.get_capmod_from_fddc(mxfe_idx, fddc_idx)
        capunit = (capmod, 0)
        param = CapParam(num_repeat=1)
        param.sections.append(CapSection(name="s0", num_capture_word=16384, num_blank_word=1))
        self._wss.config_capunit(capunit, param)
        task = self._wss.start_capunits_now({capunit})
        try:
            reader = task.result()[capunit]
        except TimeoutError as e:
            logger.warning(e)
            self._add_linkup_statistics(
                mxfe_idx,
                fddc_idx=fddc_idx,
                timeout=True,
                failure=False,
                samples=-1,
                valid_capture=False,
                max_noise_peak=-1,
            )
            return False
        except E7awgCaptureDataError as e:
            logger.warning(e)
            self._add_linkup_statistics(
                mxfe_idx,
                fddc_idx=fddc_idx,
                timeout=False,
                failure=False,
                samples=self._wss._get_capunit(capunit).get_num_captured_sample(),  # TODO: improve the design.
                valid_capture=False,
                max_noise_peak=-1,
            )
            return False

        cap_data = reader.as_wave_dict()["s0"][0]
        if save_dirpath is not None:
            os.makedirs(save_dirpath, exist_ok=True)
            np.save(str(save_dirpath / f"backgroud_{mxfe_idx}_{fddc_idx}_{int(time.time())}.npy"), cap_data)

        max_backgroud_amplitude = max(abs(cap_data))
        if max_backgroud_amplitude < background_noise_threshold:
            logger.info(
                f"successful establishment of capture link of fddc-#{fddc_idx} of mxfe-#{mxfe_idx}, "
                f"max amplitude of the capture data is "
                f"{max_backgroud_amplitude:.1f} (< {background_noise_threshold:.1f})"
            )
            self._add_linkup_statistics(
                mxfe_idx=mxfe_idx,
                fddc_idx=fddc_idx,
                timeout=False,
                failure=False,
                samples=cap_data.shape[0],
                valid_capture=True,
                max_noise_peak=max_backgroud_amplitude,
            )
            return True
        else:
            # need to link up again to make the captured data fine
            logger.warning(
                f"failed establishment of capture link of fddc-#{fddc_idx} of mxfe-#{mxfe_idx}, "
                "max amplitude of the capture data of is "
                f"{max_backgroud_amplitude:.1f} (>= {background_noise_threshold:.1f})"
            )
            self._add_linkup_statistics(
                mxfe_idx=mxfe_idx,
                fddc_idx=fddc_idx,
                timeout=False,
                failure=False,
                samples=cap_data.shape[0],
                valid_capture=True,
                max_noise_peak=max_backgroud_amplitude,
            )
            return False

    def start_temperature_control(self, restart: bool) -> None:
        tc_state = self._css.get_tempctrl_state()
        if tc_state == Quel1seTempctrlState.INIT:
            logger.info(f"the temperature control is initialized ({tc_state.tostr()} -> prerun)")
            self._css.start_tempctrl()
        elif tc_state in {Quel1seTempctrlState.PRERUN, Quel1seTempctrlState.RUN}:
            if restart:
                logger.info(f"the temperature control is initialized ({tc_state.tostr()} -> prerun)")
                self._css.start_tempctrl()
        else:
            logger.warning(f"be aware that the temperature control is in an experimental mode ({tc_state.tostr()})")
