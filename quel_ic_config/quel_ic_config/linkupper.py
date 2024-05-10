import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Final, List, Set, Tuple, Union

import numpy as np

from quel_ic_config.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config.quel1_anytype import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_wave_subsystem import CaptureReturnCode, Quel1WaveSubsystem

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
    link_status: int
    error_status: int
    adc_idx: int
    timeout: bool
    failure: bool
    samples: int
    valid_capture: bool
    max_noise_peak: float

    def categorize(self, noise_threshold: float) -> LinkupStatus:
        if self.link_status != 0xE0:
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
    _DEFAULT_BACKGROUND_NOISE_THRESHOLD: Final[float] = 256.0
    _STAT_HISTORY_MAX_LEN: Final[int] = 1000

    def __init__(
        self, css: Quel1AnyConfigSubsystem, wss: Quel1WaveSubsystem, rmap: Union[Quel1E7ResourceMapper, None] = None
    ):
        self._css = css
        self._wss = wss
        if rmap is None:
            rmap = Quel1E7ResourceMapper(css, wss)
        self._rmap = rmap
        self._statistics: Dict[int, List[LinkupStatistic]] = {}
        self._target_capmods: Dict[int, Set[int]] = {}

    def _validate_mxfe(self, mxfe_idx: int):
        self._css._validate_mxfe(mxfe_idx)

    def _lookup_capmods_of_mxfe(self, mxfe_idx: int) -> Set[int]:
        rlines = self._rmap.get_active_rlines_of_mxfe(mxfe_idx)
        return {self._rmap.get_capture_module_of_rline(mxfe_idx, rline) for _, rline in rlines}

    def _add_linkup_statistics(
        self,
        mxfe_idx: int,
        adc_idx: int,
        timeout: bool,
        failure: bool,
        samples: int,
        valid_capture: bool,
        max_noise_peak: float,
    ):
        link_status, error_status = self._css.ad9082[mxfe_idx].get_link_status()
        if mxfe_idx not in self._statistics:
            self._statistics[mxfe_idx] = []

        self._statistics[mxfe_idx].append(
            LinkupStatistic(
                link_status=link_status,
                error_status=error_status,
                adc_idx=adc_idx,
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
        self._wss.initialize_awgs(self._rmap.get_awgs_of_group(mxfe_idx))
        capunits: Set[Tuple[int, int]] = set()
        for capmod in self._lookup_capmods_of_mxfe(mxfe_idx):
            capunits.add((capmod, 0))
        self._wss.initialize_capunits(capunits)

    def linkup_and_check(
        self,
        mxfe_idx: int,
        hard_reset: bool = False,
        soft_reset: bool = True,
        use_204b: bool = False,
        use_bg_cal: bool = True,
        ignore_crc_error: bool = False,
        ignore_extraordinary_converter_select: bool = False,
        background_noise_threshold: Union[float, None] = None,
        save_dirpath: Union[Path, None] = None,
    ) -> bool:
        self._validate_mxfe(mxfe_idx)
        if background_noise_threshold is None:
            background_noise_threshold = self._DEFAULT_BACKGROUND_NOISE_THRESHOLD

        judge_system: bool = False
        for i in range(self._LINKUP_MAX_RETRY):
            if i != 0:
                sleep_duration = 0.25
                logger.info(f"waiting {sleep_duration} seconds before retrying linkup")
                time.sleep(sleep_duration)

            if not self._css.configure_mxfe(
                mxfe_idx,
                hard_reset=hard_reset,
                soft_reset=soft_reset,
                mxfe_init=True,
                use_204b=use_204b,
                use_bg_cal=use_bg_cal,
                ignore_crc_error=ignore_crc_error,
            ):
                self._add_linkup_statistics(
                    mxfe_idx,
                    adc_idx=-1,
                    timeout=False,
                    failure=False,
                    samples=-1,
                    valid_capture=False,
                    max_noise_peak=-1,
                )
                continue

            self.init_wss_resources(mxfe_idx)
            self._rmap.validate_configuration_integrity(mxfe_idx, ignore_extraordinary_converter_select)

            # Notes: it is fine to check all the available adcs of the target mxfe.
            judge_adcs: bool = True
            for _, adc_idx in self._rmap.get_active_adc_of_mxfe(mxfe_idx):
                judge_adcs &= self.check_adc(mxfe_idx, adc_idx, background_noise_threshold, save_dirpath)

            if judge_adcs:
                logger.info(f"successful system-level link-up of {self._css._css_addr}:mxfe-#{mxfe_idx}")
                judge_system = True
                break
            else:
                # Notes: info is enough
                logger.info(f"failed system-level link-up of {self._css._css_addr}:mxfe-#{mxfe_idx}")

        if judge_system:
            if self._css.tempctrl_auto_start_at_linkup:
                self._css.start_tempctrl()

        return judge_system

    def check_adc(
        self,
        mxfe_idx: int,
        adc_idx: int,
        background_noise_threshold: Union[float, None] = None,
        save_dirpath: Union[Path, None] = None,
    ) -> bool:
        if background_noise_threshold is None:
            background_noise_threshold = self._DEFAULT_BACKGROUND_NOISE_THRESHOLD
        capmod = self._rmap.get_capture_module_of_adc(mxfe_idx, adc_idx)
        status, cap_data = self._wss.simple_capture(capmod, num_words=16384)
        if status == CaptureReturnCode.SUCCESS:
            if save_dirpath is not None:
                os.makedirs(save_dirpath, exist_ok=True)
                np.save(str(save_dirpath / f"capture_{mxfe_idx}_{int(time.time())}.npy"), cap_data)

            max_backgroud_amplitude = max(abs(cap_data))
            if max_backgroud_amplitude < background_noise_threshold:
                logger.info(
                    f"successful establishment of capture link of adc{adc_idx} of mxfe-#{mxfe_idx}, "
                    f"max amplitude of the capture data is {max_backgroud_amplitude:.1f}"
                )
                self._add_linkup_statistics(
                    mxfe_idx,
                    adc_idx=adc_idx,
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
                    f"failed establishment of capture link of adc{adc_idx} of mxfe-#{mxfe_idx}, "
                    "max amplitude of the capture data of is "
                    f"{max_backgroud_amplitude:.1f} (>= {background_noise_threshold:.1f})"
                )
                self._add_linkup_statistics(
                    mxfe_idx,
                    adc_idx=adc_idx,
                    timeout=False,
                    failure=False,
                    samples=cap_data.shape[0],
                    valid_capture=True,
                    max_noise_peak=max_backgroud_amplitude,
                )
        else:
            if status == CaptureReturnCode.CAPTURE_TIMEOUT:
                self._add_linkup_statistics(
                    mxfe_idx,
                    adc_idx=adc_idx,
                    timeout=True,
                    failure=False,
                    samples=-1,
                    valid_capture=False,
                    max_noise_peak=-1,
                )
            elif status == CaptureReturnCode.CAPTURE_ERROR:
                self._add_linkup_statistics(
                    mxfe_idx,
                    adc_idx=adc_idx,
                    timeout=False,
                    failure=True,
                    samples=-1,
                    valid_capture=False,
                    max_noise_peak=-1,
                )
            elif status == CaptureReturnCode.BROKEN_DATA:
                self._add_linkup_statistics(
                    mxfe_idx,
                    adc_idx=adc_idx,
                    timeout=False,
                    failure=False,
                    samples=cap_data.shape[0],
                    valid_capture=False,
                    max_noise_peak=-1,
                )
            else:
                raise AssertionError
        return False
