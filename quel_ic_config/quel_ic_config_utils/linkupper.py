import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Final, List, Set, Tuple, Union

import numpy as np

from quel_ic_config import Quel1AnyConfigSubsystem
from quel_ic_config_utils.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config_utils.quel1_wave_subsystem import CaptureReturnCode, Quel1WaveSubsystem

logger = logging.getLogger(__name__)


class LinkupStatus(Enum):
    SUCCESS = 1
    LINKUP_FAILURE = 2
    CRC_ERROR = 3
    RX_TIMEOUT = 4
    BROKEN_RX_DATA = 5
    CHIKUCHIKU = 6


@dataclass(frozen=True)
class LinkupStatistic:
    link_status: int
    error_status: int
    target_rline: str
    timeout: bool
    valid_capture: bool
    max_noise_peak: float

    def categorize(self, noise_threshold: float) -> LinkupStatus:
        if self.link_status != 0xE0:
            return LinkupStatus.LINKUP_FAILURE
        elif self.error_status != 0x01:
            return LinkupStatus.CRC_ERROR
        elif self.timeout:
            return LinkupStatus.RX_TIMEOUT
        elif not self.valid_capture:
            return LinkupStatus.BROKEN_RX_DATA
        elif self.max_noise_peak >= noise_threshold:
            return LinkupStatus.CHIKUCHIKU
        else:
            return LinkupStatus.SUCCESS

    def is_success(self, noise_threshold: float) -> bool:
        return self.categorize(noise_threshold) == LinkupStatus.SUCCESS


class LinkupFpgaMxfe:
    _LINKUP_MAX_RETRY: Final[int] = 10
    _BACKGROUND_NOISE_THRESHOLD: Final[float] = 256.0
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

    def _validate_mxfe(self, mxfe: int):
        self._css._validate_group(mxfe)

    def _lookup_capmods(self, mxfe: int) -> Set[int]:
        rlines = self._rmap.get_active_rlines_of_group(mxfe)
        return {self._rmap.get_capture_module_of_rline(mxfe, rline) for rline in rlines}

    def _add_linkup_statistics(
        self, mxfe: int, target_rline: str, timeout: bool, valid_capture: bool, max_noise_peak: float
    ):
        link_status, error_status = self._css.ad9082[mxfe].get_link_status()
        if mxfe not in self._statistics:
            self._statistics[mxfe] = []

        self._statistics[mxfe].append(
            LinkupStatistic(
                link_status=link_status,
                error_status=error_status,
                target_rline=target_rline,
                timeout=timeout,
                valid_capture=valid_capture,
                max_noise_peak=max_noise_peak,
            )
        )

        if len(self._statistics[mxfe]) > self._STAT_HISTORY_MAX_LEN:
            self._statistics[mxfe] = self._statistics[mxfe][-self._STAT_HISTORY_MAX_LEN :]  # noqa: E203

    def clear_statistics(self, mxfe: Union[None, int] = None):
        if mxfe is None:
            self._statistics = {}
        else:
            self._validate_mxfe(mxfe)
            if mxfe not in self._statistics:
                logger.warning(f"no data exist for mxfe '{mxfe}', do nothing actually")
            self._statistics[mxfe] = []

    @property
    def linkup_statistics(self):
        return self._statistics

    def init_wss_resources(self, mxfe: int) -> None:
        self._validate_mxfe(mxfe)
        self._wss.initialize_awgs(self._rmap.get_awgs_of_group(mxfe))
        capunits: Set[Tuple[int, int]] = set()
        for capmod in self._lookup_capmods(mxfe):
            capunits.add((capmod, 0))
        self._wss.initialize_capunits(capunits)

    def linkup_and_check(
        self,
        mxfe: int,
        hard_reset: bool = False,
        soft_reset: bool = True,
        use_204b: bool = True,
        ignore_crc_error: bool = False,
        background_noise_threshold: float = _BACKGROUND_NOISE_THRESHOLD,
        save_dirpath: Union[Path, None] = None,
    ) -> bool:
        self._validate_mxfe(mxfe)

        for i in range(self._LINKUP_MAX_RETRY):
            if i != 0:
                sleep_duration = 0.25
                logger.info(f"waiting {sleep_duration} seconds before retrying linkup")
                time.sleep(sleep_duration)

            if not self._css.configure_mxfe(
                mxfe,
                hard_reset=hard_reset,
                soft_reset=soft_reset,
                mxfe_init=True,
                use_204b=use_204b,
                ignore_crc_error=ignore_crc_error,
            ):
                self._add_linkup_statistics(
                    mxfe, target_rline="", timeout=False, valid_capture=False, max_noise_peak=-1
                )
                continue

            self.init_wss_resources(mxfe)

            judge: bool = True
            for rline in self._rmap.get_active_rlines_of_group(mxfe):
                judge &= self.check_rline(mxfe, rline, background_noise_threshold, save_dirpath)

            if judge:
                return True
        return False

    def check_rline(
        self,
        mxfe: int,
        rline: str,
        background_noise_threshold: float = _BACKGROUND_NOISE_THRESHOLD,
        save_dirpath: Union[Path, None] = None,
    ) -> bool:
        capmod = self._rmap.get_capture_module_of_rline(mxfe, rline)
        status, cap_data = self._wss.simple_capture(capmod, num_words=16384)
        if status == CaptureReturnCode.SUCCESS:
            if save_dirpath is not None:
                os.makedirs(save_dirpath, exist_ok=True)
                np.save(str(save_dirpath / f"capture_{mxfe}_{int(time.time())}.npy"), cap_data)

            max_backgroud_amplitude = max(abs(cap_data))
            logger.info(f"max amplitude of capture data is {max_backgroud_amplitude}")
            if max_backgroud_amplitude < background_noise_threshold:
                logger.info(f"successful link-up of mxfe-{mxfe}")
                self._add_linkup_statistics(
                    mxfe,
                    target_rline=rline,
                    timeout=False,
                    valid_capture=True,
                    max_noise_peak=max_backgroud_amplitude,
                )
                return True
            else:
                # need to link up again to make the captured data fine
                self._add_linkup_statistics(
                    mxfe,
                    target_rline=rline,
                    timeout=False,
                    valid_capture=True,
                    max_noise_peak=max_backgroud_amplitude,
                )
        else:
            if status == CaptureReturnCode.CAPTURE_TIMEOUT:
                self._add_linkup_statistics(
                    mxfe, target_rline=rline, timeout=True, valid_capture=False, max_noise_peak=-1
                )
            elif status == CaptureReturnCode.CAPTURE_ERROR:
                self._add_linkup_statistics(
                    mxfe, target_rline=rline, timeout=False, valid_capture=False, max_noise_peak=-1
                )
            elif status == CaptureReturnCode.BROKEN_DATA:
                self._add_linkup_statistics(
                    mxfe, target_rline=rline, timeout=False, valid_capture=False, max_noise_peak=-1
                )
            else:
                raise AssertionError
        return False
