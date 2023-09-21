import copy
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Collection, Dict, Final, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam
from e7awgsw.exception import CaptureUnitTimeoutError

from testlibs.basic_scan_common import Quel1WaveSubsystem

logger = logging.getLogger(__name__)


class SimplePulseCapture:
    DEFAULT_NUM_WORDS: Final[int] = 1024
    DEFAULT_DELAY: Final[int] = 0
    DEFAULT_CAPTURE_TIMEOUT: Final[float] = 0.5
    DEFAULT_MAX_RETRY_FOR_TIMEOUT: Final[int] = 5
    DEFAULT_SLEEP_AFTER_TIMEOUT: Final[float] = 1.0

    def __init__(
        self,
        wss: Quel1WaveSubsystem,
        mxfe_c: int,
        mxfe_g: int = 0,
        line_g: int = 0,
        channel_g: int = 0,
    ):
        self._wss: Quel1WaveSubsystem = wss
        self._c: int = mxfe_c
        self._g: Tuple[int, int, int]
        self.awg_unit: int
        self.cpmd: int
        self.cpun: List[int]
        self._setup_resources(mxfe_g, line_g, channel_g)
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.waiting_thread: Union[Future, None] = None

    def _setup_resources(self, mxfe_g: int, line_g: int, channel_g: int):
        self._g = (mxfe_g, line_g, channel_g)
        self.awg_unit = self._wss._rmap.get_awg(*self._g)
        self.cpmd = self._wss._rmap.get_capture_module(self._c)
        self.cpun = self._wss._rmap.get_capture_units_of_group(self._c)[0:1]  # use the first capture unit

    def setup(
        self,
        mxfe_g: int,
        line_g: int,
        channel_g: int,
        num_words: int = DEFAULT_NUM_WORDS,
        delay: int = DEFAULT_DELAY,
    ) -> None:
        self._setup_resources(mxfe_g, line_g, channel_g)

        cprm = CaptureParam()
        cprm.num_integ_sections = 1
        cprm.add_sum_section(num_words=num_words, num_post_blank_words=1)
        cprm.capture_delay = delay

        self._wss._cap_ctrl.initialize(*self.cpun)
        for cpun in self.cpun:
            self._wss._cap_ctrl.set_capture_params(cpun, cprm)

        logger.debug(f"triggering awg for capture module {self.cpmd} is awg {self.awg_unit}")
        self._wss._cap_ctrl.select_trigger_awg(self.cpmd, self.awg_unit)
        self._wss._cap_ctrl.enable_start_trigger(*self.cpun)

    # Note: this is not always necessary for the current e7awg_sw.
    def _capture_thread_main(
        self,
        cpuns: Collection[int],
        timeout: float,
    ) -> Union[Dict[int, npt.NDArray[np.complex128]], None]:
        for i in range(self.DEFAULT_MAX_RETRY_FOR_TIMEOUT):
            if i > 0:
                time.sleep(self.DEFAULT_SLEEP_AFTER_TIMEOUT)
            try:
                self._wss._cap_ctrl.wait_for_capture_units_to_stop(timeout, *cpuns)
                break
            except CaptureUnitTimeoutError:
                logger.warning(f"timeout for waiting capture units {','.join([str(cpun) for cpun in cpuns])} to stop")
        else:
            logger.error(
                f"too many timeout of capture units {','.join([str(cpun) for cpun in cpuns])} to stop, give up"
            )
            return None

        errdict: Dict[int, List[Any]] = self._wss._cap_ctrl.check_err(*cpuns)
        errflag = False
        for cpun, errlist in errdict.items():
            for err in errlist:
                logger.error(f"unit {cpun}: {err}")
                errflag = True
        if errflag:
            return None

        data: Dict[int, npt.NDArray[np.complexfloating]] = {}
        for idx, cpun in enumerate(cpuns):
            n = self._wss._cap_ctrl.num_captured_samples(cpun)
            logger.debug(f"the capture unit {self._wss._wss_addr}:{int(cpun)} captured {n} samples")
            data_in_assq: List[Tuple[float, float]] = self._wss._cap_ctrl.get_capture_data(cpun, n)
            tmp = np.array(data_in_assq)
            data[idx] = tmp[:, 0] + tmp[:, 1] * 1j
        return data

    def start_cw(
        self,
        amplitude: float,
        num_repeats: int,
        timeout: float = DEFAULT_CAPTURE_TIMEOUT,
    ):
        if self.waiting_thread is None:
            self._wss.emit_cw(
                mxfe=self._g[0],
                line=self._g[1],
                channel=self._g[2],
                amplitude=amplitude,
                num_repeats=(1, num_repeats),
            )
            self.waiting_thread = self.executor.submit(self._capture_thread_main, copy.copy(self.cpun), timeout)
        else:
            raise RuntimeError("previous capture thread exists")

    def result(self) -> Union[npt.NDArray[np.complex128], None]:
        if self.waiting_thread is None:
            raise RuntimeError("no waiting thread is available")

        result = self.waiting_thread.result()
        self.waiting_thread = None
        return result[0]
