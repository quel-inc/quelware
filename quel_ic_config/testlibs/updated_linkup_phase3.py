import copy
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Collection, Dict, Final, List, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureCtrl, CaptureModule, CaptureParam, CaptureUnitTimeoutError

from quel_ic_config import Quel1BoxType, Quel1ConfigObjects
from testlibs.updated_linkup_phase2 import Quel1WaveGen

logger = logging.getLogger(__name__)


class Quel1WaveCap:
    DEFAULT_LO_MHZ: Final[int] = 8500
    DEFAULT_CNCO_MHZ: Final[float] = 1500.0
    DEFAULT_FNCO_MHZ: Final[float] = 0.0
    DEFAULT_NUM_WORDS: Final[int] = 16384
    DEFAULT_DELAY: Final[int] = 100
    DEFAULT_CAPTURE_TIMEOUT: Final[float] = 0.5

    def __init__(self, wss_addr: str, qco: Quel1ConfigObjects, group: int):
        self.wss_addr = wss_addr
        self.cap_ctrl = CaptureCtrl(self.wss_addr)

        self.qco: Final[Quel1ConfigObjects] = qco

        self.group = group
        if self.group == 0:
            self.cpmd: CaptureModule = CaptureModule.U1
        elif self.group == 1:
            self.cpmd = CaptureModule.U0
        else:
            raise ValueError(f"invalid group '{group}', must be either 0 or 1.")

        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.waiting_thread: Union[Future, None] = None
        self.active_units: Set[int] = set()

    def phase3a(self, input_port: str, enable_internal_loop: bool) -> None:
        if input_port == "r":
            if enable_internal_loop:
                self.activate_readloop()
            else:
                self.deactivate_readloop()
        elif input_port == "m":
            if enable_internal_loop:
                self.close_monitor_out()
            else:
                self.open_monitor_out()
        else:
            raise ValueError(f"invalid input port: {input_port}")

    def phase3b(self, input_port: str, lo_mhz: Union[int, None] = None, cnco_mhz: Union[float, None] = None):
        if input_port == "r":
            line: int = 0
            adc_ch: int = 3
        elif input_port == "m":
            line = 1
            adc_ch = 2
        else:
            raise ValueError(f"invalid input port: '{input_port}'")

        # set LO
        if lo_mhz is not None:
            if lo_mhz % 100 != 0 or lo_mhz < 8000 or lo_mhz > 15000:
                ValueError("lo_mhz must be multiple of 100, from 8000 to 15000")
            self.qco.set_lo_freq(self.group, line, lo_mhz // 100)

        # set CNCO
        if cnco_mhz is not None:
            ftw = self.qco.ad9082[self.group].calc_adc_cnco_ftw(int(cnco_mhz * 1000000 + 0.5), fractional_mode=False)
            logger.info(
                f"CNCO{adc_ch} of ADCs of {self.qco.ipaddr_css} is set to {cnco_mhz}MHz "
                f"(ftw = {ftw.ftw}, {ftw.modulus_a}, {ftw.modulus_b})"
            )
            self.qco.ad9082[self.group].set_adc_cnco({adc_ch}, ftw)

    def phase3c(self, input_port: str, fnco_mhz: Union[float, None] = None):
        if input_port == "r":
            # for both groups 0 and 1
            fnco_ch = 5
        elif input_port == "m":
            # for both groups 0 and 1
            fnco_ch = 4
        else:
            raise ValueError(f"invalid input port: '{input_port}'")

        if fnco_mhz is not None:
            ftw = self.qco.ad9082[self.group].calc_adc_fnco_ftw(int(fnco_mhz * 1000000 + 0.5), fractional_mode=False)
            logger.info(
                f"FNCO{fnco_ch} of ADCs of {self.qco.ipaddr_css} is set to {fnco_mhz}MHz "
                f"(ftw = {ftw.ftw}, {ftw.modulus_a}, {ftw.modulus_b})"
            )
            self.qco.ad9082[self.group].set_adc_fnco({fnco_ch}, ftw)

    # this function comes from phase1d(), and modifies to support "awg triggering".
    # TODO: the design of phase3d depends on the limitation that　only receiving from a single unit at the same time.
    #       So, independent operation of mxfe0 and mxfe1 is not possible with the current design, the root of that is
    #       coming from the design of e7awg_sw. More worse, in near future, both read-in and monitor-in are
    #       available at the same time. The software architecture should be revised ASAP.
    def phase3d(
        self,
        num_words: int,
        delay: int,
        active_units: Union[Set[int], None],
        triggering_awg_unit: Union[Tuple[Quel1WaveGen, int, int], None],
    ) -> None:
        if self.waiting_thread is not None:
            if self.waiting_thread.running():
                raise RuntimeError("capturing thread is still running")
            else:
                logger.warning("the unread captured data remains. it is going to be disposed")
                _ = self.waiting_thread.result()
                self.waiting_thread = None

        cpuns = CaptureModule.get_units(self.cpmd)
        if active_units is not None:
            cpuns_new = []
            for idx, cpun in enumerate(cpuns):
                if idx in active_units:
                    cpuns_new.append(cpun)
            cpuns = cpuns_new
        if len(cpuns) == 0:
            raise ValueError("no capture units is available")

        cprm = CaptureParam()
        cprm.num_integ_sections = 1
        # TODO: confirm the necessity of post blank words.
        cprm.add_sum_section(num_words=num_words, num_post_blank_words=1)
        # TODO: check the unit of the delay.
        cprm.capture_delay = delay

        self.cap_ctrl.initialize(*cpuns)
        for cpun in cpuns:
            # notes: common capture parameter is shared by all the capture units now
            self.cap_ctrl.set_capture_params(cpun, cprm)
            # notes: actually not active at this particular moment. they'll be active soon either within this method or
            #        by external trigger.
            self.active_units.add(cpun)

        if triggering_awg_unit is None:
            self.cap_ctrl.start_capture_units(*cpuns)
        else:
            wg, dac_idx, awg_idx = triggering_awg_unit
            if wg.wss_addr != self.wss_addr:
                raise ValueError("trigger works only within the same box.")

            awg_unit = wg.get_awg_unit(dac_idx, awg_idx)
            logger.info(f"triggering awg for capture module {self.cpmd} is awg {awg_unit}")
            self.cap_ctrl.select_trigger_awg(self.cpmd, awg_unit)
            self.cap_ctrl.enable_start_trigger(*cpuns)

    def phase3x(self, timeout: float = DEFAULT_CAPTURE_TIMEOUT):
        if len(self.active_units) > 0:
            self.waiting_thread = self.executor.submit(self._phase3x_thread, copy.copy(self.active_units), timeout)
        else:
            raise RuntimeError("no active capture units")
        return

    def _phase3x_thread(
        self,
        cpuns: Collection[int],
        timeout: float,
    ) -> Union[Dict[int, npt.NDArray[np.complexfloating]], None]:
        try:
            self.cap_ctrl.wait_for_capture_units_to_stop(timeout, *cpuns)
        except CaptureUnitTimeoutError:
            logger.warning(f"timeout for waiting capture units {','.join([str(cpun) for cpun in cpuns])} to stop")
            return None

        errdict: Dict[int, List[Any]] = self.cap_ctrl.check_err(*cpuns)
        errflag = False
        for cpun, errlist in errdict.items():
            for err in errlist:
                logger.error(f"unit {cpun}: {err}")
                errflag = True
        if errflag:
            return None

        data: Dict[int, npt.NDArray[np.complexfloating]] = {}
        for cpun in cpuns:
            n = self.cap_ctrl.num_captured_samples(cpun)
            logger.info(f"the capture unit {self.wss_addr}:{int(cpun)} captured {n} samples")
            data_in_assq: List[Tuple[float, float]] = self.cap_ctrl.get_capture_data(cpun, n)
            tmp = np.array(data_in_assq)
            data[int(cpun)] = tmp[:, 0] + tmp[:, 1] * 1j
        return data

    def phase3y(self) -> None:
        if len(self.active_units) == 0:
            logger.warning("no active units, do nothing")
        else:
            self.active_units.clear()

    def phase3z(self) -> Union[Dict[int, npt.NDArray[np.complexfloating]], None]:
        if self.waiting_thread is None:
            raise RuntimeError("no waiting thread is available")

        data = self.waiting_thread.result()
        self.waiting_thread = None
        return data

    def run(
        self,
        input_port: str,
        enable_internal_loop: bool,
        active_units: Union[Set[int], None] = None,
        a: bool = True,
        b: bool = True,
        c: bool = True,
        d: bool = True,
        lo_mhz: Union[int, None] = DEFAULT_LO_MHZ,
        cnco_mhz: Union[float, None] = DEFAULT_CNCO_MHZ,
        fnco_mhz: Union[float, None] = DEFAULT_FNCO_MHZ,
        num_words: int = DEFAULT_NUM_WORDS,
        delay: int = DEFAULT_DELAY,
        triggering_awg_unit: Union[Tuple[Quel1WaveGen, int, int], None] = None,
    ) -> None:
        if a:
            self.phase3a(input_port, enable_internal_loop)
        if b:
            self.phase3b(input_port, lo_mhz, cnco_mhz)
        if c:
            self.phase3c(input_port, fnco_mhz)
        if d:
            self.phase3d(num_words, delay, active_units, triggering_awg_unit)

        return None

    def complete(
        self, x: bool = True, y: bool = True, z: bool = True, timeout: float = DEFAULT_CAPTURE_TIMEOUT
    ) -> Union[Dict[int, npt.NDArray[np.complexfloating]], None]:
        data: Union[Dict[int, npt.NDArray[np.complexfloating]], None] = None
        if x:
            self.phase3x(timeout)
        if y:
            self.phase3y()
        if z:
            data = self.phase3z()
        return data

    def open_monitor_out(self):
        self.qco.open_monitor(self.group)

    def close_monitor_out(self):
        self.qco.activate_monitor_loop(self.group)

    def activate_readloop(self):
        self.qco.activate_read_loop(self.group)

    def deactivate_readloop(self):
        self.qco.open_read(self.group)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="check the sanity of a pair of input/output ports of QuEL-1")
    parser.add_argument(
        "--ipaddr_wss",
        type=str,
        required=True,
        help="IP address of the wave generation/capture subsystem of the target box",
    )
    parser.add_argument(
        "--ipaddr_css", type=str, required=True, help="IP address of the configuration subsystem of the target box"
    )
    parser.add_argument(
        "--boxtype",
        type=str,
        choices=["quel1-a", "quel1-b", "qube-a", "qube-b"],
        required=False,
        default="nonexistent",
        help="IGNORED",
    )
    parser.add_argument("--monitor", action="store_true", help="IGNORED")
    args = parser.parse_args()

    qco_ = Quel1ConfigObjects(args.ipaddr_css, Quel1BoxType.fromstr(args.boxtype), args.config_root)
    css_p3_g0 = Quel1WaveCap(args.ipaddr_wss, qco_, 0)
    css_p3_g1 = Quel1WaveCap(args.ipaddr_wss, qco_, 1)
    print(
        "How to use: data = css_p3_g0.run('m', False), you'll capture wave data from monitor-in port "
        "(i.e., port 5 of quel-1x). To use internal loopback, pass True to the second argument."
    )
