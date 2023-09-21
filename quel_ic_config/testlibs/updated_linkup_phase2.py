import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final, List, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
from e7awgsw import AWG, AwgCtrl, AwgTimeoutError, WaveSequence

from quel_ic_config import Quel1ConfigSubsystem

logger = logging.getLogger(__name__)


class Quel1WaveGen:
    DEFAULT_LO_MHZ: Final[int] = 12000  # [MHz]
    DEFAULT_SIDEBAND: Final[str] = "L"
    DEFAULT_CNCO_MHZ: Final[float] = 2000.0  # [MHz]
    DEFAULT_FNCO_MHZ: Final[float] = 0.0  # [MHz]
    DEFAULT_VATT: Final[int] = 0xA00
    MAX_VATT: Final[int] = 0xC9B
    DEFAULT_AMPLITUDE: Final[float] = 32767.0
    DEFAULT_REPEATS: Final[Tuple[int, int]] = (0xFFFFFFFF, 0xFFFFFFFF)

    AWG_STOP_TIMEOUT_AFTER_TERMINATE: Final[float] = 1.0  # [sec]

    def __init__(self, wss_addr: str, qco: Quel1ConfigSubsystem, group: int):
        self.wss_addr = wss_addr
        self.awg_ctrl = AwgCtrl(self.wss_addr)

        self.qco: Final[Quel1ConfigSubsystem] = qco

        if group not in {0, 1}:
            raise ValueError(f"invalid group '{group}', must be either 0 or 1.")
        self.group = group

        self.opened_line: Set[int] = set()
        self.activated_awgs: Set[int] = set()
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.waiting_thread: Union[Future, None] = None

    def __del__(self):
        self.force_stop()

    def phase2a(self, line: int) -> None:
        """Open RF switches of the specified line.
        :param line: the index of target line.
        :return: None
        """
        self.qco.pass_line(self.group, line)
        self.opened_line.add(line)

    def phase2b(
        self,
        line: int,
        sideband: Union[str, None] = None,
        lo_mhz: Union[int, None] = None,
        cnco_mhz: Union[float, None] = None,
        vatt: Union[int, None] = None,
    ) -> None:
        """Configure a local oscillator and a mixer of the specified line (0 -- 3) of the group.
        :param line: an index of the target line. either of 0, 1, 2, or 3.
        :param sideband: an active sideband of the mixer by "U" or "L". (optional)
        :param lo_mhz: the frequency of the local oscillator. (optional)
        :param cnco_mhz: the frequency of the coarse NCO. (optional)
        :param vatt: the voltage to be applied to the variable attenuator in the mixer. (0 -- 3227)
        :return: None
        """
        if not (0 <= line <= 3):
            raise ValueError(f"invalid index of line: {line}")

        if sideband is not None:
            self.qco.set_sideband(self.group, line, sideband)

        # set LO
        if lo_mhz is not None:
            if lo_mhz % 100 != 0:
                raise ValueError("lo_mhz must be multiple of 100")
            self.qco.set_lo_multiplier(self.group, line, lo_mhz // 100)

        # set CNCO
        if cnco_mhz is not None:
            self.qco.set_dac_cnco(self.group, line, freq_in_hz=int(cnco_mhz * 1000000 + 0.5))

        # set VATT
        if vatt is not None:
            self.qco.set_vatt(self.group, line, vatt)

    def phase2c(self, line: int, awg_idx: int, fnco_mhz: Union[float, None] = None):
        # set FNCO
        if fnco_mhz is not None:
            self.qco.set_dac_fnco(self.group, line, awg_idx, freq_in_hz=int(fnco_mhz * 1000000 + 0.5))

    def get_awg_unit(self, line: int, awg_idx: int) -> int:
        # choose AWG to activate
        if self.group == 0:
            awg_units: Tuple[int, ...] = [
                (AWG.U15,),
                (AWG.U14,),
                (AWG.U11, AWG.U12, AWG.U13),
                (AWG.U10, AWG.U9, AWG.U8),
            ][line]
        elif self.group == 1:
            awg_units = [(AWG.U2,), (AWG.U1,), (AWG.U0, AWG.U3, AWG.U4), (AWG.U5, AWG.U6, AWG.U7)][line]
        else:
            raise AssertionError

        if awg_idx < len(awg_units):
            awg_unit = awg_units[awg_idx]
        else:
            raise ValueError(f"the index of awg is out of range: {awg_idx} >= {len(awg_units)}")

        return awg_unit

    # TODO: reconsider code structure, nesting of functions looks unnecessarily deep.
    def _phase2d_pre(self, line: int, awg_idx: int) -> int:
        awg_unit = self.get_awg_unit(line, awg_idx)

        if awg_unit in self.activated_awgs:
            logger.warning(f"AWG{int(awg_unit)} is already activated, deactivate once before re-activating it.")
            self.awg_ctrl.terminate_awgs(awg_unit)
            self.activated_awgs.remove(awg_unit)

        return awg_unit

    def _phase2d_post(self, awg_unit: int, wave: WaveSequence, use_schedule: bool):
        self.awg_ctrl.initialize(awg_unit)  # TODO: this should be called at once.
        self.awg_ctrl.set_wave_sequence(awg_unit, wave)
        self.awg_ctrl.clear_awg_stop_flags(awg_unit)
        if not use_schedule:
            self.awg_ctrl.start_awgs(awg_unit)

        self.activated_awgs.add(awg_unit)
        logger.info(f"AWG{int(awg_unit):d} is activated")

    def phase2d(
        self,
        line: int,
        awg_idx: int,
        amplitude: float,
        iq: Union[npt.NDArray[np.complexfloating], None],
        num_repeats: Tuple[int, int],
        use_schedule: bool,
    ) -> int:
        awg_unit = self._phase2d_pre(line, awg_idx)

        # create wave sequence
        long_wave = WaveSequence(num_wait_words=0, num_repeats=num_repeats[0])

        if iq is None:
            iq = np.zeros(long_wave.NUM_SAMPLES_IN_WAVE_BLOCK, dtype=np.complex64)
            iq[:] = 1 + 0j

        iq[:] *= amplitude
        block_assq: List[Tuple[int, int]] = list(zip(iq.real.astype(int), iq.imag.astype(int)))
        long_wave.add_chunk(iq_samples=block_assq, num_blank_words=0, num_repeats=num_repeats[1])

        self._phase2d_post(awg_unit, long_wave, use_schedule)
        return awg_unit

    def _phase2w_thread(self, timeout: float) -> bool:
        # QuEL1WaveGen is also thread-unsafe. You shouldn't manipulate the object during this thread is running other
        # than calling phase2z().
        try:
            # a copy of activated_awgs is created at the invocation. This is desired behavior for us.
            self.awg_ctrl.wait_for_awgs_to_stop(timeout, *self.activated_awgs)
            return True
        except AwgTimeoutError:
            return False

    def phase2w(self, terminate: bool, timeout: float) -> None:
        if terminate:
            if len(self.activated_awgs) > 0:
                logger.info(f"terminating AWGs {','.join([str(x) for x in self.activated_awgs])}...")
                self.awg_ctrl.terminate_awgs(*self.activated_awgs)
            else:
                logger.info("no active AWGs")

        # Notes: There is no easy way to generate a future object for each awg units because e7awg_sw is thread-unsafe.
        #        Also, it is impossible to call terminate_awgs() during wait_for_awgs_to_stop() is executing (!).
        if self.waiting_thread is not None:
            if self.waiting_thread.running():
                raise RuntimeError("waiting thread is already running, something wrong!")
            else:
                result = self.waiting_thread.result()
                logger.warning(f"the previous waiting thread remains. it was finished with '{result}'")
                self.waiting_thread = None

        self.waiting_thread = self.executor.submit(self._phase2w_thread, timeout)

    def phase2x(self) -> bool:
        """
        :return: return if wait_for_awg_to_stop() completes successfully. False means that timeout happens.
        """
        if self.waiting_thread is None:
            raise RuntimeError("No waiting thread is available.")
        result = self.waiting_thread.result()
        self.waiting_thread = None
        return result

    def phase2y(self):
        if len(self.activated_awgs) == 0:
            logger.warning("no active AWGs, do nothing.")
        else:
            self.activated_awgs.clear()

    def phase2z(self):
        if len(self.opened_line) > 0:
            logger.info(
                f"closing RF switches of group {self.group} " f"line {', '.join([str(x) for x in self.opened_line])}..."
            )
            self.qco.block_lines(self.group, self.opened_line)
            self.opened_line.clear()
        else:
            logger.info("no open RF switches")

    def run(
        self,
        line: int,
        awg_idx: int,
        a: bool = True,
        b: bool = True,
        c: bool = True,
        d: bool = True,
        sideband: Union[str, None] = DEFAULT_SIDEBAND,
        lo_mhz: Union[int, None] = DEFAULT_LO_MHZ,
        cnco_mhz: Union[float, None] = DEFAULT_CNCO_MHZ,
        vatt: Union[int, None] = DEFAULT_VATT,
        fnco_mhz: Union[float, None] = DEFAULT_FNCO_MHZ,
        amplitude: float = DEFAULT_AMPLITUDE,
        iq: Union[npt.NDArray[np.complexfloating], None] = None,
        num_repeats: Tuple[int, int] = (0xFFFFFFFF, 0xFFFFFFFF),
        use_schedule: bool = False,
    ):
        if a:
            self.phase2a(line)
        if b:
            self.phase2b(line, sideband, lo_mhz, cnco_mhz, vatt)
        if c:
            self.phase2c(line, awg_idx, fnco_mhz)
        if d:
            self.phase2d(line, awg_idx, amplitude, iq, num_repeats, use_schedule)

    def stop(
        self,
        w: bool = True,
        x: bool = True,
        y: bool = True,
        z: bool = True,
        terminate: bool = True,
        timeout: float = AWG_STOP_TIMEOUT_AFTER_TERMINATE,
    ):
        if w:
            self.phase2w(terminate, timeout)
        if x:
            if not self.phase2x():
                raise RuntimeError("failed to stop AWGs")
        if y:
            self.phase2y()
        if z:
            self.phase2z()

    def force_stop(self):
        logger.info(f"force to stop all the AWGs of group {self.qco.ipaddr_css}:{self.group}")
        if self.group == 0:
            self.awg_ctrl.terminate_awgs(AWG.U15, AWG.U14, AWG.U13, AWG.U12, AWG.U11, AWG.U10, AWG.U9, AWG.U8)
        elif self.group == 1:
            self.awg_ctrl.terminate_awgs(AWG.U7, AWG.U6, AWG.U5, AWG.U4, AWG.U3, AWG.U2, AWG.U1, AWG.U0)
        else:
            raise AssertionError
        self.activated_awgs.clear()

        logger.info(f"force to close all the RF switches of group {self.qco.ipaddr_css}:{self.group}")
        self.qco.block_lines(self.group, {0, 1, 2, 3})
        self.qco.activate_monitor_loop(self.group)
        self.opened_line.clear()

    def open_monitor(self):
        self.qco.open_monitor(self.group)

    def block_monitor(self):
        self.qco.block_monitor(self.group)


if __name__ == "__main__":
    import argparse

    from testlibs.common_arguments import add_common_arguments

    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="check the sanity of a pair of input/output ports of QuEL-1")
    add_common_arguments(parser, use_ipaddr_sss=False)
    args = parser.parse_args()

    qco_ = Quel1ConfigSubsystem(
        css_addr=str(args.ipaddr_css),
        boxtype=args.boxtype,
        config_path=args.config_root,
        config_options=args.config_options,
    )
    css_p2_g0 = Quel1WaveGen(str(args.ipaddr_wss), qco_, 0)
    css_p2_g1 = Quel1WaveGen(str(args.ipaddr_wss), qco_, 1)
    print("How to use: css_p2_g0.run(2, 0), you'll see 10GHz signal on ports 2(3) of QuEL-1 Type-A(B) box.")
    print("          : css_p2_g1.run(2, 0), you'll see 10GHz signal on ports 9(10) of QuEL-1 Type-A(B) box.")
    print("          : css_p2_g0.stop(), for stopping activated channel.")
