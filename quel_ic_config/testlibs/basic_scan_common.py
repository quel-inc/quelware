import logging
import time
from pathlib import Path
from typing import Any, Collection, Dict, Final, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from e7awgsw import AWG, AwgCtrl, CaptureCtrl, CaptureModule, CaptureParam, WaveSequence
from e7awgsw.exception import CaptureUnitTimeoutError
from quel_inst_tool import (  # noqa: F401
    E440xb,
    E440xbTraceMode,
    E4405b,
    E4407b,
    ExpectedSpectrumPeaks,
    InstDevManager,
    MeasuredSpectrumPeak,
)

from quel_ic_config import Quel1AnyConfigSubsystem, Quel1BoxType, Quel1ConfigOption, Quel1ConfigSubsystem
from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemAd9082Mixin
from testlibs.updated_linkup_phase2 import Quel1WaveGen
from testlibs.updated_linkup_phase3 import Quel1WaveCap

logger = logging.getLogger(__name__)

MAX_CAPTURE_RETRY = 5

DEFAULT_FREQ_CENTER = 9e9
DEFAULT_FREQ_SPAN = 5e9
DEFAULT_SWEEP_POINTS = 4001
DEFAULT_RESOLUTION_BANDWIDTH = 1e5


# TODO: consider the right place for this class.
#       guessing that it can be private in an object abstracting a whole box.
#       it has hardware information defined in RTL code for Alveo U50, and connect it with AD9082 settings.
class Quel1E7ResourceMapping:
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

    _CAPTURE_MODULE_FROM_MXFE: Final[Dict[int, int]] = {
        0: CaptureModule.U1,
        1: CaptureModule.U0,
    }

    def __init__(self, css: Quel1ConfigSubsystemAd9082Mixin):
        self.dac2fduc: Final[List[List[List[int]]]] = self._parse_tx_channel_assign(css._param["ad9082"])
        self.dac_idx: Final[Dict[Tuple[int, int], int]] = css._DAC_IDX

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

    def _validate_group(self, group: int):
        if group not in {0, 1}:
            raise ValueError(f"invalid group: {group}")

    def get_awg(self, group: int, line: int, channel: int) -> int:
        try:
            fduc = self.dac2fduc[group][self.dac_idx[(group, line)]][channel]
            return self._AWGS_FROM_FDUC[(group, fduc)]
        except IndexError:
            raise ValueError(f"invalid combination of (group, line, channel) = ({group}, {line}, {channel})") from None

    def get_all_awgs_of_group(self, group: int) -> List[int]:
        self._validate_group(group)
        return [v for k, v in self._AWGS_FROM_FDUC.items() if k[0] == group]

    def get_capture_module(self, group: int) -> int:
        return self._CAPTURE_MODULE_FROM_MXFE[group]

    def get_capture_units_of_group(self, group: int):
        self._validate_group(group)
        return CaptureModule.get_units(self._CAPTURE_MODULE_FROM_MXFE[group])


class Quel1WaveSubsystem:
    __slots__ = (
        "_wss_addr",
        "_awg_ctrl",
        "_cap_ctrl",
        "_rmap",
    )

    def __init__(self, wss_addr: str, rmap: Quel1E7ResourceMapping):
        self._wss_addr: Final[str] = wss_addr
        self._awg_ctrl: Final[AwgCtrl] = AwgCtrl(self._wss_addr)
        self._cap_ctrl: Final[CaptureCtrl] = CaptureCtrl(self._wss_addr)
        self._rmap: Final[Quel1E7ResourceMapping] = rmap

    def initialize_awg_cap(self, group: int):
        awgs = self._rmap.get_all_awgs_of_group(group)
        self._awg_ctrl.initialize(*awgs)
        self._awg_ctrl.terminate_awgs(*awgs)

        cpuns = self._rmap.get_capture_units_of_group(group)
        self._cap_ctrl.initialize(*cpuns)

    def simple_capture(
        self, group: int, num_words: int = 1024, timeout: float = 0.5
    ) -> Tuple[bool, npt.NDArray[np.complexfloating]]:
        cprm = CaptureParam()
        cprm.num_integ_sections = 1
        cprm.add_sum_section(num_words=num_words, num_post_blank_words=1)
        cprm.capture_delay = 100

        cpun = self._rmap.get_capture_units_of_group(group)[0]  # use the first capture unit
        self._cap_ctrl.initialize(cpun)
        self._cap_ctrl.set_capture_params(cpun, cprm)

        self._cap_ctrl.start_capture_units(cpun)
        self._cap_ctrl.wait_for_capture_units_to_stop(timeout, cpun)
        self._cap_ctrl.check_err(cpun)

        n = self._cap_ctrl.num_captured_samples(cpun)
        is_valid = n == num_words * 4
        if is_valid:
            logger.info(f"the capture unit {int(cpun)} supposed to capture {num_words*4} samples")
        else:
            logger.warning(
                f"the capture unit {int(cpun)} supposed to capture {num_words*4} samples, "
                f"but actually capture {n} words"
            )
        data_in_assq: List[Tuple[float, float]] = self._cap_ctrl.get_capture_data(cpun, n)
        tmp = np.array(data_in_assq)
        data = tmp[:, 0] + tmp[:, 1] * 1j
        return is_valid, data

    def emit_cw(self, mxfe: int, line: int, channel: int, amplitude: float, num_repeats: Tuple[int, int]) -> None:
        au = self._rmap.get_awg(mxfe, line, channel)

        wave = WaveSequence(num_wait_words=0, num_repeats=num_repeats[0])
        iq = np.zeros(wave.NUM_SAMPLES_IN_WAVE_BLOCK, dtype=np.complex64)
        iq[:] = 1 + 0j
        iq[:] *= amplitude
        block_assq: List[Tuple[int, int]] = list(zip(iq.real.astype(int), iq.imag.astype(int)))
        wave.add_chunk(iq_samples=block_assq, num_blank_words=0, num_repeats=num_repeats[1])

        self._awg_ctrl.terminate_awgs(au)  # to override current task of the unit
        # TODO: should wait for confirming the termination.
        self._awg_ctrl.set_wave_sequence(au, wave)
        self._awg_ctrl.clear_awg_stop_flags(au)
        self._awg_ctrl.start_awgs(au)

    def emit_iq(
        self, mxfe: int, line: int, channel: int, iq: npt.NDArray[np.complex128], num_repeats: Tuple[int, int]
    ) -> None:
        au = self._rmap.get_awg(mxfe, line, channel)

        wave = WaveSequence(num_wait_words=0, num_repeats=num_repeats[0])
        block_assq: List[Tuple[int, int]] = list(zip(iq.real.astype(int), iq.imag.astype(int)))
        wave.add_chunk(iq_samples=block_assq, num_blank_words=0, num_repeats=num_repeats[1])

        self._awg_ctrl.terminate_awgs(au)  # to override current task of the unit
        # TODO: should wait for confirming the termination.
        self._awg_ctrl.set_wave_sequence(au, wave)
        self._awg_ctrl.clear_awg_stop_flags(au)
        self._awg_ctrl.start_awgs(au)

    def stop_emission(self, mxfe: int, line: int, channel: int):
        au = self._rmap.get_awg(mxfe, line, channel)
        self._awg_ctrl.terminate_awgs(au)  # to override current task of the unit
        # TODO: should wait for confirming the termination.


class LinkupFpgaMxfe:
    _LINKUP_MAX_RETRY: Final[int] = 10
    _CAPTURE_TIMEOUT_MAX_RETRY: Final[int] = 3
    _BACKGROUND_NOISE_THRESHOLD: Final[float] = 256.0

    def __init__(self, css: Quel1AnyConfigSubsystem, wss: Quel1WaveSubsystem):
        self._css = css
        self._wss = wss

    def linkup_and_check(
        self,
        mxfe: int,
        soft_reset: bool = True,
        hard_reset: bool = False,
        configure_clock: bool = False,
        ignore_crc_error: bool = False,
        background_noise_threshold: float = _BACKGROUND_NOISE_THRESHOLD,
    ) -> bool:
        if mxfe not in {0, 1}:
            raise ValueError(f"invalid mxfe to link up: {mxfe}")

        for i in range(self._LINKUP_MAX_RETRY):
            if i != 0:
                logger.info(f"waiting {i} seconds before retrying linkup")
                time.sleep(i)

            if not self._css.configure_mxfe(
                mxfe,
                soft_reset=soft_reset,
                hard_reset=hard_reset,
                configure_clock=configure_clock,
                ignore_crc_error=ignore_crc_error,
            ):
                continue

            for j in range(self._CAPTURE_TIMEOUT_MAX_RETRY):
                if j != 0:
                    time.sleep(1)

                try:
                    is_valid_cap, cap_data = self._wss.simple_capture(mxfe, 16384)
                except CaptureUnitTimeoutError as e:
                    logger.warning(e)
                    self._wss.initialize_awg_cap(mxfe)
                    continue

                if is_valid_cap:
                    max_backgroud_amplitude = max(abs(cap_data))
                    logger.info(f"max amplitude of capture data is {max_backgroud_amplitude}")
                    if max_backgroud_amplitude < background_noise_threshold:
                        logger.info(f"successful link-up of mxfe-{mxfe}")
                        return True
                break  # need to link up again to make the captured data fine

        return False


class SimpleBoxTest:
    _PORT2LINE: Final[Dict[Quel1BoxType, Dict[int, Tuple[int, int]]]] = {
        Quel1BoxType.QuEL1_TypeA: {
            1: (0, 0),
            2: (0, 2),
            3: (0, 1),
            4: (0, 3),
            8: (1, 0),
            9: (1, 3),
            10: (1, 1),
            11: (1, 2),
        },
        Quel1BoxType.QuEL1_TypeB: {
            1: (0, 0),
            2: (0, 1),
            3: (0, 2),
            4: (0, 3),
            8: (1, 0),
            9: (1, 1),
            10: (1, 3),
            11: (1, 2),
        },
    }

    # TODO: should be derived from settings in the proper object.
    _LINE2NUM_CHANNELS: Final[Dict[Tuple[int, int], int]] = {
        (0, 0): 1,
        (0, 1): 1,
        (0, 2): 3,
        (0, 3): 3,
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 3,
        (1, 3): 3,
    }

    def __init__(self, css: Quel1ConfigSubsystem, wss: Quel1WaveSubsystem):
        self._css = css
        self._boxtype = css._boxtype
        if self._boxtype not in self._PORT2LINE:
            raise ValueError(f"Unsupported boxtype; {self._boxtype}")
        self._wss = wss

    def _convert_port(self, port: int) -> Tuple[int, int]:
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid output port: {port}")
        return self._PORT2LINE[self._boxtype][port]

    def _validate_channel(self, port, channel):
        mxfe, line = self._convert_port(port)
        if channel >= self._LINE2NUM_CHANNELS[mxfe, line]:
            raise ValueError(f"invalid channel {channel} for port {port}")

    def config_all(self) -> None:
        """config all ports for your convenience of testing.
        :return: None
        """
        if self._boxtype == Quel1BoxType.QuEL1_TypeA:
            self.config_port(1, lo_freq=8.5e9, cnco_freq=1.5e9, vatt=0xA00, sideband="U")  # 10GHz
            self.config_port(2, lo_freq=11.5e9, cnco_freq=1.55e9, vatt=0xA00, sideband="L")  # 9.95GHz
            self.config_port(3, lo_freq=8.5e9, cnco_freq=1.4e9, vatt=0xA00, sideband="U")  # 9.9GHz
            self.config_port(4, lo_freq=11.5e9, cnco_freq=1.65e9, vatt=0xA00, sideband="L")  # 9.85GHz
            self.config_port(8, lo_freq=8.5e9, cnco_freq=1.3e9, vatt=0xA00, sideband="U")  # 9.8GHz
            self.config_port(9, lo_freq=11.5e9, cnco_freq=1.75e9, vatt=0xA00, sideband="L")  # 9.75GHz
            self.config_port(10, lo_freq=8.5e9, cnco_freq=1.2e9, vatt=0xA00, sideband="U")  # 9.7GHz
            self.config_port(11, lo_freq=11.5e9, cnco_freq=1.85e9, vatt=0xA00, sideband="L")  # 9.65GHz
        elif self._boxtype == Quel1BoxType.QuEL1_TypeB:
            self.config_port(1, lo_freq=11.5e9, cnco_freq=1.5e9, vatt=0xA00, sideband="L")  # 10GHz
            self.config_port(2, lo_freq=11.5e9, cnco_freq=1.55e9, vatt=0xA00, sideband="L")  # 9.95GHz
            self.config_port(3, lo_freq=11.5e9, cnco_freq=1.6e9, vatt=0xA00, sideband="L")  # 9.9GHz
            self.config_port(4, lo_freq=11.5e9, cnco_freq=1.65e9, vatt=0xA00, sideband="L")  # 9.85GHz
            self.config_port(8, lo_freq=11.5e9, cnco_freq=1.7e9, vatt=0xA00, sideband="L")  # 9.8GHz
            self.config_port(9, lo_freq=11.5e9, cnco_freq=1.75e9, vatt=0xA00, sideband="L")  # 9.75GHz
            self.config_port(10, lo_freq=11.5e9, cnco_freq=1.8e9, vatt=0xA00, sideband="L")  # 9.7GHz
            self.config_port(11, lo_freq=11.5e9, cnco_freq=1.85e9, vatt=0xA00, sideband="L")  # 9.65GHz
        else:
            raise AssertionError

    def scan_all(self, wait=4) -> None:
        """generate CW port by port.
        :param wait: duration of emittion of CW in second.
        :return: None
        """
        for p in (1, 2, 3, 4, 8, 9, 10, 11):
            logger.info(f"starting to emit CW from port #{p}")
            self.start_channel(p)
            self.open_port(p)
            time.sleep(wait)
            logger.info(f"stopping to emit CW from port #{p}")
            self.close_port(p)
            self.stop_channel(p)
            time.sleep(wait * 0.5)

    def config_port(
        self,
        port: int,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
    ) -> None:
        """config parameters of each individual port
        :param port: port to be configured
        :param lo_freq: frequency of Local Oscillator in Hz
        :param cnco_freq: frequency of Coarse Numerical Oscillator in Hz
        :param vatt: input voltage to variable attenuator of ADRF6780. see the spec sheet of the IC for details.
        :param sideband: "U" for upper side band, "L" for lower side band.
        :return: None
        """
        mxfe, line = self._convert_port(port)
        if vatt is not None:
            self._css.set_vatt(mxfe, line, vatt)
        if sideband is not None:
            self._css.set_sideband(mxfe, line, sideband)
        if lo_freq is not None:
            if lo_freq % 100000000 != 0:
                raise ValueError("lo_freq must be multiple of 100000000")
            self._css.set_lo_multiplier(mxfe, line, int(lo_freq) // 100000000)
        if cnco_freq is not None:
            self._css.set_dac_cnco(mxfe, line, int(cnco_freq + 0.5))

    def config_channel(self, port: int, channel: int = 0, fnco_freq: Union[float, None] = None):
        """set the channel specific parameter, namely frequency of Fine Numerical Oscillator in Hz.
        :param port: port to be configured.
        :param channel: channel of the port to be configured.
        :param fnco_freq: frequecny in Hz, -200e6 to 200e6
        :return: None
        """
        mxfe, line = self._convert_port(port)
        self._validate_channel(port, channel)
        if fnco_freq is not None:
            self._css.set_dac_fnco(mxfe, line, channel, int(fnco_freq + 0.5))

    def open_port(self, port):
        mxfe, line = self._convert_port(port)
        self._css.pass_line(mxfe, line)

    def close_port(self, port):
        mxfe, line = self._convert_port(port)
        self._css.block_line(mxfe, line)

    def start_channel(
        self, port: int, channel: int = 0, amplitude: float = 16384.0, num_repeat: Tuple[int, int] = (1, 0xFFFFFFFF)
    ):
        mxfe, line = self._convert_port(port)
        self._validate_channel(port, channel)
        self._wss.emit_cw(mxfe, line, channel, amplitude, num_repeat)

    def stop_channel(self, port: int, channel: Union[int, None] = None):
        mxfe, line = self._convert_port(port)
        if channel is None:
            for ch in range(self._LINE2NUM_CHANNELS[mxfe, line]):
                self._wss.stop_emission(mxfe, line, ch)
        else:
            self._wss.stop_emission(mxfe, line, channel)


def init_box(
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    mxfe_combination: str,
    config_root: Path,
    config_options: Collection[Quel1ConfigOption],
    config_clock_groupwise: bool = False,
) -> Tuple[
    bool,
    bool,
    Quel1AnyConfigSubsystem,
    Quel1WaveSubsystem,
    LinkupFpgaMxfe,
    Union[Quel1WaveGen, None],
    Union[Quel1WaveGen, None],
    Union[Quel1WaveCap, None],
    Union[Quel1WaveCap, None],
    Union[SimpleBoxTest, None],
]:
    """create QuEL testing objects and initialize ICs
    :param ipaddr_wss: IP address of the wave generation subsystem of the target box
    :param ipaddr_sss: IP address of the sequencer subsystem of the target box
    :param ipaddr_css: IP address of the configuration subsystem of the target box
    :param boxtype: type of the target box
    :param mxfe_combination: target mxfes of the target box
    :param config_root: root path of config setting files to read
    :param config_options: a collection of config options
    :return: QuEL testing objects
    """
    if boxtype in {
        Quel1BoxType.QuBE_TypeA,
        Quel1BoxType.QuBE_TypeB,
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuEL1_TypeB,
    }:
        css: Quel1AnyConfigSubsystem = Quel1ConfigSubsystem(ipaddr_css, boxtype, config_root, config_options)
    else:
        raise ValueError(f"unsupported boxtype: {boxtype}")

    if isinstance(css, Quel1ConfigSubsystemAd9082Mixin):
        rmap = Quel1E7ResourceMapping(css)
    else:
        raise AssertionError("the given ConfigSubsystem Object doesn't provide AD9082 interface")

    wss = Quel1WaveSubsystem(ipaddr_wss, rmap)
    linkupper = LinkupFpgaMxfe(css, wss)

    if isinstance(css, Quel1ConfigSubsystem):
        p2_g0: Union[Quel1WaveGen, None] = Quel1WaveGen(ipaddr_wss, css, 0)
        p2_g1: Union[Quel1WaveGen, None] = Quel1WaveGen(ipaddr_wss, css, 1)
        p3_g0: Union[Quel1WaveCap, None] = Quel1WaveCap(ipaddr_wss, css, 0)
        p3_g1: Union[Quel1WaveCap, None] = Quel1WaveCap(ipaddr_wss, css, 1)
    else:
        p2_g0 = None
        p2_g1 = None
        p3_g0 = None
        p3_g1 = None

    if isinstance(css, Quel1ConfigSubsystem):
        box: Union[SimpleBoxTest, None] = SimpleBoxTest(css, wss)
    else:
        box = None

    linkup_ok = [False, False]
    css.configure_peripherals()
    if not config_clock_groupwise:
        css.configure_all_mxfe_clocks()

    if mxfe_combination == "0":
        mxfe_list: Tuple[int, ...] = (0,)
    elif mxfe_combination == "1":
        mxfe_list = (1,)
    elif mxfe_combination in {"both", "0,1"}:
        mxfe_list = (0, 1)
    elif mxfe_combination == "1,0":
        mxfe_list = (1, 0)
    else:
        raise AssertionError

    for mxfe in mxfe_list:
        wss.initialize_awg_cap(mxfe)
        linkup_ok[mxfe] = linkupper.linkup_and_check(mxfe, configure_clock=config_clock_groupwise)

        if mxfe == 0:
            if p2_g0 is not None:
                p2_g0.force_stop()
        elif mxfe == 1:
            if p2_g1 is not None:
                p2_g1.force_stop()
        else:
            raise AssertionError

    # TODO: write scheduler object creation here.
    _ = ipaddr_sss

    return linkup_ok[0], linkup_ok[1], css, wss, linkupper, p2_g0, p2_g1, p3_g0, p3_g1, box


def init_e440xb(
    spa_type: str = "E4405B",
    freq_center: float = DEFAULT_FREQ_CENTER,
    freq_span: float = DEFAULT_FREQ_SPAN,
    sweep_points: int = DEFAULT_SWEEP_POINTS,
    resolution_bandwidth: float = DEFAULT_RESOLUTION_BANDWIDTH,
) -> E440xb:
    im = InstDevManager(ivi="/usr/lib/x86_64-linux-gnu/libiovisa.so", blacklist=["GPIB0::6::INSTR"])
    dev = im.lookup(prod_id=spa_type)
    if dev is None:
        raise RuntimeError(f"no spectrum analyzer '{spa_type}' is detected")

    if spa_type == "E4405B":
        e440xb: E440xb = E4405b(dev)
    elif spa_type == "E4407B":
        e440xb = E4407b(dev)
    else:
        raise ValueError("invalid spectrum analyzer type, it must be either E4405B or E4407B")

    e440xb.reset()
    e440xb.display_enable = False
    e440xb.trace_mode = E440xbTraceMode.WRITE
    e440xb.freq_range_set(freq_center, freq_span)
    e440xb.sweep_points = sweep_points
    e440xb.resolution_bandwidth = resolution_bandwidth
    # e4405b.resolution_bandwidth = 5e4   # floor noise < -70dBm, but spurious peaks higher than -70dBm exist
    # e4405b.resolution_bandwidth = 1e5  # floor noise < -65dBm
    # e4405b.resolution_bandwidth = 1e6   # floor noise < -55dBm
    return e440xb


def measure_floor_noise(e4405b: E440xb, n_iter: int = 5) -> float:
    # Checkout
    t0 = e4405b.trace_get()
    fln = t0[:, 1]
    for i in range(n_iter - 1):
        fln = np.maximum(fln, e4405b.trace_get()[:, 1])

    mfln = fln.max()
    logger.info(f"maximum floor noise = {mfln:.1f}dBm")
    return mfln


if __name__ == "__main__":
    import argparse

    from testlibs.common_arguments import add_common_arguments

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="check the basic functionalities about wave generation of QuEL-1 with a spectrum analyzer"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--mxfe", choices=("0", "1", "both", "0,1", "1,0"), required=True, help="combination of MxFEs under test"
    )
    parser.add_argument(
        "--groupwise_clock",
        action="store_true",
        default=False,
        help="configuring clock of MxFE one by one just before linking it up",
    )
    args = parser.parse_args()

    # Init: QuEL mxfe
    linkup_g0, linkup_g1, css, wss, linkupper, p2_g0, p2_g1, p3_g0, p3_g1, box = init_box(
        str(args.ipaddr_wss),
        str(args.ipaddr_sss),
        str(args.ipaddr_css),
        args.boxtype,
        args.mxfe,
        args.config_root,
        args.config_options,
    )

    if args.mxfe in {"0", "both", "0,1", "1,0"}:
        assert linkup_g0
    if args.mxfe in {"1", "both", "0,1", "1,0"}:
        assert linkup_g1

    """
    # Init: Spectrum Analyzer
    e4405b_ = init_e440xb()
    max_floor_noise = measure_floor_noise(e4405b_, 5)
    assert max_floor_noise < -62.0

    # Measurement Example
    e0 = ExpectedSpectrumPeaks([(9987e6, -20), (8991e6, -20)])
    assert e0.validate_with_measurement_condition(e4405b_.max_freq_error_get())

    if linkup_g0:
        css_p2_g0.run(2, 0, cnco_mhz=2000, fnco_mhz=13)
        css_p2_g0.run(3, 0, cnco_mhz=3000, fnco_mhz=9)
        m0 = MeasuredSpectrumPeak.from_spectrumanalyzer(e4405b_, -60.0)
        j0, s0, w0 = e0.match(m0)
        assert all(j0) and len(s0) == 0 and len(w0) == 0
    else:
        logger.error("linkup of mxfe0 fails, no test is conducted")
    """
