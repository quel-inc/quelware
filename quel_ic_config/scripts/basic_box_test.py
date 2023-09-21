import logging
import time
from argparse import ArgumentParser
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Collection, Dict, Final, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from e7awgsw import AWG, AwgCtrl, CaptureCtrl, CaptureModule, CaptureParam, WaveSequence
from e7awgsw.exception import CaptureUnitTimeoutError

from quel_ic_config import (
    QUEL1_BOXTYPE_ALIAS,
    Quel1BoxType,
    Quel1ConfigOption,
    Quel1ConfigSubsystem,
    Quel1ConfigSubsystemRoot,
)
from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemAd9082Mixin

logger = logging.getLogger(__name__)

MAX_CAPTURE_RETRY = 5


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

    def initialize_awg(self, group: int):
        awgs = self._rmap.get_all_awgs_of_group(group)
        self._awg_ctrl.initialize(*awgs)
        self._awg_ctrl.terminate_awgs(*awgs)

    def initialize_cap(self, group: int):
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
    _BACKGROUND_NOISE_THRESHOLD: Final[float] = 200.0

    def __init__(self, css: Quel1ConfigSubsystemRoot, wss: Quel1WaveSubsystem):
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
                    continue

                if is_valid_cap:
                    max_backgroud_amplitude = max(abs(cap_data))
                    logger.info(f"max amplitude of capture data is {max_backgroud_amplitude}")
                    if max_backgroud_amplitude < background_noise_threshold:
                        logger.info(f"successful link-up of mxfe-{mxfe}")
                        return True
                break  # need to link-up again

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

    # TODO: should be drived from settings in the proper object.
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
            self.config_adc(0, 1.5e9)
            self.config_adc(1, 1.3e9)
        elif self._boxtype == Quel1BoxType.QuEL1_TypeB:
            self.config_port(1, lo_freq=11.5e9, cnco_freq=1.5e9, vatt=0xA00, sideband="L")  # 10GHz
            self.config_port(2, lo_freq=11.5e9, cnco_freq=1.55e9, vatt=0xA00, sideband="L")  # 9.95GHz
            self.config_port(3, lo_freq=11.5e9, cnco_freq=1.6e9, vatt=0xA00, sideband="L")  # 9.9GHz
            self.config_port(4, lo_freq=11.5e9, cnco_freq=1.65e9, vatt=0xA00, sideband="L")  # 9.85GHz
            self.config_port(8, lo_freq=11.5e9, cnco_freq=1.7e9, vatt=0xA00, sideband="L")  # 9.8GHz
            self.config_port(9, lo_freq=11.5e9, cnco_freq=1.75e9, vatt=0xA00, sideband="L")  # 9.75GHz
            self.config_port(10, lo_freq=11.5e9, cnco_freq=1.8e9, vatt=0xA00, sideband="L")  # 9.7GHz
            self.config_port(11, lo_freq=11.5e9, cnco_freq=1.85e9, vatt=0xA00, sideband="L")  # 9.65GHz
            self.config_adc(0, 1.5e9)
            self.config_adc(1, 1.7e9)
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

    def open_all(self) -> None:
        for p in (1, 2, 3, 4, 8, 9, 10, 11):
            self.open_port(p)

    def close_all(self) -> None:
        for p in (1, 2, 3, 4, 8, 9, 10, 11):
            self.close_port(p)

    def start_all(self) -> None:
        for p in (1, 2, 3, 4, 8, 9, 10, 11):
            self.start_channel(p)

    def stop_all(self) -> None:
        for p in (1, 2, 3, 4, 8, 9, 10, 11):
            self.stop_channel(p)

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

    def open_port(self, port) -> None:
        """open RF switch of the output port
        :param port: port to open
        :return:
        """
        mxfe, line = self._convert_port(port)
        self._css.pass_line(mxfe, line)

    def close_port(self, port):
        """close RF switch of the output port
        :param port: port to close
        :return:
        """
        mxfe, line = self._convert_port(port)
        self._css.block_line(mxfe, line)

    def _active_input_port(self, mxfe: int) -> str:
        if mxfe == 0:
            if Quel1ConfigOption.USE_READ_IN_MXFE0 in self._css._config_options:
                input_port: str = "r"
            else:
                input_port = "m"
        elif mxfe == 1:
            if Quel1ConfigOption.USE_READ_IN_MXFE1 in self._css._config_options:
                input_port = "r"
            else:
                input_port = "m"
        else:
            raise ValueError(f"invalid mxfe: {mxfe}")
        return input_port

    def config_adc(self, mxfe: int, cnco_freq: Union[float, None]):
        input_port: str = self._active_input_port(mxfe)
        if cnco_freq is not None:
            self._css.set_adc_cnco(mxfe, input_port, freq_in_hz=int(cnco_freq + 0.5))

    def activate_loopback(self, mxfe: int) -> None:
        input_port: str = self._active_input_port(mxfe)
        if input_port == "r":
            self._css.activate_read_loop(mxfe)
            logger.info(f"the output of port {1 if mxfe == 0 else 8} is blocked to activate read loop")
        elif input_port == "m":
            self._css.activate_monitor_loop(mxfe)
        else:
            raise AssertionError

    def start_channel(
        self,
        port: int,
        channel: int = 0,
        amplitude: float = 16384.0,
        num_repeat: Tuple[int, int] = (0xFFFFFFFF, 0xFFFFFFFF),
    ):
        """start to emit CW from the given port.
        :param port: port to emit CW
        :param channel: index of a channelizer to start.
        :param amplitude: amplitude of the CW
        :param num_repeat: number of repeats, see the specification of AWG units for details.
        :return: None
        """
        mxfe, line = self._convert_port(port)
        self._validate_channel(port, channel)
        self._wss.emit_cw(mxfe, line, channel, amplitude, num_repeat)

    def stop_channel(self, port: int, channel: Union[int, None] = None) -> None:
        """stop to emit CW.
        :param port: port to stop CW
        :param channel: index of channelizer to stop. all the channelizers of the given port are stopped if omitted.
        :return: None
        """
        mxfe, line = self._convert_port(port)
        if channel is None:
            for ch in range(self._LINE2NUM_CHANNELS[mxfe, line]):
                self._wss.stop_emission(mxfe, line, ch)
        else:
            self._wss.stop_emission(mxfe, line, channel)

    def loopback_fft(self, mxfe: int, graph_filename: Union[str, None]) -> npt.NDArray[np.complex128]:
        self.activate_loopback(mxfe)
        for _ in range(5):
            try:
                is_valid_cap, cap_data = self._wss.simple_capture(mxfe, 16384)
                if is_valid_cap:
                    break
            except CaptureUnitTimeoutError as e:
                logger.warning(e)
        else:
            raise RuntimeError("failed to capture the data repeatedly, abandoned")

        fp = np.abs(np.fft.fft(cap_data))
        ff = np.fft.fftfreq(len(cap_data)) * 500e6

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(ff, fp)
        plt.savefig(graph_filename)

        return np.vstack((ff, fp))


def init_box(
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    mxfe_combination: str,
    config_root: Path,
    config_options: Collection[Quel1ConfigOption],
) -> Tuple[bool, bool, Quel1ConfigSubsystemRoot, Quel1WaveSubsystem, LinkupFpgaMxfe, SimpleBoxTest]:
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
        css: Quel1ConfigSubsystem = Quel1ConfigSubsystem(ipaddr_css, boxtype, config_root, config_options)
    else:
        raise ValueError(f"unsupported boxtype: {boxtype}")

    if isinstance(css, Quel1ConfigSubsystemAd9082Mixin):
        rmap = Quel1E7ResourceMapping(css)
    else:
        raise AssertionError("the given ConfigSubsystem Object doesn't provide AD9082 interface")

    wss = Quel1WaveSubsystem(ipaddr_wss, rmap)
    linkupper = LinkupFpgaMxfe(css, wss)
    box: SimpleBoxTest = SimpleBoxTest(css, wss)

    linkup_ok = [False, False]
    css.configure_peripherals()
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
        wss.initialize_awg(mxfe)
        wss.initialize_cap(mxfe)

    for mxfe in mxfe_list:
        linkup_ok[mxfe] = linkupper.linkup_and_check(mxfe)

        # TODO: to be modified not to rely on box, namely should be written in terms of mxfe and line.
        if mxfe == 0:
            box.stop_channel(1)
            box.stop_channel(2)
            box.stop_channel(3)
            box.stop_channel(4)
        elif mxfe == 1:
            box.stop_channel(8)
            box.stop_channel(9)
            box.stop_channel(10)
            box.stop_channel(11)
        else:
            raise AssertionError

    # TODO: write scheduler object creation here.
    _ = ipaddr_sss

    return linkup_ok[0], linkup_ok[1], css, wss, linkupper, box


def parse_boxtype(boxtypename: str) -> Quel1BoxType:
    if boxtypename not in QUEL1_BOXTYPE_ALIAS:
        raise ValueError
    return Quel1BoxType.fromstr(boxtypename)


def parse_config_options(optstr: str) -> List[Quel1ConfigOption]:
    return [Quel1ConfigOption(s) for s in optstr.split(",") if s != ""]


def add_common_arguments(
    parser: ArgumentParser,
    use_ipaddr_wss: bool = True,
    use_ipaddr_sss: bool = True,
    use_ipaddr_css: bool = True,
    use_boxtype: bool = True,
    use_config_root: bool = True,
    use_config_options: bool = True,
) -> None:
    """adding common arguments to testlibs of quel_ic_config for manual tests. allowing to accept unused arguments for
    your convenience
    :param parser: ArgumentParser object to register arguments
    :param use_ipaddr_wss:
    :param use_ipaddr_sss:
    :param use_ipaddr_css:
    :param use_boxtype:
    :param use_config_root:
    :param use_config_options:
    :return:
    """

    non_existent_ipaddress = ip_address("241.3.5.6")

    if use_ipaddr_wss:
        parser.add_argument(
            "--ipaddr_wss",
            type=ip_address,
            required=True,
            help="IP address of the wave generation/capture subsystem of the target box",
        )
    else:
        parser.add_argument(
            "--ipaddr_wss", type=ip_address, required=False, default=non_existent_ipaddress, help="IGNORED"
        )

    if use_ipaddr_sss:
        parser.add_argument(
            "--ipaddr_sss",
            type=ip_address,
            required=True,
            help="IP address of the wave sequencer subsystem of the target box",
        )
    else:
        parser.add_argument(
            "--ipaddr_sss", type=ip_address, required=False, default=non_existent_ipaddress, help="IGNORED"
        )

    if use_ipaddr_css:
        parser.add_argument(
            "--ipaddr_css",
            type=ip_address,
            required=True,
            help="IP address of the configuration subsystem of the target box",
        )
    else:
        parser.add_argument(
            "--ipaddr_css", type=ip_address, required=False, default=non_existent_ipaddress, help="IGNORED"
        )

    if use_boxtype:
        parser.add_argument(
            "--boxtype",
            type=parse_boxtype,
            required=True,
            help=f"a type of the target box: either of "
            f"{', '.join([t for t in QUEL1_BOXTYPE_ALIAS if not t.startswith('x_')])}",
        )
    else:
        raise NotImplementedError

    if use_config_root:
        parser.add_argument(
            "--config_root",
            type=Path,
            default=Path("settings"),
            help="path to configuration file root",
        )
    else:
        raise NotImplementedError

    if use_config_options:
        parser.add_argument(
            "--config_options",
            type=parse_config_options,
            default=[],
            help=f"comma separated list of config options: ("
            f"{' '.join([o for o in Quel1ConfigOption if not o.startswith('x_')])})",
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    parser = argparse.ArgumentParser(
        description="check the basic functionalities about wave generation of QuEL-1 with a spectrum analyzer"
    )
    add_common_arguments(parser, use_ipaddr_sss=False)
    parser.add_argument(
        "--mxfe", choices=("0", "1", "both", "0,1", "1,0"), default="both", help="combination of MxFEs under test"
    )
    args = parser.parse_args()

    # Init: QuEL mxfe
    linkup_g0, linkup_g1, css, wss, linkupper, box = init_box(
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
