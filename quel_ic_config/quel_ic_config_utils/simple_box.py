import logging
from pathlib import Path
from typing import Any, Collection, Dict, Final, Sequence, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from quel_ic_config import (
    QubeConfigSubsystem,
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1AnyConfigSubsystem,
    Quel1BoxType,
    Quel1ConfigOption,
    Quel1ConfigSubsystem,
    Quel1Feature,
    Quel1NecConfigSubsystem,
    Quel1SeProto8ConfigSubsystem,
    Quel1SeProto11ConfigSubsystem,
    Quel1SeProtoAddaConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
)
from quel_ic_config.quel1_config_subsystem_common import Quel1ConfigSubsystemAd9082Mixin
from quel_ic_config_utils.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config_utils.linkupper import LinkupFpgaMxfe
from quel_ic_config_utils.quel1_wave_subsystem import CaptureReturnCode, E7FwLifeStage, E7FwType, Quel1WaveSubsystem

logger = logging.getLogger(__name__)


class SimpleBoxIntrinsic:
    DEFAULT_AMPLITUDE: Final[float] = 16383.0
    DEFAULT_REPEATS: Final[Tuple[int, int]] = (0xFFFFFFFF, 0xFFFFFFFF)
    DEFAULT_NUM_CAPTURE_SAMPLES: Final[int] = 4096
    VERY_LONG_DURATION: float = 0x10000000000000000 * 128e-9

    def __init__(
        self, css: QubeConfigSubsystem, wss: Quel1WaveSubsystem, rmap: Union[Quel1E7ResourceMapper, None] = None
    ):
        self._css = css
        self._wss = wss
        if rmap is None:
            rmap = Quel1E7ResourceMapper(css, wss)
        self._rmap = rmap

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._wss._wss_addr}>"

    @property
    def css(self) -> QubeConfigSubsystem:
        return self._css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._wss

    @property
    def rmap(self) -> Quel1E7ResourceMapper:
        return self._rmap

    # TODO: update doc
    def init(
        self,
        mxfe_list: Union[Collection[int], None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
    ) -> Dict[int, bool]:
        """establish a configuration link between a box and host.
        You need to initialize all the ICs and to establish datalink between AD9082 and FPGA
        (a.k.a. linking-up the box) in advance.

        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored.
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed.
        :return: True if success
        """
        if mxfe_list is None:
            mxfe_list = self._css.get_all_mxfes()

        if ignore_crc_error_of_mxfe is None:
            ignore_crc_error_of_mxfe = {}

        if ignore_extraordinary_converter_select_of_mxfe is None:
            ignore_extraordinary_converter_select_of_mxfe = {}

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
                logger.error(f"failed to establish a configuration link with AD9082-#{mxfe_idx}")

            if not valid_link:
                logger.error(f"AD9082-#{mxfe_idx} is not working. it must be linked up in advance")
            link_ok[mxfe_idx] = valid_link

        return link_ok

    @staticmethod
    def _calc_wave_repeats(duration_in_sec: float, num_samples: int) -> Tuple[int, int]:
        unit_len = num_samples / 500e6
        duration_in_unit = duration_in_sec / unit_len
        u = round(duration_in_unit)
        v = 1
        while u > 0xFFFFFFFF and v <= 0xFFFFFFFF:
            u //= 2
            v *= 2
        if v > 0xFFFFFFFF:
            return (0xFFFFFFFF, 0xFFFFFFFF)
        else:
            return (u, v)

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
        :param amplitude: the amplitude of the sinusoidal wave to be passed to DAC.
        :param duration: the duration of wave generation in second.
        :param control_port_rfswitch: allowing the port corresponding to the line to emit the RF signal if True.
        :param control_monitor_rfswitch: allowing the monitor-out port to emit the RF signal if True.
        :return: None
        """
        self.config_line(group=group, line=line, lo_freq=lo_freq, cnco_freq=cnco_freq, vatt=vatt, sideband=sideband)
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
        self.start_channel(group, line, channel, amplitude=amplitude, num_repeat=num_repeats)

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
        self.stop_channel(group, line, channel)
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
        lo_freq: Union[float, None],
        cnco_freq: Union[float, None],
        fnco_freq: Union[float, None],
        activate_internal_loop: Union[None, bool],
        num_samples: int = DEFAULT_NUM_CAPTURE_SAMPLES,
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

    def config_line(
        self,
        group: int,
        line: int,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
    ) -> None:
        """configuring parameters of a given transmitter line.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        :param lo_freq: frequency of the corresponding local oscillator in Hz, must be multiple of 100_000_000.
        :param cnco_freq: frequency of the corresponding CNCO in Hz.
        :param vatt: controlling voltage of the corresponding VATT in unit of 3.3V / 4096. see the specification
                     sheet of the ADRF6780 for details.
        :param sideband: "U" for upper side band, "L" for lower side band.
        :return: None
        """
        if vatt is not None:
            self._css.set_vatt(group, line, vatt)
        if sideband is not None:
            self._css.set_sideband(group, line, sideband)
        if lo_freq is not None:
            if lo_freq % 100000000 != 0:
                raise ValueError("lo_freq must be multiple of 100000000")
            self._css.set_lo_multiplier(group, line, int(lo_freq) // 100000000)
        if cnco_freq is not None:
            self._css.set_dac_cnco(group, line, round(cnco_freq))

    def config_channel(self, group: int, line: int, channel: int = 0, fnco_freq: Union[float, None] = None) -> None:
        """configuring parameters of a given transmitter channel.

        :param group: a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :return: None
        """
        if fnco_freq is not None:
            self._css.set_dac_fnco(group, line, channel, round(fnco_freq))

    def config_rline(
        self, group: int, rline: str, *, lo_freq: Union[float, None], cnco_freq: Union[float, None]
    ) -> None:
        """configuring parameters of a given receiver line.

        :param group: an index of a group which the line belongs to.
        :param rline: a group-local index of the line.
        :param lo_freq: frequency of the corresponding local oscillator in Hz, must be multiple of 100_000_000.
        :param cnco_freq: frequency of the corresponding CNCO in Hz.
        :return: None
        """
        if lo_freq is not None:
            if lo_freq % 100000000 != 0:
                raise ValueError("lo_freq must be multiple of 100000000")
            self._css.set_lo_multiplier(group, rline, int(lo_freq) // 100000000)
        if cnco_freq is not None:
            self._css.set_adc_cnco(group, rline, freq_in_hz=round(cnco_freq))

    def config_runit(self, group: int, rline: str, runit: int = 0, *, fnco_freq: Union[float, None]) -> None:
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
            self._css.set_adc_fnco(group, rline, rchannel, freq_in_hz=round(fnco_freq))

    def open_rfswitch(self, group: int, line: Union[int, str]):
        """opening RF switch of the port corresponding to a given line, either of transmitter or receiver one.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        return None
        """
        if isinstance(self._css, Quel1ConfigSubsystem):
            self._css.pass_line(group, line)
        else:
            logger.info("do nothing because no RF switches are available")

    def close_rfswitch(self, group: int, line: Union[int, str]):
        """closing RF switch of the port corresponding to a given line, either of transmitter or receiver one.

        :param group: an index of a group which the line belongs to.
        :param line: a group-local index of the line.
        return None
        """
        if isinstance(self._css, Quel1ConfigSubsystem):
            self._css.block_line(group, line)
        else:
            logger.info("do nothing because no RF switches are available")

    def start_channel(
        self,
        group: int,
        line: int,
        channel: int = 0,
        amplitude: float = DEFAULT_AMPLITUDE,
        num_repeat: Tuple[int, int] = DEFAULT_REPEATS,
    ) -> None:
        """starting to generate continuous wave signal from a given channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :param amplitude: amplitude of the continuous wave, 0 -- 32767.n
        :param num_repeat: number of repetitions of a unit wave data (128ns). given as a tuple
               of two integers that specifies the number of repetition as multiple of the two.
        :return: None
        """
        awg = self._rmap.get_awg_of_channel(group, line, channel)
        self._wss.simple_cw_gen(awg, amplitude, num_repeat)

    def start_channel_with_wave(
        self,
        group: int,
        line: int,
        channel: int,
        wave: npt.NDArray[np.complex64],
        num_repeat: Tuple[int, int] = DEFAULT_REPEATS,
    ) -> None:
        """starting to generate the signal as repetitions of a given wave data from a given channel

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :param wave: complex data of the signal to generate in 500Msps. I and Q coefficients of each sample must be
                     within the range of -32768 -- 32767. its length must be a multiple of 64.
        :param num_repeat: the number of repetitions of the given wave data given as a tuple of two integers,
                           a product of the two is the number of repetitions.
        :return: None
        """
        awg = self._rmap.get_awg_of_channel(group, line, channel)
        self._wss.simple_iq_gen(awg, wave, num_repeat)

    def stop_channel(self, group: int, line: int, channel: Union[int, None] = None) -> None:
        """stopping signal generation on a given channel.

        :param group: an index of a group which the channel belongs to.
        :param line: a group-local index of a line which the channel belongs to.
        :param channel: a line-local index of the channel.
        :return: None
        """
        if channel is None:
            for ch in range(self._css.get_num_channels_of_line(group, line)):
                awg = self._rmap.get_awg_of_channel(group, line, ch)
                self._wss.stop_emission({awg})
        else:
            awg = self._rmap.get_awg_of_channel(group, line, channel)
            self._wss.stop_emission({awg})

    def activate_monitor_loop(self, group: int) -> None:
        """enabling an internal monitor loop-back path from a monitor-out port to a monitor-in port.

        :param group: an index of a group which the monitor path belongs to.
        :return: None
        """
        if isinstance(self._css, Quel1ConfigSubsystem):
            self._css.activate_monitor_loop(group)
        else:
            logger.info("do nothing because no RF switches are available")

    def deactivate_monitor_loop(self, group: int) -> None:
        """disabling an internal monitor loop-back path.

        :param group: a group which the monitor path belongs to.
        :return: None
        """
        if isinstance(self._css, Quel1ConfigSubsystem):
            self._css.deactivate_monitor_loop(group)
        else:
            logger.info("do nothing because no RF switches are available")

    def is_loopedback_monitor(self, group: int) -> bool:
        """checking if an internal monitor loop-back path is activated or not.

        :param group: an index of a group which the monitor loop-back path belongs to.
        :return: True if the monitor loop-back path is activated.
        """
        if isinstance(self._css, Quel1ConfigSubsystem):
            return self._css.is_loopedback_monitor(group)
        else:
            return False

    def activate_read_loop(self, group: int) -> None:
        """enabling an internal read loop-back path from read-out port to read-in port.

        :param group: an index of a group which the read path belongs to.
        :return: None
        """
        if isinstance(self._css, Quel1ConfigSubsystem):
            self._css.activate_read_loop(group)
        else:
            logger.info("do nothing because no RF switches are available")

    def deactivate_read_loop(self, group: int) -> None:
        """disabling an internal read loop-back.

        :param group: an index of a group which the read path belongs to.
        :return: None
        """
        if isinstance(self._css, Quel1ConfigSubsystem):
            self._css.deactivate_read_loop(group)
        else:
            logger.info("do nothing because no RF switches are available")

    def is_loopedback_read(self, group: int) -> bool:
        """checking if an internal read loop-back path is activated or not.

        :param group: an index of a group which the read loop-back path belongs to.
        :return: True if the read loop-back path is activated.
        """
        if isinstance(self._css, Quel1ConfigSubsystem):
            return self._css.is_loopedback_read(group)
        else:
            return False

    def dump_config(self) -> Dict[str, Dict[int, Dict[Union[int, str], Dict[str, Any]]]]:
        """dumping the current configuration of the box.
        :return: the current configuration of the box in dictionary.
        """
        retval: Dict[str, Dict[int, Dict[Union[int, str], Dict[str, Any]]]] = {"mxfes": {}, "lines": {}}

        for mxfe_idx in self.css.get_all_mxfes():
            retval["mxfes"][mxfe_idx] = {
                "config": {
                    "channel_interporation_rate": self._css.get_channel_interpolation_rate(mxfe_idx),
                    "main_interporation_rate": self._css.get_main_interpolation_rate(mxfe_idx),
                }
            }

        for group in self.css.get_all_groups():
            retval["lines"][group] = {}
            for line in self._css.get_all_lines_of_group(group):
                retval["lines"][group][line] = self._css.dump_line(group, line)
            for rline in self._css.get_all_rlines_of_group(group):
                retval["lines"][group][rline] = self._css.dump_rline(group, rline)

        return retval


class SimpleBox:
    _PORT2LINE_QuBE_OU_TypeA: Dict[int, Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        1: (0, "r"),
        2: (0, 1),
        5: (0, 2),
        6: (0, 3),
        7: (1, 3),
        8: (1, 2),
        11: (1, 1),
        12: (1, "r"),
        13: (1, 0),
    }

    _PORT2LINE_QuBE_OU_TypeB: Dict[int, Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        2: (0, 1),
        5: (0, 2),
        6: (0, 3),
        7: (1, 3),
        8: (1, 2),
        11: (1, 1),
        13: (1, 0),
    }

    _PORT2LINE_QuBE_RIKEN_TypeA: Dict[int, Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        1: (0, "r"),
        2: (0, 1),
        4: (0, "m"),
        5: (0, 2),
        6: (0, 3),
        7: (1, 3),
        8: (1, 2),
        9: (1, "m"),
        11: (1, 1),
        12: (1, "r"),
        13: (1, 0),
    }

    _PORT2LINE_QuBE_RIKEN_TypeB: Dict[int, Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        2: (0, 1),
        4: (0, "m"),
        5: (0, 2),
        6: (0, 3),
        7: (1, 3),
        8: (1, 2),
        9: (1, "m"),
        11: (1, 1),
        13: (1, 0),
    }

    _PORT2LINE_QuEL1_TypeA: Dict[int, Tuple[int, Union[int, str]]] = {
        0: (0, "r"),
        1: (0, 0),
        2: (0, 2),
        3: (0, 1),
        4: (0, 3),
        5: (0, "m"),
        7: (1, "r"),
        8: (1, 0),
        9: (1, 3),
        10: (1, 1),
        11: (1, 2),
        12: (1, "m"),
    }

    _PORT2LINE_QuEL1_TypeB: Dict[int, Tuple[int, Union[int, str]]] = {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (0, 3),
        5: (0, "m"),
        8: (1, 0),
        9: (1, 1),
        10: (1, 3),
        11: (1, 2),
        12: (1, "m"),
    }

    _PORT2LINE: Final[Dict[Quel1BoxType, Dict[int, Tuple[int, Union[int, str]]]]] = {
        Quel1BoxType.QuBE_OU_TypeA: _PORT2LINE_QuBE_OU_TypeA,
        Quel1BoxType.QuBE_OU_TypeB: _PORT2LINE_QuBE_OU_TypeB,
        Quel1BoxType.QuBE_RIKEN_TypeA: _PORT2LINE_QuBE_RIKEN_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeB: _PORT2LINE_QuBE_RIKEN_TypeB,
        Quel1BoxType.QuEL1_TypeA: _PORT2LINE_QuEL1_TypeA,
        Quel1BoxType.QuEL1_TypeB: _PORT2LINE_QuEL1_TypeB,
    }

    def __init__(
        self, css: QubeConfigSubsystem, wss: Quel1WaveSubsystem, rmap: Union[Quel1E7ResourceMapper, None] = None
    ):
        self._dev = SimpleBoxIntrinsic(css, wss, rmap)
        self._boxtype = css._boxtype
        if self._boxtype not in self._PORT2LINE:
            raise ValueError(f"Unsupported boxtype; {self._boxtype}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._dev._wss._wss_addr}>"

    @property
    def css(self) -> QubeConfigSubsystem:
        return self._dev.css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._dev.wss

    def init(
        self,
        mxfe_list: Union[Collection[int], None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
    ) -> Dict[int, bool]:
        """establish a configuration link between a box and host. You need to establish datalink between AD9082
        and FPGA (a.k.a. linking-up the box) in advance.
        :param ignore_crc_error_of_mxfe: list of MxFEs whose CRC error of the datalink is ignored.
        :return: True if success
        """
        return self._dev.init(
            mxfe_list=mxfe_list,
            ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinary_converter_select_of_mxfe,
        )

    def _convert_tx_port(self, port: int) -> Tuple[int, int]:
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid output port: {port}")
        group, line = self._PORT2LINE[self._boxtype][port]
        if not isinstance(line, int):
            raise ValueError(f"invalid output port: {port}")
        return group, line

    def _convert_rx_port(self, port: int) -> Tuple[int, str]:
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid output port: {port}")
        group, line = self._PORT2LINE[self._boxtype][port]
        if not isinstance(line, str):
            raise ValueError(f"invalid output port: {port}")
        return group, line

    def _convert_all_port(self, port: int) -> Tuple[int, Union[int, str]]:
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid output port: {port}")
        return self._PORT2LINE[self._boxtype][port]

    def easy_start_cw(
        self,
        port: int,
        channel: int,
        *,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        fnco_freq: Union[float, None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
        amplitude: float = SimpleBoxIntrinsic.DEFAULT_AMPLITUDE,
        duration: float = SimpleBoxIntrinsic.VERY_LONG_DURATION,
        control_port_rfswitch: bool = True,
        control_monitor_rfswitch: bool = False,
    ) -> None:
        """an easy-to-use API to generate continuous wave from a given channel.

        :param port: an index of port which the channel belongs to.
        :param channel: a port-local index of the channel in the line.
        :param lo_freq: the frequency of the corresponding local oscillator in Hz. it must be multiple of 100_000_000.
        :param cnco_freq: the frequency of the corresponding CNCO in Hz.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz.
        :param vatt: the control voltage of the corresponding VATT in unit of 3.3V / 4096.
        :param sideband: the active sideband of the corresponding mixer, "U" for upper and "L" for lower.
        :param amplitude: the amplitude of the sinusoidal wave to be passed to DAC.
        :param duration: the duration of wave generation in second.
        :param control_port_rfswitch: allowing the port corresponding to the line to emit the RF signal if True.
        :param control_monitor_rfswitch: allowing the monitor-out port to emit the RF signal if True.
        :return: None
        """

        group, line = self._convert_tx_port(port)
        self._dev.easy_start_cw(
            group,
            line,
            channel,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            fnco_freq=fnco_freq,
            vatt=vatt,
            sideband=sideband,
            amplitude=amplitude,
            duration=duration,
            control_port_rfswitch=control_port_rfswitch,
            control_monitor_rfswitch=control_monitor_rfswitch,
        )

    def easy_stop(
        self,
        port: int,
        channel: Union[int, None] = None,
        *,
        control_port_rfswitch: bool = True,
        control_monitor_rfswitch: bool = False,
    ) -> None:
        """stopping the wave generation on a given channel.

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel. stop the signal generation of all the channel of
                        the port if None.
        :param control_port_rfswitch: blocking the emission of the RF signal from the corresponding port if True.
        :param control_monitor_rfswitch: blocking the emission of the RF signal from the monitor-out port if True.
        :return: None
        """
        group, line = self._convert_tx_port(port)
        self._dev.easy_stop(
            group, line, control_port_rfswitch=control_port_rfswitch, control_monitor_rfswitch=control_monitor_rfswitch
        )

    def easy_stop_all(self, control_port_rfswitch: bool = True) -> None:
        """stopping the signal generation on all the channels of the box.

        :param control_port_rfswitch: blocking the emission of the RF signal from the corresponding port if True.
        :return: None
        """
        self._dev.easy_stop_all(control_port_rfswitch=control_port_rfswitch)

    def easy_capture(
        self,
        port: int,
        rchannel: int = 0,
        *,
        lo_freq: Union[float, None],
        cnco_freq: Union[float, None],
        fnco_freq: Union[float, None],
        activate_internal_loop: Union[None, bool],
        num_samples: int = SimpleBoxIntrinsic.DEFAULT_NUM_CAPTURE_SAMPLES,
    ) -> npt.NDArray[np.complex64]:
        """capturing the wave signal from a given receiver channel.

        :param port: an index of a port which the channel belongs to.
        :param rchannel: a port-local index of the channel.
        :param lo_freq: the frequency of the corresponding local oscillator in Hz. it must be multiple of 100_000_000.
        :param cnco_freq: the frequency of the corresponding CNCO in Hz.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz.
        :param activate_internal_loop: activate the corresponding loop-back path if True
        :param num_samples: number of samples to capture
        :return: captured wave data in NumPy array
        """

        group, rline = self._convert_rx_port(port)
        try:
            rrline: str = self._dev.rmap.resolve_rline(group, None)
            if rrline != rline:
                logger.warning(
                    f"the specified port {port} may not be connected to any ADC under the current configuration"
                )
        except ValueError:
            pass

        return self._dev.easy_capture(
            group,
            rline,
            rchannel,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            fnco_freq=fnco_freq,
            activate_internal_loop=activate_internal_loop,
            num_samples=num_samples,
        )

    def config_port(
        self,
        port: int,
        *,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
    ) -> None:
        """configuring parameters of a given port, either of transmitter or receiver one.

        :param port: an index of the target port to configure.
        :param lo_freq: the frequency of the corresponding local oscillator in Hz, must be multiple of 100_000_000.
        :param cnco_freq: the frequency of the corresponding CNCO in Hz.
        :param vatt: controlling voltage of the corresponding VATT in unit of 3.3V / 4096. see the specification sheet
                     of the ADRF6780 for details. (only for transmitter port)
        :param sideband: "U" for upper side band, "L" for lower side band. (only for transmitter port)
        :return: None
        """
        group, line = self._convert_all_port(port)
        if isinstance(line, int):
            self._dev.config_line(group, line, lo_freq=lo_freq, cnco_freq=cnco_freq, vatt=vatt, sideband=sideband)
        elif isinstance(line, str):
            if vatt is not None or sideband is not None:
                raise ValueError("No mixer is avalable for the port {port}")
            self._dev.config_rline(group, line, lo_freq=lo_freq, cnco_freq=cnco_freq)
        else:
            raise AssertionError

    def config_channel(self, port: int, channel: int = 0, *, fnco_freq: Union[float, None] = None):
        """configuring parameters of a given channel, either of transmitter or receiver one.

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :return: None
        """
        group, line = self._convert_all_port(port)
        if isinstance(line, int):
            self._dev.config_channel(group, line, channel, fnco_freq=fnco_freq)
        elif isinstance(line, str):
            self._dev.config_runit(group, line, channel, fnco_freq=fnco_freq)
        else:
            raise AssertionError

    def open_rfswitch(self, port: int):
        """opening RF switch of a given port.

        :param port: an index of the target port.
        return None
        """
        group, line = self._convert_all_port(port)
        self._dev.open_rfswitch(group, line)

    def close_rfswitch(self, port: int):
        """closing RF switch of a given port.

        :param port: an index of the target port.
        return None
        """
        group, line = self._convert_all_port(port)
        self._dev.close_rfswitch(group, line)

    def start_channel(
        self,
        port: int,
        channel: int = 0,
        amplitude: float = SimpleBoxIntrinsic.DEFAULT_AMPLITUDE,
        num_repeat: Tuple[int, int] = SimpleBoxIntrinsic.DEFAULT_REPEATS,
    ):
        """starting to generate continuous wave signal from a given channel

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel.
        :param amplitude: amplitude of the continuous wave, 0 -- 32767.
        :param num_repeat: the number of repetitions of a unit wave data (128ns). given as a tuple of two integers that
                            specifies the number of repetition as a product of the two.
        :return: None
        """
        group, line = self._convert_tx_port(port)
        self._dev.start_channel(group, line, channel, amplitude, num_repeat)

    def start_channel_with_wave(
        self,
        port: int,
        channel: int,
        wave: npt.NDArray[np.complex64],
        num_repeat: Tuple[int, int] = SimpleBoxIntrinsic.DEFAULT_REPEATS,
    ):
        """starting to generate the signal as repetitions of a given wave data from a given port with a given channel.

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel.
        :param wave: complex data of the signal to generate in 500Msps. I and Q coefficients of each sample must be
                     within the range of -32768 -- 32767. its length must be a multiple of 64.
        :param num_repeat: number of repetitions of the given wave data given as a tuple of two integers,
                           product of the two is the actual number of repetitions.
        :return: None
        """
        group, line = self._convert_tx_port(port)
        self._dev.start_channel_with_wave(group, line, channel, wave, num_repeat)

    def stop_channel(self, port: int, channel: Union[int, None] = None) -> None:
        """stopping signal generation on a given channel.

        :param group: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel.
        :return: None
        """
        group, line = self._convert_tx_port(port)
        self._dev.stop_channel(group, line, channel)

    def activate_monitor_loop(self, group: int):
        """enabling an internal monitor loop-back path from a monitor-out port to a monitor-in port.

        :param group: an index of a group which the monitor path belongs to.
        :return: None
        """
        self._dev.activate_monitor_loop(group)

    def deactivate_monitor_loop(self, group: int):
        """disabling an internal monitor loop-back path.

        :param group: a group which the monitor path belongs to.
        :return: None
        """
        self._dev.deactivate_monitor_loop(group)

    def is_loopedback_monitor(self, group: int) -> bool:
        """checking if an internal monitor loop-back path is activated or not.

        :param group: an index of a group which the monitor loop-back path belongs to.
        :return: True if the monitor loop-back path is activated.
        """
        return self._dev.is_loopedback_monitor(group)

    def activate_read_loop(self, group: int):
        """enabling an internal read loop-back path from read-out port to read-in port.

        :param group: an index of a group which the read path belongs to.
        :return: None
        """
        self._dev.activate_read_loop(group)

    def deactivate_read_loop(self, group: int):
        """disabling an internal read loop-back.

        :param group: an index of a group which the read path belongs to.
        :return: None
        """
        self._dev.deactivate_read_loop(group)

    def is_loopedback_read(self, group: int) -> bool:
        """checking if an internal read loop-back path is activated or not.

        :param group: an index of a group which the read loop-back path belongs to.
        :return: True if the read loop-back path is activated.
        """
        return self._dev.is_loopedback_read(group)

    def dump_config(self) -> Dict[str, Dict[str, Any]]:
        """dumping the current configuration of the ports

        :return: the current configuration of ports in dictionary.
        """
        retval: Dict[str, Dict[str, Any]] = {}

        for mxfe_idx in self.css.get_all_mxfes():
            mxfe_name = f"mxfe-#{mxfe_idx}"
            retval[mxfe_name] = {
                "channel_interporation_rate": self._dev._css.get_channel_interpolation_rate(mxfe_idx),
                "main_interporation_rate": self._dev._css.get_main_interpolation_rate(mxfe_idx),
            }

        groups = self.css.get_all_groups()
        for port, (group, line) in self._PORT2LINE[self._boxtype].items():
            if group not in groups:
                continue
            port_name = f"port-#{port:02d}"
            if isinstance(line, int):
                retval[port_name] = {"direction": "out"}
                retval[port_name].update(self._dev._css.dump_line(group, line))
            elif isinstance(line, str):
                retval[port_name] = {"direction": "in"}
                retval[port_name].update(self._dev._css.dump_rline(group, line))
            else:
                raise AssertionError

        return retval


def init_box_with_linkup(
    *,
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    mxfes_to_linkup: Sequence[int],
    config_root: Union[Path, None],
    config_options: Collection[Quel1ConfigOption],
    use_204b: bool = True,
    skip_init: bool = False,
    background_noise_threshold: Union[float, None] = None,
    ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
    ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
    ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ignore_extraordinal_converter_select_of_mxfe: Union[Collection[int], None] = None,
    refer_by_port: bool = True,
) -> Tuple[
    Dict[int, bool],
    Quel1AnyConfigSubsystem,
    Quel1WaveSubsystem,
    Quel1E7ResourceMapper,
    LinkupFpgaMxfe,
    Union[SimpleBox, SimpleBoxIntrinsic, None],
]:
    """create QuEL testing objects, reset all the ICs, and establish datalink.

    :param ipaddr_wss: IP address of the wave generation subsystem of the target box
    :param ipaddr_sss: IP address of the sequencer subsystem of the target box
    :param ipaddr_css: IP address of the configuration subsystem of the target box
    :param boxtype: type of the target box
    :param mxfes_to_linkup: target mxfes of the target box
    :param config_root: root path of config setting files to read
    :param config_options: a collection of config options
    :param use_204b: choose JESD204B link or 204C one
    :param refer_by_port: return an object which takes port index for specifying input and output site if True.
    :return: QuEL testing objects
    """

    css, wss, rmap, linkupper, box = create_box_objects(
        ipaddr_wss=ipaddr_wss,
        ipaddr_sss=ipaddr_sss,
        ipaddr_css=ipaddr_css,
        boxtype=boxtype,
        config_root=config_root,
        config_options=config_options,
        refer_by_port=refer_by_port,
    )

    linkup_ok = linkup(
        linkupper=linkupper,
        mxfe_list=mxfes_to_linkup,
        use_204b=use_204b,
        skip_init=skip_init,
        background_noise_threshold=background_noise_threshold,
        ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
        ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
        ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        ignore_extraordinal_converter_select_of_mxfe=ignore_extraordinal_converter_select_of_mxfe,
    )

    return linkup_ok, css, wss, rmap, linkupper, box


def init_box_with_reconnect(
    *,
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    mxfes_to_connect: Union[Sequence[int], None] = None,
    ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
    ignore_extraordinal_converter_select_of_mxfe: Union[Collection[int], None] = None,
    refer_by_port: bool = True,
) -> Tuple[
    Dict[int, bool],
    Quel1AnyConfigSubsystem,
    Quel1WaveSubsystem,
    Quel1E7ResourceMapper,
    LinkupFpgaMxfe,
    Union[SimpleBox, SimpleBoxIntrinsic, None],
]:
    css, wss, rmap, linkupper, box = create_box_objects(
        ipaddr_wss=ipaddr_wss,
        ipaddr_sss=ipaddr_sss,
        ipaddr_css=ipaddr_css,
        boxtype=boxtype,
        config_root=None,
        config_options={},
        refer_by_port=refer_by_port,
    )

    if mxfes_to_connect is None:
        mxfes_to_connect = list(css.get_all_groups())
        mxfes_to_connect.sort()

    if box is not None:
        link_ok: Dict[int, bool] = reconnect(
            box=box,
            mxfe_list=mxfes_to_connect,
            ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinal_converter_select_of_mxfe,
        )
    else:
        link_ok = reconnect(
            css=css,
            wss=wss,
            rmap=rmap,
            mxfe_list=mxfes_to_connect,
            ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinal_converter_select_of_mxfe,
        )

    # TODO: return link_ok as dict in the next minor version update.
    return link_ok, css, wss, rmap, linkupper, box


def create_box_objects(
    ipaddr_wss: str,
    ipaddr_sss: str,
    ipaddr_css: str,
    boxtype: Quel1BoxType,
    config_root: Union[Path, None],
    config_options: Collection[Quel1ConfigOption],
    refer_by_port: bool = True,
) -> Tuple[
    Quel1AnyConfigSubsystem,
    Quel1WaveSubsystem,
    Quel1E7ResourceMapper,
    LinkupFpgaMxfe,
    Union[SimpleBox, SimpleBoxIntrinsic, None],
]:
    """create QuEL testing objects and initialize it

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
        Quel1BoxType.QuBE_OU_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeA,
        Quel1BoxType.QuEL1_TypeA,
        Quel1BoxType.QuBE_OU_TypeB,
        Quel1BoxType.QuBE_RIKEN_TypeB,
        Quel1BoxType.QuEL1_TypeB,
        Quel1BoxType.QuEL1_NEC,
        Quel1BoxType.QuEL1SE_Adda,
        Quel1BoxType.QuEL1SE_Proto8,
        Quel1BoxType.QuEL1SE_Proto11,
    }:
        wss: Quel1WaveSubsystem = Quel1WaveSubsystem(ipaddr_wss)
    else:
        raise ValueError(f"unsupported boxtype: {boxtype}")

    if wss.hw_lifestage == E7FwLifeStage.TO_DEPRECATE:
        logger.warning(f"the firmware will deprecate soon, consider to update it as soon as possible: {wss.hw_version}")
    elif wss.hw_lifestage == E7FwLifeStage.EXPERIMENTAL:
        logger.warning(f"be aware that the firmware is still in an experimental stage: {wss.hw_version}")

    wss.validate_installed_e7awgsw()
    features: Set[Quel1Feature] = set()
    if wss.hw_type in {E7FwType.SIMPLEMULTI_CLASSIC}:
        features.add(Quel1Feature.SINGLE_ADC)
    elif wss.hw_type in {E7FwType.FEEDBACK_VERYEARLY}:
        features.add(Quel1Feature.BOTH_ADC_EARLY)
    elif wss.hw_type in {E7FwType.FEEDBACK_EARLY}:
        features.add(Quel1Feature.BOTH_ADC)
    else:
        raise ValueError(f"unsupported firmware is detected: {wss.hw_type}")

    if boxtype in {
        Quel1BoxType.QuBE_RIKEN_TypeA,
        Quel1BoxType.QuEL1_TypeA,
    }:
        css: Quel1AnyConfigSubsystem = Quel1TypeAConfigSubsystem(
            ipaddr_css, boxtype, features, config_root, config_options
        )
    elif boxtype in {
        Quel1BoxType.QuBE_RIKEN_TypeB,
        Quel1BoxType.QuEL1_TypeB,
    }:
        css = Quel1TypeBConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuBE_OU_TypeA:
        css = QubeOuTypeAConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuBE_OU_TypeB:
        css = QubeOuTypeBConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1_NEC:
        css = Quel1NecConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_Adda:
        css = Quel1SeProtoAddaConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_Proto8:
        css = Quel1SeProto8ConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    elif boxtype == Quel1BoxType.QuEL1SE_Proto11:
        css = Quel1SeProto11ConfigSubsystem(ipaddr_css, boxtype, features, config_root, config_options)
    else:
        raise ValueError(f"unsupported boxtype: {boxtype}")

    if not isinstance(css, Quel1ConfigSubsystemAd9082Mixin):
        raise AssertionError("the given ConfigSubsystem Object doesn't provide AD9082 interface")

    rmap = Quel1E7ResourceMapper(css, wss)
    linkupper = LinkupFpgaMxfe(css, wss, rmap)

    if isinstance(css, QubeConfigSubsystem):
        if refer_by_port:
            box: Union[SimpleBox, SimpleBoxIntrinsic, None] = SimpleBox(css, wss, rmap)
        else:
            box = SimpleBoxIntrinsic(css, wss, rmap)
    else:
        box = None

    # TODO: write scheduler object creation here.
    _ = ipaddr_sss

    return css, wss, rmap, linkupper, box


def linkup(
    *,
    linkupper: LinkupFpgaMxfe,
    mxfe_list: Sequence[int],
    use_204b: bool = True,
    skip_init: bool = False,
    background_noise_threshold: Union[float, None] = None,
    ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
    ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
    ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
    ignore_extraordinal_converter_select_of_mxfe: Union[Collection[int], None] = None,
    save_dirpath: Union[Path, None] = None,
) -> Dict[int, bool]:
    if ignore_crc_error_of_mxfe is None:
        ignore_crc_error_of_mxfe = {}

    if ignore_access_failure_of_adrf6780 is None:
        ignore_access_failure_of_adrf6780 = {}

    if ignore_lock_failure_of_lmx2594 is None:
        ignore_lock_failure_of_lmx2594 = {}

    if ignore_extraordinal_converter_select_of_mxfe is None:
        ignore_extraordinal_converter_select_of_mxfe = {}

    if not skip_init:
        linkupper._css.configure_peripherals(
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )
        linkupper._css.configure_all_mxfe_clocks(
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
        )

    linkup_ok: Dict[int, bool] = {}
    for mxfe in mxfe_list:
        linkup_ok[mxfe] = linkupper.linkup_and_check(
            mxfe,
            use_204b=use_204b,
            background_noise_threshold=background_noise_threshold,
            ignore_crc_error=mxfe in ignore_crc_error_of_mxfe,
            ignore_extraordinal_converter_select=mxfe in ignore_extraordinal_converter_select_of_mxfe,
            save_dirpath=save_dirpath,
        )

    return linkup_ok


def reconnect(
    *,
    box: Union[SimpleBox, SimpleBoxIntrinsic, None] = None,
    css: Union[Quel1AnyConfigSubsystem, None] = None,
    wss: Union[Quel1WaveSubsystem, None] = None,
    rmap: Union[Quel1E7ResourceMapper, None] = None,
    mxfe_list: Sequence[int],
    ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
    ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
) -> Dict[int, bool]:
    if ignore_crc_error_of_mxfe is None:
        ignore_crc_error_of_mxfe = {}

    if ignore_extraordinary_converter_select_of_mxfe is None:
        ignore_extraordinary_converter_select_of_mxfe = {}

    if box is not None:
        return box.init(
            mxfe_list=mxfe_list,
            ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinary_converter_select_of_mxfe,
        )
    elif css is not None and wss is not None and rmap is not None:
        if mxfe_list is None:
            mxfe_list = list(css.get_all_groups())
            mxfe_list.sort()
        link_ok: Dict[int, bool] = {}
        for g in mxfe_list:
            rmap.validate_configuration_integrity(
                g, ignore_extraordinary_converter_select=g in ignore_extraordinary_converter_select_of_mxfe
            )
            try:
                if not css.configure_mxfe(g, ignore_crc_error=g in ignore_crc_error_of_mxfe):
                    logger.error(f"AD9082-#{g} is not working, check power and link status before retrying")
                css.ad9082[g].device_chip_id_get()
                link_ok[g] = True
            except RuntimeError:
                logger.error(f"failed to establish a configuration link with AD9082-#{g}")
                link_ok[g] = False
        return link_ok
    else:
        raise ValueError("no target objects are provided")
