import collections.abc
import copy
import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Collection, Dict, Final, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from quel_ic_config.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config.linkupper import LinkupFpgaMxfe
from quel_ic_config.quel1_anytype import Quel1AnyBoxConfigSubsystem
from quel_ic_config.quel1_box_intrinsic import (
    Quel1BoxIntrinsic,
    _complete_ipaddrs,
    _create_css_object,
    _create_wss_object,
    _is_box_available_for,
)
from quel_ic_config.quel1_config_subsystem import Quel1BoxType, Quel1ConfigOption, Quel1Feature
from quel_ic_config.quel1_wave_subsystem import CaptureReturnCode, Quel1WaveSubsystem

logger = logging.getLogger(__name__)


class Quel1Box:
    _PORT2LINE_QuBE_OU_TypeA: Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        1: (0, "r"),
        2: (0, 1),
        # 3: n.c.
        # 4: n.c.
        5: (0, 2),
        6: (0, 3),
        7: (1, 3),
        8: (1, 2),
        # 9: n.c.
        # 10: n.c.
        11: (1, 1),
        12: (1, "r"),
        13: (1, 0),
    }

    _PORT2LINE_QuBE_OU_TypeB: Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        # 1: n.c.
        2: (0, 1),
        # 3: n.c.
        # 4: n.c.
        5: (0, 2),
        6: (0, 3),
        7: (1, 3),
        8: (1, 2),
        # 9: n.c.
        # 10: n.c.
        11: (1, 1),
        # 12: n.c.
        13: (1, 0),
    }

    _PORT2LINE_QuBE_RIKEN_TypeA: Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        1: (0, "r"),
        2: (0, 1),
        # 3: group-0 monitor-out
        4: (0, "m"),
        5: (0, 2),
        6: (0, 3),
        7: (1, 3),
        8: (1, 2),
        9: (1, "m"),
        # 10: group-1 monitor-out
        11: (1, 1),
        12: (1, "r"),
        13: (1, 0),
    }

    _PORT2LINE_QuBE_RIKEN_TypeB: Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        # 1: n.c.
        2: (0, 1),
        # 3: group-0 monitor-out
        4: (0, "m"),
        5: (0, 2),
        6: (0, 3),
        7: (1, 3),
        8: (1, 2),
        9: (1, "m"),
        # 10: group-1 monitor-out
        11: (1, 1),
        # 12: n.c.
        13: (1, 0),
    }

    _PORT2LINE_QuEL1_TypeA: Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]] = {
        0: (0, "r"),
        1: (0, 0),
        2: (0, 2),
        3: (0, 1),
        4: (0, 3),
        5: (0, "m"),
        # 6: group-0 monitor-out
        7: (1, "r"),
        8: (1, 0),
        9: (1, 3),
        10: (1, 1),
        11: (1, 2),
        12: (1, "m"),
        # 13: group-1 monitor-out
    }

    _PORT2LINE_QuEL1_TypeB: Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]] = {
        # 0: n.c.
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (0, 3),
        5: (0, "m"),
        # 6: group-0 monitor-out
        # 7: n.c.
        8: (1, 0),
        9: (1, 1),
        10: (1, 3),
        11: (1, 2),
        12: (1, "m"),
        # 13: group-1 monitor-out
    }

    _PORT2LINE_QuEL1_NEC: Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]] = {
        0: (0, 0),
        1: (0, 1),
        2: (0, "r"),
        3: (1, 0),
        4: (1, 1),
        5: (1, "r"),
        6: (2, 0),
        7: (2, 1),
        8: (2, "r"),
        9: (3, 0),
        10: (3, 1),
        11: (3, "r"),
    }

    _PORT2LINE_QuEL1SE_RIKEN8: Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]] = {
        0: (0, "r"),
        1: (0, 0),
        (1, 1): (0, 1),  # Notes: analog-combined port
        2: (0, 2),
        3: (0, 3),
        4: (0, "m"),
        # 5: group-0 monitor-out
        6: (1, 0),
        7: (1, 1),
        8: (1, 2),
        9: (1, 3),
        10: (1, "m"),
        # 11: group-1 monitor-out
    }

    _PORT2LINE: Final[Dict[Quel1BoxType, Dict[Union[int, Tuple[int, int]], Tuple[int, Union[int, str]]]]] = {
        Quel1BoxType.QuBE_OU_TypeA: _PORT2LINE_QuBE_OU_TypeA,
        Quel1BoxType.QuBE_OU_TypeB: _PORT2LINE_QuBE_OU_TypeB,
        Quel1BoxType.QuBE_RIKEN_TypeA: _PORT2LINE_QuBE_RIKEN_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeB: _PORT2LINE_QuBE_RIKEN_TypeB,
        Quel1BoxType.QuEL1_TypeA: _PORT2LINE_QuEL1_TypeA,
        Quel1BoxType.QuEL1_TypeB: _PORT2LINE_QuEL1_TypeB,
        Quel1BoxType.QuEL1_NEC: _PORT2LINE_QuEL1_NEC,
        Quel1BoxType.QuEL1SE_RIKEN8DBG: _PORT2LINE_QuEL1SE_RIKEN8,
        Quel1BoxType.QuEL1SE_RIKEN8: _PORT2LINE_QuEL1SE_RIKEN8,
    }

    __slots__ = (
        "_dev",
        "_boxtype",
    )

    @classmethod
    def is_applicable_to(cls, boxtype: Quel1BoxType) -> bool:
        return Quel1BoxIntrinsic.is_applicable_to(boxtype)

    @classmethod
    def create(
        cls,
        *,
        ipaddr_wss: str,
        ipaddr_sss: Union[str, None] = None,
        ipaddr_css: Union[str, None] = None,
        boxtype: Union[Quel1BoxType, str],
        config_root: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        **options: Collection[int],
    ) -> "Quel1Box":
        """create QuEL box objects
        :param ipaddr_wss: IP address of the wave generation subsystem of the target box
        :param ipaddr_sss: IP address of the sequencer subsystem of the target box (optional)
        :param ipaddr_css: IP address of the configuration subsystem of the target box (optional)
        :param boxtype: type of the target box
        :param config_root: root path of config setting files to read (optional)
        :param config_options: a collection of config options (optional)
        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored. (optional)
        :param ignore_access_failure_of_adrf6780: a list of ADRF6780 whose communication faiulre via SPI bus is
                                                  dismissed (optional)
        :param ignore_lock_failure_of_lmx2594: a list of LMX2594 whose lock failure is ignored (optional)
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed (optional)
        :return: SimpleBox objects
        """
        ipaddr_sss, ipaddr_css = _complete_ipaddrs(ipaddr_wss, ipaddr_sss, ipaddr_css)
        if isinstance(boxtype, str):
            boxtype = Quel1BoxType.fromstr(boxtype)
        if not _is_box_available_for(boxtype):
            raise ValueError(f"unsupported boxtype: {boxtype}")
        if config_options is None:
            config_options = set()

        features: Set[Quel1Feature] = set()
        wss: Quel1WaveSubsystem = _create_wss_object(ipaddr_wss, features)
        css: Quel1AnyBoxConfigSubsystem = cast(
            Quel1AnyBoxConfigSubsystem, _create_css_object(ipaddr_css, boxtype, features, config_root, config_options)
        )
        return Quel1Box(css=css, wss=wss, rmap=None, linkupper=None, **options)

    def __init__(
        self,
        *,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Union[Quel1E7ResourceMapper, None] = None,
        linkupper: Union[LinkupFpgaMxfe, None] = None,
        **options: Collection[int],
    ):
        self._dev = Quel1BoxIntrinsic(css=css, wss=wss, rmap=rmap, linkupper=linkupper, **options)
        self._boxtype = css._boxtype
        if self._boxtype not in self._PORT2LINE:
            raise ValueError(f"unsupported boxtype; {self._boxtype}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._dev._wss._wss_addr}:{self.boxtype}>"

    @property
    def css(self) -> Quel1AnyBoxConfigSubsystem:
        return self._dev.css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._dev.wss

    @property
    def rmap(self) -> Quel1E7ResourceMapper:
        return self._dev.rmap

    @property
    def linkupper(self) -> LinkupFpgaMxfe:
        return self._dev.linkupper

    @property
    def options(self) -> Dict[str, Collection[int]]:
        return self._dev.options

    @property
    def boxtype(self) -> str:
        return self._dev.boxtype

    def reconnect(
        self,
        *,
        mxfe_list: Union[Collection[int], None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
    ) -> Dict[int, bool]:
        """establish a configuration link between a box and host.
        the target box must be linked-up in advance.

        :param mxfe_list: a list of target MxFEs (optional).
        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored. (optional).
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed. (optional).
        :return: True if success.
        """
        return self._dev.reconnect(
            mxfe_list=mxfe_list,
            ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinary_converter_select_of_mxfe,
        )

    def relinkup(
        self,
        *,
        mxfes_to_linkup: Union[Collection[int], None] = None,
        hard_reset: Union[bool, None] = None,
        use_204b: Union[bool, None] = None,
        use_bg_cal: bool = False,
        skip_init: bool = False,
        background_noise_threshold: Union[float, None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
    ) -> Dict[int, bool]:
        return self._dev.relinkup(
            mxfes_to_linkup=mxfes_to_linkup,
            hard_reset=hard_reset,
            use_204b=use_204b,
            use_bg_cal=use_bg_cal,
            skip_init=skip_init,
            background_noise_threshold=background_noise_threshold,
            ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinary_converter_select_of_mxfe,
        )

    def terminate(self):
        self._dev.terminate()

    def link_status(self, ignore_crc_error_of_mxfe: Union[Collection[int], None] = None) -> Dict[int, bool]:
        return self._dev.link_status(ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe)

    def decode_port(self, port: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid port of {self.boxtype}: {port}")

        if isinstance(port, int):
            return port, 0
        elif isinstance(port, tuple) and len(port) == 2:
            return port[0], port[1]
        else:
            raise AssertionError

    def _convert_any_port(self, port: int, subport: int = 0) -> Tuple[int, Union[int, str]]:
        if subport == 0:
            if port not in self._PORT2LINE[self._boxtype]:
                raise ValueError(f"invalid port: {port}")
            group, line = self._PORT2LINE[self._boxtype][port]
        else:
            if (port, subport) not in self._PORT2LINE[self._boxtype]:
                raise ValueError(f"invalid subport-#{subport} of port-#{port}")
            group, line = self._PORT2LINE[self._boxtype][port, subport]
        return group, line

    def _convert_output_port(self, port_subport: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        p, sp = self.decode_port(port_subport)
        return self._convert_output_port_decoded(p, sp)

    def _convert_output_port_decoded(self, port: int, subport: int = 0) -> Tuple[int, int]:
        group, line = self._convert_any_port(port, subport)
        if not self._dev.is_output_line(group, line):
            raise ValueError(f"port-#{port} is not an output port")
        return group, cast(int, line)

    # Notes: currently no subport of input port is implemented.
    def _convert_input_port(self, port) -> Tuple[int, str]:
        group, line = self._convert_any_port(port, 0)
        if not self._dev.is_input_line(group, line):
            raise ValueError(f"port-#{port} is not an input port")
        return group, cast(str, line)

    def _convert_output_channel(self, channel: Tuple[Union[int, Tuple[int, int]], int]) -> Tuple[int, int, int]:
        if not (isinstance(channel, tuple) and len(channel) == 2):
            raise ValueError(f"malformed channel: {channel}")

        port_subport, ch1 = channel
        p, sp = self.decode_port(port_subport)
        return self._convert_output_channel_decoded(p, subport=sp, channel=ch1)

    def _convert_output_channel_decoded(self, port: int, channel: int, *, subport: int = 0) -> Tuple[int, int, int]:
        group, line = self._convert_output_port_decoded(port, subport)
        if channel < self.css.get_num_channels_of_line(group, line):
            ch3 = (group, line, channel)
        else:
            raise ValueError(f"invalid channel-#{channel} of subport-#{subport} of port-#{port}")
        return ch3

    def _convert_output_channels(
        self, channels: Collection[Tuple[Union[int, Tuple[int, int]], int]]
    ) -> Set[Tuple[int, int, int]]:
        ch3s: Set[Tuple[int, int, int]] = set()
        if not isinstance(channels, collections.abc.Collection):
            raise TypeError(f"malformed channels: {channels}")
        for channel in channels:
            ch3s.add(self._convert_output_channel(channel))
        return ch3s

    def _get_all_subports_of_port(self, port: int) -> Set[int]:
        subports: Set[int] = set()
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid port: {port}")

        for port_subport in self._PORT2LINE[self._boxtype]:
            if isinstance(port_subport, int) and port_subport == port:
                subports.add(0)
            elif (
                isinstance(port_subport, tuple)
                and len(port_subport) == 2
                and port_subport[0] == port
                and isinstance(port_subport[1], int)
            ):
                subports.add(port_subport[1])

        return subports

    def easy_start_cw(
        self,
        port: int,
        channel: int = 0,
        *,
        subport: int = 0,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        fnco_freq: Union[float, None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
        fullscale_current: Union[int, None] = None,
        amplitude: float = Quel1BoxIntrinsic.DEFAULT_AMPLITUDE,
        duration: float = Quel1BoxIntrinsic.VERY_LONG_DURATION,
        control_port_rfswitch: bool = True,
        control_monitor_rfswitch: bool = False,
    ) -> None:
        """an easy-to-use API to generate continuous wave from a given port.

        :param port: an index of port which the subport belongs to.
        :param channel: a (sub)port-local index of the channel of the (sub)port. (default: 0)
        :param subport: a port-local index of the DAC which the channel belongs to. (default: 0)
        :param lo_freq: the frequency of the corresponding local oscillator in Hz. it must be multiple of 100_000_000.
        :param cnco_freq: the frequency of the corresponding CNCO in Hz.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz.
        :param vatt: the control voltage of the corresponding VATT in unit of 3.3V / 4096.
        :param sideband: the active sideband of the corresponding mixer, "U" for upper and "L" for lower.
        :param fullscale_current: full-scale current of output DAC of AD9082 in uA.
        :param amplitude: the amplitude of the sinusoidal wave to be passed to DAC.
        :param duration: the duration of wave generation in second.
        :param control_port_rfswitch: allowing the port corresponding to the line to emit the RF signal if True.
        :param control_monitor_rfswitch: allowing the monitor-out port to emit the RF signal if True.
        :return: None
        """

        group, line = self._convert_output_port_decoded(port, subport)
        self._dev.easy_start_cw(
            group,
            line,
            channel,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            fnco_freq=fnco_freq,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
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
        subport: Union[int, None] = None,
        control_port_rfswitch: bool = True,
        control_monitor_rfswitch: bool = False,
    ) -> None:
        """stopping the wave generation on a given port.

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel. stop the signal generation of all the channel of
                        the port if None. (default: None)
        :param subport: a port-local index of the DAC which the channel belongs to. stop the signal generation of all
                        the sub-port of the port. (default: None)
        :param control_port_rfswitch: blocking the emission of the RF signal from the corresponding port if True.
        :param control_monitor_rfswitch: blocking the emission of the RF signal from the monitor-out port if True.
        :return: None
        """
        if subport is None:
            subports = self._get_all_subports_of_port(port)
        else:
            subports = {subport}

        for sp in subports:
            if channel is None:
                group, line = self._convert_output_port_decoded(port, subport=sp)
            else:
                group, line, channel = self._convert_output_channel_decoded(port, subport=sp, channel=channel)
            self._dev.easy_stop(
                group,
                line,
                channel=channel,
                control_port_rfswitch=control_port_rfswitch,
                control_monitor_rfswitch=control_monitor_rfswitch,
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
        runit: int = 0,
        *,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        fnco_freq: Union[float, None] = None,
        activate_internal_loop: Union[None, bool] = None,
        num_samples: int = Quel1BoxIntrinsic.DEFAULT_NUM_CAPTURE_SAMPLE,
    ) -> npt.NDArray[np.complex64]:
        """capturing the wave signal from a given receiver channel.

        :param port: an index of a port which the channel belongs to.
        :param runit: a port-local index of the capture unit.
        :param lo_freq: the frequency of the corresponding local oscillator in Hz. it must be multiple of 100_000_000.
        :param cnco_freq: the frequency of the corresponding CNCO in Hz.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz.
        :param activate_internal_loop: activate the corresponding loop-back path if True.
        :param num_samples: number of samples to capture.
        :return: captured wave data in NumPy array
        """

        group, rline = self._convert_input_port(port)
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
            runit,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            fnco_freq=fnco_freq,
            activate_internal_loop=activate_internal_loop,
            num_samples=num_samples,
        )

    def load_cw_into_channel(
        self,
        port: int,
        channel: int,
        *,
        subport: int = 0,
        amplitude: float = Quel1BoxIntrinsic.DEFAULT_AMPLITUDE,
        num_wave_sample: int = Quel1BoxIntrinsic.DEFAULT_NUM_WAVE_SAMPLE,
        num_repeats: Tuple[int, int] = Quel1BoxIntrinsic.DEFAULT_REPEATS,
        num_wait_samples: Tuple[int, int] = Quel1BoxIntrinsic.DEFAULT_NUM_WAIT_SAMPLES,
    ) -> None:
        """loading continuous wave data into a channel.

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel.
        :param subport: a port-local index of the DAC which the channel belongs to.
        :param amplitude: amplitude of the continuous wave, 0 -- 32767.
        :param num_wave_sample: number of samples in wave data to generate.
        :param num_repeats: number of repetitions of a unit wave data whose length is num_wave_sample.
                            given as a tuple of two integers that specifies the number of repetition as multiple of
                            the two.
        :param num_wait_samples: number of wait duration in samples. given as a tuple of two integers that specify the
                               length of wait at the start of the whole wave sequence and the length of wait between
                               each repeated motifs, respectively.
        :return: None
        """
        group, line = self._convert_output_port_decoded(port, subport)
        self._dev.load_cw_into_channel(
            group,
            line,
            channel,
            amplitude=amplitude,
            num_wave_sample=num_wave_sample,
            num_repeats=num_repeats,
            num_wait_samples=num_wait_samples,
        )

    def load_iq_into_channel(
        self,
        port: int,
        channel: int,
        *,
        subport: int = 0,
        iq: npt.NDArray[np.complex64],
        num_repeats: Tuple[int, int] = Quel1BoxIntrinsic.DEFAULT_REPEATS,
        num_wait_samples: Tuple[int, int] = Quel1BoxIntrinsic.DEFAULT_NUM_WAIT_SAMPLES,
    ) -> None:
        """loading arbitrary wave data into a channel.

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel.
        :param subport: a port-local index of the DAC which the channel belongs to.
        :param iq: complex data of the signal to generate in 500Msps. I and Q coefficients of each sample must be
                   within the range of -32768 -- 32767. its length must be a multiple of 64.
        :param num_repeat: the number of repetitions of the given wave data given as a tuple of two integers,
                           a product of the two is the number of repetitions.
        :param num_wait_samples: number of wait duration in samples. given as a tuple of two integers that specify the
                               length of wait at the start of the whole wave sequence and the length of wait between
                               each repeated motifs, respectively.
        :return: None
        """
        group, line = self._convert_output_port_decoded(port, subport)
        self._dev.load_iq_into_channel(
            group, line, channel, iq=iq, num_repeats=num_repeats, num_wait_samples=num_wait_samples
        )

    # TODO: reconsider name of API
    def initialize_all_awgs(self):
        self._dev.initialize_all_awgs()

    def prepare_for_emission(self, channels: Collection[Tuple[Union[int, Tuple[int, int]], int]]):
        """making preparation of signal generation of multiple channels at the same time.

        :param channels: a collection of channels to be activated. each channel is specified as a tuple of a port and
                         a channel.
        """
        self._dev.prepare_for_emission(self._convert_output_channels(channels))

    def start_emission(self, channels: Collection[Tuple[Union[int, Tuple[int, int]], int]]) -> None:
        """starting signal generation of multiple channels at the same time.

        :param channels: a collection of channels to be activated. each channel is specified as a tuple of a port and
                         a channel.
        """
        self._dev.start_emission(self._convert_output_channels(channels))

    def stop_emission(self, channels: Collection[Tuple[Union[int, Tuple[int, int]], int]]) -> None:
        """stopping signal generation on a given channel.

        :param channels: a collection of channels to be deactivated. each channel is specified as a tuple of a port and
                         a channel.
        """
        self._dev.stop_emission(self._convert_output_channels(channels))

    def simple_capture_start(
        self,
        port: int,
        runits: Union[Collection[int], None] = None,
        *,
        num_samples: int = Quel1BoxIntrinsic.DEFAULT_NUM_CAPTURE_SAMPLE,
        delay_samples: int = 0,
        triggering_channel: Union[Tuple[Union[int, Tuple[int, int]], int], None] = None,
        timeout: float = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT,
    ) -> "Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]]":
        """capturing the wave signal from a given receiver channel.

        :param port: an index of a port to capture.
        :param runits: port-local indices of the capture units of the port.
        :param num_samples: number of samples to capture, recommended to be multiple of 4.
        :param delay_samples: delay in sampling clocks before starting capture.
        :param triggering_channel: a channel which triggers this capture when it starts to emit a signal.
                                   it is specified by a tuple of port and channel. the capture starts
                                   immediately if None.
        :param timeout: waiting time in second before capturing thread quits.
        :return: captured wave data in NumPy array
        """

        cap_group, cap_rline = self._convert_input_port(port)
        try:
            rrline: str = self._dev.rmap.resolve_rline(cap_group, None)
            if rrline != cap_rline:
                logger.warning(
                    f"the specified port {port} may not be connected to any ADC under the current configuration"
                )
        except ValueError:
            # Notes: failure of resolution means the mxfe in the group has multiple capture lines.
            pass

        if triggering_channel is not None:
            if isinstance(triggering_channel[0], int):
                trg_port, trg_subport = triggering_channel[0], 0
            elif isinstance(triggering_channel[0], tuple) and len(triggering_channel[0]) == 2:
                trg_port, trg_subport = triggering_channel[0]
            else:
                raise ValueError(f"invalid triggering channel: {triggering_channel}")
            trg_group, trg_line = self._convert_output_port_decoded(trg_port, trg_subport)
            trg_ch3: Union[Tuple[int, int, int], None] = (trg_group, trg_line, triggering_channel[1])
        else:
            trg_ch3 = None

        return self._dev.simple_capture_start(
            cap_group,
            cap_rline,
            runits=runits,
            num_samples=num_samples,
            delay_samples=delay_samples,
            triggering_channel=trg_ch3,
            timeout=timeout,
        )

    def _config_box_inner(self, port_subport: Union[int, Tuple[int, int]], pc: Dict[str, Any]):
        port_conf = copy.deepcopy(pc)
        if "direction" in port_conf:
            del port_conf["direction"]  # direction will be validated at the end
        if "channels" in port_conf:
            channel_confs: Dict[int, Dict[str, Any]] = port_conf["channels"]
            del port_conf["channels"]
        else:
            channel_confs = {}
        if "runits" in port_conf:
            runit_confs: Dict[int, Dict[str, Any]] = port_conf["runits"]
            del port_conf["runits"]
        else:
            runit_confs = {}

        if isinstance(port_subport, int):
            port, subport = port_subport, 0
        elif (
            isinstance(port_subport, tuple) and len(port_subport) == 2 and all(isinstance(x, int) for x in port_subport)
        ):
            port, subport = port_subport
        else:
            raise ValueError(f"invalid port: {port_subport}")

        for ch, channel_conf in channel_confs.items():
            self.config_channel(port, subport=subport, channel=ch, **channel_conf)
        for runit, runit_conf in runit_confs.items():
            self.config_runit(port, runit=runit, **runit_conf)
        self.config_port(port, subport=subport, **port_conf)

    def config_box(self, box_conf: Dict[Union[int, Tuple[int, int]], Dict[str, Any]], ignore_validation: bool = False):
        # Notes: configure output ports before input ones to keep "cnco_locked_with" intuitive in config_box().
        for port_subport, pc in box_conf.items():
            p, sp = self.decode_port(port_subport)
            if self.is_output_port(p):
                self._config_box_inner(port_subport, pc)

        for port_subport, pc in box_conf.items():
            p, sp = self.decode_port(port_subport)
            if self.is_input_port(p):
                self._config_box_inner(port_subport, pc)

        if not ignore_validation:
            if not self.config_validate_box(box_conf):
                raise ValueError("the provided settings looks have inconsistent settings")

    def _config_validate_port(self, port_subport: Union[int, Tuple[int, int]], lc: Dict[str, Any]) -> bool:
        if isinstance(port_subport, int):
            port_name = f"port-#{port_subport:02d}"
            port, subport = port_subport, 0
        elif (
            isinstance(port_subport, tuple)
            and len(port_subport) == 2
            and isinstance(port_subport[0], int)
            and isinstance(port_subport[1], int)
        ):
            port_name = f"port-#{port_subport[0]:02d} subport-#{port_subport[1]:02d}"
            port, subport = port_subport
        else:
            raise AssertionError

        group, line = self._convert_any_port(port, subport)
        if self._dev.is_output_line(group, line):
            alc: Dict[str, Any] = self.css.dump_line(group, cast(int, line))
            ad: str = "out"
        elif self._dev.is_input_line(group, line):
            alc = self.css.dump_rline(group, cast(str, line))
            ad = "in"
        else:
            raise ValueError(f"invalid port: {port_name}")

        valid = True
        for k in lc:
            if k == "channels":
                if not self._dev._config_validate_channels(group, line, lc["channels"], port_name):
                    valid = False
            elif k == "runits":
                if not self._dev._config_validate_runits(group, line, lc["runits"], port_name):
                    valid = False
            elif k == "direction":
                if lc["direction"] != ad:
                    valid = False
                    logger.error(f"unexpected settings of {port_name}:" f"direction = {ad} (!= {lc['direction']})")
            elif k in {"cnco_freq", "fnco_freq"}:
                if not self._dev._config_validate_frequency(group, line, k, lc[k], alc[k]):
                    valid = False
                    logger.error(f"unexpected settings at {port_name}:line-{line}:{k} = {alc[k]} (!= {lc[k]})")
            elif k == "cnco_locked_with":
                dac_p, dac_sp = self.decode_port(lc[k])
                dac_g, dac_l = self._convert_output_port_decoded(dac_p, dac_sp)
                alf = alc["cnco_freq"]
                lf = self._dev._css.get_dac_cnco(dac_g, dac_l)
                if lf != alf:
                    valid = False
                    logger.error(
                        f"unexpected settings at {port_name}:cnco_freq = {alf} (!= {lf}, "
                        f"that is cnco frequency of port-#{lc[k]})"
                    )
            else:
                if lc[k] != alc[k]:
                    valid = False
                    logger.error(f"unexpected settings at {port_name}:{k} = {alc[k]} (!= {lc[k]})")
        return valid

    def config_validate_box(self, box_conf: Dict[Union[int, Tuple[int, int]], Dict[str, Any]]) -> bool:
        valid: bool = True
        for port, lc in box_conf.items():
            valid &= self._config_validate_port(port, lc)
        return valid

    def is_output_port(self, port: int):
        group, line = self._convert_any_port(port, 0)
        return self._dev.is_output_line(group, line)

    def is_input_port(self, port: int):
        group, line = self._convert_any_port(port, 0)
        return self._dev.is_input_line(group, line)

    def config_port(
        self,
        port: int,
        *,
        subport: int = 0,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        cnco_locked_with: Union[int, Tuple[int, int], None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
        fullscale_current: Union[int, None] = None,
        rfswitch: Union[str, None] = None,
    ) -> None:
        """configuring parameters of a given port, either of transmitter or receiver one.

        :param port: an index of the target port to configure.
        :param subport: a port-local index of the DAC which the channel belongs to.
        :param lo_freq: the frequency of the corresponding local oscillator in Hz, must be multiple of 100_000_000.
        :param cnco_freq: the frequency of the corresponding CNCO in Hz.
        :param cnco_locked_with: the frequency is locked to be identical to the specified port.
        :param vatt: controlling voltage of the corresponding VATT in unit of 3.3V / 4096. see the specification sheet
                     of the ADRF6780 for details. (only for transmitter port)
        :param sideband: "U" for upper side band, "L" for lower side band. (only for transmitter port)
        :param fullscale_current: full-scale current of output DAC of AD9082 in uA.
        :param rfswitch: state of RF switch, 'block' or 'pass' for output port, 'loop' or 'open' for input port.
        :return: None
        """
        group, line = self._convert_any_port(port, subport)
        if self._dev.is_output_line(group, line):
            if cnco_locked_with is not None:
                raise ValueError(f"no cnco_locked_with is available for the output port {port}")
            self._dev.config_line(
                group,
                cast(int, line),
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
                vatt=vatt,
                sideband=sideband,
                fullscale_current=fullscale_current,
                rfswitch=rfswitch,
            )
        elif self._dev.is_input_line(group, line):
            if vatt is not None or sideband is not None:
                raise ValueError(f"no mixer is available for the input port {port}")
            if fullscale_current is not None:
                raise ValueError(f"no DAC is available for the input port {port}")
            if cnco_locked_with is not None:
                p, sp = self.decode_port(cnco_locked_with)
                converted_cnco_lock_with = self._convert_output_port_decoded(p, sp)
            else:
                converted_cnco_lock_with = None
            self._dev.config_rline(
                group,
                cast(str, line),
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
                cnco_locked_with=converted_cnco_lock_with,
                rfswitch=rfswitch,
            )
        else:
            raise AssertionError

    def config_channel(self, port: int, channel: int, *, subport: int = 0, fnco_freq: Union[float, None] = None):
        """configuring parameters of a given channel, either of transmitter or receiver one.

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel.
        :param subport: a port-local index of the DAC which the channel belongs to.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :return: None
        """
        group, line = self._convert_any_port(port, subport)
        if self._dev.is_output_line(group, line):
            self._dev.config_channel(group, cast(int, line), channel, fnco_freq=fnco_freq)
        else:
            raise ValueError(f"a given port (= {port}) is not an output port, not applicable")

    def config_runit(self, port: int, runit: int, *, fnco_freq: Union[float, None]) -> None:
        """configuring parameters of a given receiver channel.

        :param port: an index of a port which the runit belongs to.
        :param runit: a line-local index of the capture unit.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :return: None
        """
        group, rline = self._convert_any_port(port, 0)
        if self._dev.is_input_line(group, rline):
            self._dev.config_runit(group, cast(str, rline), runit, fnco_freq=fnco_freq)
        else:
            raise ValueError(f"a given port (= {port}) is not an input port, not applicable")

    def block_all_output_ports(self) -> None:
        """set RF switch of all output ports to block.

        :return:
        """
        self._dev.block_all_output_lines()

    def pass_all_output_ports(self):
        """set RF switch of all output ports to pass.

        :return:
        """
        self._dev.pass_all_output_lines()

    def config_rfswitches(
        self, rfswitch_confs: Dict[Union[int, Tuple[int, int]], str], ignore_validation: bool = False
    ) -> None:
        for port_subport, rc in rfswitch_confs.items():
            p, sp = self.decode_port(port_subport)
            self.config_rfswitch(p, rfswitch=rc)

        valid = True
        for port_subport, rc in rfswitch_confs.items():
            p, sp = self.decode_port(port_subport)
            arc = self.dump_rfswitch(p)
            if rc != arc:
                valid = False
                logger.warning(f"rfswitch of port-#{p} is finally set to {arc} (!= {rc})")

        if not (ignore_validation or valid):
            raise ValueError("the specified configuration of rf switches is not realizable")

    def config_rfswitch(self, port: int, *, rfswitch: str):
        group, line = self._convert_any_port(port, 0)
        self._dev.config_rfswitch(group, line, rfswitch=rfswitch)

    def activate_monitor_loop(self, group: int) -> None:
        """enabling an internal monitor loop-back path from a monitor-out port to a monitor-in port.

        :param group: an index of a group which the monitor path belongs to.
        :return: None
        """
        self._dev.activate_monitor_loop(group)

    def deactivate_monitor_loop(self, group: int) -> None:
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

    def _dump_port(self, group: int, line: Union[int, str]) -> Dict[str, Any]:
        retval: Dict[str, Any] = {}
        if self._dev.is_output_line(group, line):
            retval["direction"] = "out"
            retval.update(self._dev._css.dump_line(group, cast(int, line)))
        elif self._dev.is_input_line(group, line):
            retval["direction"] = "in"
            retval.update(self._dev._dump_rline(group, cast(str, line)))
        else:
            raise AssertionError
        return retval

    def dump_rfswitch(self, port: int) -> str:
        """dumping the current configuration of an RF switch

        :return: the current configuration setting of the RF switch
        """
        group, line = self._convert_any_port(port, 0)
        return self._dev.dump_rfswitch(group, line)

    def dump_rfswitches(self, exclude_subordinate: bool = True) -> Dict[Union[int, Tuple[int, int]], str]:
        """dumping the current configuration of all RF switches

        :return: a mapping of a port index and the configuration of its RF switch.
        """

        # actually, any key in Tuple[int, int] key is not defined.
        retval_intrinsic = self._dev.dump_rfswitches(exclude_subordinate)
        retval: Dict[Union[int, Tuple[int, int]], str] = {}
        for port, (group, line) in self._PORT2LINE[self._boxtype].items():
            if isinstance(port, int) and (group, line) in retval_intrinsic:
                # Notes: subordinate rfswitch is not included when exclude_subordinate_switch is True.
                retval[port] = retval_intrinsic[group, line]
        return retval

    def dump_port(self, port: Union[int, Tuple[int, int]]) -> Dict[str, Any]:
        p, sp = self.decode_port(port)
        group, line = self._convert_any_port(p, sp)
        return self._dump_port(group, line)

    def dump_box(self) -> Dict[str, Dict[Union[int, Tuple[int, int]], Dict[str, Any]]]:
        """dumping the current configuration of the ports

        :return: the current configuration of ports in dictionary.
        """
        retval: Dict[str, Dict[Union[int, Tuple[int, int]], Dict[str, Any]]] = {"mxfes": {}, "ports": {}}

        for mxfe_idx in self.css.get_all_mxfes():
            retval["mxfes"][mxfe_idx] = {
                "channel_interporation_rate": self._dev._css.get_channel_interpolation_rate(mxfe_idx),
                "main_interporation_rate": self._dev._css.get_main_interpolation_rate(mxfe_idx),
            }

        for port, (group, line) in self._PORT2LINE[self._boxtype].items():
            retval["ports"][port] = self._dump_port(group, line)

        return retval
