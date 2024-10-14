import logging
from typing import Collection, Union, cast

from e7awgsw import CaptureParam, WaveSequence

from quel_clock_master import SequencerClient
from quel_ic_config.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config.linkupper import LinkupFpgaMxfe
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_box import Quel1Box, Quel1PortType
from quel_ic_config.quel1_box_intrinsic import _complete_ipaddrs, _create_css_object, _create_wss_object
from quel_ic_config.quel1_config_subsystem import Quel1BoxType
from quel_ic_config.quel1_wave_subsystem import Quel1WaveSubsystem

logger = logging.getLogger(__name__)


# Notes: the additional APIs are back-ported from Quel1Box class at quelware-0.9.0 to provide their functionalities in
# a quick and dirty way. It'll be replaced with more sophisticated APIs in near future, namely quelware-0.9.x.
class Quel1BoxWithRawWss(Quel1Box):
    @classmethod
    def create(
        cls,
        *,
        ipaddr_wss: str,
        ipaddr_sss: Union[str, None] = None,
        ipaddr_css: Union[str, None] = None,
        boxtype: Union[Quel1BoxType, str],
        skip_init: bool = False,
        **options: Collection[int],
    ) -> "Quel1BoxWithRawWss":
        """create QuEL box objects
        :param ipaddr_wss: IP address of the wave generation subsystem of the target box
        :param ipaddr_sss: IP address of the sequencer subsystem of the target box (optional)
        :param ipaddr_css: IP address of the configuration subsystem of the target box (optional)
        :param boxtype: type of the target box
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
        if boxtype not in cls._PORT2LINE:
            raise ValueError(f"unsupported boxtype for Quel1Box: {boxtype}")

        wss: Quel1WaveSubsystem = _create_wss_object(ipaddr_wss)
        sss = SequencerClient(ipaddr_sss)
        css: Quel1AnyConfigSubsystem = cast(Quel1AnyConfigSubsystem, _create_css_object(ipaddr_css, boxtype))
        return cls(css=css, sss=sss, wss=wss, rmap=None, linkupper=None, **options)

    def __init__(
        self,
        *,
        css: Quel1AnyConfigSubsystem,
        sss: SequencerClient,
        wss: Quel1WaveSubsystem,
        rmap: Union[Quel1E7ResourceMapper, None] = None,
        linkupper: Union[LinkupFpgaMxfe, None] = None,
        **options: Collection[int],
    ):
        super().__init__(css=css, sss=sss, wss=wss, rmap=rmap, linkupper=linkupper, **options)

    def config_channel(
        self,
        port: Quel1PortType,
        channel: int,
        *,
        subport: Union[int, None] = None,
        fnco_freq: Union[float, None] = None,
        wave_param: Union[WaveSequence, None] = None,
    ) -> None:
        """configuring parameters of a given channel, either of transmitter or receiver one.

        :param port: an index of the target port.
        :param channel: a port-local index of the channel.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :param wave_param: an object holding parameters of signal to be generated.
        :return: None
        """
        group, line = self._convert_any_port_flex(port, subport)
        portname = "port-" + self._portname(port, subport)
        if self._dev.is_output_line(group, line):
            try:
                self._dev.config_channel(group, cast(int, line), channel, fnco_freq=fnco_freq)
                if wave_param is not None:
                    # Notes: set_param_awg() should be called from BoxIntrinsicWithRawWss, however, I didn't do so for
                    # minimum code modification because BoxWithRawWss doesn't live long.
                    self._dev.wss.set_param_awg(
                        self._dev.rmap.get_awg_of_channel(group, cast(int, line), channel), wave_param
                    )
            except ValueError as e:
                linename = f"group:{group}, line:{line}"
                if linename in e.args[0]:
                    raise ValueError(e.args[0].replace(linename, portname))
                else:
                    raise
        else:
            raise ValueError(f"{portname} is not an output port, not applicable")

    def config_runit(
        self,
        port: Quel1PortType,
        runit: int,
        *,
        fnco_freq: Union[float, None] = None,
        capture_param: Union[CaptureParam, None] = None,
    ) -> None:
        """configuring parameters of a given receiver channel.

        :param port: an index of the target port.
        :param runit: a line-local index of the capture unit.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :param capture_param: an object keeping capture settings.
        :return: None
        """
        group, rline = self._convert_any_port_flex(port, None)
        portname = "port-" + self._portname(port, None)
        if self._dev.is_input_line(group, rline):
            try:
                self._dev.config_runit(group, cast(str, rline), runit, fnco_freq=fnco_freq)
                if capture_param is not None:
                    # Notes: set_param_capunit() should be called from BoxIntrinsicWithRawWss, however, I didn't do so
                    # for minimum code modification because BoxWithRawWss doesn't live long.
                    capmod = self._dev.rmap.get_capture_module_of_rline(group, cast(str, rline))
                    self._dev.wss.set_param_capunit(capmod=capmod, capunit=runit, capprm=capture_param)
            except ValueError as e:
                # Notes: tweaking error message
                linename = f"group:{group}, rline:{rline}"
                if linename in e.args[0]:
                    raise ValueError(e.args[0].replace(linename, portname))
                else:
                    raise
        else:
            raise ValueError(f"{portname} is not an input port, not applicable")
