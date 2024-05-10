import logging
from typing import Collection, Union, cast

from e7awgsw import CaptureParam, WaveSequence

from quel_ic_config.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config.linkupper import LinkupFpgaMxfe
from quel_ic_config.quel1_anytype import Quel1AnyBoxConfigSubsystem
from quel_ic_config.quel1_box import Quel1Box
from quel_ic_config.quel1_wave_subsystem import Quel1WaveSubsystem

logger = logging.getLogger(__name__)


# Notes: the additional APIs are back-ported from Quel1Box class at quelware-0.9.0 to provide their functionalities in
# a quick and dirty way. It'll be replaced with more sophisticated APIs in near future, namely quelware-0.9.x.
class Quel1BoxWithRawWss(Quel1Box):
    def __init__(
        self,
        *,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Union[Quel1E7ResourceMapper, None] = None,
        linkupper: Union[LinkupFpgaMxfe, None] = None,
        **options: Collection[int],
    ):
        super().__init__(css=css, wss=wss, rmap=rmap, linkupper=linkupper, **options)

    def config_channel(
        self,
        port: int,
        channel: int,
        *,
        subport: int = 0,
        fnco_freq: Union[float, None] = None,
        wave_param: Union[WaveSequence, None] = None,
    ) -> None:
        """configuring parameters of a given channel, either of transmitter or receiver one.

        :param port: an index of a port which the channel belongs to.
        :param channel: a port-local index of the channel.
        :param subport: a port-local index of the DAC which the channel belongs to.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :param wave_param: an object holding parameters of signal to be generated.
        :return: None
        """
        group, line = self._convert_any_port(port, subport)
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
                line_name = f"group:{group}, line:{line}"
                port_name = f"port-#{port:02d}"
                if line_name in e.args[0]:
                    raise ValueError(e.args[0].replace(line_name, port_name))
                else:
                    raise
        else:
            raise ValueError(f"port-#{port:02d} is not an output port, not applicable")

    def config_runit(
        self,
        port: int,
        runit: int,
        *,
        fnco_freq: Union[float, None] = None,
        capture_param: Union[CaptureParam, None] = None,
    ) -> None:
        """configuring parameters of a given receiver channel.

        :param port: an index of a port which the runit belongs to.
        :param runit: a line-local index of the capture unit.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :param capture_param: an object keeping capture settings.
        :return: None
        """
        group, rline = self._convert_any_port(port, 0)
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
                line_name = f"group:{group}, rline:{rline}"
                port_name = f"port-#{port:02d}"
                if line_name in e.args[0]:
                    raise ValueError(e.args[0].replace(line_name, port_name))
                else:
                    raise
        else:
            raise ValueError(f"port-#{port:02d} is not an input port, not applicable")
