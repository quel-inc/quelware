import collections.abc
import copy
import json
import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Collection, Dict, Final, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from quel_clock_master import SequencerClient
from quel_ic_config.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config.linkupper import LinkupFpgaMxfe
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_box_intrinsic import (
    Quel1BoxIntrinsic,
    _complete_ipaddrs,
    _create_css_object,
    _create_wss_object,
)
from quel_ic_config.quel1_config_subsystem import Quel1BoxType
from quel_ic_config.quel1_wave_subsystem import CaptureReturnCode, Quel1WaveSubsystem
from quel_ic_config.quel_config_common import Quel1ConfigOption, Quel1Feature

logger = logging.getLogger(__name__)


Quel1PortType = Union[int, Tuple[int, int]]


class Quel1Box:
    _PORT2LINE_QuBE_OU_TypeA: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
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

    _LOOPBACK_QuBE_OU_TypeA: Dict[Quel1PortType, Set[Quel1PortType]] = {}

    _PORT2LINE_QuBE_OU_TypeB: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
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

    _LOOPBACK_QuBE_OU_TypeB: Dict[Quel1PortType, Set[Quel1PortType]] = {}

    _PORT2LINE_QuBE_RIKEN_TypeA: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
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

    _LOOPBACK_QuBE_RIKEN_TypeA: Dict[Quel1PortType, Set[Quel1PortType]] = {
        1: {0},
        4: {0, 2, 5, 6},
        9: {13, 11, 8, 7},
        12: {13},
    }

    _PORT2LINE_QuBE_RIKEN_TypeB: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
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

    _LOOPBACK_QuBE_RIKEN_TypeB: Dict[Quel1PortType, Set[Quel1PortType]] = {
        4: {0, 2, 5, 6},
        9: {13, 11, 8, 7},
    }

    _PORT2LINE_QuEL1_TypeA: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
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

    _LOOPBACK_QuEL1_TypeA: Dict[Quel1PortType, Set[Quel1PortType]] = {
        0: {1},
        5: {1, 3, 2, 4},
        7: {8},
        12: {8, 10, 11, 9},
    }

    _PORT2LINE_QuEL1_TypeB: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
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

    _LOOPBACK_QuEL1_TypeB: Dict[Quel1PortType, Set[Quel1PortType]] = {
        5: {1, 2, 3, 4},
        12: {8, 9, 11, 10},
    }

    _PORT2LINE_QuEL1_NEC: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
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

    _LOOPBACK_QuEL1_NEC: Dict[Quel1PortType, Set[Quel1PortType]] = {}

    _PORT2LINE_QuEL1SE_RIKEN8: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
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

    _LOOPBACK_QuEL1SE_RIKEN8: Dict[Quel1PortType, Set[Quel1PortType]] = {
        0: {1},  # TODO: confirm (1, 1) is whether effectively loop-backed or not.
        4: {1, (1, 1), 2, 3},
        10: {6, 7, 8, 9},
    }

    _PORT2LINE_QuEL1SE_FUJITSU11_TypeA: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
        0: (0, "r"),
        1: (0, 0),
        2: (0, 2),
        3: (0, 1),
        4: (0, 3),
        5: (0, "m"),
        # 6: group-0 monitor-out
        7: (1, "r"),
        8: (1, 0),
        9: (1, 2),
        10: (1, 1),
        11: (1, 3),
        12: (1, "m"),
        # 13: group-1 monitor-out
    }

    _LOOPBACK_QuEL1SE_FUJITSU11_TypeA: Dict[Quel1PortType, Set[Quel1PortType]] = {
        0: {1},
        5: {1, 3, 2, 4},
        7: {8},
        12: {8, 10, 9, 11},
    }

    _PORT2LINE_QuEL1SE_FUJITSU11_TypeB: Dict[Quel1PortType, Tuple[int, Union[int, str]]] = {
        1: (0, 0),
        2: (0, 2),
        3: (0, 1),
        4: (0, 3),
        5: (0, "m"),
        # 6: group-0 monitor-out
        8: (1, 0),
        9: (1, 2),
        10: (1, 1),
        11: (1, 3),
        12: (1, "m"),
        # 13: group-1 monitor-out
    }

    _LOOPBACK_QuEL1SE_FUJITSU11_TypeB: Dict[Quel1PortType, Set[Quel1PortType]] = {
        5: {1, 3, 2, 4},
        12: {8, 10, 9, 11},
    }
    _PORT2LINE: Final[Dict[Quel1BoxType, Dict[Quel1PortType, Tuple[int, Union[int, str]]]]] = {
        Quel1BoxType.QuBE_OU_TypeA: _PORT2LINE_QuBE_OU_TypeA,
        Quel1BoxType.QuBE_OU_TypeB: _PORT2LINE_QuBE_OU_TypeB,
        Quel1BoxType.QuBE_RIKEN_TypeA: _PORT2LINE_QuBE_RIKEN_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeB: _PORT2LINE_QuBE_RIKEN_TypeB,
        Quel1BoxType.QuEL1_TypeA: _PORT2LINE_QuEL1_TypeA,
        Quel1BoxType.QuEL1_TypeB: _PORT2LINE_QuEL1_TypeB,
        Quel1BoxType.QuEL1_NEC: _PORT2LINE_QuEL1_NEC,
        Quel1BoxType.QuEL1SE_RIKEN8DBG: _PORT2LINE_QuEL1SE_RIKEN8,
        Quel1BoxType.QuEL1SE_RIKEN8: _PORT2LINE_QuEL1SE_RIKEN8,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA: _PORT2LINE_QuEL1SE_FUJITSU11_TypeA,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeB: _PORT2LINE_QuEL1SE_FUJITSU11_TypeB,
        Quel1BoxType.QuEL1SE_FUJITSU11_TypeA: _PORT2LINE_QuEL1SE_FUJITSU11_TypeA,
        Quel1BoxType.QuEL1SE_FUJITSU11_TypeB: _PORT2LINE_QuEL1SE_FUJITSU11_TypeB,
    }

    _LOOPBACK: Final[Dict[Quel1BoxType, Dict[Quel1PortType, Set[Quel1PortType]]]] = {
        Quel1BoxType.QuBE_OU_TypeA: _LOOPBACK_QuBE_OU_TypeA,
        Quel1BoxType.QuBE_OU_TypeB: _LOOPBACK_QuBE_OU_TypeB,
        Quel1BoxType.QuBE_RIKEN_TypeA: _LOOPBACK_QuBE_RIKEN_TypeA,
        Quel1BoxType.QuBE_RIKEN_TypeB: _LOOPBACK_QuBE_RIKEN_TypeB,
        Quel1BoxType.QuEL1_TypeA: _LOOPBACK_QuEL1_TypeA,
        Quel1BoxType.QuEL1_TypeB: _LOOPBACK_QuEL1_TypeB,
        Quel1BoxType.QuEL1_NEC: _LOOPBACK_QuEL1_NEC,
        Quel1BoxType.QuEL1SE_RIKEN8DBG: _LOOPBACK_QuEL1SE_RIKEN8,
        Quel1BoxType.QuEL1SE_RIKEN8: _LOOPBACK_QuEL1SE_RIKEN8,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeA: _LOOPBACK_QuEL1SE_FUJITSU11_TypeA,
        Quel1BoxType.QuEL1SE_FUJITSU11_TypeA: _LOOPBACK_QuEL1SE_FUJITSU11_TypeA,
        Quel1BoxType.QuEL1SE_FUJITSU11DBG_TypeB: _LOOPBACK_QuEL1SE_FUJITSU11_TypeB,
        Quel1BoxType.QuEL1SE_FUJITSU11_TypeB: _LOOPBACK_QuEL1SE_FUJITSU11_TypeB,
    }

    __slots__ = (
        "_dev",
        "_boxtype",
    )

    @classmethod
    def create(
        cls,
        *,
        ipaddr_wss: str,
        ipaddr_sss: Union[str, None] = None,
        ipaddr_css: Union[str, None] = None,
        boxtype: Quel1BoxType,
        skip_init: bool = False,
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
        if boxtype not in cls._PORT2LINE:
            raise ValueError(f"unsupported boxtype for Quel1Box: {boxtype}")

        css: Quel1AnyConfigSubsystem = cast(Quel1AnyConfigSubsystem, _create_css_object(ipaddr_css, boxtype))
        wss: Quel1WaveSubsystem = _create_wss_object(ipaddr_wss)
        sss = SequencerClient(ipaddr_sss)
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
        self._dev = Quel1BoxIntrinsic(css=css, sss=sss, wss=wss, rmap=rmap, linkupper=linkupper, **options)
        self._boxtype = css.boxtype
        if self._boxtype not in self._PORT2LINE:
            raise ValueError(f"unsupported boxtype; {self._boxtype}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._dev._wss._wss_addr}:{self.boxtype}>"

    @property
    def css(self) -> Quel1AnyConfigSubsystem:
        return self._dev.css

    @property
    def sss(self) -> SequencerClient:
        return self._dev.sss

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

    @property
    def allow_dual_modulus_nco(self) -> bool:
        return self.css.allow_dual_modulus_nco

    @allow_dual_modulus_nco.setter
    def allow_dual_modulus_nco(self, v) -> None:
        self.css.allow_dual_modulus_nco = v

    def reconnect(
        self,
        *,
        mxfe_list: Union[Collection[int], None] = None,
        skip_capture_check: bool = False,
        background_noise_threshold: Union[float, None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
        ignore_invalid_linkstatus: bool = False,
    ) -> Dict[int, bool]:
        """establish a configuration link between a box and host.
        the target box must be linked-up in advance.

        :param mxfe_list: a list of target MxFEs (optional).
        :param skip_capture_check: do not check background noise of input lines if True (optional)
        :param background_noise_threshold: the largest peak of allowable noise (optional)
        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored. (optional).
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed. (optional).
        :return: True if success.
        """
        return self._dev.reconnect(
            mxfe_list=mxfe_list,
            skip_capture_check=skip_capture_check,
            background_noise_threshold=background_noise_threshold,
            ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
            ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinary_converter_select_of_mxfe,
            ignore_invalid_linkstatus=ignore_invalid_linkstatus,
        )

    def get_wss_features(self) -> set[Quel1Feature]:
        return self._dev.get_wss_features()

    def relinkup(
        self,
        *,
        param: Union[Dict[str, Any], None] = None,
        config_root: Union[Path, None] = None,
        config_options: Union[Collection[Quel1ConfigOption], None] = None,
        mxfes_to_linkup: Union[Collection[int], None] = None,
        hard_reset: Union[bool, None] = None,
        use_204b: bool = False,
        use_bg_cal: bool = True,
        skip_init: bool = False,
        background_noise_threshold: Union[float, None] = None,
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None,
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None,
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None,
        ignore_extraordinary_converter_select_of_mxfe: Union[Collection[int], None] = None,
        restart_tempctrl: bool = False,
    ) -> Dict[int, bool]:
        return self._dev.relinkup(
            param=param,
            config_root=config_root,
            config_options=config_options,
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
            restart_tempctrl=restart_tempctrl,
        )

    def link_status(self, ignore_crc_error_of_mxfe: Union[Collection[int], None] = None) -> Dict[int, bool]:
        return self._dev.link_status(ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe)

    def _portname(self, port: Quel1PortType, subport: Union[int, None] = None) -> str:
        if isinstance(port, int):
            return f"#{port:02d}" if subport is None or subport == 0 else f"#{port:02d}-{subport:d}"
        elif self._is_port_subport(port):
            assert subport is None
            p, sp = cast(Tuple[int, int], port)
            return f"#{p:02d}-{sp:d}"
        else:
            return str(port)

    def _is_port_subport(self, port_subport: Quel1PortType) -> bool:
        return (
            isinstance(port_subport, tuple)
            and len(port_subport) == 2
            and isinstance(port_subport[0], int)
            and isinstance(port_subport[1], int)
        )

    def _decode_port(self, port_subport: Quel1PortType) -> Tuple[int, int]:
        if isinstance(port_subport, int):
            p: int = port_subport
            sp: int = 0
        elif self._is_port_subport(port_subport):
            p, sp = port_subport[0], port_subport[1]
        else:
            raise ValueError(f"malformed port: '{port_subport}'")

        portname = self._portname(p, sp)
        if sp == 0:
            if p not in self._PORT2LINE[self._boxtype]:
                raise ValueError(f"invalid port of {self.boxtype}: {portname}")
        else:
            if port_subport not in self._PORT2LINE[self._boxtype]:
                raise ValueError(f"invalid port of {self.boxtype}: {portname}")
        return p, sp

    def _convert_any_port_flex(
        self, port_subport: Quel1PortType, subport: Union[int, None]
    ) -> Tuple[int, Union[int, str]]:
        if isinstance(port_subport, int):
            return self._convert_any_port_decoded(port_subport, subport or 0)
        elif self._is_port_subport(port_subport):
            if subport is None:
                return self._convert_any_port(port_subport)
            else:
                raise ValueError("duplicated subport specifiers")
        else:
            raise ValueError(f"malformed port: '{port_subport}'")

    def _convert_any_port(self, port_subport: Quel1PortType) -> Tuple[int, Union[int, str]]:
        p, sp = self._decode_port(port_subport)
        return self._convert_any_port_decoded(p, sp)

    def _convert_any_port_decoded(self, port: int, subport: int = 0) -> Tuple[int, Union[int, str]]:
        if subport == 0:
            if port not in self._PORT2LINE[self._boxtype]:
                raise ValueError(f"invalid port of {self.boxtype}: {self._portname(port)}")
            group, line = self._PORT2LINE[self._boxtype][port]
        else:
            if (port, subport) not in self._PORT2LINE[self._boxtype]:
                raise ValueError(f"invalid port of {self.boxtype}: {self._portname(port, subport)}")
            group, line = self._PORT2LINE[self._boxtype][port, subport]
        return group, line

    def _convert_output_port_flex(self, port_subport: Quel1PortType, subport: Union[int, None]) -> Tuple[int, int]:
        if isinstance(port_subport, int):
            return self._convert_output_port_decoded(port_subport, subport or 0)
        elif self._is_port_subport(port_subport):
            if subport is None:
                return self._convert_output_port(port_subport)
            else:
                raise ValueError("duplicated subport specifiers")
        else:
            raise ValueError(f"malformed port: '{port_subport}'")

    def _convert_output_port(self, port_subport: Quel1PortType) -> Tuple[int, int]:
        p, sp = self._decode_port(port_subport)
        return self._convert_output_port_decoded(p, sp)

    def _convert_output_port_decoded(self, port: int, subport: int = 0) -> Tuple[int, int]:
        group, line = self._convert_any_port_decoded(port, subport)
        if not self._dev.is_output_line(group, line):
            raise ValueError(f"port-{self._portname(port, subport)} is not an output port")
        return group, cast(int, line)

    def _convert_input_port_flex(
        self, port_subport: Quel1PortType, subport: Union[int, None] = None
    ) -> Tuple[int, str]:
        if isinstance(port_subport, int):
            return self._convert_input_port_decoded(port_subport, subport or 0)
        elif self._is_port_subport(port_subport):
            if subport is None:
                return self._convert_input_port(port_subport)
            else:
                raise ValueError("duplicated subport specifiers")
        else:
            raise ValueError(f"malformed port: '{port_subport}'")

    def _convert_input_port(self, port_subport: Quel1PortType) -> Tuple[int, str]:
        p, sp = self._decode_port(port_subport)
        return self._convert_input_port_decoded(p, sp)

    def _convert_input_port_decoded(self, port: int, subport: int) -> Tuple[int, str]:
        group, line = self._convert_any_port_decoded(port, subport)
        if not self._dev.is_input_line(group, line):
            raise ValueError(f"port-{self._portname(port, subport)} is not an input port")
        return group, cast(str, line)

    def _convert_output_channel(self, channel: Tuple[Quel1PortType, int]) -> Tuple[int, int, int]:
        if not (isinstance(channel, tuple) and len(channel) == 2):
            raise ValueError(f"malformed channel: '{channel}'")

        port_subport, ch1 = channel
        p, sp = self._decode_port(port_subport)
        return self._convert_output_channel_decoded(p, subport=sp, channel=ch1)

    def _convert_output_channel_decoded(self, port: int, channel: int, *, subport: int = 0) -> Tuple[int, int, int]:
        group, line = self._convert_output_port_decoded(port, subport)
        if channel < self.css.get_num_channels_of_line(group, line):
            ch3 = (group, line, channel)
        else:
            raise ValueError(f"invalid channel-#{channel} of port-{self._portname(port, subport)}")
        return ch3

    def _convert_output_channels(self, channels: Collection[Tuple[Quel1PortType, int]]) -> Set[Tuple[int, int, int]]:
        ch3s: Set[Tuple[int, int, int]] = set()
        if not isinstance(channels, collections.abc.Collection):
            raise TypeError(f"malformed channels: '{channels}'")
        for channel in channels:
            ch3s.add(self._convert_output_channel(channel))
        return ch3s

    def _get_all_subports_of_port(self, port: int) -> Set[int]:
        subports: Set[int] = set()
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid port of {self.boxtype}: {self._portname(port)}")

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
        port: Quel1PortType,
        channel: int = 0,
        *,
        subport: Union[int, None] = None,
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

        :param port: an index of the target port.
        :param channel: a (sub)port-local index of the channel of the (sub)port. (default: 0)
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

        group, line = self._convert_output_port_flex(port, subport)
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
        port: Quel1PortType,
        channel: Union[int, None] = None,
        *,
        subport: Union[int, None] = None,
        control_port_rfswitch: bool = True,
        control_monitor_rfswitch: bool = False,
    ) -> None:
        """stopping the wave generation on a given port.

        :param port: an index of the target port.
        :param channel: a port-local index of the channel. all the channels of the port are subject to stop if None.
        :param control_port_rfswitch: blocking the emission of the RF signal from the corresponding port if True.
        :param control_monitor_rfswitch: blocking the emission of the RF signal from the monitor-out port if True.
        :return: None
        """
        if isinstance(port, int):
            p: int = port
            if subport is None:
                subports: Set[int] = self._get_all_subports_of_port(port)
            else:
                subports = {subport}
        elif self._is_port_subport(port):
            if subport is None:
                p, sp = self._decode_port(port)
                subports = {sp}
            else:
                raise ValueError("duplicated subport specifiers")
        else:
            raise ValueError(f"malformed port : {port}")

        for sp in subports:
            if channel is None:
                group, line = self._convert_output_port_decoded(p, subport=sp)
            else:
                group, line, channel = self._convert_output_channel_decoded(p, subport=sp, channel=channel)
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
        port: Quel1PortType,
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

        group, rline = self._convert_input_port_flex(port)
        try:
            rrline: str = self._dev.rmap.resolve_rline(group, None)
            if rrline != rline:
                logger.warning(
                    f"the specified port-{self._portname(port)} may not be connected to "
                    "any ADC under the current configuration"
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
        port: Quel1PortType,
        channel: int,
        *,
        subport: Union[int, None] = None,
        amplitude: float = Quel1BoxIntrinsic.DEFAULT_AMPLITUDE,
        num_wave_sample: int = Quel1BoxIntrinsic.DEFAULT_NUM_WAVE_SAMPLE,
        num_repeats: Tuple[int, int] = Quel1BoxIntrinsic.DEFAULT_REPEATS,
        num_wait_samples: Tuple[int, int] = Quel1BoxIntrinsic.DEFAULT_NUM_WAIT_SAMPLES,
    ) -> None:
        """loading continuous wave data into a channel.

        :param port: an index of the target port.
        :param channel: a port-local index of the channel.
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
        group, line = self._convert_output_port_flex(port, subport)
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
        port: Quel1PortType,
        channel: int,
        *,
        subport: Union[int, None] = None,
        iq: npt.NDArray[np.complex64],
        num_repeats: Tuple[int, int] = Quel1BoxIntrinsic.DEFAULT_REPEATS,
        num_wait_samples: Tuple[int, int] = Quel1BoxIntrinsic.DEFAULT_NUM_WAIT_SAMPLES,
    ) -> None:
        """loading arbitrary wave data into a channel.

        :param port: an index of the target port.
        :param channel: a port-local index of the channel.
        :param iq: complex data of the signal to generate in 500Msps. I and Q coefficients of each sample must be
                   within the range of -32768 -- 32767. its length must be a multiple of 64.
        :param num_repeat: the number of repetitions of the given wave data given as a tuple of two integers,
                           a product of the two is the number of repetitions.
        :param num_wait_samples: number of wait duration in samples. given as a tuple of two integers that specify the
                               length of wait at the start of the whole wave sequence and the length of wait between
                               each repeated motifs, respectively.
        :return: None
        """
        group, line = self._convert_output_port_flex(port, subport)
        self._dev.load_iq_into_channel(
            group, line, channel, iq=iq, num_repeats=num_repeats, num_wait_samples=num_wait_samples
        )

    def initialize_all_awgs(self):
        self._dev.initialize_all_awgs()

    def initialize_all_capunits(self):
        self._dev.initialize_all_capunits()

    def prepare_for_emission(self, channels: Collection[Tuple[Quel1PortType, int]]):
        """making preparation of signal generation of multiple channels at the same time.

        :param channels: a collection of channels to be activated. each channel is specified as a tuple of a port and
                         a channel.
        """
        self._dev.prepare_for_emission(self._convert_output_channels(channels))

    def start_emission(self, channels: Collection[Tuple[Quel1PortType, int]]) -> None:
        """starting signal generation of multiple channels at the same time.

        :param channels: a collection of channels to be activated. each channel is specified as a tuple of a port and
                         a channel.
        """
        self._dev.start_emission(self._convert_output_channels(channels))

    def read_current_clock(self) -> int:
        return self._dev.read_current_clock()

    def read_current_and_latched_clock(self) -> Tuple[int, int]:
        return self._dev.read_current_and_latched_clock()

    def reserve_emission(
        self,
        channels: Collection[Tuple[Quel1PortType, int]],
        time_count: int,
        margin: float = Quel1BoxIntrinsic.DEFAULT_SCHEDULE_DEADLINE,
        window: float = Quel1BoxIntrinsic.DEFAULT_SCHEDULE_WINDOW,
        skip_validation: bool = True,
    ) -> None:
        """scheduling to start signal generation of multiple channels at the specified timing.

        :param channels: a collection of channels to be activated. each channel is specified as a tuple of a port and
                         a channel.
        :param time_count: time to start emission in terms of the time count of the synchronization subsystem.
        :param margin: reservations with less than time `margin` in second will not be accepted.
                       default value is 0.25 seconds.
        :param window: reservations out of the time `window` in second from now is rejected.
                       default value is 300 seconds.
        :param skip_validation: skip the validation of time count if True. default is False.
        """
        self._dev.reserve_emission(
            self._convert_output_channels(channels),
            time_count,
            margin=margin,
            window=window,
            skip_validation=skip_validation,
        )

    def stop_emission(self, channels: Collection[Tuple[Quel1PortType, int]]) -> None:
        """stopping signal generation on a given channel.

        :param channels: a collection of channels to be deactivated. each channel is specified as a tuple of a port and
                         a channel.
        """
        self._dev.stop_emission(self._convert_output_channels(channels))

    def simple_capture_start(
        self,
        port: Quel1PortType,
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

        cap_group, cap_rline = self._convert_input_port_flex(port)
        try:
            rrline: str = self._dev.rmap.resolve_rline(cap_group, None)
            if rrline != cap_rline:
                logger.warning(
                    f"the specified port-{self._portname(port)} may not be connected to "
                    "any ADC under the current configuration"
                )
        except ValueError:
            # Notes: failure of resolution means the mxfe in the group has multiple capture lines.
            pass

        trg_ch3: Union[Tuple[int, int, int], None] = (
            self._convert_output_channel(triggering_channel) if triggering_channel is not None else None
        )

        return self._dev.simple_capture_start(
            cap_group,
            cap_rline,
            runits=runits,
            num_samples=num_samples,
            delay_samples=delay_samples,
            triggering_channel=trg_ch3,
            timeout=timeout,
        )

    def capture_start(
        self,
        port: Quel1PortType,
        runits: Collection[int],
        *,
        triggering_channel: Union[Tuple[Quel1PortType, int], None] = None,
        timeout: float = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT,
    ) -> "Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]]":
        """capturing the wave signal from a given receiver channel.

        :param port: an index of a port to capture.
        :param runits: port-local indices of the capture units of the port.
        :param triggering_channel: a channel which triggers this capture when it starts to emit a signal.
                                   it is specified by a tuple of port and channel. the capture starts
                                   immediately if None.
        :param timeout: waiting time in second before capturing thread quits.
        :return: captured wave data in NumPy array
        """

        cap_group, cap_rline = self._convert_input_port_flex(port)
        if cap_rline not in self._dev.rmap.get_active_rlines_of_group(cap_group):
            raise ValueError(f"the specified port-{self._portname(port)} has no active ADC")
        trg_ch3: Union[Tuple[int, int, int], None] = (
            self._convert_output_channel(triggering_channel) if triggering_channel is not None else None
        )

        return self._dev.capture_start(
            cap_group,
            cap_rline,
            runits=runits,
            triggering_channel=trg_ch3,
            timeout=timeout,
        )

    def _config_ports(self, box_conf: Dict[Quel1PortType, Dict[str, Any]]) -> None:
        # Notes: configure output ports before input ones to keep "cnco_locked_with" intuitive in config_box().
        for port_subport, pc in box_conf.items():
            if self.is_output_port(port_subport):
                self._config_box_inner(port_subport, pc)

        for port_subport, pc in box_conf.items():
            if self.is_input_port(port_subport):
                self._config_box_inner(port_subport, pc)

    def _config_box_inner(self, port_subport: Quel1PortType, pc: Dict[str, Any]):
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
        elif self._is_port_subport(port_subport):
            port, subport = port_subport
        else:
            raise ValueError(f"malformed port: {port_subport}")

        for ch, channel_conf in channel_confs.items():
            self.config_channel(port, subport=subport, channel=ch, **channel_conf)
        for runit, runit_conf in runit_confs.items():
            self.config_runit(port, runit=runit, **runit_conf)
        self.config_port(port, subport=subport, **port_conf)

    def _parse_port_str(self, port_name: str) -> Quel1PortType:
        port_idx: Union[Quel1PortType, None] = None
        if type(port_name) is str:
            if port_name.isdigit():
                port_idx = int(port_name)
            elif port_name[0] == "(" and port_name[-1] == ")":
                splitted = port_name[1:-1].split(",")
                if len(splitted) == 2 and splitted[0].strip().isdigit() and splitted[1].strip().isdigit():
                    port_idx = (int(splitted[0]), int(splitted[1]))

        if port_idx is None or port_idx not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"unexpected name of port: '{port_name}'")
        return port_idx

    def _parse_channel_conf(self, cfg0: Dict[str, Any]) -> Dict[int, Any]:
        cfg1: Dict[int, Any] = {}
        for cname, ccfg in cfg0.items():
            if cname.isdigit():
                cfg1[int(cname)] = ccfg
            else:
                raise ValueError(f"unexpected index of channel: '{cname}'")
        return cfg1

    def _parse_ports_conf(self, cfg0: Dict[str, Dict[str, Any]]) -> Dict[Quel1PortType, Dict[str, Any]]:
        cfg1: Dict[Quel1PortType, Dict[str, Any]] = {}
        for pname, pcfg in cfg0.items():
            pidx = self._parse_port_str(pname)
            cfg1[pidx] = {}
            for k, v in pcfg.items():
                if k in {"channels", "runits"}:
                    cfg1[pidx][k] = self._parse_channel_conf(v)
                else:
                    cfg1[pidx][k] = v
        return cfg1

    def _parse_box_conf(
        self, cfg0: Dict[str, Dict[str, Any]]
    ) -> Union[Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]], Dict[Quel1PortType, Dict[str, Any]]]:
        if "mxfes" in cfg0 or "ports" in cfg0:
            cfg1: Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]] = {}
            for k, v in cfg0.items():
                if k == "mxfes":
                    cfg1[k] = {int(kk): vv for kk, vv in v.items()}
                elif k == "ports":
                    cfg1[k] = self._parse_ports_conf(v)
                else:
                    raise ValueError(f"invalid key (= '{k}) is detected in the box configration data")
            return cfg1
        else:
            return self._parse_ports_conf(cfg0)

    def config_box(
        self,
        box_conf: Union[Dict[str, Dict[Quel1PortType, Dict[str, Any]]], Dict[Quel1PortType, Dict[str, Any]]],
        ignore_validation: bool = False,
    ):
        ports_conf: Union[Dict[Quel1PortType, Dict[str, Any]], None] = None
        if "ports" in box_conf:
            box_conf = cast(Dict[str, Dict[Quel1PortType, Dict[str, Any]]], box_conf)
            ports_conf = box_conf["ports"]
        elif "mxfes" not in box_conf:
            box_conf = cast(Dict[Quel1PortType, Dict[str, Any]], box_conf)
            ports_conf = box_conf

        if ports_conf is not None:
            self._config_ports(ports_conf)

        if not ignore_validation:
            if not self.config_validate_box(box_conf):
                raise ValueError("the provided settings looks to be inconsistent")

    def config_box_from_jsonfile(self, box_conf_filepath: Union[Path, str], ignore_validation: bool = False):
        with open(box_conf_filepath) as f:
            cfg = self._parse_box_conf(json.load(f))
            self.config_box(cfg, ignore_validation)

    def config_box_from_jsonstr(self, box_conf_str: str, ignore_validation: bool = False):
        cfg = self._parse_box_conf(json.loads(box_conf_str))
        self.config_box(cfg, ignore_validation)

    def _config_validate_mxfe(self, mxfe_idx: int, mxfe_conf: Dict[str, Any]) -> bool:
        validity: bool = True
        for k, v in mxfe_conf.items():
            if k == "channel_interporation_rate":
                u = self._dev._css.get_channel_interpolation_rate(mxfe_idx)
                if v != u:
                    validity = False
                    logger.error(f"given channel interpolation rate (= {v}) doens't match the actual rate (= {u})")
            elif k == "main_interporation_rate":
                u = self._dev._css.get_main_interpolation_rate(mxfe_idx)
                if v != u:
                    validity = False
                    logger.error(f"given main interpolation rate (= {v}) doens't match the actual rate (= {u})")
            else:
                logger.error(f"unknown attribute '{k}' of MxFE-#{mxfe_idx}")
                validity = False
        return validity

    def _config_validate_port(self, port_subport: Quel1PortType, lc: Dict[str, Any]) -> bool:
        group, line = self._convert_any_port(port_subport)
        portname = self._portname(port_subport)
        if self._dev.is_output_line(group, line):
            alc: Dict[str, Any] = self.css.dump_line(group, cast(int, line))
            ad: str = "out"
            line_name: str = f"group:{group}, line:{line}"
        elif self._dev.is_input_line(group, line):
            alc = self.css.dump_rline(group, cast(str, line))
            ad = "in"
            line_name = f"group:{group}, rline:{line}"
        else:
            raise ValueError(f"invalid port of {self.boxtype}: {portname}")

        try:
            # Notes: it is troublesome to call _dev._config_validate_line() due to the handling of cnco_locked_with.
            return self._config_validate_port_inner(group, line, lc, "port-" + portname, alc, ad)
        except ValueError as e:
            if line_name in e.args[0]:
                raise ValueError(e.args[0].replace(line_name, "port-" + portname))
            else:
                raise

    def _config_validate_port_inner(
        self, group: int, line: Union[int, str], lc: Dict[str, Any], portname: str, alc: Dict[str, Any], ad: str
    ) -> bool:
        valid = True
        for k in lc:
            if k == "channels":
                valid &= self._dev._config_validate_channels(group, line, lc["channels"], portname)
            elif k == "runits":
                valid &= self._dev._config_validate_runits(group, line, lc["runits"], portname)
            elif k == "direction":
                if lc["direction"] != ad:
                    valid = False
                    logger.error(f"unexpected settings of {portname}:" f"direction = {ad} (!= {lc['direction']})")
            elif k in {"cnco_freq", "fnco_freq"}:
                if k not in alc:
                    valid = False
                    logger.error(f"unexpected settings at {portname}, '{k}' is not available")
                elif not self._dev._config_validate_frequency(group, line, k, lc[k], alc[k]):
                    valid = False
                    logger.error(f"unexpected settings at {portname}:{k} = {alc[k]} (!= {lc[k]})")
            elif k == "cnco_locked_with":
                dac_p, dac_sp = self._decode_port(lc[k])
                dac_g, dac_l = self._convert_output_port_decoded(dac_p, dac_sp)
                alf = alc["cnco_freq"]
                lf = self._dev._css.get_dac_cnco(dac_g, dac_l)
                if lf != alf:
                    valid = False
                    logger.error(
                        f"unexpected settings at {portname}:cnco_freq = {alf} (!= {lf}, "
                        f"that is cnco frequency of port-{self._portname(dac_p, dac_sp)}"
                    )
            elif k == "fullscale_current":
                if k not in alc:
                    valid = False
                    logger.error(f"unexpected settings at {portname}, '{k}' is not available")
                elif not self._dev._config_validate_fsc(group, cast(int, line), lc[k], alc[k]):
                    valid = False
                    logger.error(f"unexpected settings at {portname}:{k} = {alc[k]} (!= {lc[k]})")
            else:
                if k not in alc:
                    valid = False
                    logger.error(f"unexpected settings at {portname}, '{k}' is not available")
                elif lc[k] != alc[k]:
                    valid = False
                    logger.error(f"unexpected settings at {portname}:{k} = {alc[k]} (!= {lc[k]})")
        return valid

    def config_validate_box(
        self,
        box_conf: Union[
            Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]], Dict[Quel1PortType, Dict[str, Any]]
        ],
    ) -> bool:
        valid: bool = True
        ports_conf: Union[Dict[Quel1PortType, Dict[str, Any]], None] = None

        if "ports" in box_conf:
            box_conf = cast(Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]], box_conf)
            ports_conf = box_conf["ports"]
        elif "mxfes" not in box_conf:
            box_conf = cast(Dict[Quel1PortType, Dict[str, Any]], box_conf)
            ports_conf = box_conf

        if "mxfes" in box_conf:
            box_conf = cast(Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]], box_conf)
            for mxfe_idx, mxfe_conf in box_conf["mxfes"].items():
                if not isinstance(mxfe_idx, int):
                    raise ValueError(f"invalid mxfe index: '{mxfe_idx}'")
                valid &= self._config_validate_mxfe(mxfe_idx, mxfe_conf)

        if ports_conf is not None:
            for port, lc in ports_conf.items():
                valid &= self._config_validate_port(port, lc)

        return valid

    def is_output_port(self, port_subport: Quel1PortType):
        """check whether the given port is an output port or not.

        :param port_subport: an index of the target port.
        :return: True if the port is an output port.
        """
        group, line = self._convert_any_port(port_subport)
        return self._dev.is_output_line(group, line)

    def is_input_port(self, port_subport: Quel1PortType):
        """check whether the given port is an input port or not.

        :param port_subport: an index of the target port.
        :return: True if the port is an input port.
        """
        group, line = self._convert_any_port(port_subport)
        return self._dev.is_input_line(group, line)

    def is_monitor_input_port(self, port_subport: Quel1PortType):
        """check whether the given port is a monitor input port or not.

        :param port_subport: an index of the target port.
        :return: True if the port is a monitor input port.
        """
        group, line = self._convert_any_port(port_subport)
        return self._dev.is_monitor_input_line(group, line)

    def is_read_input_port(self, port_subport: Quel1PortType):
        """check whether the given port is a read input port or not.

        :param port_subport: an index of the target port.
        :return: True if the port is a read input port.
        """
        group, line = self._convert_any_port(port_subport)
        return self._dev.is_read_input_line(group, line)

    def config_port(
        self,
        port: Quel1PortType,
        *,
        subport: Union[int, None] = None,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        cnco_locked_with: Union[Quel1PortType, None] = None,
        vatt: Union[int, None] = None,
        sideband: Union[str, None] = None,
        fullscale_current: Union[int, None] = None,
        rfswitch: Union[str, None] = None,
    ) -> None:
        """configuring parameters of a given port, either of transmitter or receiver one.

        :param port: an index of the target port to configure.
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
        group, line = self._convert_any_port_flex(port, subport)
        portname = "port-" + self._portname(port, subport)
        if self._dev.is_output_line(group, line):
            if cnco_locked_with is not None:
                raise ValueError(f"no cnco_locked_with is available for the output {portname}")
            try:
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
            except ValueError as e:
                # Notes: tweaking error message
                linename = f"group:{group}, line:{line}"
                if linename in e.args[0]:
                    raise ValueError(e.args[0].replace(linename, portname))
                else:
                    raise

        elif self._dev.is_input_line(group, line):
            if vatt is not None or sideband is not None:
                raise ValueError(f"no configurable mixer is available for the input {portname}")
            if fullscale_current is not None:
                raise ValueError(f"no DAC is available for the input {portname}")
            if cnco_locked_with is not None:
                p, sp = self._decode_port(cnco_locked_with)
                converted_cnco_lock_with = self._convert_output_port_decoded(p, sp)
            else:
                converted_cnco_lock_with = None
            try:
                self._dev.config_rline(
                    group,
                    cast(str, line),
                    lo_freq=lo_freq,
                    cnco_freq=cnco_freq,
                    cnco_locked_with=converted_cnco_lock_with,
                    rfswitch=rfswitch,
                )
            except ValueError as e:
                # Notes: tweaking error message
                linename = f"group:{group}, rline:{line}"
                if linename in e.args[0]:
                    raise ValueError(e.args[0].replace(linename, portname))
                else:
                    raise
        else:
            raise AssertionError

    def config_channel(
        self,
        port: Quel1PortType,
        channel: int,
        *,
        subport: Union[int, None] = None,
        fnco_freq: Union[float, None] = None,
    ) -> None:
        """configuring parameters of a given channel, either of transmitter or receiver one.

        :param port: an index of the target port.
        :param channel: a port-local index of the channel.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :return: None
        """
        group, line = self._convert_any_port_flex(port, subport)
        portname = "port-" + self._portname(port, subport)
        if self._dev.is_output_line(group, line):
            try:
                self._dev.config_channel(group, cast(int, line), channel, fnco_freq=fnco_freq)
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
    ) -> None:
        """configuring parameters of a given receiver channel.

        :param port: an index of the target port.
        :param runit: a line-local index of the capture unit.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :return: None
        """
        group, rline = self._convert_any_port_flex(port, None)
        portname = "port-" + self._portname(port, None)
        if self._dev.is_input_line(group, rline):
            try:
                self._dev.config_runit(group, cast(str, rline), runit, fnco_freq=fnco_freq)
            except ValueError as e:
                # Notes: tweaking error message
                linename = f"group:{group}, rline:{rline}"
                if linename in e.args[0]:
                    raise ValueError(e.args[0].replace(linename, portname))
                else:
                    raise
        else:
            raise ValueError(f"{portname} is not an input port, not applicable")

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

    def config_rfswitches(self, rfswitch_confs: Dict[Quel1PortType, str], ignore_validation: bool = False) -> None:
        for port_subport, rc in rfswitch_confs.items():
            p, sp = self._decode_port(port_subport)
            self.config_rfswitch(p, rfswitch=rc)

        valid = True
        for port_subport, rc in rfswitch_confs.items():
            p, sp = self._decode_port(port_subport)
            arc = self.dump_rfswitch(p)
            if rc != arc:
                valid = False
                logger.warning(f"rfswitch of port-{self._portname(p, sp)} is finally set to {arc} (!= {rc})")

        if not (ignore_validation or valid):
            raise ValueError("the specified configuration of rf switches is not realizable")

    def config_rfswitch(self, port: int, *, rfswitch: str):
        group, line = self._convert_any_port_decoded(port, 0)
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

    def dump_rfswitch(self, port: Quel1PortType, *, subport: Union[int, None] = None) -> str:
        """dumping the current configuration of an RF switch
        :param port: an index of the target port

        :return: the current configuration setting of the RF switch
        """
        group, line = self._convert_any_port_flex(port, subport)
        return self._dev.dump_rfswitch(group, line)

    def dump_rfswitches(self, exclude_subordinate: bool = True) -> Dict[Quel1PortType, str]:
        """dumping the current configuration of all RF switches

        :return: a mapping of a port index and the configuration of its RF switch.
        """

        # actually, any key in Tuple[int, int] key is not defined.
        retval_intrinsic = self._dev.dump_rfswitches(exclude_subordinate)
        retval: Dict[Quel1PortType, str] = {}
        for port, (group, line) in self._PORT2LINE[self._boxtype].items():
            if isinstance(port, int) and (group, line) in retval_intrinsic:
                # Notes: subordinate rfswitch is not included when exclude_subordinate_switch is True.
                retval[port] = retval_intrinsic[group, line]
        return retval

    def dump_port(self, port: Quel1PortType, *, subport: Union[int, None] = None) -> Dict[str, Any]:
        """dumping the current configuration of a port
        :param port: an index of the target port

        :return: the current configuration setting of the RF switch
        """
        group, line = self._convert_any_port_flex(port, subport)
        return self._dump_port(group, line)

    def dump_box(self) -> Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]]:
        """dumping the current configuration of the ports

        :return: the current configuration of ports in dictionary.
        """
        retval: Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]] = {"mxfes": {}, "ports": {}}

        for mxfe_idx in self.css.get_all_mxfes():
            retval["mxfes"][mxfe_idx] = {
                "channel_interporation_rate": self._dev._css.get_channel_interpolation_rate(mxfe_idx),
                "main_interporation_rate": self._dev._css.get_main_interpolation_rate(mxfe_idx),
            }

        for port, (group, line) in self._PORT2LINE[self._boxtype].items():
            retval["ports"][port] = self._dump_port(group, line)

        return retval

    def _unparse_box_conf(self, cfg0: Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]]):
        cfg1: Dict[str, Dict[str, Dict[str, Any]]] = {}
        cfg1["mxfes"] = {str(mxfeidx): mxfecfg for mxfeidx, mxfecfg in cfg0["mxfes"].items()}
        cfg1["ports"] = {}
        for pidx, pcfg0 in cfg0["ports"].items():
            pcfg1 = {}
            for k, v in pcfg0.items():
                if k in {"channels", "runits"}:
                    pcfg1[k] = {str(kk): vv for kk, vv in v.items()}
                else:
                    pcfg1[k] = v
            cfg1["ports"][str(pidx)] = pcfg1
        return cfg1

    def dump_box_to_jsonfile(self, box_conf_filepath: Union[Path, str]) -> None:
        with open(box_conf_filepath, "w") as f:
            json.dump(self._unparse_box_conf(self.dump_box()), f, indent=2)

    def dump_box_to_jsonstr(self) -> str:
        return json.dumps(self._unparse_box_conf(self.dump_box()))

    def get_output_ports(self) -> Set[Quel1PortType]:
        """show a set of output ports of this box.

        :return: a set of all the output ports.
        """
        return set([p for p in self._PORT2LINE[self._boxtype] if self.is_output_port(p)])

    def get_input_ports(self) -> Set[Quel1PortType]:
        """show a set of input ports of this box.

        :return: a set of all the input ports.
        """
        return set([p for p in self._PORT2LINE[self._boxtype] if self.is_input_port(p)])

    def get_monitor_input_ports(self) -> Set[Quel1PortType]:
        """show a set of monitor input ports of this box.

        :return: a set of all the monitor input ports.
        """
        return set([p for p in self._PORT2LINE[self._boxtype] if self.is_monitor_input_port(p)])

    def get_read_input_ports(self) -> Set[Quel1PortType]:
        """show a set of read input ports of this box.

        :return: a set of all the read input ports.
        """
        return set([p for p in self._PORT2LINE[self._boxtype] if self.is_read_input_port(p)])

    def get_loopbacks_of_port(self, port: Quel1PortType, *, subport: Union[int, None] = None) -> Set[Quel1PortType]:
        """show a set of output ports which can be loop-backed to the specified input port

        :param port: an index of the target input port
        :return: a set of output ports which has loopback path to the input port
        """
        if isinstance(port, int) and isinstance(subport, int):
            port = (port, subport)

        if self.is_input_port(port):
            lpbk = self._LOOPBACK[self._boxtype]
            if port in lpbk:
                return set(lpbk[port])
            else:
                return set()
        else:
            raise ValueError(f"port-{self._portname(port)} is not an input port")

    def get_channels_of_port(self, port: Quel1PortType) -> Set[int]:
        """show a set of channels of the specified output port

        :param port: an index of the target output port
        :return: a set of channels of the output port
        """

        group, line = self._convert_output_port(port)
        return self._dev.get_channels_of_line(group, line)

    def get_runits_of_port(self, port: Quel1PortType) -> Set[int]:
        """show a set of channels of the specified output port

        :param port: an index of the target output port
        :return: a set of channels of the output port
        """

        group, rline = self._convert_input_port(port)
        return self._dev.get_runits_of_rline(group, rline)
