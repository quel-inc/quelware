import collections.abc
import copy
import json
import logging
from pathlib import Path
from typing import Any, Collection, Dict, Final, Optional, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
from e7awghal import AwgParam, CapIqDataReader, CapParam

from quel_ic_config.box_lock import guarded_by_box_lock
from quel_ic_config.e7resource_mapper import AbstractQuel1E7ResourceMapper
from quel_ic_config.linkupper import LinkupFpgaMxfe
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_box_intrinsic import Quel1BoxIntrinsic, create_css_wss_rmap
from quel_ic_config.quel1_config_subsystem import Quel1BoxType
from quel_ic_config.quel1_wave_subsystem import (
    AbstractCancellableTaskWrapper,
    AbstractStartAwgunitsTask,
    Quel1WaveSubsystem,
    StartCapunitsByTriggerTask,
    StartCapunitsNowTask,
)
from quel_ic_config.quel_config_common import Quel1ConfigOption, Quel1Feature

logger = logging.getLogger(__name__)


Quel1PortType = Union[int, Tuple[int, int]]


def parse_port_str(port_name: str) -> Quel1PortType:
    try:
        return int(port_name)
    except ValueError:
        pass
    try:
        if not (port_name.startswith("(") and port_name.endswith(")")):
            raise ValueError("invalid format: missing parentheses")
        parts = port_name[1:-1].split(",")
        if len(parts) != 2:
            raise ValueError("invalid format: requires exactly two elements")
        return (int(parts[0]), int(parts[1]))
    except ValueError as e:
        raise ValueError(f"unexpected name of port: '{port_name}'") from e


class BoxStartCapunitsNowTask(
    AbstractCancellableTaskWrapper[
        dict[tuple[int, int], CapIqDataReader], dict[tuple[Quel1PortType, int], CapIqDataReader]
    ]
):
    def __init__(self, task: StartCapunitsNowTask, mapping: dict[tuple[int, int], tuple[Quel1PortType, int]]):
        super().__init__(task)
        self._mapping = mapping

    def _conveter(
        self, orig: dict[tuple[int, int], CapIqDataReader]
    ) -> dict[tuple[Quel1PortType, int], CapIqDataReader]:
        return {self._mapping[capunit]: reader for capunit, reader in orig.items()}


class BoxStartCapunitsByTriggerTask(
    AbstractCancellableTaskWrapper[
        dict[tuple[int, int], CapIqDataReader], dict[tuple[Quel1PortType, int], CapIqDataReader]
    ]
):
    def __init__(self, task: StartCapunitsByTriggerTask, mapping: dict[tuple[int, int], tuple[Quel1PortType, int]]):
        super().__init__(task)
        self._mapping = mapping

    def _conveter(
        self, orig: dict[tuple[int, int], CapIqDataReader]
    ) -> dict[tuple[Quel1PortType, int], CapIqDataReader]:
        return {self._mapping[capunit]: reader for capunit, reader in orig.items()}


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

    __slots__ = ("_dev", "_boxtype", "_name")

    @classmethod
    @guarded_by_box_lock
    def create(
        cls,
        *,
        ipaddr_wss: str,
        ipaddr_sss: Union[str, None] = None,
        ipaddr_css: Union[str, None] = None,
        boxtype: Quel1BoxType,
        skip_init: bool = False,
        name: Optional[str] = None,
        **options: Collection[int],
    ) -> "Quel1Box":
        """create QuEL box objects
        :param ipaddr_wss: IP address of the wave generation subsystem of the target box
        :param ipaddr_sss: IP address of the sequencer subsystem of the target box (optional)
        :param ipaddr_css: IP address of the configuration subsystem of the target box (optional)
        :param boxtype: type of the target box
        :param config_root: root path of config setting files to read (optional)
        :param config_options: a collection of config options (optional)
        :param skip_init: skip calling box.initialization(), just for debugging.
        :param name: user-definable name for identification purposes.
        :param ignore_crc_error_of_mxfe: a list of MxFEs whose CRC error of the datalink is ignored. (optional)
        :param ignore_access_failure_of_adrf6780: a list of ADRF6780 whose communication faiulre via SPI bus is
                                                  dismissed (optional)
        :param ignore_lock_failure_of_lmx2594: a list of LMX2594 whose lock failure is ignored (optional)
        :param ignore_extraordinary_converter_select_of_mxfe: a list of MxFEs whose unusual converter mapping is
                                                              dismissed (optional)
        :return: a Quel1Box object
        """
        css, wss, rmap = create_css_wss_rmap(
            ipaddr_wss=ipaddr_wss,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            boxtype=boxtype,
        )
        box = Quel1Box(css=css, wss=wss, rmap=rmap, linkupper=None, name=name, **options)
        if not skip_init:
            box.initialize()
        return box

    def __init__(
        self,
        *,
        css: Quel1AnyConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: AbstractQuel1E7ResourceMapper,
        linkupper: Union[LinkupFpgaMxfe, None] = None,
        name: Optional[str] = None,
        **options: Collection[int],
    ):
        self._dev = Quel1BoxIntrinsic(css=css, wss=wss, rmap=rmap, linkupper=linkupper, **options)
        self._boxtype = css.boxtype
        if self._boxtype not in self._PORT2LINE:
            raise ValueError(f"unsupported boxtype; {self._boxtype}")
        if name is None:
            name = f"{self._dev._wss.ipaddr_wss}:{self.boxtype}"
        self._name = name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._dev._wss.ipaddr_wss}:{self.boxtype}>"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def css(self) -> Quel1AnyConfigSubsystem:
        return self._dev.css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._dev.wss

    @property
    def rmap(self) -> AbstractQuel1E7ResourceMapper:
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
    def has_lock(self) -> bool:
        return self._dev.has_lock

    @property
    def allow_dual_modulus_nco(self) -> bool:
        return self.css.allow_dual_modulus_nco

    @allow_dual_modulus_nco.setter
    def allow_dual_modulus_nco(self, v) -> None:
        self.css.allow_dual_modulus_nco = v

    @guarded_by_box_lock
    def initialize(self):
        self._dev.initialize()

    @guarded_by_box_lock
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

    @guarded_by_box_lock
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
        hard_reset_wss: bool = False,
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
            hard_reset_wss=hard_reset_wss,
            background_noise_threshold=background_noise_threshold,
            ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe,
            ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
            ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            ignore_extraordinary_converter_select_of_mxfe=ignore_extraordinary_converter_select_of_mxfe,
            restart_tempctrl=restart_tempctrl,
        )

    @guarded_by_box_lock
    def link_status(self, ignore_crc_error_of_mxfe: Union[Collection[int], None] = None) -> Dict[int, bool]:
        return self._dev.link_status(ignore_crc_error_of_mxfe=ignore_crc_error_of_mxfe)

    def _portname(self, port: Quel1PortType) -> str:
        if isinstance(port, int):
            return f"#{port:02d}"
        elif self._is_port_subport(port):
            p, sp = cast(Tuple[int, int], port)
            return f"#{p:02d}-{sp:d}"
        else:
            return str(port)

    def _is_port_subport(self, port: Quel1PortType) -> bool:
        return isinstance(port, tuple) and len(port) == 2 and isinstance(port[0], int) and isinstance(port[1], int)

    def _decode_port(self, port: Quel1PortType) -> Tuple[int, int]:
        if isinstance(port, int):
            p: int = port
            sp: int = 0
        elif self._is_port_subport(port):
            p, sp = port[0], port[1]
        else:
            raise ValueError(f"malformed port: '{port}'")

        portname = self._portname(port)
        if sp == 0:
            if p not in self._PORT2LINE[self._boxtype]:
                raise ValueError(f"invalid port of {self.boxtype}: {portname}")
        else:
            if port not in self._PORT2LINE[self._boxtype]:
                raise ValueError(f"invalid port of {self.boxtype}: {portname}")
        return p, sp

    def _convert_any_port(self, port: Quel1PortType) -> Tuple[int, Union[int, str]]:
        p, sp = self._decode_port(port)
        if sp == 0:
            port = p
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid port of {self.boxtype}: {self._portname(port)}")
        return self._PORT2LINE[self._boxtype][port]

    def _convert_output_port(self, port: Quel1PortType) -> Tuple[int, int]:
        group, line = self._convert_any_port(port)
        if not self._dev.is_output_line(group, line):
            raise ValueError(f"port-{self._portname(port)} is not an output port")
        return group, cast(int, line)

    def _convert_input_port(self, port: Quel1PortType) -> Tuple[int, str]:
        group, line = self._convert_any_port(port)
        if not self._dev.is_input_line(group, line):
            raise ValueError(f"port-{self._portname(port)} is not an input port")
        return group, cast(str, line)

    def _convert_output_channel(self, channel: Tuple[Quel1PortType, int]) -> Tuple[int, int, int]:
        if not (isinstance(channel, tuple) and len(channel) == 2):
            raise ValueError(f"malformed channel: '{channel}'")

        port, ch1 = channel
        group, line = self._convert_output_port(port)
        if ch1 < self.css.get_num_channels_of_line(group, line):
            ch3 = (group, line, ch1)
        else:
            raise ValueError(f"invalid channel-#{ch1} of port-{self._portname(port)}")
        return ch3

    def _convert_output_channels(self, channels: Collection[Tuple[Quel1PortType, int]]) -> Set[Tuple[int, int, int]]:
        ch3s: Set[Tuple[int, int, int]] = set()
        if not isinstance(channels, collections.abc.Collection):
            raise TypeError(f"malformed channels: '{channels}'")
        for channel in channels:
            ch3s.add(self._convert_output_channel(channel))
        return ch3s

    # Notes: implement it for just in case.
    def get_ports_sharing_physical_port(self, port: Quel1PortType) -> Set[Quel1PortType]:
        subports: Set[Quel1PortType] = set()
        if port not in self._PORT2LINE[self._boxtype]:
            raise ValueError(f"invalid port of {self.boxtype}: {self._portname(port)}")
        p, _ = self._decode_port(port)

        for port1 in self._PORT2LINE[self._boxtype]:
            p1, _ = self._decode_port(port1)
            if p1 == p:
                subports.add(port1)

        return subports

    def _config_ports(self, box_conf: Dict[Quel1PortType, Dict[str, Any]]) -> None:
        # Notes: configure output ports before input ones to keep "cnco_locked_with" intuitive in config_box().
        for port, pc in box_conf.items():
            if self.is_output_port(port):
                self._config_box_inner(port, pc)

        for port, pc in box_conf.items():
            if self.is_input_port(port):
                self._config_box_inner(port, pc)

    def _config_box_inner(self, port: Quel1PortType, pc: Dict[str, Any]):
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

        for ch, channel_conf in channel_confs.items():
            self.config_channel(port, channel=ch, **channel_conf)
        for runit, runit_conf in runit_confs.items():
            self.config_runit(port, runit=runit, **runit_conf)
        self.config_port(port, **port_conf)

    def is_valid_port(self, port: Quel1PortType) -> bool:
        return port in self._PORT2LINE[self._boxtype]

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
            pidx = parse_port_str(pname)
            if not self.is_valid_port(pidx):
                raise ValueError(f"Invalid port: '{pidx}'")
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

    def _config_box(
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
            if not self._config_validate_box(box_conf):
                raise ValueError("the provided settings looks to be inconsistent")

    @guarded_by_box_lock
    def config_box(
        self,
        box_conf: Union[Dict[str, Dict[Quel1PortType, Dict[str, Any]]], Dict[Quel1PortType, Dict[str, Any]]],
        ignore_validation: bool = False,
    ):
        self._config_box(box_conf, ignore_validation)

    @guarded_by_box_lock
    def config_box_from_jsonfile(self, box_conf_filepath: Union[Path, str], ignore_validation: bool = False):
        with open(box_conf_filepath) as f:
            cfg = self._parse_box_conf(json.load(f))
            self._config_box(cfg, ignore_validation)

    @guarded_by_box_lock
    def config_box_from_jsonstr(self, box_conf_str: str, ignore_validation: bool = False):
        cfg = self._parse_box_conf(json.loads(box_conf_str))
        self._config_box(cfg, ignore_validation)

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

    def _config_validate_port(self, port: Quel1PortType, lc: Dict[str, Any]) -> bool:
        group, line = self._convert_any_port(port)
        portname = self._portname(port)
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
                dac_g, dac_l = self._convert_output_port(lc[k])
                alf = alc["cnco_freq"]
                lf = self._dev._css.get_dac_cnco(dac_g, dac_l)
                if lf != alf:
                    valid = False
                    logger.error(
                        f"unexpected settings at {portname}:cnco_freq = {alf} (!= {lf}, "
                        f"that is cnco frequency of port-{self._portname(lc[k])}"
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

    def _config_validate_box(
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

    @guarded_by_box_lock
    def config_validate_box(
        self,
        box_conf: Union[
            Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]], Dict[Quel1PortType, Dict[str, Any]]
        ],
    ) -> bool:
        return self._config_validate_box(box_conf)

    def is_output_port(self, port: Quel1PortType):
        """check whether the given port is an output port or not.

        :param port: an index of the target port.
        :return: True if the port is an output port.
        """
        group, line = self._convert_any_port(port)
        return self._dev.is_output_line(group, line)

    def is_input_port(self, port: Quel1PortType):
        """check whether the given port is an input port or not.

        :param port: an index of the target port.
        :return: True if the port is an input port.
        """
        group, line = self._convert_any_port(port)
        return self._dev.is_input_line(group, line)

    def is_monitor_input_port(self, port: Quel1PortType):
        """check whether the given port is a monitor input port or not.

        :param port: an index of the target port.
        :return: True if the port is a monitor input port.
        """
        group, line = self._convert_any_port(port)
        return self._dev.is_monitor_input_line(group, line)

    def is_read_input_port(self, port: Quel1PortType):
        """check whether the given port is a read input port or not.

        :param port: an index of the target port.
        :return: True if the port is a read input port.
        """
        group, line = self._convert_any_port(port)
        return self._dev.is_read_input_line(group, line)

    @guarded_by_box_lock
    def config_port(
        self,
        port: Quel1PortType,
        *,
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
        group, line = self._convert_any_port(port)
        portname = "port-" + self._portname(port)
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
                converted_cnco_lock_with = self._convert_output_port(cnco_locked_with)
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

    @guarded_by_box_lock
    def config_channel(
        self,
        port: Quel1PortType,
        channel: int,
        *,
        fnco_freq: Union[float, None] = None,
        awg_param: Union[AwgParam, None] = None,
    ) -> None:
        """configuring parameters of a given channel, either of transmitter or receiver one.

        :param port: an index of the target port.
        :param channel: a port-local index of the channel.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :param awg_param: an object holding parameters of signal to be generated.
        :return: None
        """
        group, line = self._convert_any_port(port)
        portname = "port-" + self._portname(port)
        if self._dev.is_output_line(group, line):
            try:
                self._dev.config_channel(group, cast(int, line), channel, fnco_freq=fnco_freq, awg_param=awg_param)
            except ValueError as e:
                linename = f"group:{group}, line:{line}"
                if linename in e.args[0]:
                    raise ValueError(e.args[0].replace(linename, portname))
                else:
                    raise
        else:
            raise ValueError(f"{portname} is not an output port, not applicable")

    @guarded_by_box_lock
    def config_runit(
        self,
        port: Quel1PortType,
        runit: int,
        *,
        fnco_freq: Union[float, None] = None,
        capture_param: Union[CapParam, None] = None,
    ) -> None:
        """configuring parameters of a given receiver channel.

        :param port: an index of the target port.
        :param runit: a line-local index of the capture unit.
        :param fnco_freq: the frequency of the corresponding FNCO in Hz. it must be within the range of -250e6 and
                          250e6.
        :param capture_param: an object keeping capture settings.
        :return: None
        """
        group, rline = self._convert_any_port(port)
        portname = "port-" + self._portname(port)
        if self._dev.is_input_line(group, rline):
            try:
                self._dev.config_runit(group, cast(str, rline), runit, fnco_freq=fnco_freq, capture_param=capture_param)
            except ValueError as e:
                # Notes: tweaking error message
                linename = f"group:{group}, rline:{rline}"
                if linename in e.args[0]:
                    raise ValueError(e.args[0].replace(linename, portname))
                else:
                    raise
        else:
            raise ValueError(f"{portname} is not an input port, not applicable")

    @guarded_by_box_lock
    def block_all_output_ports(self) -> None:
        """set RF switch of all output ports to block.

        :return:
        """
        self._dev.block_all_output_lines()

    @guarded_by_box_lock
    def pass_all_output_ports(self):
        """set RF switch of all output ports to pass.

        :return:
        """
        self._dev.pass_all_output_lines()

    @guarded_by_box_lock
    def config_rfswitches(self, rfswitch_confs: Dict[Quel1PortType, str], ignore_validation: bool = False) -> None:
        for port, rc in rfswitch_confs.items():
            p, sp = self._decode_port(port)
            self._config_rfswitch(p, rfswitch=rc)

        valid = True
        for port, rc in rfswitch_confs.items():
            p, sp = self._decode_port(port)
            arc = self.dump_rfswitch(p)
            if rc != arc:
                valid = False
                logger.warning(f"rfswitch of port-{self._portname(port)} is finally set to {arc} (!= {rc})")

        if not (ignore_validation or valid):
            raise ValueError("the specified configuration of rf switches is not realizable")

    def _config_rfswitch(self, port: Quel1PortType, *, rfswitch: str):
        group, line = self._convert_any_port(port)
        self._dev.config_rfswitch(group, line, rfswitch=rfswitch)

    @guarded_by_box_lock
    def config_rfswitch(self, port: Quel1PortType, *, rfswitch: str):
        self._config_rfswitch(port, rfswitch=rfswitch)

    @guarded_by_box_lock
    def activate_monitor_loop(self, group: int) -> None:
        """enabling an internal monitor loop-back path from a monitor-out port to a monitor-in port.

        :param group: an index of a group which the monitor path belongs to.
        :return: None
        """
        self._dev.activate_monitor_loop(group)

    @guarded_by_box_lock
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

    @guarded_by_box_lock
    def dump_rfswitch(self, port: Quel1PortType) -> str:
        """dumping the current configuration of an RF switch
        :param port: an index of the target port

        :return: the current configuration setting of the RF switch
        """
        group, line = self._convert_any_port(port)
        return self._dev.dump_rfswitch(group, line)

    @guarded_by_box_lock
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

    @guarded_by_box_lock
    def dump_port(self, port: Quel1PortType) -> Dict[str, Any]:
        """dumping the current configuration of a port
        :param port: an index of the target port

        :return: the current configuration setting of the RF switch
        """
        group, line = self._convert_any_port(port)
        return self._dump_port(group, line)

    def _dump_box(self) -> Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]]:
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

    @guarded_by_box_lock
    def dump_box(self) -> Dict[str, Dict[Union[int, Quel1PortType], Dict[str, Any]]]:
        """dumping the current configuration of the ports

        :return: the current configuration of ports in dictionary.
        """
        return self._dump_box()

    @guarded_by_box_lock
    def dump_box_to_jsonfile(self, box_conf_filepath: Union[Path, str]) -> None:
        with open(box_conf_filepath, "w") as f:
            json.dump(self._unparse_box_conf(self._dump_box()), f, indent=2)

    @guarded_by_box_lock
    def dump_box_to_jsonstr(self) -> str:
        return json.dumps(self._unparse_box_conf(self._dump_box()))

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

    def get_loopbacks_of_port(self, port: Quel1PortType) -> Set[Quel1PortType]:
        """show a set of output ports which can be loop-backed to the specified input port

        :param port: an index of the target input port
        :return: a set of output ports which has loopback path to the input port
        """
        if self.is_input_port(port):
            lpbk = self._LOOPBACK[self._boxtype]
            if port in lpbk:
                return set(lpbk[port])
            else:
                return set()
        else:
            raise ValueError(f"port-{self._portname(port)} is not an input port")

    @guarded_by_box_lock
    def get_channels_of_port(self, port: Quel1PortType) -> Set[int]:
        """show a set of channels of the specified output port

        :param port: an index of the target output port
        :return: a set of channels of the output port
        """

        group, line = self._convert_output_port(port)
        return self._dev.get_channels_of_line(group, line)

    @guarded_by_box_lock
    def get_runits_of_port(self, port: Quel1PortType) -> Set[int]:
        """show a set of channels of the specified output port

        :param port: an index of the target output port
        :return: a set of channels of the output port
        """
        group, rline = self._convert_input_port(port)
        return self._dev.get_runits_of_rline(group, rline)

    @guarded_by_box_lock
    def get_names_of_wavedata(self, port: Quel1PortType, channel: int) -> set[str]:
        group, line, channel = self._convert_output_channel((port, channel))
        return self._dev.get_names_of_wavedata(group, line, channel)

    @guarded_by_box_lock
    def register_wavedata(
        self,
        port: Quel1PortType,
        channel: int,
        name: str,
        iq: npt.NDArray[np.complex64],
        allow_update: bool = True,
        **kwdargs,
    ) -> None:
        group, line, channel = self._convert_output_channel((port, channel))
        self._dev.register_wavedata(group, line, channel, name, iq, allow_update, **kwdargs)

    def has_wavedata(self, port: Quel1PortType, channel: int, name: str) -> bool:
        group, line, channel = self._convert_output_channel((port, channel))
        return self._dev.has_wavedata(group, line, channel, name)

    def delete_wavedata(self, port: Quel1PortType, channel: int, name: str) -> None:
        group, line, channel = self._convert_output_channel((port, channel))
        self._dev.delete_wavedata(group, line, channel, name)

    @guarded_by_box_lock
    def initialize_all_awgunits(self):
        self._dev.initialize_all_awgunits()

    @guarded_by_box_lock
    def initialize_all_capunits(self):
        self._dev.initialize_all_capunits()

    @guarded_by_box_lock
    def get_current_timecounter(self) -> int:
        return self._dev.get_current_timecounter()

    @guarded_by_box_lock
    def get_latest_sysref_timecounter(self) -> int:
        return self._dev.get_latest_sysref_timecounter()

    @guarded_by_box_lock
    def get_averaged_sysref_offset(self, num_iteration: int = 100) -> float:
        return self._dev.get_averaged_sysref_offset(num_iteration)

    @guarded_by_box_lock
    def start_wavegen(
        self,
        channels: Collection[Tuple[Quel1PortType, int]],
        timecounter: Optional[int] = None,
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
        return_after_start_emission: Optional[bool] = None,
    ) -> AbstractStartAwgunitsTask:
        return self._dev.start_wavegen(
            self._convert_output_channels(channels),
            timecounter,
            timeout=timeout,
            polling_period=polling_period,
            disable_timeout=disable_timeout,
            return_after_start_emission=return_after_start_emission,
        )

    @guarded_by_box_lock
    def start_capture_now(
        self,
        runits: Collection[Tuple[Quel1PortType, int]],
        *,
        timeout: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ) -> BoxStartCapunitsNowTask:

        mapping: dict[tuple[int, int], tuple[Quel1PortType, int]] = {}
        for port, runit in runits:
            gr, rl = self._convert_input_port(port)
            cu = self._dev._get_capunit_from_runit(gr, rl, runit)
            mapping[cu] = (port, runit)

        capunit_idxs = set(mapping.keys())
        return BoxStartCapunitsNowTask(
            self.wss.start_capunits_now(
                capunit_idxs, timeout=timeout, polling_period=polling_period, disable_timeout=disable_timeout
            ),
            mapping,
        )

    @guarded_by_box_lock
    def start_capture_by_awg_trigger(
        self,
        runits: Collection[Tuple[Quel1PortType, int]],
        channels: Collection[Tuple[Quel1PortType, int]],
        timecounter: Optional[int] = None,
        *,
        timeout_before_trigger: Optional[float] = None,
        timeout_after_trigger: Optional[float] = None,
        polling_period: Optional[float] = None,
        disable_timeout: bool = False,
    ) -> tuple[BoxStartCapunitsByTriggerTask, AbstractStartAwgunitsTask]:
        if len(runits) == 0:
            raise ValueError("no capture units are specified")
        if len(channels) == 0:
            raise ValueError("no triggering channel are specified")
        if timecounter and timecounter < 0:
            raise ValueError("negative timecounter is not allowed")
        if timeout_before_trigger and timeout_before_trigger <= 0.0:
            raise ValueError("non-positive timeout_before_trigger is not allowed")
        if timeout_after_trigger and timeout_after_trigger <= 0.0:
            raise ValueError("non-positive timeout_after_trigger is not allowed")
        if polling_period and polling_period <= 0.0:
            raise ValueError("non-positive polling_period is not allowed")

        mapping: dict[tuple[int, int], tuple[Quel1PortType, int]] = {}
        capmod_idxs: set[int] = set()
        for port, runit in runits:
            gr, rl = self._convert_input_port(port)
            cu = self._dev._get_capunit_from_runit(gr, rl, runit)
            mapping[cu] = (port, runit)
            capmod_idxs.add(cu[0])

        # Notes: any channel is fine since they all will start at the same time.
        trigger_idx = self._dev._get_awg_from_channel(*self._convert_output_channel(list(channels)[0]))
        for capmod_idx in capmod_idxs:
            self.wss.set_triggering_awg_to_line(capmod_idx, trigger_idx)

        capunit_idxs = set(mapping.keys())
        cap_task = BoxStartCapunitsByTriggerTask(
            self.wss.start_capunits_by_trigger(
                capunit_idxs,
                timeout_before_trigger=timeout_before_trigger,
                timeout_after_trigger=timeout_after_trigger,
                polling_period=polling_period,
                disable_timeout=disable_timeout,
            ),
            mapping,
        )
        gen_task: AbstractStartAwgunitsTask = self.start_wavegen(channels, timecounter, polling_period=polling_period)
        return cap_task, gen_task
