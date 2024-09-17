import copy
import logging
import socket
import struct
from collections.abc import Collection
from threading import RLock
from typing import Any, Final

import numpy as np

from e7awghal.awgctrl import AwgCtrl
from e7awghal.e7awg_packet import BasePacketAccess, E7awgOutgoingPacket, E7awgPacketAccess, E7awgPacketMode
from e7awghal.fwtype import E7FwAuxAttr

logger = logging.getLogger(__name__)


class SimplemultiAwgTriggers:
    def __init__(self):
        self._triggers: list[tuple[np.uint64, set[int]]] = []
        self._awgunits: set[int] = set()

    @property
    def awgunits(self) -> set[int]:
        return set(self._awgunits)

    @property
    def triggers(self) -> tuple[tuple[np.uint64, set[int]], ...]:
        return tuple(copy.deepcopy(self._triggers))

    @property
    def num_trigger(self) -> int:
        return len(self._triggers)

    def clear(self):
        self._triggers.clear()
        self._awgunits.clear()

    def append(self, cntr: int, target_awgs: Collection[int]):
        if not (0 <= cntr <= 0xFFFF_FFFF_FFFF_FFFF):
            raise ValueError(f"cntr value (= {cntr}) is out of range")
        if self.num_trigger == 0 or self._triggers[-1][0] < cntr:
            self._triggers.append((np.uint64(cntr), set(target_awgs)))
            self._awgunits.update(target_awgs)
        else:
            raise ValueError(
                f"new cntr value (= {cntr}) is not greater than the previous one (= {self._triggers[-1][0]})"
            )


class SimplemultiSequencer:
    __slots__ = (
        "_settings",
        "_udprw",
        "_auxattr",
        "_awgctrl",
        "_lock",
        "_master_lock",
        "_latest_cmd_cntr",
        "_serial_num",
    )

    def __init__(
        self,
        ipaddrs: dict[str, str],
        settings: dict[str, Any],
        udprws: dict[tuple[str, int], BasePacketAccess],
        awgctrl: AwgCtrl,
        auxattr: set[E7FwAuxAttr],
    ):
        self._settings: Final[dict[str, Any]] = settings
        for k in ("nic", "port"):
            if k not in self._settings:
                raise ValueError(f"an essential key '{k}' in the settings is missing")
        nic, port = settings["nic"], settings["port"]
        udprw = udprws.get((nic, port))
        if udprw is None:
            udprw = E7awgPacketAccess(ipaddrs[nic], port)
            udprws[nic, port] = udprw
        else:
            if not isinstance(udprw, E7awgPacketAccess):
                raise RuntimeError(f"inconsistent packet accesses for {nic}:{port}")
        self._udprw: Final[E7awgPacketAccess] = udprw
        self._awgctrl: Final[AwgCtrl] = awgctrl
        self._auxattr: Final[set[E7FwAuxAttr]] = auxattr
        self._lock = RLock()
        self._master_lock = awgctrl.lock
        self._latest_cmd_cntr: int = 0
        self._serial_num: int = 0

    def initialize(self):
        with self._lock:
            self.cancel_triggers(at_initialization=True)

    @property
    def latest_command_timecount(self) -> int:
        return self._latest_cmd_cntr

    def add_awg_start(self, triggers: SimplemultiAwgTriggers):
        if not isinstance(triggers, SimplemultiAwgTriggers):
            raise TypeError("invalid trigger specifier, not a SimplemultiAwgTrigger object")

        pkt = E7awgOutgoingPacket(
            mode=E7awgPacketMode.SIMPLEMULTI_ADD_AWG_START,
            address=0,
            num_payload_bytes=triggers.num_trigger * 16,
        )
        with self._lock, self._master_lock:
            if self._awgctrl.are_busy_any(triggers.awgunits):
                raise RuntimeError("stop adding the triggers because some awg_units are still busy")

            for (
                i,
                (cntr, target_awgs),
            ) in enumerate(triggers.triggers):
                if cntr < self._latest_cmd_cntr:
                    raise ValueError(
                        f"cannot add a command with earlier time count (= {cntr}) "
                        f"than the previous one (= {self._latest_cmd_cntr})"
                    )
                bitmap = self._awgctrl.set2bitmap(target_awgs)
                pkt.payload[i * 16 : (i + 1) * 16] = struct.pack("<QHxxxxxB", cntr, bitmap, self._serial_num)
                self._latest_cmd_cntr = int(cntr)
                self._serial_num = (self._serial_num + 1) & 0xFF
            _ = self._udprw.send_command(pkt)

    def cancel_triggers(self, at_initialization: bool = False) -> None:
        with self._lock:
            # Notes: just CANCEL is not enough
            pkt = E7awgOutgoingPacket(
                mode=E7awgPacketMode.SIMPLEMULTI_CMD_CANCEL_AND_TERMINATE,
                address=0,
                num_payload_bytes=0,
            )
            try:
                _ = self._udprw.send_command(pkt, expect_no_reply=True)
            except socket.timeout:
                pass
            self._latest_cmd_cntr = 0
            self._serial_num = 0
            if not at_initialization:
                logger.warning(f"simplemulti sequencer of {self._udprw._dest_addrport[0]} is reset")
