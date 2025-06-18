import logging
import socket
from collections.abc import Collection
from typing import Union

from e7awghal import ClockmasterAu200Hal

from quel_ic_config.box_lock import BoxLockError
from quel_ic_config.exstickge_sock_client import FileLockKeeper
from quel_ic_config.quel1_box import Quel1Box

logger = logging.getLogger(__name__)


class QuelClockMasterV1:
    def __init__(self, ipaddr: str, boxes: Collection[Quel1Box], allow_automatic_boot: bool = True):
        self._ipaddr: str = ipaddr
        self.hal: ClockmasterAu200Hal = ClockmasterAu200Hal(ipaddr=self._ipaddr, auth_callback=self._auth_callback)
        self._boxes: set[Quel1Box] = set(boxes)
        self._allow_automatic_reboot = allow_automatic_boot
        self._lock_keeper: Union[FileLockKeeper, None] = FileLockKeeper(target=(self._ipaddr, 16384))
        if not self._lock_keeper.activate():
            raise BoxLockError(f"failed to acquire lock of {self.__class__.__name__}:{self._ipaddr}")

    def __del__(self):
        self.terminate()

    def check_availability(self) -> bool:
        available: bool = False
        try:
            self.get_current_timecounter()
            available = True
        except socket.timeout:
            logger.info(f"clockmaster {self._ipaddr} doesn't respond to a read-counter request")
        return available

    @property
    def has_lock(self) -> bool:
        return (self._lock_keeper is not None) and self._lock_keeper.has_lock

    def _auth_callback(self) -> bool:
        rv = self.has_lock
        for b in self._boxes:
            rv &= b.has_lock
        return rv

    def get_current_timecounter(self) -> int:
        return self.hal.ctrl.read_counter()

    def sync_boxes(self) -> None:
        kick_done = False
        try:
            self.hal.ctrl.kick_sync({box.wss.hal.syncintf for box in self._boxes})
            kick_done = True
        except socket.timeout:
            if self._allow_automatic_reboot:
                logger.warning(
                    f"clockmaster ({self._ipaddr}) doesn't respond to kick, reboot will be conducted before retrying"
                )
            else:
                logger.error(
                    f"clockmaster ({self._ipaddr}) doesn't respond to kick, "
                    "sending reboot command to clockmaster may solve the problem"
                )
                raise

        if not kick_done:
            self.reboot()
            try:
                self.hal.ctrl.kick_sync({box.wss.hal.syncintf for box in self._boxes})
            except socket.timeout:
                logger.error(f"clockmaster ({self._ipaddr}) doesn't respond to kick after reboot")
                raise

    def reboot(self) -> None:
        self.hal.rebooter.reboot()

    def terminate(self):
        if self._lock_keeper and self._lock_keeper.is_alive():
            if self._lock_keeper.deactivate():
                self._lock_keeper = None
