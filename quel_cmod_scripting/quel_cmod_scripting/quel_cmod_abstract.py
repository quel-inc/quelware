import logging
import telnetlib
import time
from abc import ABCMeta
from typing import Final, Set, Tuple

from packaging import version

logger = logging.getLogger(__name__)


class QuelCmodAbstract(metaclass=ABCMeta):
    CORRESPONDING_VARIANTS: Set[str] = {
        "plain",
        "quel1",
        "fancontrol",
        "switchcontrol",
        "quel1seproto",
    }
    MINIMUM_VERSION: str = "0.0.1"

    # for the oldest firmware
    DEFAULT_VARIANT: Final[str] = "quel1"
    DEFAULT_VERSION: Final[str] = "0.0.1"

    def __init__(self, host: str, port: int):
        self.host: Final[str] = host
        self.port: Final[int] = port
        self.session: Final[telnetlib.Telnet] = telnetlib.Telnet(self.host, self.port)
        self.clear_session()
        vv = self.ver()
        self.variant: Final[str] = vv[0]
        self.version: Final[str] = vv[1]
        if self.variant not in self.CORRESPONDING_VARIANTS:
            raise RuntimeError(f"unsupported variant of firmware: '{self.variant}'")
        if version.parse(self.version) < version.parse(self.MINIMUM_VERSION):
            raise RuntimeError(f"unsupported version of firmware '{self.version}'")

    def __del__(self):
        if hasattr(self, "session"):
            self.session.close()

    def init(self) -> None:
        pass

    def clear_session(self, wait0: float = 1.0, wait1: float = 0.1):
        time.sleep(wait0)
        n_empty = 0
        while True:
            reply = self.session.read_very_eager()
            if len(reply) == 0:
                n_empty += 1
                if n_empty > 3:
                    break
            time.sleep(wait1)

        if len(reply) != 0:
            logger.warning(
                f"got output of the previous session '{reply!r}', ignore it."
            )
        self.execute("")

    def execute(self, cmd: str, timeout: float = 2.0):
        self.session.write(cmd.encode("ascii") + b"\n")
        reply = self.session.read_until(b"# ", timeout).decode("ascii")
        if reply.count("# ") != 0:
            reply = reply[0 : reply.index("# ")]  # noqa: E203
        return reply

    def ver(self) -> Tuple[str, str]:
        r = self.execute("ver")
        if r.startswith("error:"):
            # this oldest release (version 0.0.1) doesn't have "ver" command.
            return self.DEFAULT_VARIANT, self.DEFAULT_VERSION
        else:
            t = r.strip().split(":")
            return t[0], t[1]
