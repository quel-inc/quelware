import logging
import socket
import telnetlib
import time
from enum import Enum
from typing import Final, List, Union

logger = logging.getLogger(__name__)


class PeSwitchState(str, Enum):
    ON = "on"
    OFF = "off"
    PENDING = "pending"


class Pe6108ava:
    _DEFAULT_USERNAME = "teladmin"
    _DEFAULT_PASSWORD = "telpwd"
    _DEFAULT_PORT = 23
    _DEFAULT_TIMEOUT: Final[float] = 1.0
    _BR = "\r\n"
    _PROMPT = "> "
    _DEFAULT_OFF_DURATION: Final[float] = 10.0
    _MINIMUM_OFF_DURATION: Final[float] = 5.0

    def __init__(
        self,
        hostname: str,
        port: int = _DEFAULT_PORT,
        timeout: float = _DEFAULT_TIMEOUT,
        username: str = _DEFAULT_USERNAME,
        password: str = _DEFAULT_PASSWORD,
    ):
        self._ipaddr = socket.gethostbyname(hostname)
        self._port = port
        self._timeout = timeout
        self._username = username
        self._password = password
        self._tn: Union[telnetlib.Telnet, None] = None

    def _parse_reply(self, data: bytes) -> List[str]:
        lines = [line for line in data.decode().split(self._BR) if len(line.strip()) > 0]
        return lines

    def _read_and_parse(self, tail: str) -> List[str]:
        if self._tn is None:
            raise AssertionError
        return self._parse_reply(self._tn.read_until(tail.encode(), timeout=self._timeout))

    def _read_until_prompt_and_parse(self) -> List[str]:
        rep = self._read_and_parse(self._BR + self._PROMPT)
        return rep

    def open(self) -> bool:
        self._tn = telnetlib.Telnet(host=self._ipaddr, port=self._port, timeout=self._timeout)
        for _ in range(3):
            rv = self._open()
            if rv:
                logger.info(f"connection is established with {self._ipaddr}")
                return rv
            else:
                self._tn.write(self._BR.encode())
        else:
            logger.info(f"failed to established connection with {self._ipaddr}")
            self._tn = None
            return False

    def _open(self) -> bool:
        if self._tn is None:
            raise AssertionError
        rep = self._read_and_parse("Login: ")
        if len(rep) == 0 or rep[-1] != "Login: ":
            return False
        self._tn.write((self._username + self._BR).encode())
        rep = self._read_and_parse("Password: ")
        if len(rep) == 0 or rep[-1] != "Password: ":
            return False
        self._tn.write(self._password.encode() + b"\r\n")
        rep = self._read_until_prompt_and_parse()
        if len(rep) == 0 or rep[-1] != self._PROMPT:
            return False
        return True

    def _exec_cmd_auto_open(self, cmd: str) -> List[str]:
        if self._tn is None:
            self.open()
        else:
            try:
                return self._exec_cmd(cmd)
            except BrokenPipeError:
                self.open()
        return self._exec_cmd(cmd)

    def _exec_cmd(self, cmd: str) -> List[str]:
        if self._tn is None:
            raise AssertionError
        self._tn.write((cmd + self._BR).encode())
        rep = self._read_until_prompt_and_parse()
        if len(rep) == 0:
            raise RuntimeError("no response")
        elif rep[-1] != "> ":
            raise RuntimeError("truncated output, no prompt is detected")
        else:
            return rep[:-1]

    def _validate_switch_index(self, idx: int) -> None:
        if not 1 <= idx <= 8:
            raise ValueError("invalid index of switch: {idx}")

    def check_switch(self, idx: int) -> PeSwitchState:
        self._validate_switch_index(idx)
        cmd = f"read status o{idx:02d} simple"
        reply = self._exec_cmd_auto_open(cmd)
        if len(reply) != 2 or reply[0] != cmd or reply[1] not in {"on", "off", "pending"}:
            msg = "/".join(reply)
            raise RuntimeError(f"failed to read status of switch, unexpected reply '{msg}' is received")
        return PeSwitchState(reply[1])

    def is_turned_on(self, idx: int) -> bool:
        return self.check_switch(idx) == PeSwitchState.ON

    def is_turned_off(self, idx: int) -> bool:
        return self.check_switch(idx) == PeSwitchState.OFF

    def turn_switch(self, idx: int, status: PeSwitchState, no_switch_ok: bool = False) -> None:
        self._validate_switch_index(idx)
        if status not in {PeSwitchState.ON, PeSwitchState.OFF}:
            raise ValueError(f"invalid swtich state: '{status.value}'")
        if self.check_switch(idx) == status:
            if not no_switch_ok:
                logger.warning(f"switch {idx} is already {status.value}, no thing happens")
            return

        cmd = f"sw o{idx:02d} imme {status.value}"
        reply = self._exec_cmd(cmd)
        if len(reply) != 2 or reply[1] != f" Outlet<{idx:02d}> command is setting ":
            msg = "/".join(reply)
            raise RuntimeError(f"failed to change the status of switch: unexpected reply '{msg}' is received")

        t0 = time.perf_counter()
        for _ in range(10):
            time.sleep(1)
            if self.check_switch(idx) == status:
                break
        else:
            raise RuntimeError(
                f"failed to turn {status.value} the switch for {int(time.perf_counter()-t0)} seconds, something wrong!"
            )
        logger.info(f"switch {self._ipaddr}:{idx} is turned {status.value}")
        return

    def turn_switch_on(self, idx: int, no_switch_ok=True):
        self.turn_switch(idx, PeSwitchState.ON, no_switch_ok)

    def turn_switch_off(self, idx: int, no_switch_ok=True):
        self.turn_switch(idx, PeSwitchState.OFF, no_switch_ok)

    def powercycle_switch(
        self, idx: int, off_duration: float = _DEFAULT_OFF_DURATION, no_switch_ok: bool = False
    ) -> None:
        self._validate_switch_index(idx)
        if off_duration < self._MINIMUM_OFF_DURATION:
            raise ValueError(f"too short off-time: {off_duration} seconds")

        self.turn_switch(idx, PeSwitchState.OFF, no_switch_ok)
        time.sleep(off_duration)
        self.turn_switch(idx, PeSwitchState.ON)
        return
