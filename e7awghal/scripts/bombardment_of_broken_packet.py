import logging
from enum import Enum
from typing import Union

from e7awghal import AbstractQuel1Au50Hal, create_quel1au50hal
from e7awghal.e7awg_packet import E7awgIncomingPacket, E7awgOutgoingPacket

logger = logging.getLogger(__name__)


class XE7awgPacketMode(Enum):
    # (mode_id, readable_mode_label, has_payload, expect_payload_in_reply, expected_modes_of_the_reply)
    BROKEN_CMD = (0xAA, "BROKEN PACKET", False, False, (0xFF,))
    UNDEFINED = (0xFF, "UNDEFINED", False, False, None)

    def __init__(
        self,
        mode_id: int,
        readable_mode_label: str,
        has_payload: bool,
        expects_payload_in_reply: bool,
        expected_modes_of_reply: Union[tuple[int, ...], None],
    ):
        self._mode_id: int = mode_id
        self._readable_mode_label: str = readable_mode_label
        self._has_payload: bool = has_payload
        self._expects_payload_in_reply: bool = expects_payload_in_reply
        self._expected_modes_of_reply: Union[tuple[int, ...], None] = expected_modes_of_reply

    def __repr__(self):
        return f"{self.__class__.__name__}:{self.readable_mode_label}"

    @property
    def mode_id(self) -> int:
        return self._mode_id

    @property
    def readable_mode_label(self) -> str:
        return self._readable_mode_label

    @property
    def has_payload(self) -> bool:
        return self._has_payload

    @property
    def expects_payload_in_reply(self) -> bool:
        return self._expects_payload_in_reply

    def is_expected_reply(self, replied_mode: "XE7awgPacketMode"):
        return (self._expected_modes_of_reply is not None) and (replied_mode.mode_id in self._expected_modes_of_reply)

    @classmethod
    def from_int(cls, v: int) -> "XE7awgPacketMode":
        for k, u in cls.__members__.items():
            if u.mode_id == v:
                return u
        else:
            raise ValueError(f"invalid mode_id for {cls.__name__}")


def send_broken_cmd(ctrl) -> E7awgIncomingPacket:
    cmd = E7awgOutgoingPacket(
        mode=XE7awgPacketMode.BROKEN_CMD,  # type: ignore
        address=0,
        num_payload_bytes=0,
    )
    rpl = ctrl._udprw.send_command(cmd)
    return rpl


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    proxy: AbstractQuel1Au50Hal = create_quel1au50hal(ipaddr_wss="10.1.0.58")
    proxy.initialize()
    ac = proxy.awgctrl
    hc = proxy.hbmctrl

    for i in range(100000):
        if i % 1000 == 999:
            logger.info(f"sending {i} packets")
        # send_broken_cmd(ac)
        send_broken_cmd(hc)
