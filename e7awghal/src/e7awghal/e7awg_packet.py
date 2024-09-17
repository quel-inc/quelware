import logging
import socket
import struct
from abc import ABCMeta
from enum import Enum
from threading import Lock
from typing import Callable, Final, Union

import numpy as np

logger = logging.getLogger(__name__)


class DeviceLockDelegationError(Exception):
    pass


class _BasePacket(metaclass=ABCMeta):
    _HEADER_SIZE: int = 0

    __slots__ = (
        "_buffer",
        "_mode",
        "_address",
        "_num_payload_bytes",
        "_valid",
    )

    def __init__(self):
        self._valid: bool = True
        self._buffer: Union[bytes, bytearray]

    @property
    def payload(self) -> memoryview:
        return memoryview(self._buffer)[self._HEADER_SIZE :]

    @property
    def is_valid(self) -> bool:
        return self._valid


class BasePacketAccess(metaclass=ABCMeta):
    _DEFAULT_TIMEOUT: Final[float] = 2.0  # sec

    def __init__(self, ip_addr: str, port: int, *, timeout: float = _DEFAULT_TIMEOUT):
        super().__init__()
        # networking
        self._dest_addrport: tuple[str, int] = (ip_addr, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout)
        self._sock.bind((_get_my_ip_addr(ip_addr), 0))  # TODO:
        self._auth_callback: Callable[[], bool] = self._default_auth_callback

        # control
        self._is_little_endian_cpu = _is_little_endian_cpu()
        self._lock = Lock()

    def _validate_sender(self, addrport):
        # TODO: implement it!
        return True

    def inject_auth_callback(self, auth_callback: Union[Callable[[], bool], None]):
        self._auth_callback = auth_callback or self._default_auth_callback

    def _default_auth_callback(self) -> bool:
        return False


class E7awgPacketMode(Enum):
    # (mode_id, readable_mode_label, has_payload, expect_payload_in_reply, expected_modes_of_the_reply)
    WAVE_RAM_READ = (0x00, "WAVE RAM READ", False, True, (0x01,))
    WAVE_RAM_READ_REPLY = (0x01, "WAVE RAM READ-REPLY", True, False, None)
    WAVE_RAM_WRITE = (0x02, "WAVE RAM WRITE", True, False, (0x03,))
    WAVE_RAM_WRITE_ACK = (0x03, "WAVE RAM WRITE-ACK", False, False, None)

    AWG_REG_READ = (0x10, "AWG REG READ", False, True, (0x11,))
    AWG_REG_READ_REPLY = (0x11, "AWG REG READ-REPLY", True, False, None)
    AWG_REG_WRITE = (0x12, "AWG REG WRITE", True, False, (0x13,))
    AWG_REG_WRITE_ACK = (0x13, "AWG REG WRITE-ACK", False, False, None)

    CAPTURE_REG_READ = (0x40, "CAP REG READ", False, True, (0x41,))
    CAPTURE_REG_READ_REPLY = (0x41, "CAP REG READ-REPLY", True, False, None)
    CAPTURE_REG_WRITE = (0x42, "CAP REG WRITE", True, False, (0x43,))
    CAPTURE_REG_WRITE_ACK = (0x43, "CAP REG WRITE-ACK", False, False, None)

    SIMPLEMULTI_ADD_AWG_START = (0x22, "SIMPLEMULTI ADD AWG START", True, False, (0x23,))
    SIMPLEMULTI_ADD_AWG_START_ACK = (0x23, "SIMPLEMULTI ADD AWG START-ACK", False, False, None)
    SIMPLEMULTI_CMD_CANCEL = (0x24, "SIMPLEMULTI CMD CANCEL", False, False, (0x25,))
    SIMPLEMULTI_CMD_CANCEL_ACK = (0x25, "SIMPLEMULTI CMD CANCEL-ACK", False, False, None)
    SIMPLEMULTI_CMD_CANCEL_AND_TERMINATE = (0x2C, "SIMPLEMULTI CMD CANCEL-AND-TERMINATE", False, False, (0x2D,))
    SIMPLEMULTI_CMD_CANCEL_AND_TERMINATE_ACK = (0x2D, "SIMPLEMULTI CMD CANCEL-AND-TERMINATE-ACK", False, False, None)

    REBOOTER_FPGA_REBOOT = (0xE0, "REBOOTER REBOOT FPGA", False, False, (0xE0,))

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
    def has_side_effect(self) -> bool:
        return self in {
            self.WAVE_RAM_WRITE,
            self.AWG_REG_WRITE,
            self.CAPTURE_REG_WRITE,
            self.SIMPLEMULTI_ADD_AWG_START,
            self.SIMPLEMULTI_ADD_AWG_START,
            self.SIMPLEMULTI_CMD_CANCEL,
            self.SIMPLEMULTI_CMD_CANCEL_AND_TERMINATE,
            self.REBOOTER_FPGA_REBOOT,
        }

    @property
    def readable_mode_label(self) -> str:
        return self._readable_mode_label

    @property
    def has_payload(self) -> bool:
        return self._has_payload

    @property
    def expects_payload_in_reply(self) -> bool:
        return self._expects_payload_in_reply

    def is_expected_reply(self, replied_mode: "E7awgPacketMode"):
        return (self._expected_modes_of_reply is not None) and (replied_mode.mode_id in self._expected_modes_of_reply)

    @classmethod
    def from_int(cls, v: int) -> "E7awgPacketMode":
        for k, u in cls.__members__.items():
            if u.mode_id == v:
                return u
        else:
            raise ValueError(f"invalid mode_id for {cls.__name__}")


class E7awgAbstractPacket(_BasePacket):
    _HEADER_SIZE: int = 8
    # Notes: assuming that a packet is sent over UDP on NIC whose MTU is 1500.
    _DEFAULT_MAX_PAYLOAD_SIZE: int = 1500 - 28 - _HEADER_SIZE

    def __init__(self):
        super().__init__()
        self._mode: E7awgPacketMode

    @property
    def mode(self) -> E7awgPacketMode:
        return self._mode


class E7awgIncomingPacket(E7awgAbstractPacket):
    def __init__(self, buffer: bytes):
        super().__init__()
        self._buffer: memoryview = memoryview(buffer)
        m = int.from_bytes(self._buffer[0:1], "big")

        try:
            self._mode = E7awgPacketMode.from_int(m)
        except ValueError:
            logger.warning("a packet with an unexpected mode 0x{m:02x} is received")
            self._mode = E7awgPacketMode.UNDEFINED
            self._valid = False

        self._address = int.from_bytes(self._buffer[1:6], "big")
        self._num_payload_bytes = int.from_bytes(self._buffer[6:8], "big")

        if self._mode.has_payload:
            if len(self._buffer) != self._num_payload_bytes + self._HEADER_SIZE:
                logger.warning(
                    f"a packet with broken header information is received, "
                    f"actual packted length(= {len(self._buffer)}) is not consistent with "
                    f"the header information ({self._HEADER_SIZE} + {self._num_payload_bytes}"
                )
                self._valid = False
        else:
            if len(self._buffer) != self._HEADER_SIZE:
                logger.warning(
                    f"a packet with unexpected length ({len(self._buffer)} != {self._HEADER_SIZE}) is received"
                )

    @property
    def buffer(self) -> memoryview:
        return self._buffer

    def is_valid_for(self, req_mode: E7awgPacketMode):
        return self.is_valid and req_mode.is_expected_reply(self._mode)


class E7awgOutgoingPacket(E7awgAbstractPacket):
    def __init__(self, mode: E7awgPacketMode, address: int, num_payload_bytes: int):
        super().__init__()
        if address < 0:
            raise ValueError("negative address is not allowed")
        if mode.has_payload or mode.expects_payload_in_reply:
            if not 0 < num_payload_bytes <= self._DEFAULT_MAX_PAYLOAD_SIZE:
                raise ValueError(f"num_payload_bytes must be positive number for packet {mode}")
        else:
            if num_payload_bytes != 0:
                raise ValueError(f"num_payload_bytes must be zero {mode}")

        self._mode = mode
        self._address = address
        self._num_payload_bytes = num_payload_bytes
        if self._mode.has_payload:
            self._buffer: bytearray = bytearray(8 + num_payload_bytes)
        else:
            self._buffer = bytearray(8)
        self._buffer[0:1] = self._mode.mode_id.to_bytes(1, "big")
        self._buffer[1:6] = self._address.to_bytes(5, "big")
        self._buffer[6:8] = self._num_payload_bytes.to_bytes(2, "big")

    @property
    def has_side_effect(self) -> bool:
        return self._mode.has_side_effect

    @property
    def buffer(self) -> bytearray:
        return self._buffer


class E7awgPacketAccess(BasePacketAccess):
    _DEFAULT_MAX_PACKET_SIZE: Final[int] = E7awgAbstractPacket._DEFAULT_MAX_PAYLOAD_SIZE + _BasePacket._HEADER_SIZE

    def __init__(self, ip_addr: str, port: int, *, timeout: float = BasePacketAccess._DEFAULT_TIMEOUT):
        super().__init__(ip_addr, port, timeout=timeout)

    def send_command(self, pkt: E7awgOutgoingPacket, expect_no_reply: bool = False) -> E7awgIncomingPacket:
        try:
            with self._lock:
                # Notes: all types of the packets are subject to access controls. (not only packet with side effect due
                #        to wrong design of e7awghw)
                if not self._auth_callback():
                    raise DeviceLockDelegationError("no device lock is available")
                self._sock.sendto(pkt._buffer, self._dest_addrport)
                while True:
                    # TODO: timeout doesn't work well if many bogus packet come periodically.
                    # Notes: raise socket.timeout if recvfrom() timeouts.
                    recv_data, recv_addr = self._sock.recvfrom(self._DEFAULT_MAX_PACKET_SIZE)
                    if self._validate_sender(recv_addr):
                        break
            rpl = E7awgIncomingPacket(recv_data)
            if not rpl.is_valid_for(pkt.mode):
                raise ValueError(f"unexpected reply: mode = {pkt.mode} -> {rpl.mode} from {recv_addr}")
        except socket.timeout as e:
            if not expect_no_reply:
                logger.error(e)
            raise
        except Exception as e:
            logger.error(e)
            raise

        return rpl


class E7awgSimple32PacketMode(Enum):
    # (mode_id, readable_mode_label, has_payload, network_byteorder, expected_modes_of_the_reply)
    CLK_CNTR_READ = (0x04, "CLOCK COUNTER READ", True, (0x05,))
    CLK_CNTR_READ_ACK = (0x05, "CLOCK COUNTER READ-ACK", True, None)

    UNDEFINED = (0xFF, "UNDEFINED", True, None)

    def __init__(
        self,
        mode_id: int,
        readable_mode_label: str,
        is_network_endian: bool,
        expected_modes_of_reply: Union[tuple[int, ...], None],
    ):
        self._mode_id: int = mode_id
        self._readable_mode_label: str = readable_mode_label
        self._is_network_endian: bool = is_network_endian
        self._expected_modes_of_reply: Union[tuple[int, ...], None] = expected_modes_of_reply

    def __repr__(self):
        return f"{self.__class__.__name__}:{self.readable_mode_label}"

    @property
    def mode_id(self) -> int:
        return self._mode_id

    @property
    def has_side_effect(self) -> bool:
        return False

    @property
    def readable_mode_label(self) -> str:
        return self._readable_mode_label

    def is_expected_reply(self, replied_mode: "E7awgSimple32PacketMode"):
        return (self._expected_modes_of_reply is not None) and (replied_mode.mode_id in self._expected_modes_of_reply)

    @property
    def is_network_endian(self) -> bool:
        return self._is_network_endian

    @classmethod
    def from_int(cls, v: int) -> "E7awgSimple32PacketMode":
        for k, u in cls.__members__.items():
            if u.mode_id == v:
                return u
        else:
            raise ValueError(f"invalid mode_id for {cls.__name__}")


class E7awgSimple32AbstractPacket(_BasePacket):
    _HEADER_SIZE: int = 4
    # Notes: assuming that a packet is sent over UDP on NIC whose MTU is 1500.
    _DEFAULT_MAX_PAYLOAD_SIZE: int = 1500 - 28 - _HEADER_SIZE

    def __init__(self):
        super().__init__()
        self._mode: E7awgSimple32PacketMode

    @property
    def mode(self) -> E7awgSimple32PacketMode:
        return self._mode


class E7awgSimple32IncomingPacket(E7awgSimple32AbstractPacket):
    def __init__(self, buffer: bytes, is_network_endian: bool):
        super().__init__()
        self._buffer: memoryview = memoryview(buffer)

        if is_network_endian:
            (m,) = struct.unpack(">xxxB", self._buffer[0:4])
        else:
            (m,) = struct.unpack("<Bxxx", self._buffer[0:4])

        try:
            self._mode = E7awgSimple32PacketMode.from_int(m)
        except ValueError:
            logger.warning("a packet with an unexpected mode 0x{m:02x} is received")
            self._mode = E7awgSimple32PacketMode.UNDEFINED
            self._valid = False

    @property
    def buffer(self) -> memoryview:
        return self._buffer

    def is_valid_for(self, req_mode: E7awgSimple32PacketMode):
        return self.is_valid and req_mode.is_expected_reply(self._mode)


class E7awgSimple32OutgoingPacket(E7awgSimple32AbstractPacket):
    def __init__(self, mode: E7awgSimple32PacketMode):
        super().__init__()

        self._mode = mode
        self._buffer: bytearray = bytearray(4)

        if self._mode.is_network_endian:
            self._buffer[0:4] = struct.pack(">xxxB", self._mode.mode_id)
        else:
            self._buffer[0:4] = struct.pack("<Bxxx", self._mode.mode_id)

    @property
    def has_side_effect(self) -> bool:
        return self.mode.has_side_effect

    @property
    def buffer(self) -> bytearray:
        return self._buffer


class E7awgSimple32PacketAccess(BasePacketAccess):
    _DEFAULT_MAX_PACKET_SIZE: Final[int] = (
        E7awgSimple32AbstractPacket._DEFAULT_MAX_PAYLOAD_SIZE + _BasePacket._HEADER_SIZE
    )

    def __init__(self, ip_addr: str, port: int, *, timeout: float = BasePacketAccess._DEFAULT_TIMEOUT):
        super().__init__(ip_addr, port, timeout=timeout)

    def send_command(self, pkt: E7awgSimple32OutgoingPacket, is_network_endian: bool) -> E7awgSimple32IncomingPacket:
        try:
            with self._lock:
                if pkt.has_side_effect:
                    if not self._auth_callback():
                        raise DeviceLockDelegationError("no device lock is available")
                self._sock.sendto(pkt._buffer, self._dest_addrport)
                # TODO: implement timeout (!)
                while True:
                    recv_data, recv_addr = self._sock.recvfrom(self._DEFAULT_MAX_PACKET_SIZE)
                    if self._validate_sender(recv_addr):
                        break
            rpl = E7awgSimple32IncomingPacket(recv_data, is_network_endian)
            if not rpl.is_valid_for(pkt.mode):
                raise ValueError(f"unexpected reply: mode = {pkt.mode} -> {rpl.mode} from {recv_addr}")
        except socket.timeout as e:
            logger.error(e)
            raise
        except Exception as e:
            logger.error(e)
            raise

        return rpl


class E7awgSimple64PacketMode(Enum):
    # (mode_id, readable_mode_label, expected_modes_of_the_reply)
    MCLK_CNTR_READ = (0x30, "CLOCKMASTER READ CLOCK", (0x33,))
    MCLK_SYNC_KICK = (0x32, "CLOCKMASTER KICK SYNC", (0x33,))
    CLOCKMASTER_CMD_ACK = (0x33, "CLOCKMASTER CMD ACK", None)
    CLOCKMASTER_REBOOT = (0x7E, "CLOCKMASTER REBOOT", (0x7E,))  # Notes: any word is OK!

    UNDEFINED = (0xFF, "UNDEFINED", None)

    def __init__(
        self,
        mode_id: int,
        readable_mode_label: str,
        expected_modes_of_reply: Union[tuple[int, ...], None],
    ):
        self._mode_id: int = mode_id
        self._readable_mode_label: str = readable_mode_label
        self._expected_modes_of_reply: Union[tuple[int, ...], None] = expected_modes_of_reply

    def __repr__(self):
        return f"{self.__class__.__name__}:{self.readable_mode_label}"

    @property
    def mode_id(self) -> int:
        return self._mode_id

    @property
    def has_side_effect(self) -> bool:
        return self in {self.MCLK_SYNC_KICK, self.CLOCKMASTER_REBOOT}

    @property
    def readable_mode_label(self) -> str:
        return self._readable_mode_label

    def is_expected_reply(self, replied_mode: "E7awgSimple64PacketMode"):
        return (self._expected_modes_of_reply is not None) and (replied_mode.mode_id in self._expected_modes_of_reply)

    @classmethod
    def from_int(cls, v: int) -> "E7awgSimple64PacketMode":
        for k, u in cls.__members__.items():
            if u.mode_id == v:
                return u
        else:
            raise ValueError(f"invalid mode_id for {cls.__name__}")


class E7awgSimple64AbstractPacket(_BasePacket):
    _HEADER_SIZE: int = 8
    # Notes: assuming that a packet is sent over UDP on NIC whose MTU is 1500.
    _DEFAULT_MAX_PAYLOAD_SIZE: int = 1500 - 28 - _HEADER_SIZE

    def __init__(self):
        super().__init__()
        self._mode: E7awgSimple64PacketMode

    @property
    def mode(self) -> E7awgSimple64PacketMode:
        return self._mode


class E7awgSimple64IncomingPacket(E7awgSimple64AbstractPacket):
    def __init__(self, buffer: bytes):
        super().__init__()
        self._buffer: memoryview = memoryview(buffer)

        (m,) = struct.unpack("<Bxxxxxxx", self._buffer[0:8])

        try:
            self._mode = E7awgSimple64PacketMode.from_int(m)
        except ValueError:
            logger.warning("a packet with an unexpected mode 0x{m:02x} is received")
            self._mode = E7awgSimple64PacketMode.UNDEFINED
            self._valid = False

    @property
    def buffer(self) -> memoryview:
        return self._buffer

    def is_valid_for(self, req_mode: E7awgSimple64PacketMode):
        return self.is_valid and req_mode.is_expected_reply(self._mode)


class E7awgSimple64OutgoingPacket(E7awgSimple64AbstractPacket):
    def __init__(self, mode: E7awgSimple64PacketMode, num_payload_bytes: int = 0):
        super().__init__()

        self._mode = mode
        self._num_payload_bytes = num_payload_bytes
        self._buffer: bytearray = bytearray(8 + num_payload_bytes)
        self._buffer[0:8] = struct.pack("<Bxxxxxxx", self._mode.mode_id)

    @property
    def has_side_effect(self) -> bool:
        return self.mode.has_side_effect

    @property
    def buffer(self) -> bytearray:
        return self._buffer


class E7awgSimple64PacketAccess(BasePacketAccess):
    _DEFAULT_MAX_PACKET_SIZE: Final[int] = (
        E7awgSimple64AbstractPacket._DEFAULT_MAX_PAYLOAD_SIZE + _BasePacket._HEADER_SIZE
    )

    def __init__(self, ip_addr: str, port: int, *, timeout: float = BasePacketAccess._DEFAULT_TIMEOUT):
        super().__init__(ip_addr, port, timeout=timeout)

    def send_command(self, pkt: E7awgSimple64OutgoingPacket) -> E7awgSimple64IncomingPacket:
        try:
            with self._lock:
                if pkt.has_side_effect:
                    if not self._auth_callback():
                        raise DeviceLockDelegationError("no device lock is available")
                self._sock.sendto(pkt._buffer, self._dest_addrport)
                # TODO: implement timeout (!)
                while True:
                    recv_data, recv_addr = self._sock.recvfrom(self._DEFAULT_MAX_PACKET_SIZE)
                    if self._validate_sender(recv_addr):
                        break
            rpl = E7awgSimple64IncomingPacket(recv_data)
            if not rpl.is_valid_for(pkt.mode):
                raise ValueError(f"unexpected reply: mode = {pkt.mode} -> {rpl.mode} from {recv_addr}")
        except socket.timeout as e:
            logger.error(e)
            raise
        except Exception as e:
            logger.error(e)
            raise

        return rpl


def _is_little_endian_cpu():
    native = 0x000094E1
    native_bytes = np.array([native], dtype=np.uint32).tobytes()

    little = np.frombuffer(native_bytes, dtype="<u4")[0]
    if native == little:
        logger.debug("the endianness of the CPU looks little")
        return True

    big = np.frombuffer(native_bytes, dtype=">u4")[0]
    if native == big:
        logger.debug("the endianness of the CPU looks big")
        return False

    logger.warning("the endianness of the CPU is unknown...")
    return False


def _get_my_ip_addr(ip_addr):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip_addr, 0))
    my_ip_addr = sock.getsockname()[0]
    sock.close()
    return my_ip_addr


if __name__ == "__main__":
    from e7awghal.common_register import E7awgVersion

    rw = E7awgPacketAccess("10.1.0.74", 16385)
    rd_ver = E7awgOutgoingPacket(mode=E7awgPacketMode.AWG_REG_READ, address=0x00000000, num_payload_bytes=4)
    rpl = rw.send_command(rd_ver)
    v = E7awgVersion.parse(np.frombuffer(rpl.payload, dtype=np.dtype("<u4"))[0])
