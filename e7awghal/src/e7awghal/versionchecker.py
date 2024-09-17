import logging
from ipaddress import IPv4Address
from typing import Final

import numpy as np
import ping3

from e7awghal.common_register import E7awgVersion
from e7awghal.e7awg_packet import E7awgOutgoingPacket, E7awgPacketAccess, E7awgPacketMode
from e7awghal.fwtype import E7FwAuxAttr, E7FwLifeStage, E7FwType

logger = logging.getLogger(__name__)


_VERSION_TO_FWTYPE: Final[dict[tuple[str, str], tuple[E7FwType, set[E7FwAuxAttr], E7FwLifeStage]]] = {
    ("a:2024/05/13-1", "a:2024/05/13-2"): (
        E7FwType.SIMPLEMULTI_STANDARD,  # TODO: update this!!
        {E7FwAuxAttr.BROKEN_AWG_RESET, E7FwAuxAttr.DSP_v0b},
        E7FwLifeStage.SUPPORTING,
    ),
    ("a:2024/01/25-1", "a:2024/01/25-2"): (
        E7FwType.SIMPLEMULTI_STANDARD,
        {E7FwAuxAttr.BROKEN_AWG_RESET, E7FwAuxAttr.DSP_v0b},
        E7FwLifeStage.SUPPORTING,
    ),
    ("K:2024/01/10-1", "K:2024/01/10-2"): (
        E7FwType.FEEDBACK_EARLY,
        {E7FwAuxAttr.BROKEN_AWG_RESET, E7FwAuxAttr.DSP_v1a},
        E7FwLifeStage.SUSPENDED,
    ),
    ("K:2023/12/28-1", "K:2023/12/28-2"): (
        E7FwType.SIMPLEMULTI_CLASSIC,
        {E7FwAuxAttr.BROKEN_AWG_RESET, E7FwAuxAttr.DSP_v0b},
        E7FwLifeStage.DEPRECATED,
    ),
    ("K:2023/12/16-1", "K:2023/12/16-2"): (
        E7FwType.SIMPLEMULTI_CLASSIC,
        {E7FwAuxAttr.BROKEN_AWG_RESET, E7FwAuxAttr.DSP_v0b},
        E7FwLifeStage.DEPRECATED,
    ),
    ("K:2023/11/08-1", "K:2023/11/08-1"): (
        E7FwType.FEEDBACK_EARLY,
        {E7FwAuxAttr.BROKEN_AWG_RESET, E7FwAuxAttr.IRREGULAR_ADC_FNCO, E7FwAuxAttr.DSP_v0b},
        E7FwLifeStage.DEPRECATED,
    ),
    # for 20230222, 20230422
    ("K:2023/02/22-1", "K:2023/02/22-2"): (
        E7FwType.SIMPLEMULTI_CLASSIC,
        {E7FwAuxAttr.BROKEN_AWG_RESET, E7FwAuxAttr.DSP_v0b},
        E7FwLifeStage.DEPRECATED,
    ),
    # for 20221023, 20230820, and 20230929
    # - 20221023: the oldestf firmware to support. it has overflow bug in DSP module.
    # - 20230820: supporting SYSREF latch (broken version identifiers, under investigation.)
    # - 20230928: sysref timing is modified (broken version identifiers, under investigation.)
    ("K:2022/07/14-1", "K:2022/07/14-2"): (
        E7FwType.SIMPLEMULTI_CLASSIC,
        {E7FwAuxAttr.BROKEN_AWG_RESET, E7FwAuxAttr.DSP_v0a},
        E7FwLifeStage.DEPRECATED,
    ),
}


class Quel1Au50HalVersionChecker:
    def __init__(self, ipaddr: str, port: int):
        self._ipaddr = IPv4Address(ipaddr)
        self._udprw = E7awgPacketAccess(str(self._ipaddr), port)

    def ping(self):
        for _ in range(5):
            if ping3.ping(str(self._ipaddr), timeout=1) is not None:
                return True
        else:
            logger.error(f"no ping response from {str(self._ipaddr)}")
            return False

    def _read_reg(self, mode: E7awgPacketMode, address: int) -> np.uint32:
        cmd = E7awgOutgoingPacket(
            mode=mode,
            address=address,
            num_payload_bytes=4,
        )
        rpl = self._udprw.send_command(cmd)
        val = np.frombuffer(rpl.payload, dtype=np.dtype("<u4"))[0]
        logger.debug(f"{self.__class__.__name__}:_read_reg({address:08x}) --> {val:08x}")
        return val

    def read_awgctrl_version(self) -> str:
        v = self._read_reg(E7awgPacketMode.AWG_REG_READ, 0)
        w = E7awgVersion.parse(v)
        return f"{chr(w.ver_char)}:20{w.ver_year:02d}/{w.ver_month:02d}/{w.ver_day:02d}-{w.ver_id:d}"

    def read_capctrl_version(self) -> str:
        v = self._read_reg(E7awgPacketMode.CAPTURE_REG_READ, 0)
        w = E7awgVersion.parse(v)
        return f"{chr(w.ver_char)}:20{w.ver_year:02d}/{w.ver_month:02d}/{w.ver_day:02d}-{w.ver_id:d}"

    def resolve_fwtype(self) -> tuple[E7FwType, set[E7FwAuxAttr], E7FwLifeStage]:
        av = self.read_awgctrl_version()
        cv = self.read_capctrl_version()
        k = _VERSION_TO_FWTYPE.get((av, cv))
        if k is not None:
            if k[2] == E7FwLifeStage.DEPRECATED:
                raise RuntimeError(f"deprecate firmware (version = {av}, {cv}), not supported any more")
            elif k[2] == E7FwLifeStage.TO_DEPRECATE:
                logger.warning(f"the installed firmware (version = {av}, {cv}) is about to be deprecated")
            elif k[2] == E7FwLifeStage.SUSPENDED:
                raise RuntimeError(f"the installed firmware (version = {av}, {cv}) is temporarily out of support")
            logger.info(f"the installed firmware is resolved as {str(k[0])}")
            return k
        else:
            raise RuntimeError(
                f"unknown firmware version (= {av}, {cv}). updating the e7awghal package may resolve this issue"
            )
