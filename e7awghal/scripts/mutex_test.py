import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np

from e7awghal import (
    AbstractCapCtrl,
    AbstractCapUnit,
    AbstractQuel1Au50Hal,
    AwgCtrl,
    AwgParam,
    AwgUnit,
    CapParam,
    CapSection,
    ClockcounterCtrl,
    WaveChunk,
    create_quel1au50hal,
)
from e7awghal.hbmctrl import HbmCtrl  # HbmCtrl is not published via __init__.py

logger = logging.getLogger()


def au_access(au: AwgUnit, postfix: int, clkcntr: Union[ClockcounterCtrl, None]):
    for i in range(10000):
        if i % 1000 == 999:
            cntr = clkcntr.read_counter()[0] if clkcntr else -1
            logger.info(f"au-#{au.unit_index} thread-#{postfix}: {i}  (cntr: {cntr})")
        if au.is_busy():
            logger.info(".")
        wp = AwgParam()
        wp.chunks.append(WaveChunk(name_of_wavedata="null"))
        au.load_parameter(wp)


def ac_access(ac: AwgCtrl, postfix: int, clkcntr: Union[ClockcounterCtrl, None]):
    for i in range(20000):
        if i % 1000 == 999:
            cntr = clkcntr.read_counter()[0] if clkcntr else -1
            logger.info(f"ac thread-#{postfix}: {i}  (cntr: {cntr})")
        if ac.are_busy_any({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}):
            logger.info(".")


def cu_access(cu: AbstractCapUnit, postfix: int, clkcntr: Union[ClockcounterCtrl, None]):
    for i in range(10000):
        if i % 1000 == 999:
            cntr = clkcntr.read_counter()[0] if clkcntr else -1
            logger.info(f"cu-#{cu.unit_index} thread-#{postfix}: {i}  (cntr: {cntr})")
        if cu.is_busy():
            logger.info(".")
        cp = CapParam()
        cp.sections.append(CapSection(num_capture_word=64))
        cu.load_parameter(cp)


def cc_access(cc: AbstractCapCtrl, clkcntr: Union[ClockcounterCtrl, None]):
    for i in range(20000):
        if i % 1000 == 999:
            cntr = clkcntr.read_counter()[0] if clkcntr else -1
            logger.info(f"cc: {i}  (cntr: {cntr})")
        if cc.are_busy_any({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}):
            logger.info(".")
        if cc.are_done_any({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}):
            logger.info(".")


def hbm_access(hc: HbmCtrl, addr, size, clkcntr: Union[ClockcounterCtrl, None]):
    hc = proxy.hbmctrl

    size_aligned = ((size + 359) // 360 + 1) * 360

    data_iq = np.array([[i, i * 2] for i in range(size)], dtype=np.int16)
    zero_iq = np.array([[0, 0] for _ in range(size_aligned)], dtype=np.int16)

    for k in range(10000):
        if k % 1000 == 999:
            cntr = clkcntr.read_counter()[0] if clkcntr else -1
            logger.info(f"hbm-{addr:09x}: {k}  (cntr: {cntr})")
        hc.write_iq32(addr, size_aligned, zero_iq)  # Notes: clear !
        d0 = hc.read_iq32(addr, size)
        assert (d0 == 0).all()
        hc.write_iq32(addr, size, data_iq)
        d1 = hc.read_iq32(addr, size)
        assert (d1 == data_iq).all()


def clkcntr_access(clkcntr: ClockcounterCtrl, postfix: int):
    cntr0 = clkcntr.read_counter()[0]
    for k in range(500000):
        if k % 10000 == 9999:
            logger.info(f"clkcntr thread-#{postfix}: {k}")
        cntr1 = clkcntr.read_counter()[0]
        assert cntr1 > cntr0
        cntr0 = cntr1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    proxy: AbstractQuel1Au50Hal = create_quel1au50hal(ipaddr_wss="10.1.0.58")
    proxy.initialize()
    ac = proxy.awgctrl
    cc = proxy.capctrl
    pool = ThreadPoolExecutor()
    clkcntr = proxy.clkcntr

    logger.info("=================== au0, au0, au1, ac, ac, cu0, cu0, cc, hbm, hbm")
    ress = []
    ress.append(pool.submit(au_access, proxy.awgunit(0), 0, clkcntr))
    ress.append(pool.submit(au_access, proxy.awgunit(0), 1, clkcntr))
    ress.append(pool.submit(au_access, proxy.awgunit(1), 0, clkcntr))
    ress.append(pool.submit(ac_access, ac, 0, clkcntr))
    ress.append(pool.submit(ac_access, ac, 1, clkcntr))
    ress.append(pool.submit(cu_access, proxy.capunit(0), 0, clkcntr))
    ress.append(pool.submit(cu_access, proxy.capunit(0), 1, clkcntr))
    ress.append(pool.submit(hbm_access, proxy.hbmctrl, 0x0_0000_2000, 1233, clkcntr))
    ress.append(pool.submit(cc_access, cc, clkcntr))
    ress.append(pool.submit(hbm_access, proxy.hbmctrl, 0x0_0001_2000, 671, clkcntr))
    ress.append(pool.submit(clkcntr_access, clkcntr, 0))

    for res in ress:
        res.result()

    ac.check_error(ac.units)
    cc.check_error(cc.units)
