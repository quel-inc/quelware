import logging
import time

import pytest

from quel_ic_config import Quel1seTempctrlState
from quel_ic_config.exstickge_proxy import LsiKindId
from testlibs.create_css_proxy import ProxyType, create_proxy

# Notes: these are test cases for confirming the validity of SOFTWARE, but not of firmware!
# Notes: some test cases disrupt the current link state.

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


DEVICE_SETTINGS = (
    {
        "label": "staging-094",
        "config": {
            "ipaddr_css": "10.5.0.94",
        },
    },
)


def _idle_tempctrl(proxy: ProxyType):
    proxy.write_tempctrl_state(Quel1seTempctrlState.IDLE)
    logger.info("tempctrl state is set to idle")
    for i in range(15):
        if proxy.read_tempctrl_state() == Quel1seTempctrlState.IDLE:
            break
        time.sleep(1)
    else:
        raise RuntimeError("failed to enter idle mode")


def _init_tempctrl(proxy: ProxyType):
    proxy.write_tempctrl_state(Quel1seTempctrlState.INIT)
    logger.info("tempctrl state is set to init")


@pytest.fixture(scope="module", params=DEVICE_SETTINGS)
def fixtures_local(request):
    param0 = request.param

    proxy = create_proxy(param0["config"]["ipaddr_css"])
    _idle_tempctrl(proxy)
    yield proxy
    _init_tempctrl(proxy)


def _check_lmx2594(proxy: ProxyType, idx: int, num_iter: int = 10000) -> bool:
    proxy.write_reg(LsiKindId.LMX2594, idx, 0x0000, 0x2612)
    proxy.write_reg(LsiKindId.LMX2594, idx, 0x0000, 0x2610)
    cnt_error = 0
    for i in range(num_iter):
        v = proxy.read_reg(LsiKindId.LMX2594, idx, 0x0000)
        if v is None:
            cnt_error += 1
            logger.info(f"LMX2594[{idx}]: read failure")
        elif v != 0x2610:
            cnt_error += 1
            logger.info(f"LMX2594[{idx}]: unexpected reading: {v:04x} (!= 0x2610)")

    if cnt_error == 0:
        logger.info(f"LMX2594[{idx}]: no error / {num_iter} access")
    else:
        logger.error(f"LMX2594[{idx}]: {cnt_error} errors!! / {num_iter} access")

    return cnt_error == 0


def _check_adrf6780(proxy: ProxyType, idx: int, num_iter: int = 10000) -> bool:
    if proxy.read_reset(LsiKindId.ADRF6780, idx) != 1:
        proxy.write_reset(LsiKindId.ADRF6780, idx, 1)
        logger.info(f"ARDF6780[{idx}]: releasing reset")

    cnt_error = 0
    for i in range(num_iter):
        v = proxy.read_reg(LsiKindId.ADRF6780, idx, 0x0000)
        if v is None:
            cnt_error += 1
            logger.info(f"ADRF6780[{idx}]: read failure")
        elif v != 0x0076:
            cnt_error += 1
            logger.info(f"ADRF6780[{idx}]: unexpected reading: {v:04x} (!= 0x0x0076)")

    if cnt_error == 0:
        logger.info(f"ADRF6780[{idx}]: no error / {num_iter} access")
    else:
        logger.error(f"ADRF6780[{idx}]: {cnt_error} errors!! / {num_iter} access")

    return cnt_error == 0


def _check_ad9082(proxy: ProxyType, idx: int, num_iter: int = 5000) -> bool:
    proxy.write_reset(LsiKindId.AD9082, idx, 0)
    proxy.write_reset(LsiKindId.AD9082, idx, 1)
    time.sleep(0.01)
    proxy.write_reg(LsiKindId.AD9082, idx, 0x0000, 0x81)
    proxy.write_reg(LsiKindId.AD9082, idx, 0x0000, 0x00)
    proxy.write_reg(LsiKindId.AD9082, idx, 0x0000, 0x3C)

    cnt_error = 0
    for i in range(num_iter):
        v3 = proxy.read_reg(LsiKindId.AD9082, idx, 0x0003)
        v4 = proxy.read_reg(LsiKindId.AD9082, idx, 0x0004)
        v5 = proxy.read_reg(LsiKindId.AD9082, idx, 0x0005)
        v6 = proxy.read_reg(LsiKindId.AD9082, idx, 0x0006)

        v = [v3, v4, v5, v6]
        exp = [0x0F, 0x82, 0x90, 0x23]
        for j, w in enumerate(v):
            if w is None:
                cnt_error += 1
                logger.info(f"AD9082[{idx}][{j+3}]: read failure")
            elif w != exp[j]:
                cnt_error += 1
                logger.info(f"AD9082[{idx}][{j+3}]: unexpected reading: 0x{w:02x} (!= {exp[j]})")

    if cnt_error == 0:
        logger.info(f"AD9082[{idx}]: no error / {num_iter * 4} access")
    else:
        logger.error(f"AD9082[{idx}]: {cnt_error} errors!! / {num_iter * 4} access")

    return cnt_error == 0


def _check_pwm(proxy: ProxyType, idx: int, num_iter: int = 10000) -> bool:
    cnt_error = 0
    for i in range(num_iter):
        v1 = proxy.read_reg(LsiKindId.POWERBOARD_PWM, idx, 0x000C)
        if v1 is None:
            cnt_error += 1
            logger.info(f"PWM[{idx}]: read failure")
        elif v1 != 0x0532:
            cnt_error += 1
            logger.info(f"PWM[{idx}]: unexpected reading: {v1:04x} (!= 0x0x532)")

        v2 = proxy.read_reg(LsiKindId.POWERBOARD_PWM, idx, 0x0006)
        if v2 is None:
            cnt_error += 1
            logger.info(f"PWM[{idx}]: read failure")
        elif v2 != 0x00FA:
            cnt_error += 1
            logger.info(f"PWM[{idx}]: unexpected reading: {v2:04x} (!= 0x0x00FA)")

    if cnt_error == 0:
        logger.info(f"PWM[{idx}]: no error / {num_iter} access")
    else:
        logger.error(f"PWM[{idx}]: {cnt_error} errors!! / {num_iter} access")

    return cnt_error == 0


def _check_ad7490(proxy: ProxyType, idx: int, num_iter: int = 1000) -> bool:
    if proxy.read_tempctrl_state() != Quel1seTempctrlState.IDLE:
        raise RuntimeError("you need to deactivate tempctrl (with idle_temp_ctrl(), for example).")

    cnt_error = 0
    for i in range(num_iter):
        proxy.write_reg(LsiKindId.AD7490, idx, 0, 0xFFB0)
        for j in range(16):
            v = proxy.read_reg(LsiKindId.AD7490, idx, 0)
            if v is None:
                cnt_error += 1
                logger.warning(f"AD7490[{idx}] access failure")
            elif (v >> 12) & 0x0F != j:
                cnt_error += 1
                logger.warning(f"AD7490[{idx}] broken data {v:04x} (!= {j:x}XXX)")

    if cnt_error == 0:
        logger.info(f"AD7490[{idx}]: no error / {num_iter * 16} access")
    else:
        logger.error(f"AD7490[{idx}]: {cnt_error} errors!! / {num_iter * 16} access")

    return cnt_error == 0


def test_ad9082(fixtures_local):
    proxy = fixtures_local
    for idx in range(2):
        assert _check_ad9082(proxy, idx, 2)


def test_lmx2594(fixtures_local):
    proxy = fixtures_local
    for idx in range(5):
        assert _check_lmx2594(proxy, idx, 5)


def test_adrf6780(fixtures_local):
    proxy = fixtures_local
    for idx in range(2):
        assert _check_adrf6780(proxy, idx, 5)


def test_pwm(fixtures_local):
    proxy = fixtures_local
    assert _check_pwm(proxy, 0, 5)


def test_ad7490(fixtures_local):
    proxy = fixtures_local
    for idx in range(8):
        assert _check_ad7490(proxy, idx, 2)
