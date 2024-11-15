import logging

import pytest

from e7awghal.quel1au50_hal import AbstractQuel1Au50Hal, create_quel1au50hal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")


def test_nonexistent():
    with pytest.raises(RuntimeError):
        _: AbstractQuel1Au50Hal = create_quel1au50hal(ipaddr_wss="10.255.254.253", auth_callback=lambda: True)
