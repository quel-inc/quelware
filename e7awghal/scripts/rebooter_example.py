import logging
import time

from e7awghal import AbstractQuel1Au50Hal, create_quel1au50hal
from quel_ic_config.quel1_box_intrinsic import _create_css_object
from quel_ic_config.quel_config_common import Quel1BoxType

logger = logging.getLogger()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    boxtype = Quel1BoxType.QuEL1_TypeA

    css = _create_css_object("10.5.0.58", boxtype)
    css.initialize()

    for mxfe_idx in css.get_all_mxfes():
        css.configure_mxfe(mxfe_idx, {})

    proxy: AbstractQuel1Au50Hal = create_quel1au50hal(ipaddr_wss="10.1.0.58", auth_callback=lambda: True)
    proxy.initialize()
    rebooter = proxy.au50rebooter

    for mxfe in css.get_all_mxfes():
        logger.info(f"linkstatus of mxfe-#{mxfe}: {css.get_link_status(mxfe)}")

    rebooter.reboot()
    time.sleep(5)

    for mxfe in css.get_all_mxfes():
        logger.info(f"linkstatus of mxfe-#{mxfe}: {css.get_link_status(mxfe)}")
