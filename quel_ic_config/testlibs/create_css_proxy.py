import logging
from typing import Union, cast

from quel_ic_config import (
    ExstickgeCoapClientQuel1seFujitsu11,
    ExstickgeCoapClientQuel1seRiken8,
    ExstickgeCoapClientQuel1seRiken8Dev1,
    ExstickgeCoapClientQuel1seRiken8Dev2,
    get_exstickge_server_info,
)

logger = logging.getLogger(__name__)


ProxyType = Union[
    ExstickgeCoapClientQuel1seRiken8,
    ExstickgeCoapClientQuel1seRiken8Dev1,
    ExstickgeCoapClientQuel1seRiken8Dev2,
    ExstickgeCoapClientQuel1seFujitsu11,
]


def create_proxy(ipaddr_css: str) -> ProxyType:
    is_coap_firmware, coap_firmware_version, boxtype = get_exstickge_server_info(ipaddr_css)

    if is_coap_firmware:
        if boxtype == "":
            raise RuntimeError("no boxtype of the CoAP firmware is available")

        for proxy_cls in (
            ExstickgeCoapClientQuel1seRiken8Dev1,
            ExstickgeCoapClientQuel1seRiken8Dev2,
            ExstickgeCoapClientQuel1seRiken8,
            ExstickgeCoapClientQuel1seFujitsu11,
        ):
            if proxy_cls.matches(boxtype, coap_firmware_version):
                return cast(ProxyType, proxy_cls(ipaddr_css))
        else:
            raise RuntimeError(f"unsupported CoAP firmware is running on exstickge: {coap_firmware_version}")
    else:
        raise RuntimeError("no CoAP firmware is running on exstickge")
