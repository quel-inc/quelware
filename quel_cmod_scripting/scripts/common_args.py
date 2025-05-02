import argparse
import logging
from typing import cast

from pyxsct import XsctClient

from quel_cmod_scripting import Quel1SeProtoCmod, QuelCmodAbstract

logger = logging.getLogger()


def add_common_arguments(parser: argparse.ArgumentParser, cls: type) -> None:
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="the name of host where xsct is running",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="the number of the port where xsct exports the CMOD console",
    )
    parser.add_argument(
        "--jtag", type=str, default="", help="the id of the jtag adapter of the CMOD"
    )
    parser.add_argument(
        "--noinit", action="store_true", help="skip to initialize the CMOD if true"
    )
    if cls == Quel1SeProtoCmod:
        parser.add_argument(
            "--mixerboard",
            type=str,
            default="0,1",
            help="comma-separated list of indices of available mixer boards",
        )


def open_cmod(args: argparse.Namespace, cls: type) -> QuelCmodAbstract:
    port = args.port
    if port == 0:
        if args.jtag != "":
            clt = XsctClient(host=args.host)
            clt.connect()
            port = clt.get_jtagterminal_by_adapter_id(args.jtag)
            logger.info(f"xsct exports the console to port {port}")
        else:
            raise RuntimeError(
                "no information is available to determine the CMOD to connect"
            )
    else:
        if args.jtag != "":
            logger.warning("the given jtag ID is ignored")

    if not issubclass(cls, QuelCmodAbstract):
        raise ValueError(
            "invalid class to instantiate, not a subclass of QuelCmodAbstract"
        )

    if cls is Quel1SeProtoCmod:
        cmod = cast(
            QuelCmodAbstract,
            Quel1SeProtoCmod(
                args.host, port, mixerboard={int(b) for b in args.mixerboard.split(",")}
            ),
        )
    else:
        cmod = cls(args.host, port)

    if not args.noinit:
        cmod.init()
        logger.info("cmod is initialized")

    return cmod
