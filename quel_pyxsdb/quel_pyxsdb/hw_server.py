import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class HwServer:
    TARGET_DICT = {
        "cmod-a7-35t": ("Digilent", "Cmod A7 - 35T"),
        "cmod": ("Digilent", "Cmod A7 - 35T"),  # alias of cmod-a7-35t
        "hs2": ("Digilent", "JTAG-HS2"),
        "exstickge": ("Digilent", "JTAG-HS2"),  # alias of hs2
        "dmbv1": ("Xilinx", "Alveo-DMBv1 FT4232H"),
        "au50": ("Xilinx", "Alveo-DMBv1 FT4232H"),  # alias of dmbv1
        "au200": ("Xilinx", "A-U200-A64G FT4232H"),
        "": ("", ""),
    }

    def __init__(self, topdir: Path, port: int, target_type: str = "", adapter_id: str = ""):
        self.topdir = topdir
        self.port = port
        self.target_type = target_type
        self.jtag_id = adapter_id
        self.hwsvr = self.open_hwsvr()

    def __del__(self):
        if hasattr(self, "hwsvr"):
            self.hwsvr.kill()
            logger.info(f"hw_server@{self.port} is killed")

    def open_hwsvr(self) -> subprocess.Popen:
        cmd = [f"{self.topdir}/bin/hw_server"]
        if self.target_type != "":
            pattern = "/".join(self.TARGET_DICT[self.target_type])
            if self.jtag_id != "":
                pattern = pattern + f"/{self.jtag_id}"
            cmd.extend(["-e", f"set jtag-port-filter {pattern}"])
        cmd.extend(["-s", f"tcp::{self.port}"])
        cmd.append("-p0")
        logger.info(f"invoking '{' '.join(cmd)}'")
        hwsvr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1.0)
        rv = hwsvr.poll()
        if rv is not None:
            if hwsvr.stderr is not None:
                raise RuntimeError(hwsvr.stderr.read().decode().strip())
            else:
                raise RuntimeError("hw_server exits unexpectedly.")
        return hwsvr

    def wait(self):
        return self.hwsvr.wait()

    def dump_out(self):
        if self.hwsvr.stdout is not None:
            stdout = self.hwsvr.stdout.read()
        else:
            stdout = ""

        if self.hwsvr.stderr is not None:
            stderr = self.hwsvr.stderr.read()
        else:
            stderr = ""

        return stdout, stderr

    def get_pid(self) -> int:
        return self.hwsvr.pid


def main_hw_server() -> None:
    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="starting xsct server")
    parser.add_argument("--hwsvr_port", type=int, requird=True, help="a port of the hwserver to connect")
    parser.add_argument(
        "--target_type", choices=tuple(HwServer.TARGET_DICT.keys()), default="", help="type of targets to grab"
    )
    parser.add_argument("--adapter_id", type=str, default="", help="Id of the adapter to grab")
    parser.add_argument(
        "--vivado_topdir", type=str, default=os.getenv("XILINX_VIVADO", ""), help="top directory of vivado to use"
    )
    args = parser.parse_args()

    if args.vivado_topdir == "":
        raise ValueError(
            "no path to vivado is specified, "
            "it may be convenient for you to source settings64.sh in one of your VIVADO directory"
        )

    hwsvr = HwServer(
        topdir=args.vivado_topdir, port=args.hwsvr_port, target_type=args.target_type, adapter_id=args.adapter_id
    )
    logger.info(f"hit Ctrl+C to stop the server (pid={hwsvr.get_pid()})...")

    try:
        rv = hwsvr.wait()
        logger.warning(f"hw_server quits with code={rv}")
        o, e = hwsvr.dump_out()
        logger.warning(o)
        logger.warning(e)
    except KeyboardInterrupt:
        pass
