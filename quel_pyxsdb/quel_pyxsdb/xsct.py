import argparse
import logging
import os
import subprocess
import sys
import telnetlib
import time
from typing import Dict, Final, List, Mapping, Tuple, Union

from quel_pyxsdb.hw_server import HwServer

logger = logging.getLogger()

# Notes: see service.examples directory for details
DEFAULT_MAPPING_BETWEEN_XSDB_AND_HWSVR: Mapping[int, int] = {
    33335: 3121,  # the default port combination
    34335: 4121,
    35335: 5121,
    36335: 6121,
}


class XsctServer:
    DEFAULT_XSDB_PORT: Final[int] = 33335

    def __init__(self, topdir: str, port: int):
        self.xsct = self.open_xsct(topdir, port)

    def __del__(self):
        if hasattr(self, "xsct"):
            if self.xsct.stdin is not None:
                self.xsct.stdin.close()
                code = self.xsct.wait()
                logger.info(f"xsct exits gracefully with code:{code}")
            else:
                self.xsct.kill()
                logger.info("xsct is killed")

    @staticmethod
    def open_xsct(vivado_path, port) -> subprocess.Popen:
        p = subprocess.Popen(
            [
                f"{vivado_path}/bin/loader",
                "-exec",
                "rdi_xsdb",
                "-eval",
                f"xsdbserver start -port {port}",
                "-interactive",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return p

    def wait(self):
        return self.xsct.wait()

    def dump_out(self):
        if self.xsct.stdout is not None:
            stdout = self.xsct.stdout.read()
        else:
            stdout = ""

        if self.xsct.stderr is not None:
            stderr = self.xsct.stderr.read()
        else:
            stderr = ""

        return stdout, stderr

    def get_pid(self) -> int:
        return self.xsct.pid


class XsctClient:
    def __init__(self, *, host: str = "localhost", xsdb_port: int, hwsvr_port: int):
        # Notes: assuming that both the servers, Xsdb and Hwsvr, runs on the same host.
        #        this restriction doesn't lose
        self._xsdb_port = xsdb_port
        self._hwsvr_port = hwsvr_port
        self.term = self.open_session(host, xsdb_port)

    @staticmethod
    def open_session(host: str, port: int) -> telnetlib.Telnet:
        for _ in range(5):
            try:
                s = telnetlib.Telnet(host, port, timeout=5)
                return s
            except ConnectionRefusedError:
                time.sleep(1)
        else:
            raise RuntimeError(f"failed to connect to xsct at {host}:{port}")

    def exec(self, cmd: str) -> Tuple[bool, List[str]]:
        self.term.write(cmd.encode("ascii"))
        self.term.write(b"\n")
        rep = self.term.read_until(b"\n").strip().decode("ascii")
        if rep[0:4] == "okay":
            return True, rep[5:].split("\\n") if len(rep) >= 5 else [""]
        elif rep[0:5] == "error":
            return False, rep[6:].split("\\n") if len(rep) >= 6 else [""]
        else:
            raise RuntimeError(f"unexpected output: '{rep}'")

    def connect(self, port: Union[int, None] = None) -> bool:
        if port is None:
            port = self._hwsvr_port

        rc, msg = self.exec(f"conn -port {port}")
        if not rc:
            logger.warning(f"'conn' is failed due with a message '{msg}'")
        return rc

    def _parse_targets(self, prop):
        if prop[0] == "{":
            prop = prop[1:]
        if prop[-1] == "}":
            prop = prop[:-1]

        d: Dict[str, str] = {}
        key: str = ""
        val: List[str] = []
        nest = 0
        for p in prop.split():
            if key == "":
                key = p
            else:
                if p[0] == "{":
                    p = p[1:]
                    nest += 1
                if p[-1] == "}":
                    p = p[:-1]
                    nest -= 1
                val.append(p)
                if nest == 0:
                    d[key] = " ".join(val)
                    key = ""
                    val = []
        return d

    def _get_jtagterminal_by_index(self, idx: int) -> int:
        rc, msg = self.exec(f"targets {idx:d}")
        if not rc:
            raise RuntimeError(f"'targets {idx:d}' is failed with a message '{msg}'")
        rc, msg = self.exec("jtagterminal -sock")
        if not rc:
            raise RuntimeError(f"'jtagterminal -sock' is failed with a message '{msg}'")
        return int(msg[0])

    def get_jtagterminal_by_adapter_id(
        self, adapter_id: str, module_name: str = "MicroBlaze Debug Module at USER2"
    ) -> int:
        rc, msg = self.exec("targets -target-properties")
        if not rc:
            raise RuntimeError("'targets -target-properties' is failed due with a message '{msg}'")
        if len(msg) != 1:
            raise RuntimeError(f"unexpected output of 'targets -target-properties': {msg}")

        for idx, prop in enumerate(msg[0].split("} {")):
            t = self._parse_targets(prop)
            if (
                "jtag_cable_serial" in t
                and t["jtag_cable_serial"] == adapter_id
                and "name" in t
                and t["name"] == module_name
                and "target_id" in t
            ):
                return self._get_jtagterminal_by_index(int(t["target_id"]))

        raise ValueError(f"no adapter {adapter_id} found.")


def get_jtagterminal_port(adapter_id: str, *, host: str = "localhost", xsdb_port: int, hwsvr_port: int) -> int:
    clt = XsctClient(host=host, xsdb_port=xsdb_port, hwsvr_port=hwsvr_port)
    term_port = clt.get_jtagterminal_by_adapter_id(adapter_id)
    del clt
    return term_port


def main_server() -> None:
    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="starting xsct server")
    parser.add_argument("--xsdb_port", type=int, required=True, help="a port of the xsdb server to launch")
    parser.add_argument("--hwsvr_port", type=int, required=True, help="a port of the hwserver to launch")
    parser.add_argument(
        "--target_type", choices=tuple(HwServer.TARGET_DICT.keys()), default="", help="type of targets to grab"
    )
    parser.add_argument(
        "--vivado_topdir", type=str, default=os.getenv("XILINX_VIVADO", ""), help="top directory of the vivado to use"
    )
    args = parser.parse_args()

    if args.vivado_topdir == "":
        raise ValueError(
            "no path to vivado is specified, "
            "it may be convenient for you to source settings64.sh in one of your VIVADO directory"
        )

    hwsvr = HwServer(topdir=args.vivado_topdir, port=args.hwsvr_port, target_type=args.target_type)
    svr = XsctServer(topdir=args.vivado_topdir, port=args.xsdb_port)
    logger.info(f"hit Ctrl+C to stop the server (pid={svr.get_pid()})...")

    clt = XsctClient(host="localhost", xsdb_port=args.xsdb_port, hwsvr_port=args.hwsvr_port)
    for _ in range(5):
        time.sleep(0.3)
        if clt.connect():
            break
    else:
        logger.error(f"failed to connect to hw_server@{hwsvr.port}")
        sys.exit(1)
    del clt

    try:
        rv = svr.wait()
        logger.warning(f"xsct quits with code={rv}")
        o, e = svr.dump_out()
        logger.warning(o)
        logger.warning(e)
    except KeyboardInterrupt:
        pass


def main_jtaglist() -> None:
    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="connecting xsct server to hw server")
    parser.add_argument("--host", type=str, default="localhost", help="the host of the xsct server")
    parser.add_argument(
        "--xsdb_port", type=int, default=XsctServer.DEFAULT_XSDB_PORT, help="the port of the xsdb server"
    )
    parser.add_argument("--hwsvr_port", type=int, default=-1, help="the port of the hw server")

    args = parser.parse_args()
    if args.hwsvr_port < 0:
        if args.xsdb_port in DEFAULT_MAPPING_BETWEEN_XSDB_AND_HWSVR:
            args.hwsvr_port = DEFAULT_MAPPING_BETWEEN_XSDB_AND_HWSVR[args.xsdb_port]
        else:
            logger.error("port to hw_server is required to work with your special Xsdb server")
            sys.exit(1)

    clt = XsctClient(host=args.host, xsdb_port=args.xsdb_port, hwsvr_port=args.hwsvr_port)
    try:
        rc, msg = clt.exec("jtag target")
        if not rc:
            raise RuntimeError(f"'jtag target' is failed with a message '{msg}'")
        for line in msg:
            logger.info(line)
    except RuntimeError as e:
        logger.error(e)


def main_jtagterminal() -> None:
    logging.basicConfig(format="%(asctime)s %(name)-8s %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="connecting xsct server to hw server")
    parser.add_argument("--host", type=str, default="localhost", help="the host of the xsct server")
    parser.add_argument(
        "--xsdb_port", type=int, default=XsctServer.DEFAULT_XSDB_PORT, help="the port of the xsct server"
    )
    parser.add_argument("--hwsvr_port", type=int, default=-1, help="the port of the hw server")
    parser.add_argument(
        "--adapter", type=str, required=True, help="ID of the adapter attached to the device to connect"
    )

    args = parser.parse_args()
    if args.hwsvr_port < 0:
        if args.xsdb_port in DEFAULT_MAPPING_BETWEEN_XSDB_AND_HWSVR:
            args.hwsvr_port = DEFAULT_MAPPING_BETWEEN_XSDB_AND_HWSVR[args.xsdb_port]
        else:
            logger.error("port to hw_server is required to work with your special Xsdb server")
            sys.exit(1)
    print(
        get_jtagterminal_port(
            adapter_id=args.adapter, host=args.host, xsdb_port=args.xsdb_port, hwsvr_port=args.hwsvr_port
        )
    )
