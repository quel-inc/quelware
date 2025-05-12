import logging
import os
import random
import subprocess
import tempfile
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Set, Union

import psutil

from quel_staging_tool.run_vivado_batch import run_vivado_batch

logger = logging.getLogger(__name__)


class PrivateHwserver:
    def __init__(self, adapter_id: str, adapter_typeid: str, timeout: int = 90):
        self._adapter_id: str = adapter_id
        self._adapter_typeid: str = adapter_typeid
        self._timeout: int = timeout
        self._port: int = 0
        self.process: Union[subprocess.Popen, None] = None

    def _start(self):
        for _ in range(3):
            used_ports: Set[int] = {
                conn.laddr.port
                for conn in psutil.net_connections()
                if conn.status == "LISTEN" and hasattr(conn, "laddr") and isinstance(conn.laddr, psutil._common.addr)
            }
            port_range = 65530 - 49152
            port0 = random.randint(0, port_range)
            for i in range(10000):
                port = 49152 + ((port0 + i) % port_range)
                if port not in used_ports:
                    break
            else:
                raise RuntimeError("No available port")

            self._port = port
            # Notes: "-I" is introduced because I cannot find a good way to kill all the child processes
            #        spawned by hw_server.
            cmd = [
                "hw_server",
                "-e",
                f"set jtag-port-filter {self._adapter_typeid}{self._adapter_id}",
                "-I",
                f"{self._timeout}",
                "-s",
                f"tcp::{self._port}",
            ]
            logger.info(f"executing {' '.join(cmd)}")
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"pid = {self.process.pid}")
            time.sleep(5)
            if self.process.poll() is None:
                break

            # Notes: may be port crash due to simultaneous spawning of hw_servers
            if self.process.stderr is not None:
                logger.warning(f"failed to start hw_server due to {self.process.stderr.read().decode()}, retrying...")
            else:
                logger.warning("failed to start hw_server, retrying...")

        else:
            raise RuntimeError("repeated failure of starting hw_server, quitting.")

    @property
    def port(self):
        if self.process is None or self.process.poll() is not None:
            self._start()
        return self._port


class QuelXilinxFpgaProgrammer(metaclass=ABCMeta):
    _BITPREFIX: str = "nonexistent_"
    _TCLCMD_POSTFIX: str = "_nonexistent.tcl"
    _UPDATEMEM_PATH = "{env_xilinx_vitis}/bin/updatemem"

    def __init__(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def _bitdir_path(self) -> Path:
        return get_valid_firmware_directory()

    def _tcldir_path(self) -> Path:
        return Path(os.path.abspath(os.path.dirname(__file__))) / "tcl_2020"

    def _validate_env(self) -> None:
        env_xilinx_vitis = os.getenv("XILINX_VITIS", None)
        if env_xilinx_vitis is None:
            raise RuntimeError("load settings64.sh of the right version of Vitis in advance")

    def get_bits(self, bitdir_path: Union[Path, None] = None, bitfile_name="top.bit") -> Dict[str, Path]:
        if bitdir_path is None:
            bitdir_path = self._bitdir_path()
        dirs = {str(p) for p in os.listdir(bitdir_path) if str(p).startswith(self._BITPREFIX)}
        bitlist = {p[len(self._BITPREFIX) :]: bitdir_path / p / bitfile_name for p in dirs}  # noqa: E203
        bitlist = {k: fn for k, fn in bitlist.items() if os.path.exists(fn)}
        return bitlist

    @abstractmethod
    def make_embedded_bit(self, bitpath: Path, **parameters) -> Path:
        pass

    def make_mcs(self, bitpath: Path, mcspath: Union[Path, None] = None) -> Path:
        self._validate_env()
        if mcspath is None:
            mcspath = Path(os.path.splitext(bitpath)[0] + ".mcs")
        retval = run_vivado_batch(self._tcldir_path(), f"create_mcs{self._TCLCMD_POSTFIX}", f"{bitpath} {mcspath}")
        if retval == 0:
            return mcspath
        else:
            raise AssertionError("not reached")

    @abstractmethod
    def make_bin(self, bitpath: Path, binpath: Union[Path, None] = None) -> Path:
        pass

    def make_mcs_with_macaddr(self, bitpath: Path, macaddrpath: Path, mcspath: Union[Path, None] = None) -> Path:
        self._validate_env()
        if mcspath is None:
            mcspath = Path(os.path.splitext(bitpath)[0] + ".mcs")
        retval = run_vivado_batch(
            self._tcldir_path(),
            f"create_mcs_with_macaddr{self._TCLCMD_POSTFIX}",
            f"{bitpath} {macaddrpath} {mcspath}",
        )
        if retval == 0:
            return mcspath
        else:
            raise AssertionError("not reached")

    def program(self, mcspath: Path, host: str, port: int, adapter_id: str, adapter_typeid: str = "") -> int:
        self._validate_env()
        if host == "localhost" and port == 0 and adapter_typeid != "":
            hwserver = PrivateHwserver(adapter_id, adapter_typeid)
            port = hwserver.port

        return run_vivado_batch(
            self._tcldir_path(), f"program_mcs{self._TCLCMD_POSTFIX}", f"{mcspath} {host}:{port} {adapter_id}"
        )

    def program_bit(self, bitpath: Path, host: str, port: int, adapter_id: str, adapter_typeid: str = "") -> int:
        self._validate_env()
        if host == "localhost" and port == 0 and adapter_typeid != "":
            hwserver = PrivateHwserver(adapter_id, adapter_typeid)
            port = hwserver.port

        return run_vivado_batch(
            self._tcldir_path(), f"program_bit{self._TCLCMD_POSTFIX}", f"{bitpath} {host}:{port} {adapter_id}"
        )

    def reboot(self, host: str, port: int, adapter_id: str, adapter_typeid: str = "") -> int:
        self._validate_env()
        if host == "localhost" and port == 0 and adapter_typeid != "":
            hwserver = PrivateHwserver(adapter_id, adapter_typeid)
            port = hwserver.port

        return run_vivado_batch(self._tcldir_path(), "reboot.tcl", f"{host}:{port} {adapter_id}")

    def dry_run(self, host: str, port: int, adapter_id: str, adapter_typeid: str = "") -> int:
        self._validate_env()
        if host == "localhost" and port == 0 and adapter_typeid != "":
            hwserver = PrivateHwserver(adapter_id, adapter_typeid)
            port = hwserver.port

        return run_vivado_batch(self._tcldir_path(), "dry_run.tcl", f"{host}:{port} {adapter_id}")


def get_valid_firmware_directory() -> Path:
    """
    Checks for the existence of the firmware directory, "$XDG_DATA_HOME/quelware/firmwares".
    Returns the valid firmware directory path if found.
    """
    if "XDG_DATA_HOME" in os.environ:
        firmware_dir = Path(os.path.join(os.environ["XDG_DATA_HOME"], "quelware", "firmwares", "plain_bits"))
    else:
        firmware_dir = Path(os.path.join(os.path.expanduser("~"), ".local", "share", "quelware", "firmwares", "plain_bits"))

    if not firmware_dir.exists():
        raise FileNotFoundError(f"Firmware directory not found. Please ensure firmware have been installed.")

    return firmware_dir
