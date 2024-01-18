import logging
import os
import subprocess
import tempfile
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Final, Tuple, Union

logger = logging.getLogger(__name__)


def run_vivado_batch(script_dir: Path, script_file: str, tclargs: str) -> int:
    with tempfile.TemporaryDirectory() as dname:
        cmd = (
            f"vivado -mode batch -nolog -nojournal -notrace -tempDir {dname} "
            f"-source {script_dir / script_file} -tclargs {tclargs}"
        )
        logger.info(f"executing {cmd}")
        retcode = subprocess.run(cmd.split(), capture_output=True)
        for msg in retcode.stdout.decode().split("\n"):
            if msg.startswith("INFO:"):
                logger.debug(msg[5:].strip())  # Notes: shown only when --verbose option is provided in cli commands.
            elif msg.startswith("XINFO:"):
                logger.info(msg[6:].strip())
            elif msg.startswith("ERROR:"):
                logger.error(msg[6:].strip())
        if retcode.returncode != 0:
            raise RuntimeError(f"failed execution of {script_file}")
        return retcode.returncode


class QuelXilinxFpgaProgrammer(metaclass=ABCMeta):
    _BITPREFIX: str = "nonexistent_"
    _TCLCMD_POSTFIX: str = "_nonexistent.tcl"
    _UPDATEMEM_PATH = "{env_xilinx_vitis}/bin/updatemem"

    def __init__(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    @staticmethod
    def encode_macaddr(macaddr: str) -> Tuple[str, str]:
        b = bytearray([int(o, 16) for o in macaddr.split("-")])
        if len(b) != 6:
            raise ValueError(f"malformed macaddress: '{macaddr}'")
        b.append(0)
        b.append(0)
        w0 = b[0:4]
        w1 = b[4:8]
        w0.reverse()
        w1.reverse()
        return w0.hex(), w1.hex()

    @staticmethod
    def encode_ipaddr(ipaddr: str) -> str:
        b = bytearray([int(o, 10) for o in ipaddr.split(".")])
        if len(b) != 4:
            raise ValueError(f"malformed IP aaddress: '{ipaddr}'")
        b.reverse()
        return b.hex()

    def _bitdir_path(self) -> Path:
        return Path(os.path.abspath(os.path.dirname(__file__))) / "plain_bits"

    def _tcldir_path(self) -> Path:
        return Path(os.path.abspath(os.path.dirname(__file__))) / "tcl_2020"

    def _validate_env(self) -> None:
        env_xilinx_vitis = os.getenv("XILINX_VITIS", None)
        if env_xilinx_vitis is None:
            raise RuntimeError("load settings64.sh of the right version of Vitis in advance")

    def get_bits(self, bitdir_path: Union[Path, None] = None) -> Dict[str, Path]:
        if bitdir_path is None:
            bitdir_path = self._bitdir_path()
        dirs = {str(p) for p in os.listdir(bitdir_path) if str(p).startswith(self._BITPREFIX)}
        bitlist = {p[len(self._BITPREFIX) :]: bitdir_path / p / "top.bit" for p in dirs}  # noqa: E203
        return bitlist

    @abstractmethod
    def make_mem(self, **params: str) -> Path:
        pass

    def make_embedded_bit(self, bitpath: Path, mempath: Path, mmipath: Union[Path, None] = None) -> Path:
        self._validate_env()
        if mmipath is None:
            mmipath = Path(os.path.dirname(bitpath)) / "ram_loc.mmi"
        outpath = Path(self._tmpdir.name) / (os.path.splitext(mempath)[0] + ".bit")
        retcode = subprocess.run(
            f"updatemem -force -meminfo {mmipath} -data {mempath} -bit {bitpath} -proc dummy -out {outpath}".split(),
            capture_output=True,
        )
        if retcode.returncode != 0:
            raise RuntimeError("failed execution of updatemem")
        return outpath

    def make_mcs(self, bitpath: Path, mcspath: Union[Path, None] = None) -> Path:
        self._validate_env()
        if mcspath is None:
            mcspath = Path(os.path.splitext(bitpath)[0] + ".mcs")
        retval = run_vivado_batch(self._tcldir_path(), f"create_mcs{self._TCLCMD_POSTFIX}", f"{bitpath} {mcspath}")
        if retval == 0:
            return mcspath
        else:
            raise AssertionError("not reached")

    def program(self, mcspath: Path, host: str, port: int, adapter_id: str) -> int:
        self._validate_env()
        return run_vivado_batch(
            self._tcldir_path(), f"program_mcs{self._TCLCMD_POSTFIX}", f"{mcspath} {host}:{port} {adapter_id}"
        )

    def program_bit(self, bitpath: Path, host: str, port: int, adapter_id: str) -> int:
        self._validate_env()
        return run_vivado_batch(
            self._tcldir_path(), f"program_bit{self._TCLCMD_POSTFIX}", f"{bitpath} {host}:{port} {adapter_id}"
        )

    def reboot(self, host: str, port: int, adapter_id: str) -> int:
        self._validate_env()
        return run_vivado_batch(self._tcldir_path(), "reboot.tcl", f"{host}:{port} {adapter_id}")

    def dry_run(self, host: str, port: int, adapter_id: str) -> int:
        self._validate_env()
        return run_vivado_batch(self._tcldir_path(), "dry_run.tcl", f"{host}:{port} {adapter_id}")


class ExstickgeProgrammer(QuelXilinxFpgaProgrammer):
    _TEMPLATE = """\
@00000000
{ipaddr:s}
{netmask:s}
{default_gateway:s}
{target_server:s}
{macaddr_3210:s}
{macaddr_zz54:s}
00000000
00000000
00000000
00000000
00000000
00000000
00000000
00000000
00000000
00000000
"""

    _BITPREFIX: str = "exstickge_"
    _TCLCMD_POSTFIX: str = "_exstickge.tcl"

    def __init__(self):
        super().__init__()

    def make_mem(self, **params: str) -> Path:
        macaddr: str = params["macaddr"]
        ipaddr: str = params["ipaddr"]
        netmask: str = params["netmask"]
        default_gateway: str = params["default_gateway"]
        target_server: Union[str, None] = params.get("target_server")
        if target_server is None:
            target_server = default_gateway

        macaddr_3210, macaddr_zz54 = self.encode_macaddr(macaddr)

        mem = self._TEMPLATE.format(
            ipaddr=self.encode_ipaddr(ipaddr),
            netmask=self.encode_ipaddr(netmask),
            default_gateway=self.encode_ipaddr(default_gateway),
            target_server=self.encode_ipaddr(target_server),
            macaddr_3210=macaddr_3210,
            macaddr_zz54=macaddr_zz54,
        )

        filename = Path(self._tmpdir.name) / f"{ipaddr}.mem"
        with open(filename, "w") as f:
            f.write(mem)

        return filename


class Au50Programmer(QuelXilinxFpgaProgrammer):
    _TEMPLATE = """\
@00000000
{ipaddr_a:s}
{netmask_a:s}
{default_gateway_a:s}
{target_server_a:s}
{macaddr_3210_a:s}
{macaddr_zz54_a:s}
{ipaddr_b:s}
{netmask_b:s}
{default_gateway_b:s}
{target_server_b:s}
{macaddr_3210_b:s}
{macaddr_zz54_b:s}
00000000
00000000
00000000
00000000
"""
    _BITPREFIX: str = "au50_"
    _TCLCMD_POSTFIX: str = "_au50.tcl"
    _IP_VARIABLE_DIGITS: Final[int] = 3
    _MAC_VARIABLE_DIGITS: Final[int] = 3

    def __init__(self):
        super().__init__()

    def _addr_a_to_b(self, addr_in_hexstr: str, num_octets: int, diff: int, variable_octets: int) -> str:
        if num_octets != (len(addr_in_hexstr) + 1) // 2:
            raise ValueError(f"unexpected number of octets in address: '{addr_in_hexstr}'")
        a_ba = bytearray.fromhex(addr_in_hexstr)
        a_i = int.from_bytes(a_ba, "little")
        a_i += diff
        a_ba2 = a_i.to_bytes(num_octets, "little")
        for i in range(variable_octets, num_octets):
            if a_ba[i] != a_ba2[i]:
                raise ValueError("upper digits changed unexpectedly!")
        return a_ba2.hex()

    def ipaddr_a_to_b(self, ipaddr_in_hexstr: str, variable_octets: int = _IP_VARIABLE_DIGITS) -> str:
        # Note: want to increment the second octet.
        return self._addr_a_to_b(ipaddr_in_hexstr, 4, 0x010000, variable_octets)

    def macaddr_a_to_b(
        self, macadr_in_hexstr_3210: str, macadr_in_hexstr_zz54: str, variable_octets: int = _MAC_VARIABLE_DIGITS
    ) -> Tuple[str, str]:
        # Note: want to just increment by one, however, zzzz should be considered.
        a = self._addr_a_to_b(macadr_in_hexstr_zz54 + macadr_in_hexstr_3210, 8, 0x010000, variable_octets)
        return a[8:16], a[0:8]

    def make_mem(self, **params: str) -> Path:
        macaddr: str = params["macaddr"]
        ipaddr: str = params["ipaddr"]
        netmask: str = params["netmask"]
        default_gateway: str = params["default_gateway"]
        target_server: Union[str, None] = params.get("target_server")
        if target_server is None:
            target_server = default_gateway

        ipaddr_a = self.encode_ipaddr(ipaddr)
        ipaddr_b = self.ipaddr_a_to_b(ipaddr_a)
        macaddr_3210_a, macaddr_zz54_a = self.encode_macaddr(macaddr)
        macaddr_3210_b, macaddr_zz54_b = self.macaddr_a_to_b(macaddr_3210_a, macaddr_zz54_a)

        mem = self._TEMPLATE.format(
            ipaddr_a=ipaddr_a,
            netmask_a=self.encode_ipaddr(netmask),
            default_gateway_a=self.encode_ipaddr(default_gateway),
            target_server_a=self.encode_ipaddr(target_server),
            macaddr_3210_a=macaddr_3210_a,
            macaddr_zz54_a=macaddr_zz54_a,
            ipaddr_b=ipaddr_b,
            netmask_b=self.encode_ipaddr(netmask),
            default_gateway_b=self.encode_ipaddr(default_gateway),
            target_server_b=self.encode_ipaddr(target_server),
            macaddr_3210_b=macaddr_3210_b,
            macaddr_zz54_b=macaddr_zz54_b,
        )

        filename = Path(self._tmpdir.name) / f"{ipaddr}.mem"
        with open(filename, "w") as f:
            f.write(mem)

        return filename
