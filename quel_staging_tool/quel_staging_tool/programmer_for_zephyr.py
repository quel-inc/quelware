import ipaddress
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Union

from quel_staging_tool.quel_xilinx_fpga_programmer import QuelXilinxFpgaProgrammer
from quel_staging_tool.run_vivado_batch import run_vivado_batch

logger = logging.getLogger(__name__)


class QuelXilinxFpgaProgrammerZephyr(QuelXilinxFpgaProgrammer):
    _BITPREFIX: str = "zephyr-exstickge_"
    _TCLCMD_POSTFIX: str = "_exstickge.tcl"
    _DUMMY_IPADDR = b"10.255.254.253\x00"

    @staticmethod
    def encode_macaddr(macaddr: str) -> bytes:
        b = bytearray([int(o, 16) for o in macaddr.split("-")])
        if len(b) != 6:
            raise ValueError(f"malformed macaddress: '{macaddr}'")
        return bytes(b)

    @classmethod
    def encode_ipaddr(cls, ipaddr: ipaddress.IPv4Address) -> bytes:
        ipaddr_str = str(ipaddr)
        body = ipaddr_str.encode()
        space = b"\x00" * (len(cls._DUMMY_IPADDR) - len(body))
        return body + space

    def make_embedded_elf(self, elfpath: Path, ipaddr: ipaddress.IPv4Address, patch_dict: Dict[str, str]) -> Path:
        with open(elfpath, "rb") as f:
            obj = bytearray(f.read())

        # Notes: embedding IP address
        pos = obj.find(self._DUMMY_IPADDR)
        if pos < 0:
            raise RuntimeError("failed to find IP ADDRESS marker")
        encoded_ipaddr = self.encode_ipaddr(ipaddr)
        obj[pos : pos + len(encoded_ipaddr)] = encoded_ipaddr

        # Notes: applying replace_dict
        for orig, mod in patch_dict.items():
            pos = obj.find(orig.encode())
            if pos < 0:
                raise RuntimeError(f"failed to find marker '{orig}', which is supposed to be replaced with '{mod}'")
            if len(orig) < len(mod):
                raise RuntimeError(f"it is impossible to fit '{mod}' at the marker '{orig}'")
            elif len(orig) == len(mod):
                bmod = mod.encode()
            else:
                bmod = mod.encode()
                bmod += b"\x00" * (len(orig) - len(bmod))
            obj[pos : pos + len(orig)] = bmod

        # Notes: writing modified elf binary
        outpath = Path(self._tmpdir.name) / f"{str(ipaddr)}.elf"
        with open(outpath, "wb") as g:
            g.write(obj)

        return outpath

    def make_macaddr_bin(self, macaddr: str) -> Path:
        outpath = Path(self._tmpdir.name) / "macaddr.bin"
        macaddr_bin = self.encode_macaddr(macaddr)
        with open(outpath, "wb") as f:
            f.write(macaddr_bin)
        return outpath

    def make_embedded_bit(self, bitpath: Path, **parameters) -> Path:
        self._validate_env()
        for pathname in ("elfpath", "mmipath"):
            if pathname not in parameters:
                raise ValueError(f"lacking parameter '{pathname}'")
            if not isinstance(parameters[pathname], Path):
                raise TypeError("unexpected type of parameter '{pathname}'")
        elfpath = parameters["elfpath"]
        mmipath = parameters["mmipath"]

        outfile = os.path.splitext(elfpath)[0] + ".bit"
        outpath = Path(self._tmpdir.name) / outfile
        logger.info(f"generating bit file: {outfile}")
        retcode = subprocess.run(
            f"updatemem -force -bit {bitpath} -meminfo {mmipath} -data {elfpath} "
            f"-proc cm3_ss/itcm/mem_reg -out {outpath}".split(),
            capture_output=True,
        )
        if os.path.exists(outpath) and retcode.returncode != 0:
            raise RuntimeError("failed execution of updatemem")
        return outpath

    def make_bin(self, bitpath: Path, binpath: Union[Path, None] = None) -> Path:
        self._validate_env()
        if binpath is None:
            binpath = Path(os.path.splitext(bitpath)[0] + ".bin")
        retval = run_vivado_batch(self._tcldir_path(), f"create_bin{self._TCLCMD_POSTFIX}", f"{bitpath} {binpath}")
        if retval == 0:
            return binpath
        else:
            raise AssertionError("not reached")
