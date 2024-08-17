import logging
import subprocess
import tempfile
from pathlib import Path

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
