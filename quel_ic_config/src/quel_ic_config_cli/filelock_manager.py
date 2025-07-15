#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Final

LOCK_DIR: Final[Path] = Path("/run/quelware")
TIMEOUT_DEFAULT: Final[int] = 30

logger = logging.getLogger(__name__)


def list_lockfiles():
    """
    Lists lock files (files that do not contain '|').
    """
    if not LOCK_DIR.is_dir():
        raise FileNotFoundError(f"Lock directory not found: {LOCK_DIR}")

    for f in LOCK_DIR.iterdir():
        if f.is_file() and "|" not in f.name:
            print(f.name)


def kill(lockfile_path: Path, timeout: int = TIMEOUT_DEFAULT):
    """
    Sends a request to release the specified lock (by creating a .kill file)
    and waits for it to be released.
    """

    if not lockfile_path.is_file():
        raise FileNotFoundError(f"Lock file not found: {lockfile_path}")

    try:
        claimfile: str = lockfile_path.read_text().strip()
        if not claimfile:
            raise ValueError(f"Lock file is empty: {lockfile_path}")
        if not claimfile.startswith(str(LOCK_DIR)):
            raise ValueError(f"Invalid content in lock file: '{claimfile}'")
        if not Path(claimfile).exists():
            raise ValueError(f"Invalid content in lock file: '{claimfile}'")
    except IOError as e:
        raise ValueError(f"Error reading lock file {lockfile_path}: {e}")

    killfile_path = Path(claimfile + ".kill")

    try:
        try:
            killfile_path.touch()
        except IOError as e:
            raise RuntimeError(f"Could not create kill file: {killfile_path}, reason: {e}")

        start_time = time.time()
        is_removed = False

        logger.info(f"Waiting for lock file '{lockfile_path}' to be removed (timeout: {timeout}s)...")

        while time.time() - start_time < timeout:
            if not lockfile_path.exists():
                is_removed = True
                break
            time.sleep(0.4)

        if is_removed:
            logger.info(f"Successfully confirmed removal of lock: {lockfile_path}")
        else:
            logger.info("Failed to kill the lock, but the specified lock file may no longer be active.")
            raise TimeoutError("Timeout occurred while waiting for the lock file to be removed.")

    finally:
        killfile_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        prog="quel_filelock_manager", description="A command-line tool to manage Quelware file locks."
    )

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    subparsers = parser.add_subparsers(title="Commands", dest="command", metavar="COMMAND")
    subparsers.required = True

    parser_list = subparsers.add_parser("list", help="List active lock files.")
    parser_list.set_defaults(func=lambda _: list_lockfiles())

    parser_kill = subparsers.add_parser("kill", help="Request to kill a thread holding a lock.")
    parser_kill.add_argument("lockfile", help="The name of the lock file to target (e.g., 10.5.0.167).")
    parser_kill.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=TIMEOUT_DEFAULT,
        help=f"Seconds to wait for the lock file to be removed (default: {TIMEOUT_DEFAULT}).",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)

    def handle_kill(args):
        lockfile_path: Path = LOCK_DIR / args.lockfile
        timeout = args.timeout
        kill(lockfile_path, timeout)

    parser_kill.set_defaults(func=handle_kill)

    try:
        args = parser.parse_args()
        args.func(args)
    except Exception as e:
        logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
