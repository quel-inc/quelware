import logging
import os
import shutil
from pathlib import Path
from typing import Final, List

import matplotlib as mpl
import pytest
from web_server_availability import QuelInstWebServer, get_available_devices

logger = logging.getLogger(__name__)
DEVICES: Final[List[str]] = [QuelInstWebServer.E4405B, QuelInstWebServer.E4407B]
OUTDIR: Final[Path] = Path("./artifacts/web/")


@pytest.fixture(scope="module", params=DEVICES)
def e440xb_name(request) -> str:
    available = get_available_devices()
    spa_name = request.param

    logger.info(f"try to acquire {spa_name}")
    if spa_name not in available:
        pytest.skip(f"{spa_name} is not available")

    return spa_name


@pytest.fixture(scope="session")
def outdir() -> Path:
    mpl.use("Gtk3Agg")
    if os.path.exists(OUTDIR):
        shutil.rmtree(OUTDIR)
        os.makedirs(OUTDIR)
    return Path(OUTDIR)
