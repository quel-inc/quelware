import logging
from contextlib import contextmanager
from typing import Any

import quel_ic_config as qi

logger = logging.getLogger(__name__)


@contextmanager
def modified_config(box: qi.Quel1Box, config: dict[str, Any]):
    original = box.dump_box()
    logger.info(f"Temporarily modifying config for box '{box.name}': {config}")
    box.config_box(config)
    yield box
    box.config_box(original)
    logger.info(f"Config for {box.name} has been restored.")
