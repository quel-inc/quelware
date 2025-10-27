import json
import os

from pydantic import BaseModel, Field

import quel_ic_config as qi


class Port(BaseModel):
    port: qi.Quel1PortType
    wait_ps_offset: int  # picosecond


class Box(BaseModel):
    name: str
    timecounter_offset: int = Field(default=0)
    wait_ps: int = Field(default=0)  # picosecond
    ports: list[Port]


class DeskewConfiguration(BaseModel):
    boxes: list[Box]


def load_default_configuration() -> DeskewConfiguration:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))

    config_dir = os.path.join(xdg_config_home, "quelware")
    config_file = os.path.join(config_dir, "deskew.json")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found at {config_file}.")

    try:
        with open(config_file, "r") as f:
            obj = json.load(f)
    except Exception as e:
        raise ValueError(f"Error reading {config_file}: {e}") from e

    conf = DeskewConfiguration.model_validate(obj)
    return conf
