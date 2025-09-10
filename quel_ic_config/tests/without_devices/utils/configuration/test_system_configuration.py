import os

import yaml

from quel_ic_config_utils import configuration


def test_validate_example():
    path_to_example = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../examples/sysconf.yaml"))
    with open(path_to_example) as io:
        obj = yaml.safe_load(io)
    _ = configuration.SystemConfiguration.model_validate(obj)  # without error
