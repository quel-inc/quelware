import copy
import json
import logging
import os.path as osp
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Union

from pydantic.v1.utils import deep_update

from quel_ic_config.quel_config_common import Quel1BoxType, Quel1ConfigOption, Quel1Feature

logger = logging.getLogger(__name__)


class Quel1ConfigLoader:
    DEFAULT_CONFIG_ROOTPATH: Final[Path] = Path(osp.dirname(__file__)) / "settings"

    def __init__(
        self,
        boxtype: Quel1BoxType,
        num_ic: dict[str, int],
        config_options: Collection[Quel1ConfigOption],
        features: Collection[Quel1Feature],
        config_filename: Path,
        config_rootpath: Union[Path, None] = None,
    ):
        self._boxtype: Quel1BoxType = boxtype
        self._num_ic: dict[str, int] = copy.copy(num_ic)
        self._config_options: set[Quel1ConfigOption] = set(config_options)
        self._features: set[Quel1Feature] = set(features)
        self._config_filename: Path = config_filename
        self._config_rootpath: Path = config_rootpath or self.DEFAULT_CONFIG_ROOTPATH

    def _remove_comments(self, settings: dict[str, Any]) -> dict[str, Any]:
        s1: dict[str, Any] = {}
        for k, v in settings.items():
            if not k.startswith("#"):
                if isinstance(v, dict):
                    s1[k] = self._remove_comments(v)
                else:
                    s1[k] = v
        return s1

    def _match_conditional_include(self, directive: Mapping[str, Union[str, Sequence[str]]]) -> bool:
        flag: bool = True
        file: str = ""
        for k, v in directive.items():
            if k == "file":
                if isinstance(v, str):
                    file = v
                else:
                    raise TypeError(f"invalid type of 'file': {k}")
            elif k == "boxtype":  # OR
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise TypeError(f"invalid type of 'boxtype': {k}")
                if self._boxtype.value[1] not in v:
                    flag = False
            elif k == "option":  # AND
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise TypeError(f"invalid type of 'option': {k}")
                for op1 in v:
                    if Quel1ConfigOption(op1) not in self._config_options:
                        flag = False
            elif k == "feature":  # AND
                if isinstance(v, str):
                    v = [v]
                if not isinstance(v, list):
                    raise TypeError(f"invalid type of 'option': {k}")
                for ft1 in v:
                    if Quel1Feature(ft1) not in self._features:
                        flag = False
            elif k == "otherwise":
                if not isinstance(v, str):
                    raise TypeError(f"invalid type of 'otherwise': {k}")
            else:
                raise ValueError(f"invalid key of conditional include: {k}")

        if file == "":
            raise ValueError(f"no file is specified in conditional include: {directive}")
        return flag

    def _include_config(
        self, directive: Union[str, Mapping[str, str], Sequence[Union[str, Mapping[str, str]]]], label_for_log: str
    ) -> tuple[dict[str, Any], set[Quel1ConfigOption]]:
        fired_options: set[Quel1ConfigOption] = set()

        if isinstance(directive, str) or isinstance(directive, dict):
            directive = [directive]

        config: dict[str, Any] = {}
        for d1 in directive:
            if isinstance(d1, str):
                with open(self._config_rootpath / d1) as f:
                    logger.info(f"basic config applied to {label_for_log}: {d1}")
                    config = deep_update(config, json.load(f))
            elif isinstance(d1, dict):
                if self._match_conditional_include(d1):
                    with open(self._config_rootpath / d1["file"]) as f:
                        logger.info(f"conditional config applied to {label_for_log}: {d1}")
                        config = deep_update(config, json.load(f))
                        if "option" in d1:
                            option = d1["option"]
                            if isinstance(option, str):
                                fired_options.add(Quel1ConfigOption(option))
                            elif isinstance(option, list):
                                fired_options.update({Quel1ConfigOption(o) for o in option})
                            else:
                                raise AssertionError
                elif "otherwise" in d1:
                    with open(self._config_rootpath / d1["otherwise"]) as f:
                        logger.info(f"'otherwise' part of conditional config applied to {label_for_log}: {d1}")
                        config = deep_update(config, json.load(f))
            else:
                raise TypeError(f"malformed template at {label_for_log}: '{d1}'")
        if "meta" in config:
            del config["meta"]
        return self._remove_comments(config), fired_options

    def load_config(self) -> dict[str, Any]:
        logger.info(f"loading configuration settings from '{self._config_rootpath / self._config_filename}'")
        logger.info(f"boxtype = {self._boxtype}")
        logger.info(f"config_options = {self._config_options}")
        fired_options: set[Quel1ConfigOption] = set()

        with open(self._config_rootpath / self._config_filename) as f:
            root: dict[str, Any] = self._remove_comments(json.load(f))

        config = copy.copy(root)
        for k0, directive0 in root.items():
            if k0 == "meta":
                pass
            elif k0 in self._num_ic.keys():
                for idx, directive1 in enumerate(directive0):
                    if idx >= self._num_ic[k0]:
                        raise ValueError(f"too many {k0.upper()}s are found")
                    config[k0][idx], fired_options_1 = self._include_config(directive1, label_for_log=f"{k0}[{idx}]")
                    fired_options.update(fired_options_1)
            else:
                raise ValueError(f"invalid name of IC: '{k0}'")

        for k1, n1 in self._num_ic.items():
            if len(config[k1]) != n1:
                raise ValueError(
                    f"lacking config, there should be {n1} instances of '{k1}', "
                    f"but actually have {len(config[k1])} ones"
                )

        for option in self._config_options:
            if option not in fired_options:
                logger.warning(f"config option '{str(option)}' is not applicable")

        return config
