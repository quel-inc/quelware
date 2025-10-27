import logging
from collections.abc import Collection

from . import _config

logger = logging.getLogger(__name__)

_CLOCK_FREQ = 125_000_000
_TRIGGER_GRID_STEP_DEFAULT = 16

BoxName = str


def _round_up_to_grid(val: float, grid_step: int) -> int:
    return int(val + (grid_step - 1)) // grid_step * grid_step


class StableCountProposer:
    def __init__(self, trigger_grid_step: int = _TRIGGER_GRID_STEP_DEFAULT):
        self._name_to_offset: dict[str, int] = {}
        self._trigger_grid_step: int = trigger_grid_step

    def get_offset(self, box_name) -> int:
        return self._name_to_offset[box_name]

    def set_offset(self, box_name: str, offset: int):
        self._name_to_offset[box_name] = offset

    def propose_trigger_counts(
        self,
        current_count: int,
        target_names: Collection[str],
        delay_sec: float = 0.15,
    ) -> dict[BoxName, int]:
        reference = _round_up_to_grid(current_count + _CLOCK_FREQ * delay_sec, self._trigger_grid_step)
        return {name: reference + self._name_to_offset.get(name, 0) for name in target_names}

    @classmethod
    def from_deskew_configuration(
        cls, deskew_conf: _config.DeskewConfiguration, trigger_grid_step: int = _TRIGGER_GRID_STEP_DEFAULT
    ):
        inst = cls(trigger_grid_step=trigger_grid_step)
        for box in deskew_conf.boxes:
            inst.set_offset(box.name, box.timecounter_offset)
        return inst
