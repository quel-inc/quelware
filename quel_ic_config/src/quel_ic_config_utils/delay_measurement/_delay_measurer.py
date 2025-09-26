from collections.abc import Callable
from typing import Any, Optional, Protocol
from typing_extensions import Self

import quel_ic_config as qi


class DelayMeasurer(Protocol):
    """
    A class to measure the time delay between a generated signal and a captured signal.
    """

    def measure(
        self,
        source_init_blank_word: int = 0,
        dest_init_blank_word: int = 0,
        source_timecounter_shift: int = 0,
        dest_timecounter_shift: int = 0,
        callback: Optional[Callable[[Self, dict[str, Any]], None]] = None,
    ) -> tuple[int, list[dict[str, Any]]]: ...


class RelevantPeakNotFoundError(Exception):
    def __init__(
        self,
        *args,
        source_box_name: str = "unprovided",
        source_port: qi.Quel1PortType = -1,
        dest_box_name: str = "unprovided",
        dest_port: qi.Quel1PortType = -1,
    ):
        self.source_box_name = source_box_name
        self.source_port = source_port
        self.dest_box_name = dest_box_name
        self.dest_port = dest_port
        super().__init__(*args)
