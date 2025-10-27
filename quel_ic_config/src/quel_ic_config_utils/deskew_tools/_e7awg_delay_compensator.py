import logging
from typing import Final

import numpy as np
import numpy.typing as npt
from e7awghal.capdata import CapIqDataReader

import quel_ic_config as qi

from . import _config

logger = logging.getLogger(__name__)

_SAMPLING_RATE = 500e6
_SAMPLES_PER_WORD = 4

_BLANK_WAVEDATA_NAME = "__quel_deskew_tools_blank"
_BLANK_WAVEDATA_LENGTH = 64
_BLANK_WAVEDATA = np.zeros((_BLANK_WAVEDATA_LENGTH,), dtype=np.complex64)
_DUMMY_CAPSECTION_NAME = "__quel_deskew_tools_dummy_capsection"
_AWG_INIT_BLANK_REFERENCE_WORD_DEFAULT = 1280

BoxName = str


def register_blank_wavedata(box: qi.Quel1Box, port, channel):
    box.register_wavedata(port, channel, _BLANK_WAVEDATA_NAME, _BLANK_WAVEDATA)


class E7awgDelayCompensator:
    """Compensates delays by adjusting parameters for e7awg."""

    _MINIMUM_BLANK_IN_LAST_CAP_SECTION: Final[int] = 21

    def __init__(self, awg_init_blank_reference_word: int = _AWG_INIT_BLANK_REFERENCE_WORD_DEFAULT):
        """Initializes the E7awgParameterAdjuster.

        :param awg_init_blank_reference_word: The reference number of blank words
            at the beginning of the AWG sequence. This value serves as a baseline
            for all subsequent adjustments.
        """
        self._awg_init_blank_reference_word = awg_init_blank_reference_word

    def adjust_awg_param(self, awg_param: qi.AwgParam, init_blank_offset_word: int) -> qi.AwgParam:
        """Adjusts AwgParam by inserting a blank chunk at the beginning.

        This method calculates the necessary blank word count based on a reference
        and an offset, then inserts a `WaveChunk` with this blank count to the
        front of the AWG sequence.

        :param awg_param: The ``AwgParam`` object to be adjusted.
        :param init_blank_offset_word: The offset value to be added to the reference
                                       blank word count.
        :returns: The modified ``AwgParam`` object with the blank chunk inserted.
        """
        copied = awg_param.model_copy(deep=True)
        if self._check_if_awg_param_already_adjusted(awg_param):
            raise ValueError("AwgParam is already adjusted.")

        if len(copied.chunks) > 15:
            raise ValueError("The number of chunks must be less than 16.")

        blank_chunk = qi.WaveChunk(
            name_of_wavedata=_BLANK_WAVEDATA_NAME,
            num_repeat=1,
            num_blank_word=self._awg_init_blank_reference_word
            + init_blank_offset_word
            - _BLANK_WAVEDATA_LENGTH // _SAMPLES_PER_WORD,
        )
        copied.chunks.insert(0, blank_chunk)
        return copied

    def _check_if_awg_param_already_adjusted(self, awg_param: qi.AwgParam):
        return awg_param.chunks[0].name_of_wavedata == _BLANK_WAVEDATA_NAME

    def adjust_cap_param(self, cap_param: qi.CapParam, init_blank_offset_word: int) -> qi.CapParam:
        """Adjusts CapParam.

        This method inserts a dummy section for timing adjustment at the beginning without
        changing the total length of all sections combined.

        :param cap_param: The ``CapParam`` object to be adjusted.
        :param init_blank_offset_word: The offset value used to calculate the total
            blank words for the CAP sequence.
        :returns: The modified ``CapParam`` object.
        :raises ValueError: If the number of ``num_blank_word`` in the last section is not
            greater than 20, which is required for the adjustment logic.
        """
        copied = cap_param.model_copy(deep=True)
        if self._check_if_cap_param_already_adjusted(cap_param):
            raise ValueError("CapParam is already adjusted.")

        last_section = copied.sections[-1]

        copied.sections.insert(0, qi.CapSection(name=_DUMMY_CAPSECTION_NAME, num_capture_word=4, num_blank_word=1))

        cap_init_blank_word = self._awg_init_blank_reference_word + init_blank_offset_word - (4 + 1)
        num_wait_word_div_by_16, num_blank_word_of_dummy_section = divmod(cap_init_blank_word, 16)
        copied.num_wait_word += 16 * num_wait_word_div_by_16
        copied.sections[0].num_blank_word += num_blank_word_of_dummy_section
        if copied.num_repeat > 1:
            if last_section.num_blank_word < self._MINIMUM_BLANK_IN_LAST_CAP_SECTION:
                raise ValueError(
                    f"When num_repeat > 1, the number of num_blank_word in the last section must be equal to or greater than {self._MINIMUM_BLANK_IN_LAST_CAP_SECTION} for adjustment."
            )
            last_section.num_blank_word -= 4 + 1
            last_section.num_blank_word -= num_blank_word_of_dummy_section

        return copied

    def _check_if_cap_param_already_adjusted(self, cap_param: qi.CapParam):
        return cap_param.sections[0].name == _DUMMY_CAPSECTION_NAME

    def get_minimum_blank_in_last_cap_section(self):
        return self._MINIMUM_BLANK_IN_LAST_CAP_SECTION


def _picosec_to_word(ps: int) -> int:
    word_index = (ps * (1e-12 * _SAMPLING_RATE)) / _SAMPLES_PER_WORD
    return int(round(word_index))


class WaitAmountResolver:
    def __init__(self):
        self._name_to_wait_word: dict[str, int] = {}
        self._name_port_to_wait_word_offset: dict[tuple[str, qi.Quel1PortType], int] = {}

    def register_wait_word(self, box_name: BoxName, wait_word: int):
        self._name_to_wait_word[box_name] = wait_word

    def register_wait_word_port_offset(self, box_name: BoxName, port: qi.Quel1PortType, wait_word: int):
        self._name_port_to_wait_word_offset[(box_name, port)] = wait_word

    @classmethod
    def from_deskew_configuration(cls, deskew_conf: _config.DeskewConfiguration):
        inst = cls()
        for box in deskew_conf.boxes:
            inst.register_wait_word(box.name, _picosec_to_word(box.wait_ps))
            for port in box.ports:
                inst.register_wait_word_port_offset(box.name, port.port, _picosec_to_word(port.wait_ps_offset))
        return inst

    def get_word_to_wait(self, box_name: str, port: qi.Quel1PortType) -> int:
        if (box_name, port) in self._name_port_to_wait_word_offset:
            return self._name_to_wait_word.get(box_name, 0) + self._name_port_to_wait_word_offset[(box_name, port)]
        else:
            return self._name_to_wait_word.get(box_name, 0)


def extract_wave_dict(iq_reader: CapIqDataReader) -> dict[str, npt.NDArray[np.complex64]]:
    d = iq_reader.as_wave_dict()
    d.pop(_DUMMY_CAPSECTION_NAME, None)
    return d


def extract_wave_list(iq_reader: CapIqDataReader) -> list[npt.NDArray[np.complex64]]:
    l = iq_reader.as_wave_list()
    return l[1:]
