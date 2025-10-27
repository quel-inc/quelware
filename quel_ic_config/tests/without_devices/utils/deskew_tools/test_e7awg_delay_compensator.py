import pytest

import quel_ic_config as qi
from quel_ic_config_utils import deskew_tools


class TestE7awgDelayCompensator:
    def test_adjust_awg_param(self):
        awg_param = qi.AwgParam(num_wait_word=16)
        first_chunk = qi.WaveChunk(name_of_wavedata="first", num_blank_word=10, num_repeat=1)
        mid_chunk = qi.WaveChunk(name_of_wavedata="mid", num_blank_word=20, num_repeat=3)
        last_chunk = qi.WaveChunk(name_of_wavedata="last", num_blank_word=30, num_repeat=1)
        awg_param.chunks = [first_chunk, mid_chunk, last_chunk]

        compensator = deskew_tools.E7awgDelayCompensator(awg_init_blank_reference_word=100)
        init_offset = 50
        adjusted = compensator.adjust_awg_param(awg_param, init_offset)

        assert len(adjusted.chunks) == 4
        assert adjusted.chunks[1] == awg_param.chunks[0]
        assert adjusted.chunks[2] == awg_param.chunks[1]

        inserted_chunk = adjusted.chunks[0]
        assert inserted_chunk.num_blank_word + (64 // 4) + last_chunk.num_blank_word == (100 + 50) + 30
        assert adjusted.num_wait_word == awg_param.num_wait_word

        # An exception will be thrown for a duplicate adjustment.
        with pytest.raises(ValueError):
            compensator.adjust_awg_param(adjusted, init_offset)

    def test_adjust_awg_param_single_chunk(self):
        awg_param = qi.AwgParam(num_wait_word=16)
        chunk = qi.WaveChunk(name_of_wavedata="chunk", num_blank_word=20, num_repeat=3)
        awg_param.chunks = [chunk]

        compensator = deskew_tools.E7awgDelayCompensator(awg_init_blank_reference_word=100)
        init_offset = 50
        adjusted = compensator.adjust_awg_param(awg_param, init_offset)

        assert len(adjusted.chunks) == 2
        assert adjusted.chunks[1] == awg_param.chunks[0]

        inserted_chunk = adjusted.chunks[0]
        assert inserted_chunk.num_blank_word + (64 // 4) + chunk.num_blank_word == (100 + 50) + 20
        assert adjusted.num_wait_word == awg_param.num_wait_word

        # An exception will be thrown for a duplicate adjustment.
        with pytest.raises(ValueError):
            compensator.adjust_awg_param(adjusted, init_offset)

    def test_adjust_cap_param(self):
        cap_param = qi.CapParam(num_wait_word=16, num_repeat=1)
        first_capsec = qi.CapSection(name="first", num_capture_word=64, num_blank_word=10)
        mid_capsec = qi.CapSection(name="mid", num_capture_word=128, num_blank_word=20)
        last_capsec = qi.CapSection(name="last", num_capture_word=192, num_blank_word=30)
        cap_param.sections = [first_capsec, mid_capsec, last_capsec]

        compensator = deskew_tools.E7awgDelayCompensator(awg_init_blank_reference_word=1000)
        init_offset = 50
        adjusted = compensator.adjust_cap_param(cap_param, init_offset)

        assert len(adjusted.sections) == 4
        assert adjusted.sections[1] == cap_param.sections[0]
        assert adjusted.sections[2] == cap_param.sections[1]
        inserted_section = adjusted.sections[0]

        assert (
            adjusted.num_wait_word + inserted_section.num_capture_word + inserted_section.num_blank_word
            == 1000 + 50 + 16
        )

        # An exception will be thrown for a duplicate adjustment.
        with pytest.raises(ValueError):
            compensator.adjust_cap_param(adjusted, init_offset)

    def test_adjust_cap_param_with_repetition(self):
        cap_param = qi.CapParam(num_wait_word=16, num_repeat=2)
        first_capsec = qi.CapSection(name="first", num_capture_word=64, num_blank_word=10)
        mid_capsec = qi.CapSection(name="mid", num_capture_word=128, num_blank_word=20)
        last_capsec = qi.CapSection(name="last", num_capture_word=192, num_blank_word=30)
        cap_param.sections = [first_capsec, mid_capsec, last_capsec]

        compensator = deskew_tools.E7awgDelayCompensator(awg_init_blank_reference_word=1000)
        init_offset = 50
        adjusted = compensator.adjust_cap_param(cap_param, init_offset)

        assert len(adjusted.sections) == 4
        assert adjusted.sections[1] == cap_param.sections[0]
        assert adjusted.sections[2] == cap_param.sections[1]
        inserted_section = adjusted.sections[0]

        assert (
            adjusted.num_wait_word + inserted_section.num_capture_word + inserted_section.num_blank_word
            == 1000 + 50 + 16
        )
        assert (
            inserted_section.num_capture_word
            + adjusted.sections[0].num_blank_word
            + adjusted.sections[-1].num_blank_word
            == cap_param.sections[-1].num_blank_word
        )

        # An exception will be thrown for a duplicate adjustment.
        with pytest.raises(ValueError):
            compensator.adjust_cap_param(adjusted, init_offset)

    def test_adjust_cap_single_section(self):
        cap_param = qi.CapParam(num_wait_word=16)
        capsec = qi.CapSection(name="mid", num_capture_word=128, num_blank_word=10)
        cap_param.sections = [capsec]

        compensator = deskew_tools.E7awgDelayCompensator(awg_init_blank_reference_word=1000)
        init_offset = 50
        adjusted = compensator.adjust_cap_param(cap_param, init_offset)

        assert len(adjusted.sections) == 2
        assert adjusted.sections[1] == cap_param.sections[0]
        inserted_section = adjusted.sections[0]

        assert (
            adjusted.num_wait_word + inserted_section.num_capture_word + inserted_section.num_blank_word
            == 1000 + 50 + 16
        )

        # An exception will be thrown for a duplicate adjustment.
        with pytest.raises(ValueError):
            compensator.adjust_cap_param(adjusted, init_offset)

    def test_adjust_cap_param_with_repetition_and_too_short_num_blank_raises_value_error(self):
        cap_param = qi.CapParam(num_wait_word=16, num_repeat=2)
        cap_param.sections = [qi.CapSection(name="bad", num_capture_word=1, num_blank_word=20)]

        compensator = deskew_tools.E7awgDelayCompensator(awg_init_blank_reference_word=100)

        with pytest.raises(ValueError):
            compensator.adjust_cap_param(cap_param, 10)


class TestWaitAmountResolver:
    def test_from_deskew_configuration(self):
        deskew_conf = deskew_tools.DeskewConfiguration(
            boxes=[
                deskew_tools.Box(
                    name="box1",
                    ports=[
                        deskew_tools.Port(port=1, wait_ps_offset=20 * 8000),
                        deskew_tools.Port(port=2, wait_ps_offset=30 * 8000),
                    ],
                    wait_ps=15 * 8000,
                ),
                deskew_tools.Box(
                    name="box2",
                    ports=[
                        deskew_tools.Port(port=3, wait_ps_offset=50 * 8000),
                    ],
                    wait_ps=-1 * 8000,
                ),
            ]
        )
        resolver = deskew_tools.WaitAmountResolver.from_deskew_configuration(deskew_conf)

        assert resolver.get_word_to_wait("box1", 1) == 35
        assert resolver.get_word_to_wait("box1", 2) == 45
        assert resolver.get_word_to_wait("box2", 3) == 49

    def test_register_wait_word(self):
        resolver = deskew_tools.WaitAmountResolver()
        resolver.register_wait_word_port_offset("box1", 4, 40)
        assert resolver.get_word_to_wait("box1", 4) == 40
