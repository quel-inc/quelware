from collections.abc import Collection

from e7awghal import AwgParam, CapParam, CapSection, WaveChunk

from quel_ic_config.quel1_box_intrinsic import Quel1BoxIntrinsic


def config_awgs_gen_seconds(boxi: Quel1BoxIntrinsic, glcs: dict[tuple[int, int, int], int]) -> set[int]:
    wss, css, rmap = boxi.wss, boxi.css, boxi.rmap

    awg_idxs: set[int] = set()
    for (group, line, channel), duration in glcs.items():
        cwp = AwgParam(num_wait_word=0, num_repeat=duration)
        cwp.chunks.append(
            WaveChunk(name_of_wavedata="test_wave_generation:cw16383", num_blank_word=0, num_repeat=7812500)
        )  # just 1 second

        awg_idx = rmap.get_awg_from_fduc(*css.get_fduc_idx(group, line, channel))
        wss.config_awgunit(awg_idx, cwp)
        awg_idxs.add(awg_idx)
    return awg_idxs


def check_awgs_are_clear(boxi: Quel1BoxIntrinsic, awg_idxs: Collection[int]):
    for i in awg_idxs:
        u = boxi.wss.hal.awgunit(i)
        assert not u.is_busy()  # should be completed
        assert not u.is_done()  # should be cleared


def config_caps_cap_now_seconds(
    boxi: Quel1BoxIntrinsic,
    glus: dict[tuple[int, str, int], tuple[int, int]],
    period: int = 125_000_000,
) -> set[tuple[int, int]]:
    wss, css, rmap = boxi.wss, boxi.css, boxi.rmap

    capunit_idxs: set[tuple[int, int]] = set()
    for (group, rline, runit), (num_capword, duration) in glus.items():
        lcp = CapParam(num_repeat=duration)
        lcp.sections.append(
            CapSection(name="s0", num_capture_word=num_capword, num_blank_word=period - num_capword)
        )  # 1 second
        capmod_idx = rmap.get_capmod_from_fddc(
            *css.get_fddc_idx(group, rline, 0)
        )  # Notes: this will be wrong in future
        capunit_idx = (capmod_idx, runit)  # Notes: this will be wrong in future
        wss.config_capunit(capunit_idx, lcp)
        capunit_idxs.add(capunit_idx)
    return capunit_idxs


def check_caps_are_clear(boxi: Quel1BoxIntrinsic, capunit_idxs: Collection[int]):
    for i in capunit_idxs:
        u = boxi.wss.hal.capunit(i)
        assert not u.is_busy()  # should be completed
        assert not u.is_done()  # should be cleared
