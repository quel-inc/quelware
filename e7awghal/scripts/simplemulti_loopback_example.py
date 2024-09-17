import logging
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from e7awghal import (
    AbstractCapCtrl,
    AbstractCapUnit,
    AbstractQuel1Au50Hal,
    AwgCtrl,
    AwgParam,
    AwgUnit,
    CapCtrlClassic,
    CapCtrlStandard,
    CapParam,
    CapSection,
    ClockcounterCtrl,
    E7FwType,
    SimplemultiAwgTriggers,
    SimplemultiSequencer,
    WaveChunk,
)
from quel_ic_config.e7resource_mapper import AbstractQuel1E7ResourceMapper, create_rmap_object
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_box_intrinsic import _create_css_object
from quel_ic_config.quel_config_common import Quel1Feature
from testlibs.awgctrl_with_hlapi import AwgCtrlHL
from testlibs.capunit_with_hlapi import CapUnitSimplifiedHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger()


def config_output_line(css: Quel1AnyConfigSubsystem, group: int, line: int) -> None:
    # Notes: 11500 - 2000 = 9500MHz
    css.set_lo_multiplier(group=group, line=line, freq_multiplier=115)
    css.set_divider_ratio(group=group, line=line, divide_ratio=1)
    css.set_dac_cnco(group=group, line=line, freq_in_hz=2.000e9)
    css.set_vatt(group=group, line=line, vatt=0xC00)
    css.set_sideband(group=group, line=line, sideband="L")
    css.set_fullscale_current(group=group, line=line, fsc=40000)


def get_awg_idx(
    css: Quel1AnyConfigSubsystem, rmap: AbstractQuel1E7ResourceMapper, group: int, line: int, channel: int
) -> int:
    return rmap.get_awg_from_fduc(*css.get_fduc_idx(group, line, channel))


def config_input_line(
    css: Quel1AnyConfigSubsystem, group: int, rline: str, rchannel: int, use_loop: Union[bool, None]
) -> None:
    # Notes: 11500 - 2000 = 9500MHz
    css.set_lo_multiplier(group=group, line=rline, freq_multiplier=115)
    css.set_divider_ratio(group=group, line=rline, divide_ratio=1)
    css.set_adc_cnco(group=group, rline=rline, freq_in_hz=2.000e9)
    css.set_adc_fnco(group=group, rline=rline, rchannel=rchannel, freq_in_hz=0)
    if use_loop is not None:
        if use_loop:
            css.block_line(group=group, line=rline)
        else:
            css.pass_line(group=group, line=rline)


def _get_rchannel_from_runit_tentative(group: int, rline: str, runit: int) -> int:
    # Notes: the mapping should be managed by box-layer, but no box-layer exists in this example.
    #        so, assuming that all runits are referring to rchannel 0.
    # TODO: consider better way before this assumption becomes not true.
    # TODO: runit will have a pair of (subport, channel) in near future.
    return 0


def get_cap_idx(
    css: Quel1AnyConfigSubsystem,
    whal: AbstractQuel1Au50Hal,
    rmap: AbstractQuel1E7ResourceMapper,
    group: int,
    rline: str,
    runit: int,
) -> tuple[int, int]:
    rchannel = _get_rchannel_from_runit_tentative(group, rline, runit)
    mxfe_idx, fddc_idx = css.get_fddc_idx(group, rline, rchannel)
    capmod_idx = rmap.get_capmod_from_fddc(mxfe_idx, fddc_idx)
    capunt_idx = whal.capctrl.get_all_capunts_of_capmod(capmod_idx)[runit]
    return capunt_idx, rchannel


def find_chunks(
    iq: npt.NDArray[np.complex64], power_thr=1000.0, space_thr=16, minimal_length=16
) -> tuple[tuple[int, int], ...]:
    chunk = (abs(iq) > power_thr).nonzero()[0]
    if len(chunk) == 0:
        logger.info("no pulse!")
        return ()

    gaps = (chunk[1:] - chunk[:-1]) > space_thr
    start_edges = list(chunk[1:][gaps])
    start_edges.insert(0, chunk[0])
    last_edges = list(chunk[:-1][gaps])
    last_edges.append(chunk[-1])
    chunks = tuple([(s, e) for s, e in zip(start_edges, last_edges) if e - s >= minimal_length])

    n_chunks = len(chunks)
    logger.info(f"number_of_chunks: {n_chunks}")
    for i, chunk in enumerate(chunks):
        s, e = chunk
        iq0 = np.average(iq[s:e])
        angle = round(np.arctan2(iq0.real, iq0.imag) * 180.0 / np.pi, 1)
        logger.info(f"  chunk {i}: {e - s} samples, ({s} -- {e}),  mean phase = {angle:.1f}")
    return chunks


def calc_angle(iq) -> tuple[float, float, float]:
    angle = np.angle(iq)
    min_angle = min(angle)
    max_angle = max(angle)
    if max_angle - min_angle > 6.0:
        angle = (angle + 2 * np.pi) % np.pi

    avg = np.mean(angle) * 180.0 / np.pi
    sd = np.sqrt(np.var(angle)) * 180.0 / np.pi
    delta = (max(angle) - min(angle)) * 180.0 / np.pi
    return avg, sd, delta


def plot_iqs(iq_dict, t_offset: int = 0) -> None:
    n_plot = len(iq_dict)

    m = 0
    for _, iq in iq_dict.items():
        m = max(m, np.max(abs(np.real(iq))))
        m = max(m, np.max(abs(np.imag(iq))))

    fig, axs = plt.subplots(n_plot, sharex="col")
    if n_plot == 1:
        axs = [axs]
    fig.set_size_inches(10.0, 2.0 * n_plot)
    fig.subplots_adjust(bottom=max(0.025, 0.125 / n_plot), top=min(0.975, 1.0 - 0.05 / n_plot))
    for idx, (title, iq) in enumerate(iq_dict.items()):
        t = np.arange(0, len(iq)) - t_offset
        axs[idx].plot(t, np.real(iq))
        axs[idx].plot(t, np.imag(iq))
        axs[idx].set_ylim((-m * 1.1, m * 1.1))
        axs[idx].text(0.05, 0.1, title, transform=axs[idx].transAxes)
    plt.show()


if __name__ == "__main__":
    import argparse

    from quel_ic_config_utils import add_common_arguments, complete_ipaddrs

    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = argparse.ArgumentParser(description="testing e7awghal APIs with normal firmware and internal RF loopback")
    add_common_arguments(parser)
    parser.add_argument("--skip_config", action="store_true", help="disabling re-configuration of CSS parameters")
    parser.add_argument("--verbose", action="store_true", help="show verbose log")
    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Notes: assuming WSS firmware with implements BOTH_ADC feature.
    css: Quel1AnyConfigSubsystem = _create_css_object(ipaddr_css=str(args.ipaddr_css), boxtype=args.boxtype)
    proxy: AbstractQuel1Au50Hal = create_quel1au50hal_for_test(
        ipaddr_wss=str(args.ipaddr_wss), ipaddr_sss=str(args.ipaddr_sss), auth_callback=lambda: True
    )
    # Notes: configuration settings are hard coded here instead of retrieving it from the device.
    assert proxy.fw_type() == E7FwType.SIMPLEMULTI_STANDARD
    css.initialize({Quel1Feature.BOTH_ADC})
    rmap: AbstractQuel1E7ResourceMapper = create_rmap_object(str(args.ipaddr_wss), E7FwType.SIMPLEMULTI_STANDARD)

    for mxfe_idx in css.get_all_mxfes():
        css.configure_mxfe(mxfe_idx)
    proxy.initialize()

    GR = 0
    LN, CH = 0, 0
    RLN, RUNT = "r", 0
    SKIP_CSS_CONFIG = args.skip_config

    NUM_REPEAT = 5
    DELAY_B4_START_SEC = 0.1  # [s]
    DELAY_B4_START_COUNT = round(DELAY_B4_START_SEC * ClockcounterCtrl.CLOCK_FREQUENCY)

    au: AwgUnit = proxy.awgunit(get_awg_idx(css, rmap, GR, LN, CH))
    capunit_idx, rchannel = get_cap_idx(css, proxy, rmap, GR, RLN, RUNT)
    cu: AbstractCapUnit = proxy.capunit(capunit_idx)

    # configure
    if not SKIP_CSS_CONFIG:
        config_output_line(css, GR, LN)
        config_input_line(css, GR, RLN, rchannel, use_loop=True)

    # setup awg unit
    cw = np.zeros(64, dtype=np.complex64)
    cw[:] = 32767 + 0j
    au.register_wavedata_from_complex64vector("cw", cw)

    param_cw128 = AwgParam(num_wait_word=0, num_repeat=NUM_REPEAT)
    param_cw128.chunks.append(WaveChunk(name_of_wavedata="cw", num_blank_word=16, num_repeat=3))
    param_cw128.chunks.append(WaveChunk(name_of_wavedata="null", num_blank_word=144, num_repeat=1))
    au.load_parameter(param_cw128)  # (16+16)*3 + (16+144)*1

    # setup capture unit
    cp0 = CapParam(num_wait_word=80, num_repeat=NUM_REPEAT)
    cp0.sections.append(CapSection(name="s0", num_capture_word=255, num_blank_word=1))  # 255+1 = 256 [word]
    cu.load_parameter(cp0)
    assert isinstance(cu, CapUnitSimplifiedHL)

    # setup trigger
    cc: AbstractCapCtrl = proxy.capctrl
    assert isinstance(cc, CapCtrlStandard) or isinstance(cc, CapCtrlClassic)
    cc.set_triggering_awgunit_idx(cu.module_index, au.unit_index)
    cc.add_triggerable_unit(cu.unit_index)

    ac: AwgCtrl = proxy.awgctrl
    assert isinstance(ac, AwgCtrlHL)
    cntr: ClockcounterCtrl = proxy.clkcntr
    sqr: SimplemultiSequencer = proxy.sqrctrl

    fut0 = cu.wait_for_triggered_capture()

    trig = SimplemultiAwgTriggers()
    cur, last_sysref = cntr.read_counter()
    logger.info(f"current counter is {cur}")
    tts = cur + DELAY_B4_START_COUNT
    logger.info(f"adding a trigger at {tts}")
    trig.append(tts, {au.unit_index})
    # Notes: confirm the successful start of awg units. actually, this is not perfect since the detection of ready bit
    #        is too difficult to implement.
    # Notes: in cases that multiple triggers are put into the queue, this kind of check doesn't work.
    sqr.add_awg_start(trig)

    fut1 = ac.wait_to_start({au.unit_index}, timeout=DELAY_B4_START_SEC * 2)
    fut1.result()
    logger.info("the units has started")

    rdr = fut0.result()
    logger.info("the capture completed")
    data = rdr.as_wave_dict()

    # Notes: confirm the completion of the wave generation
    # Notes: in cases that multiple triggers are put into the queue, this kind of check doesn't work, either.
    fut2 = ac.wait_done({au.unit_index})
    fut2.result()

    matplotlib.use("Gtk3agg")
    for i in range(NUM_REPEAT):
        find_chunks(data["s0"][i], 1500)
    plot_iqs({f"loop-{i}": data["s0"][i] for i in range(NUM_REPEAT)})
