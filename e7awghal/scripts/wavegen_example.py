import logging
from typing import Any

import numpy as np

from e7awghal import AbstractQuel1Au50Hal, AwgParam, AwgUnit, E7FwType, WaveChunk
from quel_ic_config.e7resource_mapper import AbstractQuel1E7ResourceMapper, create_rmap_object
from quel_ic_config.quel1_any_config_subsystem import Quel1AnyConfigSubsystem
from quel_ic_config.quel1_box_intrinsic import _create_css_object
from quel_ic_config.quel_config_common import Quel1Feature
from testlibs.awgctrl_with_hlapi import AwgUnitHL
from testlibs.quel1au50_hal_for_test import create_quel1au50hal_for_test

logger = logging.getLogger()

if __name__ == "__main__":
    import argparse

    from quel_ic_config_utils import add_common_arguments, complete_ipaddrs

    logging.basicConfig(level=logging.WARNING, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")

    parser = argparse.ArgumentParser(description="testing e7awghal APIs with normal firmware and internal RF loopback")
    add_common_arguments(parser)
    parser.add_argument("--verbose", action="store_true", help="show verbose log")
    args = parser.parse_args()
    complete_ipaddrs(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

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

    # configure channel
    GR, LN, CH = 0, 0, 0

    css.set_lo_multiplier(group=GR, line=LN, freq_multiplier=115)
    css.set_divider_ratio(group=GR, line=LN, divide_ratio=1)
    css.set_dac_cnco(group=GR, line=LN, freq_in_hz=2.000e9)
    css.set_vatt(group=GR, line=LN, vatt=0xC00)
    css.set_sideband(group=GR, line=LN, sideband="L")
    css.set_fullscale_current(group=GR, line=LN, fsc=40000)
    css.pass_line(group=GR, line=LN)

    _: Any
    # configure awg (actually, awg should be a part of channel)
    mxfe_idx, _ = css.get_dac_idx(GR, LN)
    awg_idx = rmap.get_awg_from_fduc(*css.get_fduc_idx(GR, LN, CH))

    ac = proxy.awgctrl
    au: AwgUnit = proxy.awgunit(awg_idx)
    assert isinstance(au, AwgUnitHL)
    # -------------------------------- now ready to describe examples -----------------------------------------------

    cw = np.zeros(64, dtype=np.complex64)
    cw[:] = 32767 + 0j
    au.register_wavedata_from_complex64vector("cw", cw)

    cw_weak = np.zeros(64, dtype=np.complex64)
    cw_weak[:] = 8192 + 0j
    au.register_wavedata_from_complex64vector("cw_weak", cw_weak)

    cw_param0 = AwgParam(num_wait_word=0, num_repeat=0xFFFFFFFF)
    cw_param0.chunks.append(WaveChunk(name_of_wavedata="cw", num_blank_word=0, num_repeat=0xFFFFFFFF))

    cw_param1 = AwgParam(num_wait_word=0, num_repeat=0xFFFFFFFF)
    cw_param1.chunks.append(WaveChunk(name_of_wavedata="cw_weak", num_blank_word=0, num_repeat=0xFFFFFFFF))

    _ = input("hit any key to start emitting cw >>>")
    au.load_parameter(cw_param0)
    au.start_now().result()

    _ = input("hit any key to start emitting weak cw >>>")
    au.terminate().result()
    au.load_parameter(cw_param1)
    au.start_now().result()

    _ = input("hit any key to stop >>>")
    au.terminate().result()
