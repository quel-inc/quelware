import copy
import json
import logging
from typing import Any, Dict, Tuple

from pydantic.v1.utils import deep_update

import adi_ad9082_v170 as adi_ad9082
from quel_ic_config.ad9082 import Ad9082Config, Ad9082Mixin

with open("src/quel_ic_config/settings/quel-1/ad9082.json") as f:
    function_setting: Dict[str, Any] = json.load(f)

with open("src/quel_ic_config/settings/quel-1/ad9082_dac_channel_assign_for_mxfe0.json") as f:
    function_setting_additional: Dict[str, Any] = json.load(f)


default_setting = copy.copy(function_setting)
default_setting = deep_update(default_setting, function_setting_additional)
del default_setting["meta"]


addon_tx_mxfe0 = {
    # "apply_to": "dac",
    "channel_assign": {
        "dac0": [0],
        "dac1": [1],
        "dac2": [4, 3, 2],
        "dac3": [7, 6, 5],
    },
}

addon_tx_mxfe1 = {
    # "apply_to": "dac",
    "channel_assign": {
        "dac0": [2, 1, 0],
        "dac1": [5, 4, 3],
        "dac2": [6],
        "dac3": [7],
    },
}

addon_rx_readout = {
    # "apply_to": "adc",
    "converter_mappings": [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ],
}

addon_rx_monitor = {
    # "apply_to": "adc",
    "converter_mappings": [
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 8, 9, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ],
}

logging.basicConfig(level=logging.DEBUG)

cfg_desc = copy.copy(default_setting)
# cfg_desc['dac'].update(addon_tx_mxfe0)
# cfg_desc['adc'].update(addon_rx_monitor)
cfg_obj = Ad9082Config.model_validate(cfg_desc)


class Ad9082Dummy(Ad9082Mixin):
    def _read_reg_cb(self, address: int) -> Tuple[bool, int]:
        return True, 0

    def _write_reg_cb(self, address: int, value: int) -> Tuple[bool]:
        return (True,)

    def _reset_pin_ctrl_cb(self, level: int) -> Tuple[bool]:
        return (True,)


x = Ad9082Dummy("dummy")
# Notes: cfg_obj is already validated
# Notes: usually cfg_obj is passed to x.configure(). in the following test cases, a subtree of cfg_obj is directly
#        passed to callees of x.configure().
# x._set_spi_settings(cfg_obj.spi)  # Note: no corresponding data structure exists in x.
assert cfg_obj.spi.pin.as_cpptype() == adi_ad9082.SPI_SDO
assert cfg_obj.spi.msb.as_cpptype() == adi_ad9082.SPI_MSB_FIRST
assert cfg_obj.spi.addr_next.as_cpptype() == adi_ad9082.SPI_ADDR_INC_AUTO

x._set_serdes_settings(cfg_obj.serdes)
assert x.device.serdes_info.des_settings.boost_mask == 0xFF
assert x.device.serdes_info.des_settings.invert_mask == 0x00
assert all(x.device.serdes_info.des_settings.ctle_filter == (4, 4, 4, 4, 4, 4, 4, 4))
assert all(x.device.serdes_info.des_settings.lane_mapping[0] == (0, 1, 2, 3, 4, 5, 6, 7))
assert all(x.device.serdes_info.des_settings.lane_mapping[1] == (0, 1, 2, 3, 4, 5, 6, 7))

assert x.device.serdes_info.ser_settings.invert_mask == 0x00
for i in range(8):
    for j in range(3):
        assert x.device.serdes_info.ser_settings.lane_settings[i][j] == 1
assert all(x.device.serdes_info.ser_settings.lane_mapping[0] == (0, 1, 2, 3, 4, 5, 6, 7))
assert all(x.device.serdes_info.ser_settings.lane_mapping[1] == (0, 1, 2, 3, 4, 5, 6, 7))

param_tx = cfg_obj.dac
assert param_tx.interpolation_rate.channel == 4
assert param_tx.interpolation_rate.main == 6
assert param_tx.channel_assign.as_cpptype() == [0x01, 0x02, 0x1C, 0xE0]
assert param_tx.shift_freq.main == (1500000000, 1500000000, 1500000000, 1500000000)
assert param_tx.shift_freq.channel == (0, 0, 0, 0, 0, 0, 0, 0)
jesd204_0 = param_tx.jesd204.as_cpptype()
assert jesd204_0.l == 8
assert jesd204_0.f == 4
assert jesd204_0.m == 16
assert jesd204_0.s == 1
assert jesd204_0.hd == 0
assert jesd204_0.k == 64
assert jesd204_0.n == 16
assert jesd204_0.np == 16
assert jesd204_0.cf == 0
assert jesd204_0.cs == 0
assert jesd204_0.did == 0
assert jesd204_0.did == 0
assert jesd204_0.lid0 == 0
assert jesd204_0.subclass == 1
assert jesd204_0.scr == 1
assert jesd204_0.duallink == 0
assert jesd204_0.jesdv == 2
assert jesd204_0.mode_id == 16
assert jesd204_0.mode_c2r_en == 0
assert jesd204_0.mode_s_sel == 0
assert param_tx.lane_xbar == (0, 1, 2, 3, 4, 5, 6, 7)
assert param_tx.fullscale_current == (40527, 40527, 40527, 40527)

param_rx = cfg_obj.adc
assert param_rx.shift_freq.main == (1500000000, 1500000000, 1500000000, 1500000000)
assert param_rx.decimation_rate.main.as_cpptype() == [int(adi_ad9082.ADC_CDDC_DCM_6) for _ in range(4)]
assert param_rx.c2r_enable.main == (0, 0, 0, 0)

assert param_rx.shift_freq.channel == (0, 0, 0, 0, 0, 0, 0, 0)
assert param_rx.decimation_rate.channel.as_cpptype() == [int(adi_ad9082.ADC_FDDC_DCM_2) for _ in range(8)]

assert param_rx.c2r_enable.channel == (0, 0, 0, 0, 0, 0, 0, 0)
jesd204_1 = param_rx.jesd204[0].as_cpptype()
jesd204_2 = param_rx.jesd204[1].as_cpptype()
convmap_0 = param_rx.converter_mappings[0].as_cpptype()
convmap_1 = param_rx.converter_mappings[1].as_cpptype()
assert param_rx.lane_xbar == (0, 1, 2, 3, 4, 5, 6, 7)
