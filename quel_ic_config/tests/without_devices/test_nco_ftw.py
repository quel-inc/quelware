import copy
import json
import logging
from typing import Any, Dict, Tuple, Union

import pytest
from pydantic.v1.utils import deep_update

from quel_ic_config.ad9082_v106 import Ad9082Config, Ad9082V106Mixin

logger = logging.getLogger(__name__)


class Ad9082V106Dummy(Ad9082V106Mixin):
    def _read_reg_cb(self, address: int) -> Tuple[bool, int]:
        return True, 0

    def _write_reg_cb(self, address: int, value: int) -> Tuple[bool]:
        return (True,)

    def initialize(
        self,
        reset: bool = False,
        link_init: bool = False,
        use_204b: bool = False,
        use_bg_cal: bool = True,
        wait_after_device_init: float = 0.1,
    ):
        self.device.dev_info.dev_freq_hz = self.param.clock.ref
        self.device.dev_info.dac_freq_hz = self.param.clock.dac
        self.device.dev_info.adc_freq_hz = self.param.clock.adc

        self._interp_cache = (
            int(self.param.dac.interpolation_rate.main),
            int(self.param.dac.interpolation_rate.channel),
        )
        self._fduc_map_cache: Union[Tuple[Tuple[int, ...], ...], None] = (
            tuple([int(i) for i in self.param.dac.channel_assign.dac0]),
            tuple([int(i) for i in self.param.dac.channel_assign.dac1]),
            tuple([int(i) for i in self.param.dac.channel_assign.dac2]),
            tuple([int(i) for i in self.param.dac.channel_assign.dac3]),
        )


@pytest.fixture(scope="session")
def ad9082_obj_4x6():
    with open("src/quel_ic_config/settings/quel-1/ad9082.json") as f:
        base_setting: Dict[str, Any] = json.load(f)

    with open("src/quel_ic_config/settings/quel-1/ad9082_dac_channel_assign_for_mxfe0.json") as f:
        additional_setting: Dict[str, Any] = json.load(f)

    setting = copy.copy(base_setting)
    setting = deep_update(setting, base_setting)
    setting = deep_update(setting, additional_setting)
    del setting["meta"]
    cfg_obj = Ad9082Config.model_validate(setting)
    ic_obj = Ad9082V106Dummy("dummy", cfg_obj)
    ic_obj.initialize()
    return ic_obj


def test_dac_cnco_ftw(ad9082_obj_4x6):
    ad9082_obj = ad9082_obj_4x6

    for i in (1, -1):
        for j in range(0, 9800):
            freq = int(i * (10 ** (j * 0.001)))
            if -6000000000 < freq < 6000000000:
                ftw_c = ad9082_obj.calc_dac_cnco_ftw(freq, fractional_mode=False)
                ftw_cf = ad9082_obj.calc_dac_cnco_ftw(freq, fractional_mode=True)

                ftw_c_ = ftw_c.to_ftw()
                ftw_cf_ = ftw_cf.to_ftw()

                freq_restored_c = ad9082_obj.calc_dac_cnco_freq(ftw_c)
                assert abs(freq_restored_c - freq) < 2.15e-5 * 2, f"(i, j, freq) = {i}, {j}, {freq}  ftw = {ftw_c:s}"

                freq_restored_cf = ad9082_obj.calc_dac_cnco_freq(ftw_cf)
                assert abs(freq_restored_cf - freq) == 0, f"(i, j, freq) = {i}, {j}, {freq}  ftw = {ftw_cf:s}"

                if i == 1:
                    assert 0 <= ftw_c_.ftw < (1 << 47)
                    assert 0 <= ftw_cf_.ftw < (1 << 47)
                else:
                    assert (1 << 47) <= ftw_c_.ftw < (1 << 48)
                    assert (1 << 47) <= ftw_cf_.ftw < (1 << 48)
            else:
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_dac_cnco_ftw(freq, fractional_mode=False)
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_dac_cnco_ftw(freq, fractional_mode=True)


def test_dac_fnco_ftw(ad9082_obj_4x6):
    ad9082_obj = ad9082_obj_4x6

    for i in (1, -1):
        for j in range(0, 900):
            freq = int(i * (10 ** (j * 0.001)))
            if -750000000 < freq < 750000000:
                ftw_c = ad9082_obj.calc_dac_fnco_ftw(freq, fractional_mode=False)
                ftw_cf = ad9082_obj.calc_dac_fnco_ftw(freq, fractional_mode=True)

                ftw_c_ = ftw_c.to_ftw()
                ftw_cf_ = ftw_cf.to_ftw()

                freq_restored_c = ad9082_obj.calc_dac_fnco_freq(ftw_c)
                assert abs(freq_restored_c - freq) < 2.15e-5 * 2, f"(i, j, freq) = {i}, {j}, {freq}  ftw = {ftw_c:s}"

                freq_restored_cf = ad9082_obj.calc_dac_fnco_freq(ftw_cf)
                assert abs(freq_restored_cf - freq) == 0, f"(i, j, freq) = {i}, {j}, {freq}  ftw = {ftw_cf:s}"

                if i == 1:
                    assert 0 <= ftw_c_.ftw < (1 << 47)
                    assert 0 <= ftw_cf_.ftw < (1 << 47)
                else:
                    assert (1 << 47) <= ftw_c_.ftw < (1 << 48)
                    assert (1 << 47) <= ftw_cf_.ftw < (1 << 48)
            else:
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_dac_fnco_ftw(freq, fractional_mode=False)
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_dac_fnco_ftw(freq, fractional_mode=True)
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_dac_fnco_ftw_rational(freq, 1, fractional_mode=False)


def test_adc_cnco_ftw(ad9082_obj_4x6):
    ad9082_obj = ad9082_obj_4x6

    for i in (1, -1):
        for j in range(0, 9800):
            freq = int(i * (10 ** (j * 0.001))) // 2
            if -3000000000 < freq < 3000000000:
                ftw_c = ad9082_obj.calc_adc_cnco_ftw(freq, fractional_mode=False)
                ftw_cf = ad9082_obj.calc_adc_cnco_ftw(freq, fractional_mode=True)

                ftw_c_ = ftw_c.to_ftw()
                ftw_cf_ = ftw_cf.to_ftw()

                freq_restored_c = ad9082_obj.calc_adc_cnco_freq(ftw_c)
                assert abs(freq_restored_c - freq) < 2.15e-5 * 2, f"(i, j, freq) = {i}, {j}, {freq}  ftw = {ftw_c:s}"

                freq_restored_cf = ad9082_obj.calc_adc_cnco_freq(ftw_cf)
                assert abs(freq_restored_cf - freq) == 0, f"(i, j, freq) = {i}, {j}, {freq}  ftw = {ftw_cf:s}"

                if i == 1:
                    assert 0 <= ftw_c_.ftw < (1 << 47)
                    assert 0 <= ftw_cf_.ftw < (1 << 47)
                else:
                    assert (1 << 47) <= ftw_c_.ftw < (1 << 48)
                    assert (1 << 47) <= ftw_cf_.ftw < (1 << 48)
            else:
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_adc_cnco_ftw(freq, fractional_mode=False)
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_adc_cnco_ftw(freq, fractional_mode=True)


def test_adc_fnco_ftw(ad9082_obj_4x6):
    ad9082_obj = ad9082_obj_4x6

    for i in (1, -1):
        for j in range(0, 900):
            freq = int(i * (10 ** (j * 0.001)))
            if -500000000 < freq < 500000000:
                ftw_c = ad9082_obj.calc_adc_fnco_ftw(freq, fractional_mode=False)
                ftw_cf = ad9082_obj.calc_adc_fnco_ftw(freq, fractional_mode=True)

                ftw_c_ = ftw_c.to_ftw()
                ftw_cf_ = ftw_cf.to_ftw()

                freq_restored_c = ad9082_obj.calc_adc_fnco_freq(ftw_c)
                assert abs(freq_restored_c - freq) < 2.15e-5 * 2, f"(i, j, freq) = {i}, {j}, {freq}  ftw = {ftw_c:s}"

                freq_restored_cf = ad9082_obj.calc_adc_fnco_freq(ftw_cf)
                assert abs(freq_restored_cf - freq) == 0, f"(i, j, freq) = {i}, {j}, {freq}  ftw = {ftw_cf:s}"

                if i == 1:
                    assert 0 <= ftw_c_.ftw < (1 << 47)
                    assert 0 <= ftw_cf_.ftw < (1 << 47)
                else:
                    assert (1 << 47) <= ftw_c_.ftw < (1 << 48)
                    assert (1 << 47) <= ftw_cf_.ftw < (1 << 48)
            else:
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_dac_fnco_ftw(freq, fractional_mode=False)
                with pytest.raises(ValueError):
                    _ = ad9082_obj.calc_dac_fnco_ftw(freq, fractional_mode=True)
