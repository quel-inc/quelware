#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "adi_ad9081_config.h"
#include "adi_ad9081_hal.h"

namespace py = pybind11;

typedef struct api_revision {
  uint8_t rev_major;
  uint8_t rev_minor;
  uint8_t rev_rc;
} api_revision_t;

typedef struct addr_data {
  uint32_t addr;
  uint8_t data;
} addr_data_t;

typedef struct nco_ftw {
  uint64_t ftw;
  uint64_t modulus_a;
  uint64_t modulus_b;
} nco_ftw_t;

typedef struct ad9081_callbacks {
  py::function regread_cb;
  py::function regwrite_cb;
  py::function delayus_cb;
  py::function logwrite_cb;
} ad9081_callbacks_t;

extern "C" {
int ad9081_callback_regread(void *handle, uint32_t address, uint8_t *value);
int ad9081_callback_regwrite(void *handle, uint32_t address, uint8_t value);
int ad9081_callback_delay_us(void *handle, uint32_t us);
int ad9081_callback_log_write(void *handle, adi_cms_log_type_e log_type,
                              const char *msg);
}

int ad9081_callback_regread(void *handle, uint32_t address, uint8_t *value) {
  assert(handle != nullptr);

  PyGILState_STATE state = PyGILState_Ensure();
  py::tuple result =
      ((ad9081_callbacks_t *)handle)->regread_cb(static_cast<int>(address));
  auto retcode = py::cast<bool>(result[0]);
  auto v = py::cast<int>(result[1]);
  PyGILState_Release(state);

  if (retcode) {
    *value = (v & 0xff);
  }
  return retcode ? 0 : 1;
}

int ad9081_callback_regwrite(void *handle, uint32_t address, uint8_t value) {
  assert(handle != nullptr);
  PyGILState_STATE state = PyGILState_Ensure();
  py::tuple result =
      ((ad9081_callbacks_t *)handle)
          ->regwrite_cb(static_cast<int>(address), static_cast<int>(value));
  auto retcode = py::cast<bool>(result[0]);
  PyGILState_Release(state);
  return retcode ? 0 : 1;
}

int ad9081_callback_delay_us(void *handle, uint32_t us) {
  assert(handle != nullptr);
  PyGILState_STATE state = PyGILState_Ensure();
  py::tuple result =
      ((ad9081_callbacks_t *)handle)->delayus_cb(static_cast<int>(us));
  auto retcode = py::cast<bool>(result[0]);
  PyGILState_Release(state);
  return retcode ? 0 : 1;
}

int ad9081_callback_log_write(void *handle, adi_cms_log_type_e log_type,
                              const char *msg) {
  assert(handle != nullptr);
  PyGILState_STATE state = PyGILState_Ensure();
  py::tuple result = ((ad9081_callbacks_t *)handle)
                         ->logwrite_cb(static_cast<int>(log_type), msg);
  auto retcode = py::cast<bool>(result[0]);
  PyGILState_Release(state);
  return retcode ? 0 : 1;
}

PYBIND11_MODULE(adi_ad9081_v106, m) {
  py::enum_<adi_cms_error_e>(m, "CmsError")
      .value("API_CMS_ERROR_OK", API_CMS_ERROR_OK, "No Error")
      .value("API_CMS_ERROR_ERROR", API_CMS_ERROR_ERROR, "General Error")
      .value("API_CMS_ERROR_NULL_PARAM", API_CMS_ERROR_NULL_PARAM,
             "Null parameter")
      .value("API_CMS_ERROR_SPI_SDO", API_CMS_ERROR_SPI_SDO,
             "Wrong value assigned to the SDO in device structure")
      .value("API_CMS_ERROR_INVALID_HANDLE_PTR",
             API_CMS_ERROR_INVALID_HANDLE_PTR,
             "Device handler pointer is invalid")
      .value("API_CMS_ERROR_INVALID_XFER_PTR", API_CMS_ERROR_INVALID_XFER_PTR,
             "Invalid pointer to the SPI xfer function assigned")
      .value("API_CMS_ERROR_INVALID_DELAYUS_PTR",
             API_CMS_ERROR_INVALID_DELAYUS_PTR,
             "Invalid pointer to the delay_us function assigned")
      .value("API_CMS_ERROR_INVALID_PARAM", API_CMS_ERROR_INVALID_PARAM,
             "Invalid parameter passed")
      .value("API_CMS_ERROR_INVALID_RESET_CTRL_PTR",
             API_CMS_ERROR_INVALID_RESET_CTRL_PTR,
             "Invalid pointer to the reset control function assigned")
      .value("API_CMS_ERROR_NOT_SUPPORTED", API_CMS_ERROR_NOT_SUPPORTED,
             "Not supported")
      .value("API_CMS_ERROR_VCO_OUT_OF_RANGE", API_CMS_ERROR_VCO_OUT_OF_RANGE,
             "The VCO is out of range")
      .value("API_CMS_ERROR_PLL_NOT_LOCKED", API_CMS_ERROR_PLL_NOT_LOCKED,
             "PLL is not locked")
      .value("API_CMS_ERROR_DLL_NOT_LOCKED", API_CMS_ERROR_DLL_NOT_LOCKED,
             "DLL is not locked")
      .value("API_CMS_ERROR_MODE_NOT_IN_TABLE", API_CMS_ERROR_MODE_NOT_IN_TABLE,
             "JESD Mode not in table")
      .value("API_CMS_ERROR_FTW_LOAD_ACK", API_CMS_ERROR_FTW_LOAD_ACK,
             "FTW acknowledge not received")
      .value("API_CMS_ERROR_NCO_NOT_ENABLED", API_CMS_ERROR_NCO_NOT_ENABLED,
             "The NCO is not enabled")
      .value("API_CMS_ERROR_INIT_SEQ_FAIL", API_CMS_ERROR_INIT_SEQ_FAIL,
             "Initialization sequence failed")
      .value("API_CMS_ERROR_TEST_FAILED", API_CMS_ERROR_TEST_FAILED,
             "Test failed")
      .value("API_CMS_ERROR_SPI_XFER", API_CMS_ERROR_SPI_XFER,
             "SPI transfer error")
      .value("API_CMS_ERROR_TX_EN_PIN_CTRL", API_CMS_ERROR_TX_EN_PIN_CTRL,
             "TX enable function error")
      .value("API_CMS_ERROR_RESET_PIN_CTRL", API_CMS_ERROR_RESET_PIN_CTRL,
             "HW reset function error")
      .value("API_CMS_ERROR_EVENT_HNDL", API_CMS_ERROR_EVENT_HNDL,
             "Event handling error")
      .value("API_CMS_ERROR_HW_OPEN", API_CMS_ERROR_HW_OPEN,
             "HW open function error")
      .value("API_CMS_ERROR_HW_CLOSE", API_CMS_ERROR_HW_CLOSE,
             "HW close function error")
      .value("API_CMS_ERROR_LOG_OPEN", API_CMS_ERROR_LOG_OPEN, "Log open error")
      .value("API_CMS_ERROR_LOG_WRITE", API_CMS_ERROR_LOG_WRITE,
             "Log write error")
      .value("API_CMS_ERROR_LOG_CLOSE", API_CMS_ERROR_LOG_CLOSE,
             "Log close error")
      .value("API_CMS_ERROR_DELAY_US", API_CMS_ERROR_DELAY_US, "Delay error")
      .export_values();

  py::class_<adi_cms_chip_id_t>(m, "CmsChipId")
      .def(py::init<>())
      .def_readonly("chip_type", &adi_cms_chip_id_t::chip_type)
      .def_readonly("prod_id", &adi_cms_chip_id_t::prod_id)
      .def_readonly("prod_grade", &adi_cms_chip_id_t::prod_grade)
      .def_readonly("dev_revision", &adi_cms_chip_id_t::dev_revision);

  py::enum_<adi_ad9081_reset_e>(m, "Reset", py::arithmetic())
      .value("SOFT_RESET", AD9081_SOFT_RESET, "Soft Reset")
      .value("HARD_RESET", AD9081_HARD_RESET, "Hard Reset")
      .value("SOFT_RESET_AND_INIT", AD9081_SOFT_RESET_AND_INIT,
             "Soft Reset Then Init")
      .value("HARD_RESET_AND_INIT", AD9081_HARD_RESET_AND_INIT,
             "Hard Reset Then Init")
      .export_values();

  py::class_<adi_ad9081_info_t>(m, "Info")
      .def(py::init<>())
      .def_readwrite("dev_freq_hz", &adi_ad9081_info_t::dev_freq_hz)
      .def_readwrite("dac_freq_hz", &adi_ad9081_info_t::dac_freq_hz)
      .def_readwrite("adc_freq_hz", &adi_ad9081_info_t::adc_freq_hz)
      .def_readwrite("dev_rev", &adi_ad9081_info_t::dev_rev);

  py::enum_<adi_ad9081_ser_swing_e>(m, "SerSwing", py::arithmetic())
      .value("SER_SWING_1000", AD9081_SER_SWING_1000, "1000 mV Swing")
      .value("SER_SWING_850", AD9081_SER_SWING_850, "850 mV Swing")
      .value("SER_SWING_750", AD9081_SER_SWING_750, "750 mV Swing")
      .value("SER_SWING_500", AD9081_SER_SWING_500, "500 mV Swing")
      .export_values();

  py::enum_<adi_ad9081_ser_pre_emp_e>(m, "SerPreEmp", py::arithmetic())
      .value("SER_PRE_EMP_0DB", AD9081_SER_PRE_EMP_0DB, "0 dB Pre-Emphasis")
      .value("SER_PRE_EMP_3DB", AD9081_SER_PRE_EMP_3DB, "3 dB Pre-Emphasis")
      .value("SER_PRE_EMP_6DB", AD9081_SER_PRE_EMP_6DB, "6 dB Pre-Emphasis")
      .export_values();

  py::enum_<adi_ad9081_ser_post_emp_e>(m, "SerPostEmp", py::arithmetic())
      .value("SER_POST_EMP_0DB", AD9081_SER_POST_EMP_0DB, "0 dB Post-Emphasis")
      .value("SER_POST_EMP_3DB", AD9081_SER_POST_EMP_3DB, "3 dB Post-Emphasis")
      .value("SER_POST_EMP_6DB", AD9081_SER_POST_EMP_6DB, "6 dB Post-Emphasis")
      .value("SER_POST_EMP_9DB", AD9081_SER_POST_EMP_9DB, "9 dB Post-Emphasis")
      .value("SER_POST_EMP_12DB", AD9081_SER_POST_EMP_12DB,
             "12 dB Post-Emphasis")
      .export_values();

  py::class_<adi_ad9081_ser_lane_settings_t>(m, "SerLaneSettings")
      .def(py::init<>())
      .def_readwrite("swing_setting",
                     &adi_ad9081_ser_lane_settings_t::swing_setting)
      .def_readwrite("pre_emp_setting",
                     &adi_ad9081_ser_lane_settings_t::pre_emp_setting)
      .def_readwrite("post_emp_setting",
                     &adi_ad9081_ser_lane_settings_t::post_emp_setting);
  PYBIND11_NUMPY_DTYPE(adi_ad9081_ser_lane_settings_t, swing_setting,
                       pre_emp_setting, post_emp_setting);
  typedef enum {
    SWING_SETTING = 0,
    PRE_EMP_SETTING = 1,
    POST_EMP_SETTING = 2
  } ser_lane_settings_field_e;
  py::enum_<ser_lane_settings_field_e>(m, "SerLaneSettingsField",
                                       py::arithmetic())
      .value("SWING_SETTING", SWING_SETTING)
      .value("PRE_EMP_SETTING", PRE_EMP_SETTING)
      .value("POST_EMP_SETTING", POST_EMP_SETTING)
      .export_values();

  py::class_<adi_ad9081_ser_settings_t>(m, "SerSettings")
      .def(py::init<>())
      .def_readwrite("invert_mask", &adi_ad9081_ser_settings_t::invert_mask)
      .def_property_readonly("lane_mapping0",
                             [](py::object &obj) {
                               auto &o =
                                   obj.cast<adi_ad9081_ser_settings_t &>();
                               return py::array{8, o.lane_mapping[0], obj};
                             })
      .def_property_readonly("lane_mapping1",
                             [](py::object &obj) {
                               auto &o =
                                   obj.cast<adi_ad9081_ser_settings_t &>();
                               return py::array{8, o.lane_mapping[1], obj};
                             })
      .def_property_readonly("lane_settings", [](py::object &obj) {
        auto &o = obj.cast<adi_ad9081_ser_settings_t &>();
        return py::array{8, o.lane_settings, obj};
      });

  py::class_<adi_ad9081_des_settings_t>(m, "DesSettings")
      .def(py::init<>())
      .def_readwrite("boost_mask", &adi_ad9081_des_settings_t::boost_mask)
      .def_readwrite("invert_mask", &adi_ad9081_des_settings_t::invert_mask)
      .def_property_readonly("ctle_filter",
                             [](py::object &obj) {
                               auto &o =
                                   obj.cast<adi_ad9081_des_settings_t &>();
                               return py::array{8, o.ctle_filter, obj};
                             })
      .def_property_readonly("lane_mapping0",
                             [](py::object &obj) {
                               auto &o =
                                   obj.cast<adi_ad9081_des_settings_t &>();
                               return py::array{8, o.lane_mapping[0], obj};
                             })
      .def_property_readonly("lane_mapping1", [](py::object &obj) {
        auto &o = obj.cast<adi_ad9081_des_settings_t &>();
        return py::array{8, o.lane_mapping[1], obj};
      });

  py::class_<adi_ad9081_serdes_settings_t>(m, "SerdesSettings")
      .def(py::init<>())
      .def_readwrite("ser_settings",
                     &adi_ad9081_serdes_settings_t::ser_settings)
      .def_readwrite("des_settings",
                     &adi_ad9081_serdes_settings_t::des_settings);

  py::enum_<adi_cms_spi_sdo_config_e>(m, "CmsSpiSdoConfig", py::arithmetic())
      .value("SPI_NONE", SPI_NONE, "keep this for test")
      .value("SPI_SDO", SPI_SDO, "SDO active, 4-wire only")
      .value("SPI_SDIO", SPI_SDIO, "SDIO active, 3-wire only")
      .export_values();

  py::enum_<adi_cms_spi_msb_config_e>(m, "CmsSpiMsbConfig", py::arithmetic())
      .value("SPI_MSB_LAST", SPI_MSB_LAST, "LSB first")
      .value("SPI_MSB_FIRST", SPI_MSB_FIRST, "MSB first")
      .export_values();

  py::enum_<adi_cms_spi_addr_inc_e>(m, "CmsSpiAddrInc", py::arithmetic())
      .value("SPI_ADDR_DEC_AUTO", SPI_ADDR_DEC_AUTO, "auto decremented")
      .value("SPI_ADDR_INC_AUTO", SPI_ADDR_INC_AUTO, "auto incremented")
      .export_values();

  py::class_<adi_ad9081_device_t>(m, "Device")
      .def(py::init<>())
      .def_readwrite("dev_info", &adi_ad9081_device_t::dev_info)
      .def_readwrite("serdes_info", &adi_ad9081_device_t::serdes_info)
      .def("spi_conf_set",
           [](adi_ad9081_device_t &self, adi_cms_spi_sdo_config_e sdo,
              adi_cms_spi_msb_config_e msb, adi_cms_spi_addr_inc_e addr_inc) {
             self.hal_info.sdo = sdo;
             self.hal_info.msb = msb;
             self.hal_info.addr_inc = addr_inc;
           })
      .def("callback_set",
           [](adi_ad9081_device_t &self, py::function regread_cb,
              py::function regwrite_cb, py::function delayus_cb,
              py::function logwrite_cb) {
             if (self.hal_info.callback_handle == nullptr) {
               self.hal_info.callback_handle =
                   (void *)(new ad9081_callbacks_t());
             }
             auto *ptr = (ad9081_callbacks_t *)(self.hal_info.callback_handle);
             ptr->regread_cb = std::move(regread_cb);
             ptr->regwrite_cb = std::move(regwrite_cb);
             ptr->delayus_cb = std::move(delayus_cb);
             ptr->logwrite_cb = std::move(logwrite_cb);
           })
      .def("callback_unset", [](adi_ad9081_device_t &self) {
        delete (ad9081_callbacks_t *)(self.hal_info.callback_handle);
        self.hal_info.callback_handle = nullptr;
      });

  py::class_<api_revision_t>(m, "ApiRevision")
      .def(py::init<>())
      .def_readonly("major", &api_revision_t::rev_major)
      .def_readonly("minor", &api_revision_t::rev_minor)
      .def_readonly("rc", &api_revision_t::rev_rc);
  m.def("device_api_revision_get",
        [](adi_ad9081_device_t *device, api_revision_t *rev) {
          return adi_ad9081_device_api_revision_get(
              device, &(rev->rev_major), &(rev->rev_minor), &(rev->rev_rc));
        });

  m.def("device_reset", adi_ad9081_device_reset);
  m.def("device_init", adi_ad9081_device_init);
  m.def("device_chip_id_get", adi_ad9081_device_chip_id_get);
  m.def("device_clk_config_set", adi_ad9081_device_clk_config_set);

  py::class_<addr_data_t>(m, "AddrData")
      .def(py::init([](uint32_t addr) {
        auto p = std::make_unique<addr_data_t>(addr_data_t{addr, 0U});
        return p;
      }))
      .def(py::init([](uint32_t addr, uint8_t data) {
        auto p = std::make_unique<addr_data_t>(addr_data_t{addr, data});
        return p;
      }))
      .def_readwrite("addr", &addr_data_t::addr)
      .def_readwrite("data", &addr_data_t::data);
  m.def("hal_reg_get", [](adi_ad9081_device_t *device, addr_data_t *reg) {
    return adi_ad9081_hal_reg_get(device, reg->addr, &(reg->data));
  });
  m.def("hal_reg_set", [](adi_ad9081_device_t *device, addr_data_t *reg) {
    // adi_ad9081_hal_reg_set() takes int32_t as its third argument.
    // however, it is fine for us to use int8_t because revision 3 chip doesn't
    // use 32bit registers.
    return adi_ad9081_hal_reg_set(device, reg->addr, reg->data);
  });
  m.def("hal_delay_us", [](adi_ad9081_device_t *device, uint32_t delay_us) {
    return adi_ad9081_hal_delay_us(device, delay_us);
  });

  py::class_<nco_ftw_t>(m, "NcoFtw")
      .def(py::init<>())
      .def_readwrite("ftw", &nco_ftw_t::ftw)
      .def_readwrite("modulus_a", &nco_ftw_t::modulus_a)
      .def_readwrite("modulus_b", &nco_ftw_t::modulus_b);
  m.def("hal_calc_tx_nco_ftw",
        [](adi_ad9081_device_t *device, uint64_t dac_freq, int64_t nco_shift,
           nco_ftw_t *nco_ftw) {
          nco_ftw->modulus_a = 0;
          nco_ftw->modulus_b = 0;
          return adi_ad9081_hal_calc_tx_nco_ftw(device, dac_freq, nco_shift,
                                                &(nco_ftw->ftw));
        });
  m.def("hal_calc_rx_nco_ftw",
        [](adi_ad9081_device_t *device, uint64_t adc_freq, int64_t nco_shift,
           nco_ftw_t *nco_ftw) {
          nco_ftw->modulus_a = 0;
          nco_ftw->modulus_b = 0;
          return adi_ad9081_hal_calc_rx_nco_ftw(device, adc_freq, nco_shift,
                                                &(nco_ftw->ftw));
        });
  m.def("hal_calc_nco_ftw", [](adi_ad9081_device_t *device, uint64_t freq,
                               int64_t nco_shift, nco_ftw_t *nco_ftw) {
    return adi_ad9081_hal_calc_nco_ftw(device, freq, nco_shift, &(nco_ftw->ftw),
                                       &(nco_ftw->modulus_a),
                                       &(nco_ftw->modulus_b));
  });
  m.def("hal_calc_nco_ftw_f", [](adi_ad9081_device_t *device, double freq,
                                 double nco_shift, nco_ftw_t *nco_ftw) {
    return adi_ad9081_hal_calc_nco_ftw_f(device, freq, nco_shift,
                                         &(nco_ftw->ftw), &(nco_ftw->modulus_a),
                                         &(nco_ftw->modulus_b));
  });

  py::enum_<adi_ad9081_dac_select_e>(m, "DacSelect", py::arithmetic())
      .value("DAC_NONE", AD9081_DAC_NONE)
      .value("DAC_0", AD9081_DAC_0)
      .value("DAC_1", AD9081_DAC_1)
      .value("DAC_2", AD9081_DAC_2)
      .value("DAC_3", AD9081_DAC_3)
      .value("DAC_ALL", AD9081_DAC_ALL)
      .export_values();
  py::enum_<adi_ad9081_dac_channel_select_e>(m, "DacChannelSelect",
                                             py::arithmetic())
      .value("DAC_CH_NONE", AD9081_DAC_CH_NONE)
      .value("DAC_CH_0", AD9081_DAC_CH_0)
      .value("DAC_CH_1", AD9081_DAC_CH_1)
      .value("DAC_CH_2", AD9081_DAC_CH_2)
      .value("DAC_CH_3", AD9081_DAC_CH_3)
      .value("DAC_CH_4", AD9081_DAC_CH_4)
      .value("DAC_CH_5", AD9081_DAC_CH_5)
      .value("DAC_CH_6", AD9081_DAC_CH_6)
      .value("DAC_CH_7", AD9081_DAC_CH_7)
      .value("DAC_CH_ALL", AD9081_DAC_CH_ALL)
      .export_values();
  m.def("dac_duc_chan_skew_set", &adi_ad9081_dac_duc_chan_skew_set);
  m.def("dac_duc_nco_reset_set", &adi_ad9081_dac_duc_nco_reset_set);
  m.def("dac_duc_nco_enable_set", &adi_ad9081_dac_duc_nco_enable_set);
  m.def("dac_duc_nco_phase_offset_set",
        &adi_ad9081_dac_duc_nco_phase_offset_set);
  m.def("dac_duc_nco_ftw_set", [](adi_ad9081_device_t *device, uint8_t dacs,
                                  uint8_t channels, nco_ftw_t *nco_ftw) {
    return adi_ad9081_dac_duc_nco_ftw_set(device, dacs, channels, nco_ftw->ftw,
                                          nco_ftw->modulus_a,
                                          nco_ftw->modulus_b);
  });

  py::enum_<adi_ad9081_dac_mode_switch_group_select_e>(
      m, "DacModeSwitchGroupSelect", py::arithmetic())
      .value("DAC_MODE_SWITCH_GROUP_NONE", AD9081_DAC_MODE_SWITCH_GROUP_NONE,
             "No Group")
      .value("DAC_MODE_SWITCH_GROUP_0", AD9081_DAC_MODE_SWITCH_GROUP_0,
             "Group 0 (DAC0 & DAC1)")
      .value("DAC_MODE_SWITCH_GROUP_1", AD9081_DAC_MODE_SWITCH_GROUP_1,
             "Group 1 (DAC2 & DAC3)")
      .value("DAC_MODE_SWITCH_GROUP_ALL", AD9081_DAC_MODE_SWITCH_GROUP_ALL,
             "All Groups")
      .export_values();
  py::enum_<adi_ad9081_dac_mode_e>(m, "DacMode", py::arithmetic())
      .value("DAC_MODE_0", AD9081_DAC_MODE_0, "I0.Q0 -> DAC0, I1.Q1 -> DAC1")
      .value("DAC_MODE_1", AD9081_DAC_MODE_1,
             "(I0 + I1) / 2 -> DAC0, (Q0 + Q1) / 2 -> DAC1, Data Path NCOs "
             "Bypassed")
      .value("DAC_MODE_2", AD9081_DAC_MODE_2,
             "I0 -> DAC0, Q0 -> DAC1, Datapath0 NCO Bypassed, Datapath1 Unused")
      .value("DAC_MODE_3", AD9081_DAC_MODE_3,
             "(I0 + I1) / 2 -> DAC0, DAC1 Output Tied To Midscale")
      .export_values();
  m.def("dac_mode_set", adi_ad9081_dac_mode_set);

  py::class_<adi_cms_jesd_param_t>(m, "AdiCmsJesdParam")
      .def(py::init<>())
      .def_readwrite("l", &adi_cms_jesd_param_t::jesd_l)
      .def_readwrite("f", &adi_cms_jesd_param_t::jesd_f)
      .def_readwrite("m", &adi_cms_jesd_param_t::jesd_m)
      .def_readwrite("s", &adi_cms_jesd_param_t::jesd_s)
      .def_readwrite("hd", &adi_cms_jesd_param_t::jesd_hd)
      .def_readwrite("k", &adi_cms_jesd_param_t::jesd_k)
      .def_readwrite("n", &adi_cms_jesd_param_t::jesd_n)
      .def_readwrite("np", &adi_cms_jesd_param_t::jesd_np)
      .def_readwrite("cf", &adi_cms_jesd_param_t::jesd_cf)
      .def_readwrite("cs", &adi_cms_jesd_param_t::jesd_cs)
      .def_readwrite("did", &adi_cms_jesd_param_t::jesd_did)
      .def_readwrite("bid", &adi_cms_jesd_param_t::jesd_bid)
      .def_readwrite("lid0", &adi_cms_jesd_param_t::jesd_lid0)
      .def_readwrite("subclass", &adi_cms_jesd_param_t::jesd_subclass)
      .def_readwrite("scr", &adi_cms_jesd_param_t::jesd_scr)
      .def_readwrite("duallink", &adi_cms_jesd_param_t::jesd_duallink)
      .def_readwrite("jesdv", &adi_cms_jesd_param_t::jesd_jesdv)
      .def_readwrite("mode_id", &adi_cms_jesd_param_t::jesd_mode_id)
      .def_readwrite("mode_c2r_en", &adi_cms_jesd_param_t::jesd_mode_c2r_en)
      .def_readwrite("mode_s_sel", &adi_cms_jesd_param_t::jesd_mode_s_sel);
  m.def(
      "device_startup_tx",
      [](adi_ad9081_device_t *device, uint8_t main_interp, uint8_t chan_interp,
         std::array<uint8_t, 4> &dac_chan, std::array<int64_t, 4> &main_shift,
         std::array<int64_t, 8> &chan_shift, adi_cms_jesd_param_t *jesd_param) {
        return adi_ad9081_device_startup_tx(device, main_interp, chan_interp,
                                            dac_chan.data(), main_shift.data(),
                                            chan_shift.data(), jesd_param);
      });

  py::enum_<adi_ad9081_jesd_link_select_e>(m, "JesdLinkSelect",
                                           py::arithmetic())
      .value("LINK_NONE", AD9081_LINK_NONE, "No Link")
      .value("LINK_0", AD9081_LINK_0, "Link 0")
      .value("LINK_1", AD9081_LINK_1, "Link 1")
      .value("LINK_ALL", AD9081_LINK_ALL, "All Links")
      .export_values();
  m.def("jesd_rx_lane_xbar_set", adi_ad9081_jesd_rx_lane_xbar_set);
  m.def("jesd_rx_link_enable_set", adi_ad9081_jesd_rx_link_enable_set);

  py::enum_<adi_ad9081_adc_coarse_ddc_select_e>(m, "AdcCoarseDdcSelect",
                                                py::arithmetic())
      .value("ADC_CDDC_NONE", AD9081_ADC_CDDC_NONE)
      .value("ADC_CDDC_0", AD9081_ADC_CDDC_0)
      .value("ADC_CDDC_1", AD9081_ADC_CDDC_1)
      .value("ADC_CDDC_2", AD9081_ADC_CDDC_2)
      .value("ADC_CDDC_3", AD9081_ADC_CDDC_3)
      .value("ADC_CDDC_ALL", AD9081_ADC_CDDC_ALL)
      .export_values();
  py::enum_<adi_ad9081_adc_fine_ddc_select_e>(m, "AdcFineDdcSelect",
                                              py::arithmetic())
      .value("ADC_FDDC_NONE", AD9081_ADC_FDDC_NONE)
      .value("ADC_FDDC_0", AD9081_ADC_FDDC_0)
      .value("ADC_FDDC_1", AD9081_ADC_FDDC_1)
      .value("ADC_FDDC_2", AD9081_ADC_FDDC_2)
      .value("ADC_FDDC_3", AD9081_ADC_FDDC_3)
      .value("ADC_FDDC_4", AD9081_ADC_FDDC_4)
      .value("ADC_FDDC_5", AD9081_ADC_FDDC_5)
      .value("ADC_FDDC_6", AD9081_ADC_FDDC_6)
      .value("ADC_FDDC_7", AD9081_ADC_FDDC_7)
      .value("ADC_FDDC_ALL", AD9081_ADC_FDDC_ALL)
      .export_values();
  py::enum_<adi_ad9081_adc_coarse_ddc_dcm_e>(m, "AdcCoarseDdcDcm")
      .value("ADC_CDDC_DCM_1", AD9081_CDDC_DCM_1)
      .value("ADC_CDDC_DCM_2", AD9081_CDDC_DCM_2)
      .value("ADC_CDDC_DCM_3", AD9081_CDDC_DCM_3)
      .value("ADC_CDDC_DCM_4", AD9081_CDDC_DCM_4)
      .value("ADC_CDDC_DCM_6", AD9081_CDDC_DCM_6)
      .value("ADC_CDDC_DCM_8", AD9081_CDDC_DCM_8)
      .value("ADC_CDDC_DCM_9", AD9081_CDDC_DCM_9)
      .value("ADC_CDDC_DCM_12", AD9081_CDDC_DCM_12)
      .value("ADC_CDDC_DCM_16", AD9081_CDDC_DCM_16)
      .value("ADC_CDDC_DCM_18", AD9081_CDDC_DCM_18)
      .value("ADC_CDDC_DCM_24", AD9081_CDDC_DCM_24)
      .value("ADC_CDDC_DCM_36", AD9081_CDDC_DCM_36)
      .export_values();
  py::enum_<adi_ad9081_adc_fine_ddc_dcm_e>(m, "AdcFineDdcDcm")
      .value("ADC_FDDC_DCM_1", AD9081_FDDC_DCM_1)
      .value("ADC_FDDC_DCM_2", AD9081_FDDC_DCM_2)
      .value("ADC_FDDC_DCM_3", AD9081_FDDC_DCM_3)
      .value("ADC_FDDC_DCM_4", AD9081_FDDC_DCM_4)
      .value("ADC_FDDC_DCM_6", AD9081_FDDC_DCM_6)
      .value("ADC_FDDC_DCM_8", AD9081_FDDC_DCM_8)
      .value("ADC_FDDC_DCM_12", AD9081_FDDC_DCM_12)
      .value("ADC_FDDC_DCM_16", AD9081_FDDC_DCM_16)
      .value("ADC_FDDC_DCM_24", AD9081_FDDC_DCM_24)
      .export_values();
  py::class_<adi_ad9081_jtx_conv_sel_t>(m, "JtxConvSel")
      .def(py::init<>())
      .def_readwrite("virtual_converter0_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter0_index)
      .def_readwrite("virtual_converter1_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter1_index)
      .def_readwrite("virtual_converter2_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter2_index)
      .def_readwrite("virtual_converter3_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter3_index)
      .def_readwrite("virtual_converter4_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter4_index)
      .def_readwrite("virtual_converter5_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter5_index)
      .def_readwrite("virtual_converter6_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter6_index)
      .def_readwrite("virtual_converter7_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter7_index)
      .def_readwrite("virtual_converter8_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter8_index)
      .def_readwrite("virtual_converter9_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converter9_index)
      .def_readwrite("virtual_convertera_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_convertera_index)
      .def_readwrite("virtual_converterb_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converterb_index)
      .def_readwrite("virtual_converterc_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converterc_index)
      .def_readwrite("virtual_converterd_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converterd_index)
      .def_readwrite("virtual_convertere_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_convertere_index)
      .def_readwrite("virtual_converterf_index",
                     &adi_ad9081_jtx_conv_sel_t::virtual_converterf_index);
  m.def(
      "device_startup_rx",
      [](adi_ad9081_device_t *device, uint8_t cddcs, uint8_t fddcs,
         std::array<int64_t, 4> &cddc_shift, std::array<int64_t, 8> &fddc_shift,
         std::array<uint8_t, 4> &cddc_dcm, std::array<uint8_t, 8> &fddc_dcm,
         std::array<uint8_t, 4> &cc2r_en, std::array<uint8_t, 8> &fc2r_en,
         std::array<adi_cms_jesd_param_t, 2> &jesd_param,
         std::array<adi_ad9081_jtx_conv_sel_t, 2> &jesd_conv_sel) {
        return adi_ad9081_device_startup_rx(
            device, cddcs, fddcs, cddc_shift.data(), fddc_shift.data(),
            cddc_dcm.data(), fddc_dcm.data(), cc2r_en.data(), fc2r_en.data(),
            jesd_param.data(), jesd_conv_sel.data());
      });

  py::enum_<adi_ad9081_adc_nco_mode_e>(m, "AdcNcoMode", py::arithmetic())
      .value("ADC_NCO_VIF", AD9081_ADC_NCO_VIF, "Variable IF Mode")
      .value("ADC_NCO_ZIF", AD9081_ADC_NCO_ZIF, "Zero IF Mode")
      .value("ADC_NCO_FS_4_IF", AD9081_ADC_NCO_FS_4_IF, "Fs/4 Hz IF Mode")
      .value("ADC_NCO_TEST", AD9081_ADC_NCO_TEST, "Test Mode")
      .export_values();

  m.def("adc_ddc_coarse_nco_mode_set", &adi_ad9081_adc_ddc_coarse_nco_mode_set);
  m.def("adc_ddc_coarse_nco_ftw_set", [](adi_ad9081_device_t *device,
                                         uint8_t cddcs, nco_ftw_t *nco_ftw) {
    return adi_ad9081_adc_ddc_coarse_nco_ftw_set(
        device, cddcs, nco_ftw->ftw, nco_ftw->modulus_a, nco_ftw->modulus_b);
  });
  m.def("adc_ddc_coarse_nco_ftw_get",
        [](adi_ad9081_device_t *device, uint8_t cddc, nco_ftw_t *nco_ftw) {
          return adi_ad9081_adc_ddc_coarse_nco_ftw_get(
              device, cddc, &(nco_ftw->ftw), &(nco_ftw->modulus_a),
              &(nco_ftw->modulus_b));
        });
  m.def("adc_ddc_fine_nco_mode_set", &adi_ad9081_adc_ddc_fine_nco_mode_set);
  m.def("adc_ddc_fine_nco_ftw_set", [](adi_ad9081_device_t *device,
                                       uint8_t fddcs, nco_ftw_t *nco_ftw) {
    return adi_ad9081_adc_ddc_fine_nco_ftw_set(
        device, fddcs, nco_ftw->ftw, nco_ftw->modulus_a, nco_ftw->modulus_b);
  });
  m.def("adc_ddc_fine_nco_ftw_get",
        [](adi_ad9081_device_t *device, uint8_t fddc, nco_ftw_t *nco_ftw) {
          return adi_ad9081_adc_ddc_fine_nco_ftw_get(
              device, fddc, &(nco_ftw->ftw), &(nco_ftw->modulus_a),
              &(nco_ftw->modulus_b));
        });

  m.def("jesd_tx_lanes_xbar_set",
        [](adi_ad9081_device_t *device, adi_ad9081_jesd_link_select_e links,
           std::array<uint8_t, 8> &logical_lanes) {
          return adi_ad9081_jesd_tx_lanes_xbar_set(device, links,
                                                   logical_lanes.data());
        });
  m.def("jesd_tx_lids_cfg_set",
        [](adi_ad9081_device_t *device, adi_ad9081_jesd_link_select_e links,
           std::array<uint8_t, 8> &lids) {
          return adi_ad9081_jesd_tx_lids_cfg_set(device, links, lids.data());
        });
  m.def("jesd_tx_link_enable_set", adi_ad9081_jesd_tx_link_enable_set);
}
