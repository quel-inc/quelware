#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "adi_ad9082_config.h"
#include "adi_ad9082_hal.h"
extern "C" {
int32_t adi_ad9082_dac_mode_switch_group_select_set(
    adi_ad9082_device_t *device, adi_ad9082_dac_pair_select_e dac_pair);
}

namespace py = pybind11;

typedef struct ad9081_callbacks {
  py::function regread_cb;
  py::function regwrite_cb;
  py::function delayus_cb;
  py::function logwrite_cb;
  py::function resetpinctrl_cb;
} ad9081_callbacks_t;

extern "C" {
int ad9081_callback_spi_xfer(void *handle, uint8_t *in_data, uint8_t *out_data,
                             uint32_t size_bytes);
int ad9081_callback_delay_us(void *handle, uint32_t us);
int ad9081_callback_log_write(void *handle, int32_t log_type,
                              const char *message, va_list argp);
int ad9081_callback_reset_pin_ctrl(void *handle, uint8_t enable);
}

int ad9081_callback_spi_xfer(void *handle, uint8_t *in_data, uint8_t *out_data,
                             uint32_t size_bytes) {
  assert(handle != nullptr);
  assert(size_bytes == 3);
  bool retcode = false;
  int address = static_cast<int>(((in_data[0] << 8) + in_data[1]) & 0x7FFF);

  PyGILState_STATE state = PyGILState_Ensure();
  // Notes: the design of the underlying CoAP API is not well-suited for the
  // ADI's API.
  if (in_data[0] & 0x80) {
    py::tuple result = ((ad9081_callbacks_t *)handle)->regread_cb(address);
    retcode = py::cast<bool>(result[0]);
    out_data[2] = py::cast<int>(result[1]) & 0xff;
  } else {
    py::tuple result = ((ad9081_callbacks_t *)handle)
                           ->regwrite_cb(address, static_cast<int>(in_data[2]));
    retcode = py::cast<bool>(result[0]);
  }

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

int ad9081_callback_log_write(void *handle, int32_t log_type,
                              const char *message, va_list argp) {
  assert(handle != nullptr);
  PyGILState_STATE state = PyGILState_Ensure();

  char buf[1024];
  vsnprintf(buf, sizeof(buf), message, argp);
  py::tuple result = ((ad9081_callbacks_t *)handle)
                         ->logwrite_cb(static_cast<int>(log_type), buf);
  auto retcode = py::cast<bool>(result[0]);
  PyGILState_Release(state);
  return retcode ? 0 : 1;
}

int ad9081_callback_reset_pin_ctrl(void *handle, uint8_t enable) {
  assert(handle != nullptr);
  PyGILState_STATE state = PyGILState_Ensure();
  py::tuple result =
      ((ad9081_callbacks_t *)handle)->resetpinctrl_cb(static_cast<int>(enable));
  auto retcode = py::cast<bool>(result[0]);
  PyGILState_Release(state);
  return retcode ? 0 : 1;
}

typedef struct api_revision {
  uint8_t rev_major;
  uint8_t rev_minor;
  uint8_t rev_rc;
} api_revision_t;

PYBIND11_MODULE(adi_ad9082_v161, m) {
  /* "exporting adi_cms_api_common.h" */
  // clang-format off
  py::enum_<adi_cms_error_e>(m, "CmsError")
      .value("API_CMS_ERROR_OK", API_CMS_ERROR_OK, "No Error")
      .value("API_CMS_ERROR_ERROR", API_CMS_ERROR_ERROR, "General Error")
      .value("API_CMS_ERROR_NULL_PARAM", API_CMS_ERROR_NULL_PARAM, "Null parameter")
      .value("API_CMS_ERROR_SPI_SDO", API_CMS_ERROR_SPI_SDO,"Wrong value assigned to the SDO in device structure")
      .value("API_CMS_ERROR_INVALID_HANDLE_PTR",API_CMS_ERROR_INVALID_HANDLE_PTR,"Device handler pointer is invalid")
      .value("API_CMS_ERROR_INVALID_XFER_PTR", API_CMS_ERROR_INVALID_XFER_PTR, "Invalid pointer to the SPI xfer function assigned")
      .value("API_CMS_ERROR_INVALID_DELAYUS_PTR", API_CMS_ERROR_INVALID_DELAYUS_PTR, "Invalid pointer to the delay_us function assigned")
      .value("API_CMS_ERROR_INVALID_PARAM", API_CMS_ERROR_INVALID_PARAM, "Invalid parameter passed")
      .value("API_CMS_ERROR_INVALID_RESET_CTRL_PTR",API_CMS_ERROR_INVALID_RESET_CTRL_PTR,"Invalid pointer to the reset control function assigned")
      .value("API_CMS_ERROR_NOT_SUPPORTED", API_CMS_ERROR_NOT_SUPPORTED,"Not supported")
      .value("API_CMS_ERROR_VCO_OUT_OF_RANGE", API_CMS_ERROR_VCO_OUT_OF_RANGE,"The VCO is out of range")
      .value("API_CMS_ERROR_PLL_NOT_LOCKED", API_CMS_ERROR_PLL_NOT_LOCKED,"PLL is not locked")
      .value("API_CMS_ERROR_DLL_NOT_LOCKED", API_CMS_ERROR_DLL_NOT_LOCKED,"DLL is not locked")
      .value("API_CMS_ERROR_MODE_NOT_IN_TABLE", API_CMS_ERROR_MODE_NOT_IN_TABLE, "JESD Mode not in table")
      .value("API_CMS_ERROR_JESD_PLL_NOT_LOCKED", API_CMS_ERROR_JESD_PLL_NOT_LOCKED, "PD STBY function error")
      .value("API_CMS_ERROR_JESD_SYNC_NOT_DONE",API_CMS_ERROR_JESD_SYNC_NOT_DONE, "JESD_SYNC_NOT_DONE")
      .value("API_CMS_ERROR_FTW_LOAD_ACK", API_CMS_ERROR_FTW_LOAD_ACK, "FTW acknowledge not received")
      .value("API_CMS_ERROR_NCO_NOT_ENABLED", API_CMS_ERROR_NCO_NOT_ENABLED,"The NCO is not enabled")
      .value("API_CMS_ERROR_INIT_SEQ_FAIL", API_CMS_ERROR_INIT_SEQ_FAIL,"Initialization sequence failed")
      .value("API_CMS_ERROR_TEST_FAILED", API_CMS_ERROR_TEST_FAILED,"Test failed")
      .value("API_CMS_ERROR_SPI_XFER", API_CMS_ERROR_SPI_XFER,"SPI transfer error")
      .value("API_CMS_ERROR_TX_EN_PIN_CTRL", API_CMS_ERROR_TX_EN_PIN_CTRL,"TX enable function error")
      .value("API_CMS_ERROR_RESET_PIN_CTRL", API_CMS_ERROR_RESET_PIN_CTRL,"HW reset function error")
      .value("API_CMS_ERROR_EVENT_HNDL", API_CMS_ERROR_EVENT_HNDL,"Event handling error")
      .value("API_CMS_ERROR_HW_OPEN", API_CMS_ERROR_HW_OPEN,"HW open function error")
      .value("API_CMS_ERROR_HW_CLOSE", API_CMS_ERROR_HW_CLOSE,"HW close function error")
      .value("API_CMS_ERROR_LOG_OPEN", API_CMS_ERROR_LOG_OPEN, "Log open error")
      .value("API_CMS_ERROR_LOG_WRITE", API_CMS_ERROR_LOG_WRITE,"Log write error")
      .value("API_CMS_ERROR_LOG_CLOSE", API_CMS_ERROR_LOG_CLOSE,"Log close error")
      .value("API_CMS_ERROR_DELAY_US", API_CMS_ERROR_DELAY_US, "Delay error")
      .value("API_CMS_ERROR_PD_STBY_PIN_CTRL", API_CMS_ERROR_PD_STBY_PIN_CTRL,"STBY function error")
      .value("API_CMS_ERROR_SYSREF_CTRL", API_CMS_ERROR_SYSREF_CTRL, "SYSREF enable function error")
      .export_values();
  // clang-format on

  py::class_<adi_cms_chip_id_t>(m, "CmsChipId")
      .def(py::init<>())
      .def_readonly("chip_type", &adi_cms_chip_id_t::chip_type)
      .def_readonly("prod_id", &adi_cms_chip_id_t::prod_id)
      .def_readonly("prod_grade", &adi_cms_chip_id_t::prod_grade)
      .def_readonly("dev_revision", &adi_cms_chip_id_t::dev_revision);

  py::class_<adi_cms_reg_data_t>(m, "RegData")
      .def(py::init([](uint16_t addr) {
        auto p =
            std::make_unique<adi_cms_reg_data_t>(adi_cms_reg_data_t{addr, 0U});
        return p;
      }))
      .def(py::init([](uint16_t addr, uint8_t data) {
        auto p = std::make_unique<adi_cms_reg_data_t>(
            adi_cms_reg_data_t{addr, data});
        return p;
      }))
      .def_readwrite("addr", &adi_cms_reg_data_t::reg)
      .def_readwrite("data", &adi_cms_reg_data_t::val);

  py::enum_<adi_cms_spi_sdo_config_e>(m, "CmsSpiSdoConfig")
      .value("SPI_NONE", SPI_NONE, "keep this for test")
      .value("SPI_SDO", SPI_SDO, "SDO active, 4-wire only")
      .value("SPI_SDIO", SPI_SDIO, "SDIO active, 3-wire only")
      .export_values();

  py::enum_<adi_cms_spi_msb_config_e>(m, "CmsSpiMsbConfig")
      .value("SPI_MSB_LAST", SPI_MSB_LAST, "LSB first")
      .value("SPI_MSB_FIRST", SPI_MSB_FIRST, "MSB first")
      .export_values();

  py::enum_<adi_cms_spi_addr_inc_e>(m, "CmsSpiAddrInc")
      .value("SPI_ADDR_DEC_AUTO", SPI_ADDR_DEC_AUTO, "auto decremented")
      .value("SPI_ADDR_INC_AUTO", SPI_ADDR_INC_AUTO, "auto incremented")
      .export_values();

  py::enum_<adi_cms_jesd_link_e>(m, "CmdJesdLink", py::arithmetic())
      .value("JESD_LINK_NONE", JESD_LINK_NONE, "JESD link none")
      .value("JESD_LINK_0", JESD_LINK_0, "JESD link 0")
      .value("JESD_LINK_1", JESD_LINK_1, "JESD link 1")
      .value("JESD_LINK_ALL", JESD_LINK_ALL, "ALL JESD links")
      .export_values();

  py::enum_<adi_cms_jesd_syncoutb_e>(m, "CmsJesdSyncoutb", py::arithmetic())
      .value("SYNCOUTB_0", SYNCOUTB_0, "SYNCOUTB 0")
      .value("SYNCOUTB_1", SYNCOUTB_1, "SYNCOUTB 1")
      .value("SYNCOUTB_ALL", SYNCOUTB_ALL, "ALL SYNCOUTB SIGNALS")
      .export_values();

  py::enum_<adi_cms_jesd_sysref_mode_e>(m, "CmsJesdSysrefMode")
      .value("SYSREF_NONE", SYSREF_NONE, "No SYSREF SUPPORT")
      .value("SYSREF_ONESHOT", SYSREF_ONESHOT, "ONE-SHOT SYSREF")
      .value("SYSREF_CONT", SYSREF_CONT, "Continuous SysRef sync.")
      .value("SYSREF_MON", SYSREF_MON, "SYSREF monitor mode")
      .value("SYSREF_MODE_INVALID", SYSREF_MODE_INVALID, "")
      .export_values();

  py::enum_<adi_cms_jesd_prbs_pattern_e>(m, "CmsJesdPrbsPattern")
      .value("PRBS_NONE", PRBS_NONE, "PRBS off")
      .value("PRBS7", PRBS7, "PRBS7 pattern")
      .value("PRBS9", PRBS9, "PRBS9 pattern")
      .value("PRBS15", PRBS15, "PRBS15 pattern")
      .value("PRBS23", PRBS23, "PRBS23 pattern")
      .value("PRBS31", PRBS31, "PRBS31 pattern")
      .value("PRBS_MAX", PRBS_MAX, "Number of member")
      .export_values();

  py::enum_<adi_cms_jesd_subclass_e>(m, "CmsJesdSubclass")
      .value("JESD_SUBCLASS_0", JESD_SUBCLASS_0, "JESD SUBCLASS 0")
      .value("JESD_SUBCLASS_1", JESD_SUBCLASS_1, "JESD SUBCLASS 1")
      .value("JESD_SUBCLASS_INVALID", JESD_SUBCLASS_INVALID, "")
      .export_values();

  py::class_<adi_cms_jesd_param_t>(m, "CmsJesdParam")
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

  py::enum_<adi_cms_chip_op_mode_t>(m, "CmsChioOpMode", py::arithmetic())
      .value("TX_ONLY", TX_ONLY, "Chip using Tx path only")
      .value("RX_ONLY", RX_ONLY, "Chip using Rx path only")
      .value("TX_RX_ONLY", TX_RX_ONLY, "Chip using Tx + Rx both paths")
      .export_values();

  py::class_<adi_ad9082_info_t>(m, "Info")
      .def(py::init<>())
      .def_readwrite("dev_freq_hz", &adi_ad9082_info_t::dev_freq_hz)
      .def_readwrite("dac_freq_hz", &adi_ad9082_info_t::dac_freq_hz)
      .def_readwrite("adc_freq_hz", &adi_ad9082_info_t::adc_freq_hz)
      .def_readwrite("dev_rev", &adi_ad9082_info_t::dev_rev)
      .def_readwrite("jesd_rx_lane_rate",
                     &adi_ad9082_info_t::jesd_rx_lane_rate);

  py::enum_<adi_ad9082_ser_swing_e>(m, "SerSwing")
      .value("SER_SWING_1000", AD9082_SER_SWING_1000, "1000 mV Swing")
      .value("SER_SWING_850", AD9082_SER_SWING_850, "850 mV Swing")
      .value("SER_SWING_750", AD9082_SER_SWING_750, "750 mV Swing")
      .value("SER_SWING_500", AD9082_SER_SWING_500, "500 mV Swing")
      .export_values();

  py::enum_<adi_ad9082_ser_pre_emp_e>(m, "SerPreEmp")
      .value("SER_PRE_EMP_0DB", AD9082_SER_PRE_EMP_0DB, "0 dB Pre-Emphasis")
      .value("SER_PRE_EMP_3DB", AD9082_SER_PRE_EMP_3DB, "3 dB Pre-Emphasis")
      .value("SER_PRE_EMP_6DB", AD9082_SER_PRE_EMP_6DB, "6 dB Pre-Emphasis")
      .export_values();

  py::enum_<adi_ad9082_ser_post_emp_e>(m, "SerPostEmp")
      .value("SER_POST_EMP_0DB", AD9082_SER_POST_EMP_0DB, "0 dB Post-Emphasis")
      .value("SER_POST_EMP_3DB", AD9082_SER_POST_EMP_3DB, "3 dB Post-Emphasis")
      .value("SER_POST_EMP_6DB", AD9082_SER_POST_EMP_6DB, "6 dB Post-Emphasis")
      .value("SER_POST_EMP_9DB", AD9082_SER_POST_EMP_9DB, "9 dB Post-Emphasis")
      .value("SER_POST_EMP_12DB", AD9082_SER_POST_EMP_12DB,
             "12 dB Post-Emphasis")
      .export_values();

  typedef enum ad9082_des_ctle_il {
    AD9082_DES_CTLE_IL_4DB_10DB = 1,
    AD9082_DES_CTLE_IL_0DB_6DB = 2,
    AD9082_DES_CTLE_IL_LT_0DB = 3,
    AD9082_DES_CTLE_IL_FAR_LT_0DB = 4,
  } ad9082_des_ctle_il_e;
  py::enum_<ad9082_des_ctle_il_e>(m, "DesCtleInsertionLoss")
      .value("DES_CTLE_IL_4DB_10DB", AD9082_DES_CTLE_IL_4DB_10DB,
             "Insertion Loss from 4dB to 10dB")
      .value("DES_CTLE_IL_0DB_6DB", AD9082_DES_CTLE_IL_0DB_6DB,
             "Insertion Loss from 0dB to 6dB")
      .value("DES_CTLE_IL_LT_0DB", AD9082_DES_CTLE_IL_LT_0DB,
             "Insertion Loss less than 0dB")
      .value("DES_CTLE_IL_FAR_LT_0DB", AD9082_DES_CTLE_IL_FAR_LT_0DB,
             "Insertion Loss far less than 0dB")
      .export_values();

  py::enum_<adi_ad9082_cal_mode_e>(m, "CalMode")
      .value("AD9082_CAL_MODE_RUN", AD9082_CAL_MODE_RUN,
             "Run 204C QR Calibration")
      .value("AD9082_CAL_MODE_RUN_AND_SAVE", AD9082_CAL_MODE_RUN_AND_SAVE,
             "Run 204C QR Calibration and save CTLE Coefficients")
      .value("AD9082_CAL_MODE_BYPASS", AD9082_CAL_MODE_BYPASS,
             "Bypass 204C QR Calibration and load CTLE Coefficients")
      .export_values();

  py::class_<adi_ad9082_ser_lane_settings_t>(m, "SerLaneSettings")
      .def(py::init<>())
      .def_readwrite("swing_setting",
                     &adi_ad9082_ser_lane_settings_t::swing_setting)
      .def_readwrite("pre_emp_setting",
                     &adi_ad9082_ser_lane_settings_t::pre_emp_setting)
      .def_readwrite("post_emp_setting",
                     &adi_ad9082_ser_lane_settings_t::post_emp_setting);
  PYBIND11_NUMPY_DTYPE(adi_ad9082_ser_lane_settings_t, swing_setting,
                       pre_emp_setting, post_emp_setting);

  typedef enum {
    SWING_SETTING = 0,
    PRE_EMP_SETTING = 1,
    POST_EMP_SETTING = 2
  } ser_lane_settings_field_e;

  py::enum_<ser_lane_settings_field_e>(m, "SerLaneSettingsField")
      .value("SWING_SETTING", SWING_SETTING)
      .value("PRE_EMP_SETTING", PRE_EMP_SETTING)
      .value("POST_EMP_SETTING", POST_EMP_SETTING)
      .export_values();

  py::class_<adi_ad9082_ser_settings_t>(m, "SerSettings")
      .def(py::init<>())
      .def_readwrite("invert_mask", &adi_ad9082_ser_settings_t::invert_mask)
      .def_property_readonly("lane_mapping",
                             [](py::object &obj) {
                               auto &o =
                                   obj.cast<adi_ad9082_ser_settings_t &>();
                               return py::array{2, o.lane_mapping, obj};
                             })
      .def_property_readonly("lane_settings", [](py::object &obj) {
        auto &o = obj.cast<adi_ad9082_ser_settings_t &>();
        return py::array{8, o.lane_settings, obj};
      });

  py::class_<adi_ad9082_des_settings_t>(m, "DesSettings")
      .def(py::init<>())
      .def_readwrite("boost_mask", &adi_ad9082_des_settings_t::boost_mask)
      .def_readwrite("invert_mask", &adi_ad9082_des_settings_t::invert_mask)
      .def_property_readonly("ctle_filter",
                             [](py::object &obj) {
                               auto &o =
                                   obj.cast<adi_ad9082_des_settings_t &>();
                               return py::array{8, o.ctle_filter, obj};
                             })
      .def_readwrite("cal_mode", &adi_ad9082_des_settings_t::cal_mode)
      .def_property_readonly("ctle_coeffs",
                             [](py::object &obj) {
                               auto &o =
                                   obj.cast<adi_ad9082_des_settings_t &>();
                               return py::array{8, o.ctle_coeffs, obj};
                             })
      .def_property_readonly("lane_mapping", [](py::object &obj) {
        auto &o = obj.cast<adi_ad9082_des_settings_t &>();
        return py::array{2, o.lane_mapping, obj};
      });

  py::class_<adi_ad9082_serdes_settings_t>(m, "SerdesSettings")
      .def(py::init<>())
      .def_readwrite("ser_settings",
                     &adi_ad9082_serdes_settings_t::ser_settings)
      .def_readwrite("des_settings",
                     &adi_ad9082_serdes_settings_t::des_settings);

  py::class_<adi_ad9082_clk_t>(m, "Clk")
      .def(py::init<>())
      .def_readwrite("sysref_mode", &adi_ad9082_clk_t::sysref_mode);

  py::class_<adi_ad9082_device_t>(m, "Device")
      .def(py::init<>())
      .def_readwrite("dev_info", &adi_ad9082_device_t::dev_info)
      .def_readwrite("serdes_info", &adi_ad9082_device_t::serdes_info)
      .def_readwrite("clk_info", &adi_ad9082_device_t::clk_info)
      .def("spi_conf_set",
           [](adi_ad9082_device_t &self, adi_cms_spi_sdo_config_e sdo,
              adi_cms_spi_msb_config_e msb, adi_cms_spi_addr_inc_e addr_inc) {
             self.hal_info.sdo = sdo;
             self.hal_info.msb = msb;
             self.hal_info.addr_inc = addr_inc;
           })
      .def("callback_set",
           [](adi_ad9082_device_t &self, py::function regread_cb,
              py::function regwrite_cb, py::function delayus_cb,
              py::function logwrite_cb, py::function resetpinctrl_cb) {
             if (self.hal_info.user_data == nullptr) {
               self.hal_info.user_data = (void *)(new ad9081_callbacks_t());
             }
             auto *ptr = (ad9081_callbacks_t *)(self.hal_info.user_data);
             ptr->regread_cb = std::move(regread_cb);
             ptr->regwrite_cb = std::move(regwrite_cb);
             ptr->delayus_cb = std::move(delayus_cb);
             ptr->logwrite_cb = std::move(logwrite_cb);
             ptr->resetpinctrl_cb = std::move(resetpinctrl_cb);
             self.hal_info.spi_xfer = ad9081_callback_spi_xfer;
             self.hal_info.delay_us = ad9081_callback_delay_us;
             self.hal_info.log_write = ad9081_callback_log_write;
             self.hal_info.reset_pin_ctrl = ad9081_callback_reset_pin_ctrl;
           })
      .def("callback_unset", [](adi_ad9082_device_t &self) {
        self.hal_info.spi_xfer = nullptr;
        self.hal_info.delay_us = nullptr;
        self.hal_info.log_write = nullptr;
        self.hal_info.reset_pin_ctrl = nullptr;
        delete (ad9081_callbacks_t *)(self.hal_info.user_data);
        self.hal_info.user_data = nullptr;
      });

  m.def("hal_reg_get",
        [](adi_ad9082_device_t *device, adi_cms_reg_data_t *reg) {
          return adi_ad9082_hal_reg_get(device, reg->reg, &(reg->val));
        });
  m.def("hal_reg_set",
        [](adi_ad9082_device_t *device, adi_cms_reg_data_t *reg) {
          // adi_ad9081_hal_reg_set() takes int32_t as its third argument.
          // however, it is fine for us to use int8_t because revision 3 chip
          // doesn't use 32bit registers.
          return adi_ad9082_hal_reg_set(device, reg->reg, reg->val);
        });
  m.def("hal_delay_us", [](adi_ad9082_device_t *device, uint32_t delay_us) {
    return adi_ad9082_hal_delay_us(device, delay_us);
  });
  m.def("hal_reset_pin_ctrl", [](adi_ad9082_device_t *device, uint8_t enable) {
    return adi_ad9082_hal_reset_pin_ctrl(device, enable);
  });

  // DEVICE top-level
  py::class_<api_revision_t>(m, "ApiRevision")
      .def(py::init<>())
      .def_readonly("major", &api_revision_t::rev_major)
      .def_readonly("minor", &api_revision_t::rev_minor)
      .def_readonly("rc", &api_revision_t::rev_rc);
  m.def("device_api_revision_get",
        [](adi_ad9082_device_t *device, api_revision_t *rev) {
          return adi_ad9082_device_api_revision_get(
              device, &(rev->rev_major), &(rev->rev_minor), &(rev->rev_rc));
        });

  py::enum_<adi_ad9082_reset_e>(m, "Reset", py::arithmetic())
      .value("SOFT_RESET", AD9082_SOFT_RESET, "Soft Reset")
      .value("HARD_RESET", AD9082_HARD_RESET, "Hard Reset")
      .value("SOFT_RESET_AND_INIT", AD9082_SOFT_RESET_AND_INIT,
             "Soft Reset Then Init")
      .value("HARD_RESET_AND_INIT", AD9082_HARD_RESET_AND_INIT,
             "Hard Reset Then Init")
      .export_values();
  m.def("device_reset", adi_ad9082_device_reset);
  m.def("device_init", adi_ad9082_device_init);
  m.def("device_clk_config_set", adi_ad9082_device_clk_config_set);

  // DAC primitives
  py::enum_<adi_ad9082_dac_select_e>(m, "DacSelect", py::arithmetic())
      .value("DAC_NONE", AD9082_DAC_NONE)
      .value("DAC_0", AD9082_DAC_0)
      .value("DAC_1", AD9082_DAC_1)
      .value("DAC_2", AD9082_DAC_2)
      .value("DAC_3", AD9082_DAC_3)
      .value("DAC_ALL", AD9082_DAC_ALL)
      .export_values();
  py::enum_<adi_ad9082_dac_channel_select_e>(m, "DacChannelSelect",
                                             py::arithmetic())
      .value("DAC_CH_NONE", AD9082_DAC_CH_NONE)
      .value("DAC_CH_0", AD9082_DAC_CH_0)
      .value("DAC_CH_1", AD9082_DAC_CH_1)
      .value("DAC_CH_2", AD9082_DAC_CH_2)
      .value("DAC_CH_3", AD9082_DAC_CH_3)
      .value("DAC_CH_4", AD9082_DAC_CH_4)
      .value("DAC_CH_5", AD9082_DAC_CH_5)
      .value("DAC_CH_6", AD9082_DAC_CH_6)
      .value("DAC_CH_7", AD9082_DAC_CH_7)
      .value("DAC_CH_ALL", AD9082_DAC_CH_ALL)
      .export_values();
  py::enum_<adi_ad9082_dac_pair_select_e>(m, "DacPairSelect", py::arithmetic())
      .value("DAC_PAIR_NONE", AD9082_DAC_PAIR_NONE, "No Group")
      .value("DAC_PAIR_0_1", AD9082_DAC_PAIR_0_1, "Group 0 (DAC0 & DAC1)")
      .value("DAC_PAIR_2_3", AD9082_DAC_PAIR_2_3, "Group 1 (DAC2 & DAC3)")
      .value("DAC_PAIR_ALL", AD9082_DAC_PAIR_ALL, "All Groups")
      .export_values();
  m.def("dac_select_set", &adi_ad9082_dac_select_set);
  m.def("dac_chan_select_set", &adi_ad9082_dac_chan_select_set);
  m.def("dac_mode_switch_group_select_set",
        &adi_ad9082_dac_mode_switch_group_select_set);

  // DAC top-level
  m.def(
      "device_startup_tx",
      [](adi_ad9082_device_t *device, uint8_t main_interp, uint8_t chan_interp,
         std::array<uint8_t, 4> &dac_chan, std::array<int64_t, 4> &main_shift,
         std::array<int64_t, 8> &chan_shift, adi_cms_jesd_param_t *jesd_param) {
        return adi_ad9082_device_startup_tx(device, main_interp, chan_interp,
                                            dac_chan.data(), main_shift.data(),
                                            chan_shift.data(), jesd_param);
      });

  m.def("dac_duc_nco_gains_set",
        [](adi_ad9082_device_t *device, std::array<uint16_t, 8> &gains) {
          return adi_ad9082_dac_duc_nco_gains_set(device, gains.data());
        });

  py::enum_<adi_ad9082_dac_mod_mux_mode_e>(m, "DacModMuxMode")
      .value("DAC_MUX_MODE_0", AD9082_DAC_MUX_MODE_0,
             "I0.Q0 -> DAC0, I1.Q1 -> DAC1")
      .value("DAC_MUX_MODE_1", AD9082_DAC_MUX_MODE_1,
             "(I0 + I1) / 2 -> DAC0, (Q0 + Q1) / 2 -> DAC1, Data Path NCOs "
             "Bypassed")
      .value("DAC_MUX_MODE_2", AD9082_DAC_MUX_MODE_2,
             "I0 -> DAC0, Q0 -> DAC1, Datapath0 NCO Bypassed, Datapath1 Unused")
      .value("DAC_MUX_MODE_3", AD9082_DAC_MUX_MODE_3,
             "(I0 + I1) / 2 -> DAC0, DAC1 Output Tied To Midscale")
      .export_values();
  m.def("dac_modulation_mux_mode_set", adi_ad9082_dac_modulation_mux_mode_set);

  // DAC auxiliary
  m.def("dac_duc_chan_skew_set", &adi_ad9082_dac_duc_chan_skew_set);
  m.def("dac_xbar_set", &adi_ad9082_dac_xbar_set);
  m.def("dac_fsc_set", &adi_ad9082_dac_fsc_set);

  // ADC top-level
  py::enum_<adi_ad9082_adc_coarse_ddc_select_e>(m, "AdcCoarseDdcSelect",
                                                py::arithmetic())
      .value("ADC_CDDC_NONE", AD9082_ADC_CDDC_NONE)
      .value("ADC_CDDC_0", AD9082_ADC_CDDC_0)
      .value("ADC_CDDC_1", AD9082_ADC_CDDC_1)
      .value("ADC_CDDC_2", AD9082_ADC_CDDC_2)
      .value("ADC_CDDC_3", AD9082_ADC_CDDC_3)
      .value("ADC_CDDC_ALL", AD9082_ADC_CDDC_ALL)
      .export_values();
  py::enum_<adi_ad9082_adc_fine_ddc_select_e>(m, "AdcFineDdcSelect",
                                              py::arithmetic())
      .value("ADC_FDDC_NONE", AD9082_ADC_FDDC_NONE)
      .value("ADC_FDDC_0", AD9082_ADC_FDDC_0)
      .value("ADC_FDDC_1", AD9082_ADC_FDDC_1)
      .value("ADC_FDDC_2", AD9082_ADC_FDDC_2)
      .value("ADC_FDDC_3", AD9082_ADC_FDDC_3)
      .value("ADC_FDDC_4", AD9082_ADC_FDDC_4)
      .value("ADC_FDDC_5", AD9082_ADC_FDDC_5)
      .value("ADC_FDDC_6", AD9082_ADC_FDDC_6)
      .value("ADC_FDDC_7", AD9082_ADC_FDDC_7)
      .value("ADC_FDDC_ALL", AD9082_ADC_FDDC_ALL)
      .export_values();
  py::enum_<adi_ad9082_adc_coarse_ddc_dcm_e>(m, "AdcCoarseDdcDcm")
      .value("ADC_CDDC_DCM_1", AD9082_CDDC_DCM_1)
      .value("ADC_CDDC_DCM_2", AD9082_CDDC_DCM_2)
      .value("ADC_CDDC_DCM_3", AD9082_CDDC_DCM_3)
      .value("ADC_CDDC_DCM_4", AD9082_CDDC_DCM_4)
      .value("ADC_CDDC_DCM_6", AD9082_CDDC_DCM_6)
      .value("ADC_CDDC_DCM_8", AD9082_CDDC_DCM_8)
      .value("ADC_CDDC_DCM_9", AD9082_CDDC_DCM_9)
      .value("ADC_CDDC_DCM_12", AD9082_CDDC_DCM_12)
      .value("ADC_CDDC_DCM_16", AD9082_CDDC_DCM_16)
      .value("ADC_CDDC_DCM_18", AD9082_CDDC_DCM_18)
      .value("ADC_CDDC_DCM_24", AD9082_CDDC_DCM_24)
      .value("ADC_CDDC_DCM_36", AD9082_CDDC_DCM_36)
      .export_values();
  py::enum_<adi_ad9082_adc_fine_ddc_dcm_e>(m, "AdcFineDdcDcm")
      .value("ADC_FDDC_DCM_1", AD9082_FDDC_DCM_1)
      .value("ADC_FDDC_DCM_2", AD9082_FDDC_DCM_2)
      .value("ADC_FDDC_DCM_3", AD9082_FDDC_DCM_3)
      .value("ADC_FDDC_DCM_4", AD9082_FDDC_DCM_4)
      .value("ADC_FDDC_DCM_6", AD9082_FDDC_DCM_6)
      .value("ADC_FDDC_DCM_8", AD9082_FDDC_DCM_8)
      .value("ADC_FDDC_DCM_12", AD9082_FDDC_DCM_12)
      .value("ADC_FDDC_DCM_16", AD9082_FDDC_DCM_16)
      .value("ADC_FDDC_DCM_24", AD9082_FDDC_DCM_24)
      .export_values();

  py::class_<adi_ad9082_jtx_conv_sel_t>(m, "JtxConvSel")
      .def(py::init<>())
      .def_readwrite("virtual_converter0_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter0_index)
      .def_readwrite("virtual_converter1_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter1_index)
      .def_readwrite("virtual_converter2_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter2_index)
      .def_readwrite("virtual_converter3_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter3_index)
      .def_readwrite("virtual_converter4_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter4_index)
      .def_readwrite("virtual_converter5_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter5_index)
      .def_readwrite("virtual_converter6_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter6_index)
      .def_readwrite("virtual_converter7_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter7_index)
      .def_readwrite("virtual_converter8_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter8_index)
      .def_readwrite("virtual_converter9_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converter9_index)
      .def_readwrite("virtual_convertera_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_convertera_index)
      .def_readwrite("virtual_converterb_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converterb_index)
      .def_readwrite("virtual_converterc_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converterc_index)
      .def_readwrite("virtual_converterd_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converterd_index)
      .def_readwrite("virtual_convertere_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_convertere_index)
      .def_readwrite("virtual_converterf_index",
                     &adi_ad9082_jtx_conv_sel_t::virtual_converterf_index);
  m.def(
      "device_startup_rx",
      [](adi_ad9082_device_t *device, uint8_t cddcs, uint8_t fddcs,
         std::array<int64_t, 4> &cddc_shift, std::array<int64_t, 8> &fddc_shift,
         std::array<uint8_t, 4> &cddc_dcm, std::array<uint8_t, 8> &fddc_dcm,
         std::array<uint8_t, 4> &cc2r_en, std::array<uint8_t, 8> &fc2r_en,
         std::array<adi_cms_jesd_param_t, 2> &jesd_param,
         std::array<adi_ad9082_jtx_conv_sel_t, 2> &jesd_conv_sel) {
        return adi_ad9082_device_startup_rx(
            device, cddcs, fddcs, cddc_shift.data(), fddc_shift.data(),
            cddc_dcm.data(), fddc_dcm.data(), cc2r_en.data(), fc2r_en.data(),
            jesd_param.data(), jesd_conv_sel.data());
      });

  // JESD204 primitives
  py::enum_<adi_ad9082_jesd_link_select_e>(m, "JesdLinkSelect",
                                           py::arithmetic())
      .value("LINK_NONE", AD9082_LINK_NONE, "No Link")
      .value("LINK_0", AD9082_LINK_0, "Link 0")
      .value("LINK_1", AD9082_LINK_1, "Link 1")
      .value("LINK_ALL", AD9082_LINK_ALL, "All Links")
      .export_values();
  m.def("jesd_rx_link_select_set", adi_ad9082_jesd_rx_link_select_set);

  // JESD204 top-level
  m.def("jesd_tx_link_enable_set", adi_ad9082_jesd_tx_link_enable_set);
  m.def("jesd_rx_link_enable_set", adi_ad9082_jesd_rx_link_enable_set);
  m.def("jesd_rx_calibrate_204c", adi_ad9082_jesd_rx_calibrate_204c);

  // JESD204 auxiliary
  m.def("jesd_rx_config_status_get", [](adi_ad9082_device_t *device) {
    uint8_t cfg_valid = 255;
    int32_t rc = adi_ad9082_jesd_rx_config_status_get(device, &cfg_valid);
    return std::pair<int32_t, uint8_t>(rc, cfg_valid);
  });

  typedef enum ad9082_link_status {
    AD9082_LINK_STATUS_RESET = 0,
    AD9082_LINK_STATUS_SH_FAILURE = 1,
    AD9082_LINK_STATUS_SH_ALIGNED = 2,
    AD9082_LINK_STATUS_EMB_SYNCED = 3,
    AD9082_LINK_STATUS_EMB_ALIGNED = 4,
    AD9082_LINK_STATUS_LOCK_FAILURE = 5,
    AD9082_LINK_STATUS_LOCKED = 6,
    AD9082_LINK_STATUS_UNKNOWN = 255,
  } ad9082_link_status_e;
  py::enum_<ad9082_link_status_e>(m, "LinkStatus")
      .value("LINK_STATUS_RESET", AD9082_LINK_STATUS_RESET)
      .value("LINK_STATUS_SH_FAILURE", AD9082_LINK_STATUS_SH_FAILURE)
      .value("LINK_STATUS_SH_ALIGNED", AD9082_LINK_STATUS_SH_ALIGNED)
      .value("LINK_STATUS_EMB_SYNCED", AD9082_LINK_STATUS_EMB_SYNCED)
      .value("LINK_STATUS_EMB_ALIGNED", AD9082_LINK_STATUS_EMB_ALIGNED)
      .value("LINK_STATUS_LOCK_FAILURE", AD9082_LINK_STATUS_LOCK_FAILURE)
      .value("LINK_STATUS_LOCKED", AD9082_LINK_STATUS_LOCKED)
      .value("LINK_STATUS_UNKNOWN", AD9082_LINK_STATUS_UNKNOWN)
      .export_values();
  m.def("jesd_rx_link_status_get", [](adi_ad9082_device_t *device,
                                      adi_ad9082_jesd_link_select_e links) {
    uint16_t status = 65535;
    int32_t rc = adi_ad9082_jesd_rx_link_status_get(device, links, &status);

    uint8_t s204c = (status >> 8) & 0x7;
    if (s204c >= 7) {
      adi_ad9082_hal_log_write(device, ADI_CMS_LOG_MSG,
                               "unexpected link status (= %d) is acquired",
                               s204c);
      s204c = 255;
    }
    return std::tuple<int32_t, enum ad9082_link_status, uint8_t>(
        rc, static_cast<ad9082_link_status_e>(s204c), status & 0xff);
  });

  m.def("jesd_rx_204c_crc_irq_enable", adi_ad9082_jesd_rx_204c_crc_irq_enable);

  m.def("jesd_rx_204c_crc_irq_status_get",
        [](adi_ad9082_device_t *device, adi_ad9082_jesd_link_select_e links) {
          uint8_t status = 255;
          int32_t rc = adi_ad9082_jesd_rx_204c_crc_irq_status_get(device, links,
                                                                  &status);
          return std::pair<int32_t, uint8_t>(rc, status);
        });

  m.def("jesd_rx_204c_crc_irq_clr", adi_ad9082_jesd_rx_204c_crc_irq_clr);

  m.def("jesd_rx_204c_mb_irq_enable", &adi_ad9082_jesd_rx_204c_mb_irq_enable);

  m.def("jesd_rx_204c_mb_irq_status_get",
        [](adi_ad9082_device_t *device, adi_ad9082_jesd_link_select_e links) {
          uint8_t status = 255;
          int32_t rc =
              adi_ad9082_jesd_rx_204c_mb_irq_status_get(device, links, &status);
          return std::pair<int32_t, uint8_t>(rc, status);
        });

  m.def("jesd_rx_204c_mb_irq_clr", adi_ad9082_jesd_rx_204c_mb_irq_clr);

  m.def("jesd_rx_204c_sh_irq_enable", adi_ad9082_jesd_rx_204c_sh_irq_enable);

  m.def("jesd_rx_204c_sh_irq_status_get",
        [](adi_ad9082_device_t *device, adi_ad9082_jesd_link_select_e links) {
          uint8_t status = 255;
          int32_t rc =
              adi_ad9082_jesd_rx_204c_sh_irq_status_get(device, links, &status);
          return std::pair<int32_t, uint8_t>(rc, status);
        });

  m.def("jesd_rx_204c_sh_irq_clr", adi_ad9082_jesd_rx_204c_mb_irq_clr);

  m.def("rx_ctle_manual_config_get",
        [](adi_ad9082_device_t *device, uint8_t lane) {
          if (lane < 0 || lane > 7) {
            return static_cast<int32_t>(API_CMS_ERROR_INVALID_PARAM);
          }
          return adi_ad9082_jesd_rx_ctle_manual_config_get(device, lane);
        });
};
