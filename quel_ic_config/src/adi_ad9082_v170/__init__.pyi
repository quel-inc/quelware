from __future__ import annotations

import typing

import numpy
import pybind11_stubgen.typing_ext

__all__ = [
    "AD9082_CAL_MODE_BYPASS",
    "AD9082_CAL_MODE_RUN",
    "AD9082_CAL_MODE_RUN_AND_SAVE",
    "ADC_CDDC_0",
    "ADC_CDDC_1",
    "ADC_CDDC_2",
    "ADC_CDDC_3",
    "ADC_CDDC_ALL",
    "ADC_CDDC_DCM_1",
    "ADC_CDDC_DCM_12",
    "ADC_CDDC_DCM_16",
    "ADC_CDDC_DCM_18",
    "ADC_CDDC_DCM_2",
    "ADC_CDDC_DCM_24",
    "ADC_CDDC_DCM_3",
    "ADC_CDDC_DCM_36",
    "ADC_CDDC_DCM_4",
    "ADC_CDDC_DCM_6",
    "ADC_CDDC_DCM_8",
    "ADC_CDDC_DCM_9",
    "ADC_CDDC_NONE",
    "ADC_FDDC_0",
    "ADC_FDDC_1",
    "ADC_FDDC_2",
    "ADC_FDDC_3",
    "ADC_FDDC_4",
    "ADC_FDDC_5",
    "ADC_FDDC_6",
    "ADC_FDDC_7",
    "ADC_FDDC_ALL",
    "ADC_FDDC_DCM_1",
    "ADC_FDDC_DCM_12",
    "ADC_FDDC_DCM_16",
    "ADC_FDDC_DCM_2",
    "ADC_FDDC_DCM_24",
    "ADC_FDDC_DCM_3",
    "ADC_FDDC_DCM_4",
    "ADC_FDDC_DCM_6",
    "ADC_FDDC_DCM_8",
    "ADC_FDDC_NONE",
    "ADC_NCO_FS_4_IF",
    "ADC_NCO_TEST",
    "ADC_NCO_VIF",
    "ADC_NCO_ZIF",
    "API_CMS_ERROR_DELAY_US",
    "API_CMS_ERROR_DLL_NOT_LOCKED",
    "API_CMS_ERROR_ERROR",
    "API_CMS_ERROR_EVENT_HNDL",
    "API_CMS_ERROR_FTW_LOAD_ACK",
    "API_CMS_ERROR_HW_CLOSE",
    "API_CMS_ERROR_HW_OPEN",
    "API_CMS_ERROR_INIT_SEQ_FAIL",
    "API_CMS_ERROR_INVALID_DELAYUS_PTR",
    "API_CMS_ERROR_INVALID_HANDLE_PTR",
    "API_CMS_ERROR_INVALID_PARAM",
    "API_CMS_ERROR_INVALID_RESET_CTRL_PTR",
    "API_CMS_ERROR_INVALID_XFER_PTR",
    "API_CMS_ERROR_JESD_PLL_NOT_LOCKED",
    "API_CMS_ERROR_JESD_SYNC_NOT_DONE",
    "API_CMS_ERROR_LOG_CLOSE",
    "API_CMS_ERROR_LOG_OPEN",
    "API_CMS_ERROR_LOG_WRITE",
    "API_CMS_ERROR_MODE_NOT_IN_TABLE",
    "API_CMS_ERROR_NCO_NOT_ENABLED",
    "API_CMS_ERROR_NOT_SUPPORTED",
    "API_CMS_ERROR_NULL_PARAM",
    "API_CMS_ERROR_OK",
    "API_CMS_ERROR_PD_STBY_PIN_CTRL",
    "API_CMS_ERROR_PLL_NOT_LOCKED",
    "API_CMS_ERROR_RESET_PIN_CTRL",
    "API_CMS_ERROR_SPI_SDO",
    "API_CMS_ERROR_SPI_XFER",
    "API_CMS_ERROR_SYSREF_CTRL",
    "API_CMS_ERROR_TEST_FAILED",
    "API_CMS_ERROR_TX_EN_PIN_CTRL",
    "API_CMS_ERROR_VCO_OUT_OF_RANGE",
    "AdcCoarseDdcDcm",
    "AdcCoarseDdcSelect",
    "AdcFineDdcDcm",
    "AdcFineDdcSelect",
    "AdcNcoMode",
    "ApiRevision",
    "COUPLING_AC",
    "COUPLING_DC",
    "COUPLING_UNKNOWN",
    "CalMode",
    "ChipTemperatures",
    "Clk",
    "CmdJesdLink",
    "CmsChioOpMode",
    "CmsChipId",
    "CmsError",
    "CmsJesdParam",
    "CmsJesdPrbsPattern",
    "CmsJesdSubclass",
    "CmsJesdSyncoutb",
    "CmsJesdSysrefMode",
    "CmsSignalType",
    "CmsSingalCoupling",
    "CmsSpiAddrInc",
    "CmsSpiMsbConfig",
    "CmsSpiSdoConfig",
    "DAC_0",
    "DAC_1",
    "DAC_2",
    "DAC_3",
    "DAC_ALL",
    "DAC_CH_0",
    "DAC_CH_1",
    "DAC_CH_2",
    "DAC_CH_3",
    "DAC_CH_4",
    "DAC_CH_5",
    "DAC_CH_6",
    "DAC_CH_7",
    "DAC_CH_ALL",
    "DAC_CH_NONE",
    "DAC_MUX_MODE_0",
    "DAC_MUX_MODE_1",
    "DAC_MUX_MODE_2",
    "DAC_MUX_MODE_3",
    "DAC_NONE",
    "DAC_PAIR_0_1",
    "DAC_PAIR_2_3",
    "DAC_PAIR_ALL",
    "DAC_PAIR_NONE",
    "DES_CTLE_IL_0DB_6DB",
    "DES_CTLE_IL_4DB_10DB",
    "DES_CTLE_IL_FAR_LT_0DB",
    "DES_CTLE_IL_LT_0DB",
    "DacChannelSelect",
    "DacModMuxMode",
    "DacPairSelect",
    "DacSelect",
    "DesCtleInsertionLoss",
    "DesSettings",
    "Device",
    "HARD_RESET",
    "HARD_RESET_AND_INIT",
    "Info",
    "JESD_LINK_0",
    "JESD_LINK_1",
    "JESD_LINK_ALL",
    "JESD_LINK_NONE",
    "JESD_SUBCLASS_0",
    "JESD_SUBCLASS_1",
    "JESD_SUBCLASS_INVALID",
    "JesdLinkSelect",
    "JtxConvSel",
    "LINK_0",
    "LINK_1",
    "LINK_ALL",
    "LINK_NONE",
    "LINK_STATUS_EMB_ALIGNED",
    "LINK_STATUS_EMB_SYNCED",
    "LINK_STATUS_LOCKED",
    "LINK_STATUS_LOCK_FAILURE",
    "LINK_STATUS_RESET",
    "LINK_STATUS_SH_ALIGNED",
    "LINK_STATUS_SH_FAILURE",
    "LINK_STATUS_UNKNOWN",
    "LinkStatus",
    "NcoFtw",
    "POST_EMP_SETTING",
    "PRBS15",
    "PRBS23",
    "PRBS31",
    "PRBS7",
    "PRBS9",
    "PRBS_MAX",
    "PRBS_NONE",
    "PRE_EMP_SETTING",
    "RX_ONLY",
    "RegData",
    "Reset",
    "SER_POST_EMP_0DB",
    "SER_POST_EMP_12DB",
    "SER_POST_EMP_3DB",
    "SER_POST_EMP_6DB",
    "SER_POST_EMP_9DB",
    "SER_PRE_EMP_0DB",
    "SER_PRE_EMP_3DB",
    "SER_PRE_EMP_6DB",
    "SER_SWING_1000",
    "SER_SWING_500",
    "SER_SWING_750",
    "SER_SWING_850",
    "SIGNAL_CML",
    "SIGNAL_CMOS",
    "SIGNAL_LVDS",
    "SIGNAL_LVPECL",
    "SIGNAL_UNKNOWN",
    "SOFT_RESET",
    "SOFT_RESET_AND_INIT",
    "SPI_ADDR_DEC_AUTO",
    "SPI_ADDR_INC_AUTO",
    "SPI_MSB_FIRST",
    "SPI_MSB_LAST",
    "SPI_NONE",
    "SPI_SDIO",
    "SPI_SDO",
    "SWING_SETTING",
    "SYNCOUTB_0",
    "SYNCOUTB_1",
    "SYNCOUTB_ALL",
    "SYSREF_CONT",
    "SYSREF_MODE_INVALID",
    "SYSREF_MON",
    "SYSREF_NONE",
    "SYSREF_ONESHOT",
    "SerLaneSettings",
    "SerLaneSettingsField",
    "SerPostEmp",
    "SerPreEmp",
    "SerSettings",
    "SerSwing",
    "SerdesSettings",
    "TX_ONLY",
    "TX_RX_ONLY",
    "adc_ddc_coarse_nco_ftw_get",
    "adc_ddc_coarse_nco_ftw_set",
    "adc_ddc_coarse_nco_mode_set",
    "adc_ddc_coarse_select_set",
    "adc_ddc_coarse_sync_enable_set",
    "adc_ddc_coarse_sync_next_set",
    "adc_ddc_coarse_trig_nco_reset_enable_set",
    "adc_ddc_fine_nco_ftw_get",
    "adc_ddc_fine_nco_ftw_set",
    "adc_ddc_fine_nco_mode_set",
    "adc_ddc_fine_select_set",
    "adc_ddc_fine_sync_enable_set",
    "adc_ddc_fine_sync_next_set",
    "adc_ddc_fine_trig_nco_reset_enable_set",
    "dac_chan_select_set",
    "dac_duc_chan_skew_set",
    "dac_duc_nco_enable_set",
    "dac_duc_nco_ftw_set",
    "dac_duc_nco_gains_set",
    "dac_fsc_set",
    "dac_mode_switch_group_select_set",
    "dac_modulation_mux_mode_set",
    "dac_select_set",
    "dac_xbar_set",
    "device_api_revision_get",
    "device_chip_id_get",
    "device_clk_config_set",
    "device_clk_pll_lock_status_get",
    "device_get_temperature",
    "device_init",
    "device_reset",
    "device_startup_rx",
    "device_startup_tx",
    "hal_calc_rx_nco_ftw",
    "hal_calc_tx_nco_ftw",
    "hal_delay_us",
    "hal_reg_get",
    "hal_reg_set",
    "hal_reset_pin_ctrl",
    "jesd_oneshot_sync",
    "jesd_rx_204c_crc_irq_clr",
    "jesd_rx_204c_crc_irq_enable",
    "jesd_rx_204c_crc_irq_status_get",
    "jesd_rx_204c_mb_irq_clr",
    "jesd_rx_204c_mb_irq_enable",
    "jesd_rx_204c_mb_irq_status_get",
    "jesd_rx_204c_sh_irq_clr",
    "jesd_rx_204c_sh_irq_enable",
    "jesd_rx_204c_sh_irq_status_get",
    "jesd_rx_bit_rate_get",
    "jesd_rx_calibrate_204c",
    "jesd_rx_config_status_get",
    "jesd_rx_ctle_manual_config_get",
    "jesd_rx_lanes_xbar_set",
    "jesd_rx_link_enable_set",
    "jesd_rx_link_select_set",
    "jesd_rx_link_status_get",
    "jesd_sysref_average_set",
    "jesd_sysref_monitor_phase_get",
    "jesd_sysref_setup_hold_get",
    "jesd_tx_lanes_xbar_set",
    "jesd_tx_link_enable_set",
    "jesd_tx_link_select_set",
    "sync_calc_jrx_lmfc_lemc",
    "sync_calc_jtx_lmfc_lemc",
    "sync_sysref_frequency_set",
    "sync_sysref_input_config_set",
]

class AdcCoarseDdcDcm:
    """
    Members:

      ADC_CDDC_DCM_1

      ADC_CDDC_DCM_2

      ADC_CDDC_DCM_3

      ADC_CDDC_DCM_4

      ADC_CDDC_DCM_6

      ADC_CDDC_DCM_8

      ADC_CDDC_DCM_9

      ADC_CDDC_DCM_12

      ADC_CDDC_DCM_16

      ADC_CDDC_DCM_18

      ADC_CDDC_DCM_24

      ADC_CDDC_DCM_36
    """

    ADC_CDDC_DCM_1: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_1: 12>
    ADC_CDDC_DCM_12: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_12: 6>
    ADC_CDDC_DCM_16: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_16: 3>
    ADC_CDDC_DCM_18: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_18: 10>
    ADC_CDDC_DCM_2: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_2: 0>
    ADC_CDDC_DCM_24: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_24: 7>
    ADC_CDDC_DCM_3: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_3: 8>
    ADC_CDDC_DCM_36: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_36: 11>
    ADC_CDDC_DCM_4: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_4: 1>
    ADC_CDDC_DCM_6: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_6: 5>
    ADC_CDDC_DCM_8: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_8: 2>
    ADC_CDDC_DCM_9: typing.ClassVar[AdcCoarseDdcDcm]  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_9: 9>
    __members__: typing.ClassVar[
        dict[str, AdcCoarseDdcDcm]
    ]  # value = {'ADC_CDDC_DCM_1': <AdcCoarseDdcDcm.ADC_CDDC_DCM_1: 12>, 'ADC_CDDC_DCM_2': <AdcCoarseDdcDcm.ADC_CDDC_DCM_2: 0>, 'ADC_CDDC_DCM_3': <AdcCoarseDdcDcm.ADC_CDDC_DCM_3: 8>, 'ADC_CDDC_DCM_4': <AdcCoarseDdcDcm.ADC_CDDC_DCM_4: 1>, 'ADC_CDDC_DCM_6': <AdcCoarseDdcDcm.ADC_CDDC_DCM_6: 5>, 'ADC_CDDC_DCM_8': <AdcCoarseDdcDcm.ADC_CDDC_DCM_8: 2>, 'ADC_CDDC_DCM_9': <AdcCoarseDdcDcm.ADC_CDDC_DCM_9: 9>, 'ADC_CDDC_DCM_12': <AdcCoarseDdcDcm.ADC_CDDC_DCM_12: 6>, 'ADC_CDDC_DCM_16': <AdcCoarseDdcDcm.ADC_CDDC_DCM_16: 3>, 'ADC_CDDC_DCM_18': <AdcCoarseDdcDcm.ADC_CDDC_DCM_18: 10>, 'ADC_CDDC_DCM_24': <AdcCoarseDdcDcm.ADC_CDDC_DCM_24: 7>, 'ADC_CDDC_DCM_36': <AdcCoarseDdcDcm.ADC_CDDC_DCM_36: 11>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class AdcCoarseDdcSelect:
    """
    Members:

      ADC_CDDC_NONE

      ADC_CDDC_0

      ADC_CDDC_1

      ADC_CDDC_2

      ADC_CDDC_3

      ADC_CDDC_ALL
    """

    ADC_CDDC_0: typing.ClassVar[AdcCoarseDdcSelect]  # value = <AdcCoarseDdcSelect.ADC_CDDC_0: 1>
    ADC_CDDC_1: typing.ClassVar[AdcCoarseDdcSelect]  # value = <AdcCoarseDdcSelect.ADC_CDDC_1: 2>
    ADC_CDDC_2: typing.ClassVar[AdcCoarseDdcSelect]  # value = <AdcCoarseDdcSelect.ADC_CDDC_2: 4>
    ADC_CDDC_3: typing.ClassVar[AdcCoarseDdcSelect]  # value = <AdcCoarseDdcSelect.ADC_CDDC_3: 8>
    ADC_CDDC_ALL: typing.ClassVar[AdcCoarseDdcSelect]  # value = <AdcCoarseDdcSelect.ADC_CDDC_ALL: 15>
    ADC_CDDC_NONE: typing.ClassVar[AdcCoarseDdcSelect]  # value = <AdcCoarseDdcSelect.ADC_CDDC_NONE: 0>
    __members__: typing.ClassVar[
        dict[str, AdcCoarseDdcSelect]
    ]  # value = {'ADC_CDDC_NONE': <AdcCoarseDdcSelect.ADC_CDDC_NONE: 0>, 'ADC_CDDC_0': <AdcCoarseDdcSelect.ADC_CDDC_0: 1>, 'ADC_CDDC_1': <AdcCoarseDdcSelect.ADC_CDDC_1: 2>, 'ADC_CDDC_2': <AdcCoarseDdcSelect.ADC_CDDC_2: 4>, 'ADC_CDDC_3': <AdcCoarseDdcSelect.ADC_CDDC_3: 8>, 'ADC_CDDC_ALL': <AdcCoarseDdcSelect.ADC_CDDC_ALL: 15>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class AdcFineDdcDcm:
    """
    Members:

      ADC_FDDC_DCM_1

      ADC_FDDC_DCM_2

      ADC_FDDC_DCM_3

      ADC_FDDC_DCM_4

      ADC_FDDC_DCM_6

      ADC_FDDC_DCM_8

      ADC_FDDC_DCM_12

      ADC_FDDC_DCM_16

      ADC_FDDC_DCM_24
    """

    ADC_FDDC_DCM_1: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_1: 8>
    ADC_FDDC_DCM_12: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_12: 6>
    ADC_FDDC_DCM_16: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_16: 3>
    ADC_FDDC_DCM_2: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_2: 0>
    ADC_FDDC_DCM_24: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_24: 7>
    ADC_FDDC_DCM_3: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_3: 4>
    ADC_FDDC_DCM_4: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_4: 1>
    ADC_FDDC_DCM_6: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_6: 5>
    ADC_FDDC_DCM_8: typing.ClassVar[AdcFineDdcDcm]  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_8: 2>
    __members__: typing.ClassVar[
        dict[str, AdcFineDdcDcm]
    ]  # value = {'ADC_FDDC_DCM_1': <AdcFineDdcDcm.ADC_FDDC_DCM_1: 8>, 'ADC_FDDC_DCM_2': <AdcFineDdcDcm.ADC_FDDC_DCM_2: 0>, 'ADC_FDDC_DCM_3': <AdcFineDdcDcm.ADC_FDDC_DCM_3: 4>, 'ADC_FDDC_DCM_4': <AdcFineDdcDcm.ADC_FDDC_DCM_4: 1>, 'ADC_FDDC_DCM_6': <AdcFineDdcDcm.ADC_FDDC_DCM_6: 5>, 'ADC_FDDC_DCM_8': <AdcFineDdcDcm.ADC_FDDC_DCM_8: 2>, 'ADC_FDDC_DCM_12': <AdcFineDdcDcm.ADC_FDDC_DCM_12: 6>, 'ADC_FDDC_DCM_16': <AdcFineDdcDcm.ADC_FDDC_DCM_16: 3>, 'ADC_FDDC_DCM_24': <AdcFineDdcDcm.ADC_FDDC_DCM_24: 7>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class AdcFineDdcSelect:
    """
    Members:

      ADC_FDDC_NONE

      ADC_FDDC_0

      ADC_FDDC_1

      ADC_FDDC_2

      ADC_FDDC_3

      ADC_FDDC_4

      ADC_FDDC_5

      ADC_FDDC_6

      ADC_FDDC_7

      ADC_FDDC_ALL
    """

    ADC_FDDC_0: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_0: 1>
    ADC_FDDC_1: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_1: 2>
    ADC_FDDC_2: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_2: 4>
    ADC_FDDC_3: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_3: 8>
    ADC_FDDC_4: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_4: 16>
    ADC_FDDC_5: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_5: 32>
    ADC_FDDC_6: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_6: 64>
    ADC_FDDC_7: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_7: 128>
    ADC_FDDC_ALL: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_ALL: 255>
    ADC_FDDC_NONE: typing.ClassVar[AdcFineDdcSelect]  # value = <AdcFineDdcSelect.ADC_FDDC_NONE: 0>
    __members__: typing.ClassVar[
        dict[str, AdcFineDdcSelect]
    ]  # value = {'ADC_FDDC_NONE': <AdcFineDdcSelect.ADC_FDDC_NONE: 0>, 'ADC_FDDC_0': <AdcFineDdcSelect.ADC_FDDC_0: 1>, 'ADC_FDDC_1': <AdcFineDdcSelect.ADC_FDDC_1: 2>, 'ADC_FDDC_2': <AdcFineDdcSelect.ADC_FDDC_2: 4>, 'ADC_FDDC_3': <AdcFineDdcSelect.ADC_FDDC_3: 8>, 'ADC_FDDC_4': <AdcFineDdcSelect.ADC_FDDC_4: 16>, 'ADC_FDDC_5': <AdcFineDdcSelect.ADC_FDDC_5: 32>, 'ADC_FDDC_6': <AdcFineDdcSelect.ADC_FDDC_6: 64>, 'ADC_FDDC_7': <AdcFineDdcSelect.ADC_FDDC_7: 128>, 'ADC_FDDC_ALL': <AdcFineDdcSelect.ADC_FDDC_ALL: 255>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class AdcNcoMode:
    """
    Members:

      ADC_NCO_VIF : Variable IF Mode

      ADC_NCO_ZIF : Zero IF Mode

      ADC_NCO_FS_4_IF : Fs/4 Hz IF Mode

      ADC_NCO_TEST : Test Mode
    """

    ADC_NCO_FS_4_IF: typing.ClassVar[AdcNcoMode]  # value = <AdcNcoMode.ADC_NCO_FS_4_IF: 2>
    ADC_NCO_TEST: typing.ClassVar[AdcNcoMode]  # value = <AdcNcoMode.ADC_NCO_TEST: 3>
    ADC_NCO_VIF: typing.ClassVar[AdcNcoMode]  # value = <AdcNcoMode.ADC_NCO_VIF: 0>
    ADC_NCO_ZIF: typing.ClassVar[AdcNcoMode]  # value = <AdcNcoMode.ADC_NCO_ZIF: 1>
    __members__: typing.ClassVar[
        dict[str, AdcNcoMode]
    ]  # value = {'ADC_NCO_VIF': <AdcNcoMode.ADC_NCO_VIF: 0>, 'ADC_NCO_ZIF': <AdcNcoMode.ADC_NCO_ZIF: 1>, 'ADC_NCO_FS_4_IF': <AdcNcoMode.ADC_NCO_FS_4_IF: 2>, 'ADC_NCO_TEST': <AdcNcoMode.ADC_NCO_TEST: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ApiRevision:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...
    @property
    def major(self) -> int: ...
    @property
    def minor(self) -> int: ...
    @property
    def rc(self) -> int: ...

class CalMode:
    """
    Members:

      AD9082_CAL_MODE_RUN : Run 204C QR Calibration

      AD9082_CAL_MODE_RUN_AND_SAVE : Run 204C QR Calibration and save CTLE Coefficients

      AD9082_CAL_MODE_BYPASS : Bypass 204C QR Calibration and load CTLE Coefficients
    """

    AD9082_CAL_MODE_BYPASS: typing.ClassVar[CalMode]  # value = <CalMode.AD9082_CAL_MODE_BYPASS: 2>
    AD9082_CAL_MODE_RUN: typing.ClassVar[CalMode]  # value = <CalMode.AD9082_CAL_MODE_RUN: 0>
    AD9082_CAL_MODE_RUN_AND_SAVE: typing.ClassVar[CalMode]  # value = <CalMode.AD9082_CAL_MODE_RUN_AND_SAVE: 1>
    __members__: typing.ClassVar[
        dict[str, CalMode]
    ]  # value = {'AD9082_CAL_MODE_RUN': <CalMode.AD9082_CAL_MODE_RUN: 0>, 'AD9082_CAL_MODE_RUN_AND_SAVE': <CalMode.AD9082_CAL_MODE_RUN_AND_SAVE: 1>, 'AD9082_CAL_MODE_BYPASS': <CalMode.AD9082_CAL_MODE_BYPASS: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ChipTemperatures:
    temp_max: int
    temp_min: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...

class Clk:
    sysref_mode: CmsJesdSysrefMode
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...

class CmdJesdLink:
    """
    Members:

      JESD_LINK_NONE : JESD link none

      JESD_LINK_0 : JESD link 0

      JESD_LINK_1 : JESD link 1

      JESD_LINK_ALL : ALL JESD links
    """

    JESD_LINK_0: typing.ClassVar[CmdJesdLink]  # value = <CmdJesdLink.JESD_LINK_0: 1>
    JESD_LINK_1: typing.ClassVar[CmdJesdLink]  # value = <CmdJesdLink.JESD_LINK_1: 2>
    JESD_LINK_ALL: typing.ClassVar[CmdJesdLink]  # value = <CmdJesdLink.JESD_LINK_ALL: 3>
    JESD_LINK_NONE: typing.ClassVar[CmdJesdLink]  # value = <CmdJesdLink.JESD_LINK_NONE: 0>
    __members__: typing.ClassVar[
        dict[str, CmdJesdLink]
    ]  # value = {'JESD_LINK_NONE': <CmdJesdLink.JESD_LINK_NONE: 0>, 'JESD_LINK_0': <CmdJesdLink.JESD_LINK_0: 1>, 'JESD_LINK_1': <CmdJesdLink.JESD_LINK_1: 2>, 'JESD_LINK_ALL': <CmdJesdLink.JESD_LINK_ALL: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsChioOpMode:
    """
    Members:

      TX_ONLY : Chip using Tx path only

      RX_ONLY : Chip using Rx path only

      TX_RX_ONLY : Chip using Tx + Rx both paths
    """

    RX_ONLY: typing.ClassVar[CmsChioOpMode]  # value = <CmsChioOpMode.RX_ONLY: 2>
    TX_ONLY: typing.ClassVar[CmsChioOpMode]  # value = <CmsChioOpMode.TX_ONLY: 1>
    TX_RX_ONLY: typing.ClassVar[CmsChioOpMode]  # value = <CmsChioOpMode.TX_RX_ONLY: 3>
    __members__: typing.ClassVar[
        dict[str, CmsChioOpMode]
    ]  # value = {'TX_ONLY': <CmsChioOpMode.TX_ONLY: 1>, 'RX_ONLY': <CmsChioOpMode.RX_ONLY: 2>, 'TX_RX_ONLY': <CmsChioOpMode.TX_RX_ONLY: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsChipId:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...
    @property
    def chip_type(self) -> int: ...
    @property
    def dev_revision(self) -> int: ...
    @property
    def prod_grade(self) -> int: ...
    @property
    def prod_id(self) -> int: ...

class CmsError:
    """
    Members:

      API_CMS_ERROR_OK : No Error

      API_CMS_ERROR_ERROR : General Error

      API_CMS_ERROR_NULL_PARAM : Null parameter

      API_CMS_ERROR_SPI_SDO : Wrong value assigned to the SDO in device structure

      API_CMS_ERROR_INVALID_HANDLE_PTR : Device handler pointer is invalid

      API_CMS_ERROR_INVALID_XFER_PTR : Invalid pointer to the SPI xfer function assigned

      API_CMS_ERROR_INVALID_DELAYUS_PTR : Invalid pointer to the delay_us function assigned

      API_CMS_ERROR_INVALID_PARAM : Invalid parameter passed

      API_CMS_ERROR_INVALID_RESET_CTRL_PTR : Invalid pointer to the reset control function assigned

      API_CMS_ERROR_NOT_SUPPORTED : Not supported

      API_CMS_ERROR_VCO_OUT_OF_RANGE : The VCO is out of range

      API_CMS_ERROR_PLL_NOT_LOCKED : PLL is not locked

      API_CMS_ERROR_DLL_NOT_LOCKED : DLL is not locked

      API_CMS_ERROR_MODE_NOT_IN_TABLE : JESD Mode not in table

      API_CMS_ERROR_JESD_PLL_NOT_LOCKED : PD STBY function error

      API_CMS_ERROR_JESD_SYNC_NOT_DONE : JESD_SYNC_NOT_DONE

      API_CMS_ERROR_FTW_LOAD_ACK : FTW acknowledge not received

      API_CMS_ERROR_NCO_NOT_ENABLED : The NCO is not enabled

      API_CMS_ERROR_INIT_SEQ_FAIL : Initialization sequence failed

      API_CMS_ERROR_TEST_FAILED : Test failed

      API_CMS_ERROR_SPI_XFER : SPI transfer error

      API_CMS_ERROR_TX_EN_PIN_CTRL : TX enable function error

      API_CMS_ERROR_RESET_PIN_CTRL : HW reset function error

      API_CMS_ERROR_EVENT_HNDL : Event handling error

      API_CMS_ERROR_HW_OPEN : HW open function error

      API_CMS_ERROR_HW_CLOSE : HW close function error

      API_CMS_ERROR_LOG_OPEN : Log open error

      API_CMS_ERROR_LOG_WRITE : Log write error

      API_CMS_ERROR_LOG_CLOSE : Log close error

      API_CMS_ERROR_DELAY_US : Delay error

      API_CMS_ERROR_PD_STBY_PIN_CTRL : STBY function error

      API_CMS_ERROR_SYSREF_CTRL : SYSREF enable function error
    """

    API_CMS_ERROR_DELAY_US: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_DELAY_US: -70>
    API_CMS_ERROR_DLL_NOT_LOCKED: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_DLL_NOT_LOCKED: -22>
    API_CMS_ERROR_ERROR: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_ERROR: -1>
    API_CMS_ERROR_EVENT_HNDL: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_EVENT_HNDL: -64>
    API_CMS_ERROR_FTW_LOAD_ACK: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_FTW_LOAD_ACK: -30>
    API_CMS_ERROR_HW_CLOSE: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_HW_CLOSE: -66>
    API_CMS_ERROR_HW_OPEN: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_HW_OPEN: -65>
    API_CMS_ERROR_INIT_SEQ_FAIL: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_INIT_SEQ_FAIL: -40>
    API_CMS_ERROR_INVALID_DELAYUS_PTR: typing.ClassVar[
        CmsError
    ]  # value = <CmsError.API_CMS_ERROR_INVALID_DELAYUS_PTR: -13>
    API_CMS_ERROR_INVALID_HANDLE_PTR: typing.ClassVar[
        CmsError
    ]  # value = <CmsError.API_CMS_ERROR_INVALID_HANDLE_PTR: -11>
    API_CMS_ERROR_INVALID_PARAM: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_INVALID_PARAM: -14>
    API_CMS_ERROR_INVALID_RESET_CTRL_PTR: typing.ClassVar[
        CmsError
    ]  # value = <CmsError.API_CMS_ERROR_INVALID_RESET_CTRL_PTR: -15>
    API_CMS_ERROR_INVALID_XFER_PTR: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_INVALID_XFER_PTR: -12>
    API_CMS_ERROR_JESD_PLL_NOT_LOCKED: typing.ClassVar[
        CmsError
    ]  # value = <CmsError.API_CMS_ERROR_JESD_PLL_NOT_LOCKED: -24>
    API_CMS_ERROR_JESD_SYNC_NOT_DONE: typing.ClassVar[
        CmsError
    ]  # value = <CmsError.API_CMS_ERROR_JESD_SYNC_NOT_DONE: -25>
    API_CMS_ERROR_LOG_CLOSE: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_LOG_CLOSE: -69>
    API_CMS_ERROR_LOG_OPEN: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_LOG_OPEN: -67>
    API_CMS_ERROR_LOG_WRITE: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_LOG_WRITE: -68>
    API_CMS_ERROR_MODE_NOT_IN_TABLE: typing.ClassVar[
        CmsError
    ]  # value = <CmsError.API_CMS_ERROR_MODE_NOT_IN_TABLE: -23>
    API_CMS_ERROR_NCO_NOT_ENABLED: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_NCO_NOT_ENABLED: -31>
    API_CMS_ERROR_NOT_SUPPORTED: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_NOT_SUPPORTED: -16>
    API_CMS_ERROR_NULL_PARAM: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_NULL_PARAM: -2>
    API_CMS_ERROR_OK: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_OK: 0>
    API_CMS_ERROR_PD_STBY_PIN_CTRL: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_PD_STBY_PIN_CTRL: -71>
    API_CMS_ERROR_PLL_NOT_LOCKED: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_PLL_NOT_LOCKED: -21>
    API_CMS_ERROR_RESET_PIN_CTRL: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_RESET_PIN_CTRL: -63>
    API_CMS_ERROR_SPI_SDO: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_SPI_SDO: -10>
    API_CMS_ERROR_SPI_XFER: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_SPI_XFER: -60>
    API_CMS_ERROR_SYSREF_CTRL: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_SYSREF_CTRL: -72>
    API_CMS_ERROR_TEST_FAILED: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_TEST_FAILED: -50>
    API_CMS_ERROR_TX_EN_PIN_CTRL: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_TX_EN_PIN_CTRL: -62>
    API_CMS_ERROR_VCO_OUT_OF_RANGE: typing.ClassVar[CmsError]  # value = <CmsError.API_CMS_ERROR_VCO_OUT_OF_RANGE: -20>
    __members__: typing.ClassVar[
        dict[str, CmsError]
    ]  # value = {'API_CMS_ERROR_OK': <CmsError.API_CMS_ERROR_OK: 0>, 'API_CMS_ERROR_ERROR': <CmsError.API_CMS_ERROR_ERROR: -1>, 'API_CMS_ERROR_NULL_PARAM': <CmsError.API_CMS_ERROR_NULL_PARAM: -2>, 'API_CMS_ERROR_SPI_SDO': <CmsError.API_CMS_ERROR_SPI_SDO: -10>, 'API_CMS_ERROR_INVALID_HANDLE_PTR': <CmsError.API_CMS_ERROR_INVALID_HANDLE_PTR: -11>, 'API_CMS_ERROR_INVALID_XFER_PTR': <CmsError.API_CMS_ERROR_INVALID_XFER_PTR: -12>, 'API_CMS_ERROR_INVALID_DELAYUS_PTR': <CmsError.API_CMS_ERROR_INVALID_DELAYUS_PTR: -13>, 'API_CMS_ERROR_INVALID_PARAM': <CmsError.API_CMS_ERROR_INVALID_PARAM: -14>, 'API_CMS_ERROR_INVALID_RESET_CTRL_PTR': <CmsError.API_CMS_ERROR_INVALID_RESET_CTRL_PTR: -15>, 'API_CMS_ERROR_NOT_SUPPORTED': <CmsError.API_CMS_ERROR_NOT_SUPPORTED: -16>, 'API_CMS_ERROR_VCO_OUT_OF_RANGE': <CmsError.API_CMS_ERROR_VCO_OUT_OF_RANGE: -20>, 'API_CMS_ERROR_PLL_NOT_LOCKED': <CmsError.API_CMS_ERROR_PLL_NOT_LOCKED: -21>, 'API_CMS_ERROR_DLL_NOT_LOCKED': <CmsError.API_CMS_ERROR_DLL_NOT_LOCKED: -22>, 'API_CMS_ERROR_MODE_NOT_IN_TABLE': <CmsError.API_CMS_ERROR_MODE_NOT_IN_TABLE: -23>, 'API_CMS_ERROR_JESD_PLL_NOT_LOCKED': <CmsError.API_CMS_ERROR_JESD_PLL_NOT_LOCKED: -24>, 'API_CMS_ERROR_JESD_SYNC_NOT_DONE': <CmsError.API_CMS_ERROR_JESD_SYNC_NOT_DONE: -25>, 'API_CMS_ERROR_FTW_LOAD_ACK': <CmsError.API_CMS_ERROR_FTW_LOAD_ACK: -30>, 'API_CMS_ERROR_NCO_NOT_ENABLED': <CmsError.API_CMS_ERROR_NCO_NOT_ENABLED: -31>, 'API_CMS_ERROR_INIT_SEQ_FAIL': <CmsError.API_CMS_ERROR_INIT_SEQ_FAIL: -40>, 'API_CMS_ERROR_TEST_FAILED': <CmsError.API_CMS_ERROR_TEST_FAILED: -50>, 'API_CMS_ERROR_SPI_XFER': <CmsError.API_CMS_ERROR_SPI_XFER: -60>, 'API_CMS_ERROR_TX_EN_PIN_CTRL': <CmsError.API_CMS_ERROR_TX_EN_PIN_CTRL: -62>, 'API_CMS_ERROR_RESET_PIN_CTRL': <CmsError.API_CMS_ERROR_RESET_PIN_CTRL: -63>, 'API_CMS_ERROR_EVENT_HNDL': <CmsError.API_CMS_ERROR_EVENT_HNDL: -64>, 'API_CMS_ERROR_HW_OPEN': <CmsError.API_CMS_ERROR_HW_OPEN: -65>, 'API_CMS_ERROR_HW_CLOSE': <CmsError.API_CMS_ERROR_HW_CLOSE: -66>, 'API_CMS_ERROR_LOG_OPEN': <CmsError.API_CMS_ERROR_LOG_OPEN: -67>, 'API_CMS_ERROR_LOG_WRITE': <CmsError.API_CMS_ERROR_LOG_WRITE: -68>, 'API_CMS_ERROR_LOG_CLOSE': <CmsError.API_CMS_ERROR_LOG_CLOSE: -69>, 'API_CMS_ERROR_DELAY_US': <CmsError.API_CMS_ERROR_DELAY_US: -70>, 'API_CMS_ERROR_PD_STBY_PIN_CTRL': <CmsError.API_CMS_ERROR_PD_STBY_PIN_CTRL: -71>, 'API_CMS_ERROR_SYSREF_CTRL': <CmsError.API_CMS_ERROR_SYSREF_CTRL: -72>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsJesdParam:
    bid: int
    cf: int
    cs: int
    did: int
    duallink: int
    f: int
    hd: int
    jesdv: int
    k: int
    l: int
    lid0: int
    m: int
    mode_c2r_en: int
    mode_id: int
    mode_s_sel: int
    n: int
    np: int
    s: int
    scr: int
    subclass: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...

class CmsJesdPrbsPattern:
    """
    Members:

      PRBS_NONE : PRBS off

      PRBS7 : PRBS7 pattern

      PRBS9 : PRBS9 pattern

      PRBS15 : PRBS15 pattern

      PRBS23 : PRBS23 pattern

      PRBS31 : PRBS31 pattern

      PRBS_MAX : Number of member
    """

    PRBS15: typing.ClassVar[CmsJesdPrbsPattern]  # value = <CmsJesdPrbsPattern.PRBS15: 3>
    PRBS23: typing.ClassVar[CmsJesdPrbsPattern]  # value = <CmsJesdPrbsPattern.PRBS23: 4>
    PRBS31: typing.ClassVar[CmsJesdPrbsPattern]  # value = <CmsJesdPrbsPattern.PRBS31: 5>
    PRBS7: typing.ClassVar[CmsJesdPrbsPattern]  # value = <CmsJesdPrbsPattern.PRBS7: 1>
    PRBS9: typing.ClassVar[CmsJesdPrbsPattern]  # value = <CmsJesdPrbsPattern.PRBS9: 2>
    PRBS_MAX: typing.ClassVar[CmsJesdPrbsPattern]  # value = <CmsJesdPrbsPattern.PRBS_MAX: 6>
    PRBS_NONE: typing.ClassVar[CmsJesdPrbsPattern]  # value = <CmsJesdPrbsPattern.PRBS_NONE: 0>
    __members__: typing.ClassVar[
        dict[str, CmsJesdPrbsPattern]
    ]  # value = {'PRBS_NONE': <CmsJesdPrbsPattern.PRBS_NONE: 0>, 'PRBS7': <CmsJesdPrbsPattern.PRBS7: 1>, 'PRBS9': <CmsJesdPrbsPattern.PRBS9: 2>, 'PRBS15': <CmsJesdPrbsPattern.PRBS15: 3>, 'PRBS23': <CmsJesdPrbsPattern.PRBS23: 4>, 'PRBS31': <CmsJesdPrbsPattern.PRBS31: 5>, 'PRBS_MAX': <CmsJesdPrbsPattern.PRBS_MAX: 6>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsJesdSubclass:
    """
    Members:

      JESD_SUBCLASS_0 : JESD SUBCLASS 0

      JESD_SUBCLASS_1 : JESD SUBCLASS 1

      JESD_SUBCLASS_INVALID : JESD_SUBCLASS_INVALID
    """

    JESD_SUBCLASS_0: typing.ClassVar[CmsJesdSubclass]  # value = <CmsJesdSubclass.JESD_SUBCLASS_0: 0>
    JESD_SUBCLASS_1: typing.ClassVar[CmsJesdSubclass]  # value = <CmsJesdSubclass.JESD_SUBCLASS_1: 1>
    JESD_SUBCLASS_INVALID: typing.ClassVar[CmsJesdSubclass]  # value = <CmsJesdSubclass.JESD_SUBCLASS_INVALID: 2>
    __members__: typing.ClassVar[
        dict[str, CmsJesdSubclass]
    ]  # value = {'JESD_SUBCLASS_0': <CmsJesdSubclass.JESD_SUBCLASS_0: 0>, 'JESD_SUBCLASS_1': <CmsJesdSubclass.JESD_SUBCLASS_1: 1>, 'JESD_SUBCLASS_INVALID': <CmsJesdSubclass.JESD_SUBCLASS_INVALID: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsJesdSyncoutb:
    """
    Members:

      SYNCOUTB_0 : SYNCOUTB 0

      SYNCOUTB_1 : SYNCOUTB 1

      SYNCOUTB_ALL : ALL SYNCOUTB SIGNALS
    """

    SYNCOUTB_0: typing.ClassVar[CmsJesdSyncoutb]  # value = <CmsJesdSyncoutb.SYNCOUTB_0: 0>
    SYNCOUTB_1: typing.ClassVar[CmsJesdSyncoutb]  # value = <CmsJesdSyncoutb.SYNCOUTB_1: 1>
    SYNCOUTB_ALL: typing.ClassVar[CmsJesdSyncoutb]  # value = <CmsJesdSyncoutb.SYNCOUTB_ALL: 255>
    __members__: typing.ClassVar[
        dict[str, CmsJesdSyncoutb]
    ]  # value = {'SYNCOUTB_0': <CmsJesdSyncoutb.SYNCOUTB_0: 0>, 'SYNCOUTB_1': <CmsJesdSyncoutb.SYNCOUTB_1: 1>, 'SYNCOUTB_ALL': <CmsJesdSyncoutb.SYNCOUTB_ALL: 255>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsJesdSysrefMode:
    """
    Members:

      SYSREF_NONE : No SYSREF SUPPORT

      SYSREF_ONESHOT : ONE-SHOT SYSREF

      SYSREF_CONT : Continuous SysRef sync.

      SYSREF_MON : SYSREF monitor mode

      SYSREF_MODE_INVALID :
    """

    SYSREF_CONT: typing.ClassVar[CmsJesdSysrefMode]  # value = <CmsJesdSysrefMode.SYSREF_CONT: 2>
    SYSREF_MODE_INVALID: typing.ClassVar[CmsJesdSysrefMode]  # value = <CmsJesdSysrefMode.SYSREF_MODE_INVALID: 4>
    SYSREF_MON: typing.ClassVar[CmsJesdSysrefMode]  # value = <CmsJesdSysrefMode.SYSREF_MON: 3>
    SYSREF_NONE: typing.ClassVar[CmsJesdSysrefMode]  # value = <CmsJesdSysrefMode.SYSREF_NONE: 0>
    SYSREF_ONESHOT: typing.ClassVar[CmsJesdSysrefMode]  # value = <CmsJesdSysrefMode.SYSREF_ONESHOT: 1>
    __members__: typing.ClassVar[
        dict[str, CmsJesdSysrefMode]
    ]  # value = {'SYSREF_NONE': <CmsJesdSysrefMode.SYSREF_NONE: 0>, 'SYSREF_ONESHOT': <CmsJesdSysrefMode.SYSREF_ONESHOT: 1>, 'SYSREF_CONT': <CmsJesdSysrefMode.SYSREF_CONT: 2>, 'SYSREF_MON': <CmsJesdSysrefMode.SYSREF_MON: 3>, 'SYSREF_MODE_INVALID': <CmsJesdSysrefMode.SYSREF_MODE_INVALID: 4>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsSignalType:
    """
    Members:

      SIGNAL_CMOS : CMOS signal

      SIGNAL_LVDS : LVDS signal

      SIGNAL_CML : CML signal

      SIGNAL_LVPECL : LVPECL signal

      SIGNAL_UNKNOWN : UNKNOWN signal
    """

    SIGNAL_CML: typing.ClassVar[CmsSignalType]  # value = <CmsSignalType.SIGNAL_CML: 2>
    SIGNAL_CMOS: typing.ClassVar[CmsSignalType]  # value = <CmsSignalType.SIGNAL_CMOS: 0>
    SIGNAL_LVDS: typing.ClassVar[CmsSignalType]  # value = <CmsSignalType.SIGNAL_LVDS: 1>
    SIGNAL_LVPECL: typing.ClassVar[CmsSignalType]  # value = <CmsSignalType.SIGNAL_LVPECL: 3>
    SIGNAL_UNKNOWN: typing.ClassVar[CmsSignalType]  # value = <CmsSignalType.SIGNAL_UNKNOWN: 4>
    __members__: typing.ClassVar[
        dict[str, CmsSignalType]
    ]  # value = {'SIGNAL_CMOS': <CmsSignalType.SIGNAL_CMOS: 0>, 'SIGNAL_LVDS': <CmsSignalType.SIGNAL_LVDS: 1>, 'SIGNAL_CML': <CmsSignalType.SIGNAL_CML: 2>, 'SIGNAL_LVPECL': <CmsSignalType.SIGNAL_LVPECL: 3>, 'SIGNAL_UNKNOWN': <CmsSignalType.SIGNAL_UNKNOWN: 4>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsSingalCoupling:
    """
    Members:

      COUPLING_AC : AC coupled signal

      COUPLING_DC : DC signal

      COUPLING_UNKNOWN : UNKNOWN coupling
    """

    COUPLING_AC: typing.ClassVar[CmsSingalCoupling]  # value = <CmsSingalCoupling.COUPLING_AC: 0>
    COUPLING_DC: typing.ClassVar[CmsSingalCoupling]  # value = <CmsSingalCoupling.COUPLING_DC: 1>
    COUPLING_UNKNOWN: typing.ClassVar[CmsSingalCoupling]  # value = <CmsSingalCoupling.COUPLING_UNKNOWN: 2>
    __members__: typing.ClassVar[
        dict[str, CmsSingalCoupling]
    ]  # value = {'COUPLING_AC': <CmsSingalCoupling.COUPLING_AC: 0>, 'COUPLING_DC': <CmsSingalCoupling.COUPLING_DC: 1>, 'COUPLING_UNKNOWN': <CmsSingalCoupling.COUPLING_UNKNOWN: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsSpiAddrInc:
    """
    Members:

      SPI_ADDR_DEC_AUTO : auto decremented

      SPI_ADDR_INC_AUTO : auto incremented
    """

    SPI_ADDR_DEC_AUTO: typing.ClassVar[CmsSpiAddrInc]  # value = <CmsSpiAddrInc.SPI_ADDR_DEC_AUTO: 0>
    SPI_ADDR_INC_AUTO: typing.ClassVar[CmsSpiAddrInc]  # value = <CmsSpiAddrInc.SPI_ADDR_INC_AUTO: 1>
    __members__: typing.ClassVar[
        dict[str, CmsSpiAddrInc]
    ]  # value = {'SPI_ADDR_DEC_AUTO': <CmsSpiAddrInc.SPI_ADDR_DEC_AUTO: 0>, 'SPI_ADDR_INC_AUTO': <CmsSpiAddrInc.SPI_ADDR_INC_AUTO: 1>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsSpiMsbConfig:
    """
    Members:

      SPI_MSB_LAST : LSB first

      SPI_MSB_FIRST : MSB first
    """

    SPI_MSB_FIRST: typing.ClassVar[CmsSpiMsbConfig]  # value = <CmsSpiMsbConfig.SPI_MSB_FIRST: 1>
    SPI_MSB_LAST: typing.ClassVar[CmsSpiMsbConfig]  # value = <CmsSpiMsbConfig.SPI_MSB_LAST: 0>
    __members__: typing.ClassVar[
        dict[str, CmsSpiMsbConfig]
    ]  # value = {'SPI_MSB_LAST': <CmsSpiMsbConfig.SPI_MSB_LAST: 0>, 'SPI_MSB_FIRST': <CmsSpiMsbConfig.SPI_MSB_FIRST: 1>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CmsSpiSdoConfig:
    """
    Members:

      SPI_NONE : keep this for test

      SPI_SDO : SDO active, 4-wire only

      SPI_SDIO : SDIO active, 3-wire only
    """

    SPI_NONE: typing.ClassVar[CmsSpiSdoConfig]  # value = <CmsSpiSdoConfig.SPI_NONE: 0>
    SPI_SDIO: typing.ClassVar[CmsSpiSdoConfig]  # value = <CmsSpiSdoConfig.SPI_SDIO: 2>
    SPI_SDO: typing.ClassVar[CmsSpiSdoConfig]  # value = <CmsSpiSdoConfig.SPI_SDO: 1>
    __members__: typing.ClassVar[
        dict[str, CmsSpiSdoConfig]
    ]  # value = {'SPI_NONE': <CmsSpiSdoConfig.SPI_NONE: 0>, 'SPI_SDO': <CmsSpiSdoConfig.SPI_SDO: 1>, 'SPI_SDIO': <CmsSpiSdoConfig.SPI_SDIO: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DacChannelSelect:
    """
    Members:

      DAC_CH_NONE

      DAC_CH_0

      DAC_CH_1

      DAC_CH_2

      DAC_CH_3

      DAC_CH_4

      DAC_CH_5

      DAC_CH_6

      DAC_CH_7

      DAC_CH_ALL
    """

    DAC_CH_0: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_0: 1>
    DAC_CH_1: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_1: 2>
    DAC_CH_2: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_2: 4>
    DAC_CH_3: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_3: 8>
    DAC_CH_4: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_4: 16>
    DAC_CH_5: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_5: 32>
    DAC_CH_6: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_6: 64>
    DAC_CH_7: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_7: 128>
    DAC_CH_ALL: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_ALL: 255>
    DAC_CH_NONE: typing.ClassVar[DacChannelSelect]  # value = <DacChannelSelect.DAC_CH_NONE: 0>
    __members__: typing.ClassVar[
        dict[str, DacChannelSelect]
    ]  # value = {'DAC_CH_NONE': <DacChannelSelect.DAC_CH_NONE: 0>, 'DAC_CH_0': <DacChannelSelect.DAC_CH_0: 1>, 'DAC_CH_1': <DacChannelSelect.DAC_CH_1: 2>, 'DAC_CH_2': <DacChannelSelect.DAC_CH_2: 4>, 'DAC_CH_3': <DacChannelSelect.DAC_CH_3: 8>, 'DAC_CH_4': <DacChannelSelect.DAC_CH_4: 16>, 'DAC_CH_5': <DacChannelSelect.DAC_CH_5: 32>, 'DAC_CH_6': <DacChannelSelect.DAC_CH_6: 64>, 'DAC_CH_7': <DacChannelSelect.DAC_CH_7: 128>, 'DAC_CH_ALL': <DacChannelSelect.DAC_CH_ALL: 255>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DacModMuxMode:
    """
    Members:

      DAC_MUX_MODE_0 : I0.Q0 -> DAC0, I1.Q1 -> DAC1

      DAC_MUX_MODE_1 : (I0 + I1) / 2 -> DAC0, (Q0 + Q1) / 2 -> DAC1, Data Path NCOs Bypassed

      DAC_MUX_MODE_2 : I0 -> DAC0, Q0 -> DAC1, Datapath0 NCO Bypassed, Datapath1 Unused

      DAC_MUX_MODE_3 : (I0 + I1) / 2 -> DAC0, DAC1 Output Tied To Midscale
    """

    DAC_MUX_MODE_0: typing.ClassVar[DacModMuxMode]  # value = <DacModMuxMode.DAC_MUX_MODE_0: 0>
    DAC_MUX_MODE_1: typing.ClassVar[DacModMuxMode]  # value = <DacModMuxMode.DAC_MUX_MODE_1: 1>
    DAC_MUX_MODE_2: typing.ClassVar[DacModMuxMode]  # value = <DacModMuxMode.DAC_MUX_MODE_2: 2>
    DAC_MUX_MODE_3: typing.ClassVar[DacModMuxMode]  # value = <DacModMuxMode.DAC_MUX_MODE_3: 3>
    __members__: typing.ClassVar[
        dict[str, DacModMuxMode]
    ]  # value = {'DAC_MUX_MODE_0': <DacModMuxMode.DAC_MUX_MODE_0: 0>, 'DAC_MUX_MODE_1': <DacModMuxMode.DAC_MUX_MODE_1: 1>, 'DAC_MUX_MODE_2': <DacModMuxMode.DAC_MUX_MODE_2: 2>, 'DAC_MUX_MODE_3': <DacModMuxMode.DAC_MUX_MODE_3: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DacPairSelect:
    """
    Members:

      DAC_PAIR_NONE : No Group

      DAC_PAIR_0_1 : Group 0 (DAC0 & DAC1)

      DAC_PAIR_2_3 : Group 1 (DAC2 & DAC3)

      DAC_PAIR_ALL : All Groups
    """

    DAC_PAIR_0_1: typing.ClassVar[DacPairSelect]  # value = <DacPairSelect.DAC_PAIR_0_1: 1>
    DAC_PAIR_2_3: typing.ClassVar[DacPairSelect]  # value = <DacPairSelect.DAC_PAIR_2_3: 2>
    DAC_PAIR_ALL: typing.ClassVar[DacPairSelect]  # value = <DacPairSelect.DAC_PAIR_ALL: 3>
    DAC_PAIR_NONE: typing.ClassVar[DacPairSelect]  # value = <DacPairSelect.DAC_PAIR_NONE: 0>
    __members__: typing.ClassVar[
        dict[str, DacPairSelect]
    ]  # value = {'DAC_PAIR_NONE': <DacPairSelect.DAC_PAIR_NONE: 0>, 'DAC_PAIR_0_1': <DacPairSelect.DAC_PAIR_0_1: 1>, 'DAC_PAIR_2_3': <DacPairSelect.DAC_PAIR_2_3: 2>, 'DAC_PAIR_ALL': <DacPairSelect.DAC_PAIR_ALL: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DacSelect:
    """
    Members:

      DAC_NONE

      DAC_0

      DAC_1

      DAC_2

      DAC_3

      DAC_ALL
    """

    DAC_0: typing.ClassVar[DacSelect]  # value = <DacSelect.DAC_0: 1>
    DAC_1: typing.ClassVar[DacSelect]  # value = <DacSelect.DAC_1: 2>
    DAC_2: typing.ClassVar[DacSelect]  # value = <DacSelect.DAC_2: 4>
    DAC_3: typing.ClassVar[DacSelect]  # value = <DacSelect.DAC_3: 8>
    DAC_ALL: typing.ClassVar[DacSelect]  # value = <DacSelect.DAC_ALL: 15>
    DAC_NONE: typing.ClassVar[DacSelect]  # value = <DacSelect.DAC_NONE: 0>
    __members__: typing.ClassVar[
        dict[str, DacSelect]
    ]  # value = {'DAC_NONE': <DacSelect.DAC_NONE: 0>, 'DAC_0': <DacSelect.DAC_0: 1>, 'DAC_1': <DacSelect.DAC_1: 2>, 'DAC_2': <DacSelect.DAC_2: 4>, 'DAC_3': <DacSelect.DAC_3: 8>, 'DAC_ALL': <DacSelect.DAC_ALL: 15>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DesCtleInsertionLoss:
    """
    Members:

      DES_CTLE_IL_4DB_10DB : Insertion Loss from 4dB to 10dB

      DES_CTLE_IL_0DB_6DB : Insertion Loss from 0dB to 6dB

      DES_CTLE_IL_LT_0DB : Insertion Loss less than 0dB

      DES_CTLE_IL_FAR_LT_0DB : Insertion Loss far less than 0dB
    """

    DES_CTLE_IL_0DB_6DB: typing.ClassVar[DesCtleInsertionLoss]  # value = <DesCtleInsertionLoss.DES_CTLE_IL_0DB_6DB: 2>
    DES_CTLE_IL_4DB_10DB: typing.ClassVar[
        DesCtleInsertionLoss
    ]  # value = <DesCtleInsertionLoss.DES_CTLE_IL_4DB_10DB: 1>
    DES_CTLE_IL_FAR_LT_0DB: typing.ClassVar[
        DesCtleInsertionLoss
    ]  # value = <DesCtleInsertionLoss.DES_CTLE_IL_FAR_LT_0DB: 4>
    DES_CTLE_IL_LT_0DB: typing.ClassVar[DesCtleInsertionLoss]  # value = <DesCtleInsertionLoss.DES_CTLE_IL_LT_0DB: 3>
    __members__: typing.ClassVar[
        dict[str, DesCtleInsertionLoss]
    ]  # value = {'DES_CTLE_IL_4DB_10DB': <DesCtleInsertionLoss.DES_CTLE_IL_4DB_10DB: 1>, 'DES_CTLE_IL_0DB_6DB': <DesCtleInsertionLoss.DES_CTLE_IL_0DB_6DB: 2>, 'DES_CTLE_IL_LT_0DB': <DesCtleInsertionLoss.DES_CTLE_IL_LT_0DB: 3>, 'DES_CTLE_IL_FAR_LT_0DB': <DesCtleInsertionLoss.DES_CTLE_IL_FAR_LT_0DB: 4>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DesSettings:
    boost_mask: int
    cal_mode: CalMode
    invert_mask: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...
    @property
    def ctle_coeffs(self) -> numpy.ndarray: ...
    @property
    def ctle_filter(self) -> numpy.ndarray: ...
    @property
    def lane_mapping(self) -> numpy.ndarray: ...

class Device:
    clk_info: Clk
    dev_info: Info
    serdes_info: SerdesSettings
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...
    def callback_set(
        self,
        arg0: typing.Callable,
        arg1: typing.Callable,
        arg2: typing.Callable,
        arg3: typing.Callable,
        arg4: typing.Callable,
    ) -> None: ...
    def callback_unset(self) -> None: ...
    def spi_conf_set(self, arg0: CmsSpiSdoConfig, arg1: CmsSpiMsbConfig, arg2: CmsSpiAddrInc) -> None: ...

class Info:
    adc_freq_hz: int
    dac_freq_hz: int
    dev_freq_hz: int
    dev_rev: int
    jesd_rx_lane_rate: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...

class JesdLinkSelect:
    """
    Members:

      LINK_NONE : No Link

      LINK_0 : Link 0

      LINK_1 : Link 1

      LINK_ALL : All Links
    """

    LINK_0: typing.ClassVar[JesdLinkSelect]  # value = <JesdLinkSelect.LINK_0: 1>
    LINK_1: typing.ClassVar[JesdLinkSelect]  # value = <JesdLinkSelect.LINK_1: 2>
    LINK_ALL: typing.ClassVar[JesdLinkSelect]  # value = <JesdLinkSelect.LINK_ALL: 3>
    LINK_NONE: typing.ClassVar[JesdLinkSelect]  # value = <JesdLinkSelect.LINK_NONE: 0>
    __members__: typing.ClassVar[
        dict[str, JesdLinkSelect]
    ]  # value = {'LINK_NONE': <JesdLinkSelect.LINK_NONE: 0>, 'LINK_0': <JesdLinkSelect.LINK_0: 1>, 'LINK_1': <JesdLinkSelect.LINK_1: 2>, 'LINK_ALL': <JesdLinkSelect.LINK_ALL: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class JtxConvSel:
    virtual_converter0_index: int
    virtual_converter1_index: int
    virtual_converter2_index: int
    virtual_converter3_index: int
    virtual_converter4_index: int
    virtual_converter5_index: int
    virtual_converter6_index: int
    virtual_converter7_index: int
    virtual_converter8_index: int
    virtual_converter9_index: int
    virtual_convertera_index: int
    virtual_converterb_index: int
    virtual_converterc_index: int
    virtual_converterd_index: int
    virtual_convertere_index: int
    virtual_converterf_index: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...

class LinkStatus:
    """
    Members:

      LINK_STATUS_RESET

      LINK_STATUS_SH_FAILURE

      LINK_STATUS_SH_ALIGNED

      LINK_STATUS_EMB_SYNCED

      LINK_STATUS_EMB_ALIGNED

      LINK_STATUS_LOCK_FAILURE

      LINK_STATUS_LOCKED

      LINK_STATUS_UNKNOWN
    """

    LINK_STATUS_EMB_ALIGNED: typing.ClassVar[LinkStatus]  # value = <LinkStatus.LINK_STATUS_EMB_ALIGNED: 4>
    LINK_STATUS_EMB_SYNCED: typing.ClassVar[LinkStatus]  # value = <LinkStatus.LINK_STATUS_EMB_SYNCED: 3>
    LINK_STATUS_LOCKED: typing.ClassVar[LinkStatus]  # value = <LinkStatus.LINK_STATUS_LOCKED: 6>
    LINK_STATUS_LOCK_FAILURE: typing.ClassVar[LinkStatus]  # value = <LinkStatus.LINK_STATUS_LOCK_FAILURE: 5>
    LINK_STATUS_RESET: typing.ClassVar[LinkStatus]  # value = <LinkStatus.LINK_STATUS_RESET: 0>
    LINK_STATUS_SH_ALIGNED: typing.ClassVar[LinkStatus]  # value = <LinkStatus.LINK_STATUS_SH_ALIGNED: 2>
    LINK_STATUS_SH_FAILURE: typing.ClassVar[LinkStatus]  # value = <LinkStatus.LINK_STATUS_SH_FAILURE: 1>
    LINK_STATUS_UNKNOWN: typing.ClassVar[LinkStatus]  # value = <LinkStatus.LINK_STATUS_UNKNOWN: 255>
    __members__: typing.ClassVar[
        dict[str, LinkStatus]
    ]  # value = {'LINK_STATUS_RESET': <LinkStatus.LINK_STATUS_RESET: 0>, 'LINK_STATUS_SH_FAILURE': <LinkStatus.LINK_STATUS_SH_FAILURE: 1>, 'LINK_STATUS_SH_ALIGNED': <LinkStatus.LINK_STATUS_SH_ALIGNED: 2>, 'LINK_STATUS_EMB_SYNCED': <LinkStatus.LINK_STATUS_EMB_SYNCED: 3>, 'LINK_STATUS_EMB_ALIGNED': <LinkStatus.LINK_STATUS_EMB_ALIGNED: 4>, 'LINK_STATUS_LOCK_FAILURE': <LinkStatus.LINK_STATUS_LOCK_FAILURE: 5>, 'LINK_STATUS_LOCKED': <LinkStatus.LINK_STATUS_LOCKED: 6>, 'LINK_STATUS_UNKNOWN': <LinkStatus.LINK_STATUS_UNKNOWN: 255>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class NcoFtw:
    delta_b: int
    ftw: int
    modulus_a: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...

class RegData:
    addr: int
    data: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    @typing.overload
    def __init__(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: int, arg1: int) -> None: ...

class Reset:
    """
    Members:

      SOFT_RESET : Soft Reset

      HARD_RESET : Hard Reset

      SOFT_RESET_AND_INIT : Soft Reset Then Init

      HARD_RESET_AND_INIT : Hard Reset Then Init
    """

    HARD_RESET: typing.ClassVar[Reset]  # value = <Reset.HARD_RESET: 1>
    HARD_RESET_AND_INIT: typing.ClassVar[Reset]  # value = <Reset.HARD_RESET_AND_INIT: 3>
    SOFT_RESET: typing.ClassVar[Reset]  # value = <Reset.SOFT_RESET: 0>
    SOFT_RESET_AND_INIT: typing.ClassVar[Reset]  # value = <Reset.SOFT_RESET_AND_INIT: 2>
    __members__: typing.ClassVar[
        dict[str, Reset]
    ]  # value = {'SOFT_RESET': <Reset.SOFT_RESET: 0>, 'HARD_RESET': <Reset.HARD_RESET: 1>, 'SOFT_RESET_AND_INIT': <Reset.SOFT_RESET_AND_INIT: 2>, 'HARD_RESET_AND_INIT': <Reset.HARD_RESET_AND_INIT: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __and__(self, other: typing.Any) -> typing.Any: ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __ge__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> typing.Any: ...
    def __le__(self, other: typing.Any) -> bool: ...
    def __lt__(self, other: typing.Any) -> bool: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __or__(self, other: typing.Any) -> typing.Any: ...
    def __rand__(self, other: typing.Any) -> typing.Any: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: typing.Any) -> typing.Any: ...
    def __rxor__(self, other: typing.Any) -> typing.Any: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    def __xor__(self, other: typing.Any) -> typing.Any: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SerLaneSettings:
    post_emp_setting: SerPostEmp
    pre_emp_setting: SerPreEmp
    swing_setting: SerSwing
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...

class SerLaneSettingsField:
    """
    Members:

      SWING_SETTING

      PRE_EMP_SETTING

      POST_EMP_SETTING
    """

    POST_EMP_SETTING: typing.ClassVar[SerLaneSettingsField]  # value = <SerLaneSettingsField.POST_EMP_SETTING: 2>
    PRE_EMP_SETTING: typing.ClassVar[SerLaneSettingsField]  # value = <SerLaneSettingsField.PRE_EMP_SETTING: 1>
    SWING_SETTING: typing.ClassVar[SerLaneSettingsField]  # value = <SerLaneSettingsField.SWING_SETTING: 0>
    __members__: typing.ClassVar[
        dict[str, SerLaneSettingsField]
    ]  # value = {'SWING_SETTING': <SerLaneSettingsField.SWING_SETTING: 0>, 'PRE_EMP_SETTING': <SerLaneSettingsField.PRE_EMP_SETTING: 1>, 'POST_EMP_SETTING': <SerLaneSettingsField.POST_EMP_SETTING: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SerPostEmp:
    """
    Members:

      SER_POST_EMP_0DB : 0 dB Post-Emphasis

      SER_POST_EMP_3DB : 3 dB Post-Emphasis

      SER_POST_EMP_6DB : 6 dB Post-Emphasis

      SER_POST_EMP_9DB : 9 dB Post-Emphasis

      SER_POST_EMP_12DB : 12 dB Post-Emphasis
    """

    SER_POST_EMP_0DB: typing.ClassVar[SerPostEmp]  # value = <SerPostEmp.SER_POST_EMP_0DB: 0>
    SER_POST_EMP_12DB: typing.ClassVar[SerPostEmp]  # value = <SerPostEmp.SER_POST_EMP_12DB: 4>
    SER_POST_EMP_3DB: typing.ClassVar[SerPostEmp]  # value = <SerPostEmp.SER_POST_EMP_3DB: 1>
    SER_POST_EMP_6DB: typing.ClassVar[SerPostEmp]  # value = <SerPostEmp.SER_POST_EMP_6DB: 2>
    SER_POST_EMP_9DB: typing.ClassVar[SerPostEmp]  # value = <SerPostEmp.SER_POST_EMP_9DB: 3>
    __members__: typing.ClassVar[
        dict[str, SerPostEmp]
    ]  # value = {'SER_POST_EMP_0DB': <SerPostEmp.SER_POST_EMP_0DB: 0>, 'SER_POST_EMP_3DB': <SerPostEmp.SER_POST_EMP_3DB: 1>, 'SER_POST_EMP_6DB': <SerPostEmp.SER_POST_EMP_6DB: 2>, 'SER_POST_EMP_9DB': <SerPostEmp.SER_POST_EMP_9DB: 3>, 'SER_POST_EMP_12DB': <SerPostEmp.SER_POST_EMP_12DB: 4>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SerPreEmp:
    """
    Members:

      SER_PRE_EMP_0DB : 0 dB Pre-Emphasis

      SER_PRE_EMP_3DB : 3 dB Pre-Emphasis

      SER_PRE_EMP_6DB : 6 dB Pre-Emphasis
    """

    SER_PRE_EMP_0DB: typing.ClassVar[SerPreEmp]  # value = <SerPreEmp.SER_PRE_EMP_0DB: 0>
    SER_PRE_EMP_3DB: typing.ClassVar[SerPreEmp]  # value = <SerPreEmp.SER_PRE_EMP_3DB: 1>
    SER_PRE_EMP_6DB: typing.ClassVar[SerPreEmp]  # value = <SerPreEmp.SER_PRE_EMP_6DB: 2>
    __members__: typing.ClassVar[
        dict[str, SerPreEmp]
    ]  # value = {'SER_PRE_EMP_0DB': <SerPreEmp.SER_PRE_EMP_0DB: 0>, 'SER_PRE_EMP_3DB': <SerPreEmp.SER_PRE_EMP_3DB: 1>, 'SER_PRE_EMP_6DB': <SerPreEmp.SER_PRE_EMP_6DB: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SerSettings:
    invert_mask: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...
    @property
    def lane_mapping(self) -> numpy.ndarray: ...
    @property
    def lane_settings(self) -> numpy.ndarray: ...

class SerSwing:
    """
    Members:

      SER_SWING_1000 : 1000 mV Swing

      SER_SWING_850 : 850 mV Swing

      SER_SWING_750 : 750 mV Swing

      SER_SWING_500 : 500 mV Swing
    """

    SER_SWING_1000: typing.ClassVar[SerSwing]  # value = <SerSwing.SER_SWING_1000: 0>
    SER_SWING_500: typing.ClassVar[SerSwing]  # value = <SerSwing.SER_SWING_500: 3>
    SER_SWING_750: typing.ClassVar[SerSwing]  # value = <SerSwing.SER_SWING_750: 2>
    SER_SWING_850: typing.ClassVar[SerSwing]  # value = <SerSwing.SER_SWING_850: 1>
    __members__: typing.ClassVar[
        dict[str, SerSwing]
    ]  # value = {'SER_SWING_1000': <SerSwing.SER_SWING_1000: 0>, 'SER_SWING_850': <SerSwing.SER_SWING_850: 1>, 'SER_SWING_750': <SerSwing.SER_SWING_750: 2>, 'SER_SWING_500': <SerSwing.SER_SWING_500: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SerdesSettings:
    des_settings: DesSettings
    ser_settings: SerSettings
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __init__(self) -> None: ...

def adc_ddc_coarse_nco_ftw_get(arg0: Device, arg1: int, arg2: NcoFtw) -> int: ...
def adc_ddc_coarse_nco_ftw_set(arg0: Device, arg1: int, arg2: NcoFtw) -> int: ...
def adc_ddc_coarse_nco_mode_set(arg0: Device, arg1: int, arg2: AdcNcoMode) -> int: ...
def adc_ddc_coarse_select_set(arg0: Device, arg1: int) -> int: ...
def adc_ddc_coarse_sync_enable_set(arg0: Device, arg1: int, arg2: int) -> int: ...
def adc_ddc_coarse_sync_next_set(arg0: Device, arg1: int, arg2: int) -> int: ...
def adc_ddc_coarse_trig_nco_reset_enable_set(arg0: Device, arg1: int, arg2: int) -> int: ...
def adc_ddc_fine_nco_ftw_get(arg0: Device, arg1: int, arg2: NcoFtw) -> int: ...
def adc_ddc_fine_nco_ftw_set(arg0: Device, arg1: int, arg2: NcoFtw) -> int: ...
def adc_ddc_fine_nco_mode_set(arg0: Device, arg1: int, arg2: AdcNcoMode) -> int: ...
def adc_ddc_fine_select_set(arg0: Device, arg1: int) -> int: ...
def adc_ddc_fine_sync_enable_set(arg0: Device, arg1: int, arg2: int) -> int: ...
def adc_ddc_fine_sync_next_set(arg0: Device, arg1: int, arg2: int) -> int: ...
def adc_ddc_fine_trig_nco_reset_enable_set(arg0: Device, arg1: int, arg2: int) -> int: ...
def dac_chan_select_set(arg0: Device, arg1: int) -> int: ...
def dac_duc_chan_skew_set(arg0: Device, arg1: int, arg2: int) -> int: ...
def dac_duc_nco_enable_set(arg0: Device, arg1: int, arg2: int, arg3: int) -> int: ...
def dac_duc_nco_ftw_set(arg0: Device, arg1: int, arg2: int, arg3: NcoFtw) -> int: ...
def dac_duc_nco_gains_set(
    arg0: Device, arg1: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)]
) -> int: ...
def dac_fsc_set(arg0: Device, arg1: int, arg2: int, arg3: int) -> int: ...
def dac_mode_switch_group_select_set(arg0: Device, arg1: DacPairSelect) -> int: ...
def dac_modulation_mux_mode_set(arg0: Device, arg1: DacPairSelect, arg2: DacModMuxMode) -> int: ...
def dac_select_set(arg0: Device, arg1: int) -> int: ...
def dac_xbar_set(arg0: Device, arg1: int, arg2: int) -> int: ...
def device_api_revision_get(arg0: Device, arg1: ApiRevision) -> int: ...
def device_chip_id_get(arg0: Device, arg1: CmsChipId) -> int: ...
def device_clk_config_set(arg0: Device, arg1: int, arg2: int, arg3: int) -> int: ...
def device_clk_pll_lock_status_get(arg0: Device) -> tuple[int, int]: ...
def device_get_temperature(arg0: Device, arg1: ChipTemperatures) -> int: ...
def device_init(arg0: Device) -> int: ...
def device_reset(arg0: Device, arg1: Reset) -> int: ...
def device_startup_rx(
    arg0: Device,
    arg1: int,
    arg2: int,
    arg3: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)],
    arg4: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)],
    arg5: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)],
    arg6: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)],
    arg7: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)],
    arg8: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)],
    arg9: typing.Annotated[list[CmsJesdParam], pybind11_stubgen.typing_ext.FixedSize(2)],
    arg10: typing.Annotated[list[JtxConvSel], pybind11_stubgen.typing_ext.FixedSize(2)],
) -> int: ...
def device_startup_tx(
    arg0: Device,
    arg1: int,
    arg2: int,
    arg3: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)],
    arg4: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)],
    arg5: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)],
    arg6: CmsJesdParam,
) -> int: ...
def hal_calc_rx_nco_ftw(arg0: Device, arg1: int, arg2: int, arg3: NcoFtw) -> int: ...
def hal_calc_tx_nco_ftw(arg0: Device, arg1: int, arg2: int, arg3: NcoFtw) -> int: ...
def hal_delay_us(arg0: Device, arg1: int) -> int: ...
def hal_reg_get(arg0: Device, arg1: RegData) -> int: ...
def hal_reg_set(arg0: Device, arg1: RegData) -> int: ...
def hal_reset_pin_ctrl(arg0: Device, arg1: int) -> int: ...
def jesd_oneshot_sync(arg0: Device, arg1: CmsJesdSubclass) -> int: ...
def jesd_rx_204c_crc_irq_clr(arg0: Device, arg1: JesdLinkSelect) -> int: ...
def jesd_rx_204c_crc_irq_enable(arg0: Device, arg1: JesdLinkSelect, arg2: int) -> int: ...
def jesd_rx_204c_crc_irq_status_get(arg0: Device, arg1: JesdLinkSelect) -> tuple[int, int]: ...
def jesd_rx_204c_mb_irq_clr(arg0: Device, arg1: JesdLinkSelect) -> int: ...
def jesd_rx_204c_mb_irq_enable(arg0: Device, arg1: JesdLinkSelect, arg2: int) -> int: ...
def jesd_rx_204c_mb_irq_status_get(arg0: Device, arg1: JesdLinkSelect) -> tuple[int, int]: ...
def jesd_rx_204c_sh_irq_clr(arg0: Device, arg1: JesdLinkSelect) -> int: ...
def jesd_rx_204c_sh_irq_enable(arg0: Device, arg1: JesdLinkSelect, arg2: int) -> int: ...
def jesd_rx_204c_sh_irq_status_get(arg0: Device, arg1: JesdLinkSelect) -> tuple[int, int]: ...
def jesd_rx_bit_rate_get(arg0: Device) -> tuple[int, int]: ...
def jesd_rx_calibrate_204c(arg0: Device, arg1: int, arg2: int, arg3: int) -> int: ...
def jesd_rx_config_status_get(arg0: Device) -> tuple[int, int]: ...
def jesd_rx_ctle_manual_config_get(arg0: Device, arg1: int) -> int: ...
def jesd_rx_lanes_xbar_set(
    arg0: Device, arg1: JesdLinkSelect, arg2: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)]
) -> int: ...
def jesd_rx_link_enable_set(arg0: Device, arg1: JesdLinkSelect, arg2: int) -> int: ...
def jesd_rx_link_select_set(arg0: Device, arg1: JesdLinkSelect) -> int: ...
def jesd_rx_link_status_get(arg0: Device, arg1: JesdLinkSelect) -> tuple[int, LinkStatus, int]: ...
def jesd_sysref_average_set(arg0: Device, arg1: int) -> int: ...
def jesd_sysref_monitor_phase_get(arg0: Device) -> tuple[int, int]: ...
def jesd_sysref_setup_hold_get(arg0: Device) -> tuple[int, int, int]: ...
def jesd_tx_lanes_xbar_set(
    arg0: Device, arg1: JesdLinkSelect, arg2: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)]
) -> int: ...
def jesd_tx_link_enable_set(arg0: Device, arg1: JesdLinkSelect, arg2: int) -> int: ...
def jesd_tx_link_select_set(arg0: Device, arg1: JesdLinkSelect) -> int: ...
def sync_calc_jrx_lmfc_lemc(arg0: int, arg1: int, arg2: int, arg3: CmsJesdParam) -> tuple[int, int]: ...
def sync_calc_jtx_lmfc_lemc(
    arg0: int,
    arg1: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)],
    arg2: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)],
    arg3: JesdLinkSelect,
    arg4: typing.Annotated[list[CmsJesdParam], pybind11_stubgen.typing_ext.FixedSize(2)],
) -> tuple[int, int]: ...
def sync_sysref_frequency_set(
    arg0: Device,
    arg1: int,
    arg2: int,
    arg3: int,
    arg4: int,
    arg5: int,
    arg6: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)],
    arg7: typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(8)],
    arg8: JesdLinkSelect,
    arg9: CmsJesdParam,
    arg10: typing.Annotated[list[CmsJesdParam], pybind11_stubgen.typing_ext.FixedSize(2)],
) -> tuple[int, int]: ...
def sync_sysref_input_config_set(
    arg0: Device, arg1: CmsSingalCoupling, arg2: CmsSignalType, arg3: int, arg4: int
) -> int: ...

AD9082_CAL_MODE_BYPASS: CalMode  # value = <CalMode.AD9082_CAL_MODE_BYPASS: 2>
AD9082_CAL_MODE_RUN: CalMode  # value = <CalMode.AD9082_CAL_MODE_RUN: 0>
AD9082_CAL_MODE_RUN_AND_SAVE: CalMode  # value = <CalMode.AD9082_CAL_MODE_RUN_AND_SAVE: 1>
ADC_CDDC_0: AdcCoarseDdcSelect  # value = <AdcCoarseDdcSelect.ADC_CDDC_0: 1>
ADC_CDDC_1: AdcCoarseDdcSelect  # value = <AdcCoarseDdcSelect.ADC_CDDC_1: 2>
ADC_CDDC_2: AdcCoarseDdcSelect  # value = <AdcCoarseDdcSelect.ADC_CDDC_2: 4>
ADC_CDDC_3: AdcCoarseDdcSelect  # value = <AdcCoarseDdcSelect.ADC_CDDC_3: 8>
ADC_CDDC_ALL: AdcCoarseDdcSelect  # value = <AdcCoarseDdcSelect.ADC_CDDC_ALL: 15>
ADC_CDDC_DCM_1: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_1: 12>
ADC_CDDC_DCM_12: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_12: 6>
ADC_CDDC_DCM_16: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_16: 3>
ADC_CDDC_DCM_18: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_18: 10>
ADC_CDDC_DCM_2: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_2: 0>
ADC_CDDC_DCM_24: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_24: 7>
ADC_CDDC_DCM_3: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_3: 8>
ADC_CDDC_DCM_36: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_36: 11>
ADC_CDDC_DCM_4: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_4: 1>
ADC_CDDC_DCM_6: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_6: 5>
ADC_CDDC_DCM_8: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_8: 2>
ADC_CDDC_DCM_9: AdcCoarseDdcDcm  # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_9: 9>
ADC_CDDC_NONE: AdcCoarseDdcSelect  # value = <AdcCoarseDdcSelect.ADC_CDDC_NONE: 0>
ADC_FDDC_0: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_0: 1>
ADC_FDDC_1: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_1: 2>
ADC_FDDC_2: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_2: 4>
ADC_FDDC_3: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_3: 8>
ADC_FDDC_4: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_4: 16>
ADC_FDDC_5: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_5: 32>
ADC_FDDC_6: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_6: 64>
ADC_FDDC_7: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_7: 128>
ADC_FDDC_ALL: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_ALL: 255>
ADC_FDDC_DCM_1: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_1: 8>
ADC_FDDC_DCM_12: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_12: 6>
ADC_FDDC_DCM_16: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_16: 3>
ADC_FDDC_DCM_2: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_2: 0>
ADC_FDDC_DCM_24: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_24: 7>
ADC_FDDC_DCM_3: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_3: 4>
ADC_FDDC_DCM_4: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_4: 1>
ADC_FDDC_DCM_6: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_6: 5>
ADC_FDDC_DCM_8: AdcFineDdcDcm  # value = <AdcFineDdcDcm.ADC_FDDC_DCM_8: 2>
ADC_FDDC_NONE: AdcFineDdcSelect  # value = <AdcFineDdcSelect.ADC_FDDC_NONE: 0>
ADC_NCO_FS_4_IF: AdcNcoMode  # value = <AdcNcoMode.ADC_NCO_FS_4_IF: 2>
ADC_NCO_TEST: AdcNcoMode  # value = <AdcNcoMode.ADC_NCO_TEST: 3>
ADC_NCO_VIF: AdcNcoMode  # value = <AdcNcoMode.ADC_NCO_VIF: 0>
ADC_NCO_ZIF: AdcNcoMode  # value = <AdcNcoMode.ADC_NCO_ZIF: 1>
API_CMS_ERROR_DELAY_US: CmsError  # value = <CmsError.API_CMS_ERROR_DELAY_US: -70>
API_CMS_ERROR_DLL_NOT_LOCKED: CmsError  # value = <CmsError.API_CMS_ERROR_DLL_NOT_LOCKED: -22>
API_CMS_ERROR_ERROR: CmsError  # value = <CmsError.API_CMS_ERROR_ERROR: -1>
API_CMS_ERROR_EVENT_HNDL: CmsError  # value = <CmsError.API_CMS_ERROR_EVENT_HNDL: -64>
API_CMS_ERROR_FTW_LOAD_ACK: CmsError  # value = <CmsError.API_CMS_ERROR_FTW_LOAD_ACK: -30>
API_CMS_ERROR_HW_CLOSE: CmsError  # value = <CmsError.API_CMS_ERROR_HW_CLOSE: -66>
API_CMS_ERROR_HW_OPEN: CmsError  # value = <CmsError.API_CMS_ERROR_HW_OPEN: -65>
API_CMS_ERROR_INIT_SEQ_FAIL: CmsError  # value = <CmsError.API_CMS_ERROR_INIT_SEQ_FAIL: -40>
API_CMS_ERROR_INVALID_DELAYUS_PTR: CmsError  # value = <CmsError.API_CMS_ERROR_INVALID_DELAYUS_PTR: -13>
API_CMS_ERROR_INVALID_HANDLE_PTR: CmsError  # value = <CmsError.API_CMS_ERROR_INVALID_HANDLE_PTR: -11>
API_CMS_ERROR_INVALID_PARAM: CmsError  # value = <CmsError.API_CMS_ERROR_INVALID_PARAM: -14>
API_CMS_ERROR_INVALID_RESET_CTRL_PTR: CmsError  # value = <CmsError.API_CMS_ERROR_INVALID_RESET_CTRL_PTR: -15>
API_CMS_ERROR_INVALID_XFER_PTR: CmsError  # value = <CmsError.API_CMS_ERROR_INVALID_XFER_PTR: -12>
API_CMS_ERROR_JESD_PLL_NOT_LOCKED: CmsError  # value = <CmsError.API_CMS_ERROR_JESD_PLL_NOT_LOCKED: -24>
API_CMS_ERROR_JESD_SYNC_NOT_DONE: CmsError  # value = <CmsError.API_CMS_ERROR_JESD_SYNC_NOT_DONE: -25>
API_CMS_ERROR_LOG_CLOSE: CmsError  # value = <CmsError.API_CMS_ERROR_LOG_CLOSE: -69>
API_CMS_ERROR_LOG_OPEN: CmsError  # value = <CmsError.API_CMS_ERROR_LOG_OPEN: -67>
API_CMS_ERROR_LOG_WRITE: CmsError  # value = <CmsError.API_CMS_ERROR_LOG_WRITE: -68>
API_CMS_ERROR_MODE_NOT_IN_TABLE: CmsError  # value = <CmsError.API_CMS_ERROR_MODE_NOT_IN_TABLE: -23>
API_CMS_ERROR_NCO_NOT_ENABLED: CmsError  # value = <CmsError.API_CMS_ERROR_NCO_NOT_ENABLED: -31>
API_CMS_ERROR_NOT_SUPPORTED: CmsError  # value = <CmsError.API_CMS_ERROR_NOT_SUPPORTED: -16>
API_CMS_ERROR_NULL_PARAM: CmsError  # value = <CmsError.API_CMS_ERROR_NULL_PARAM: -2>
API_CMS_ERROR_OK: CmsError  # value = <CmsError.API_CMS_ERROR_OK: 0>
API_CMS_ERROR_PD_STBY_PIN_CTRL: CmsError  # value = <CmsError.API_CMS_ERROR_PD_STBY_PIN_CTRL: -71>
API_CMS_ERROR_PLL_NOT_LOCKED: CmsError  # value = <CmsError.API_CMS_ERROR_PLL_NOT_LOCKED: -21>
API_CMS_ERROR_RESET_PIN_CTRL: CmsError  # value = <CmsError.API_CMS_ERROR_RESET_PIN_CTRL: -63>
API_CMS_ERROR_SPI_SDO: CmsError  # value = <CmsError.API_CMS_ERROR_SPI_SDO: -10>
API_CMS_ERROR_SPI_XFER: CmsError  # value = <CmsError.API_CMS_ERROR_SPI_XFER: -60>
API_CMS_ERROR_SYSREF_CTRL: CmsError  # value = <CmsError.API_CMS_ERROR_SYSREF_CTRL: -72>
API_CMS_ERROR_TEST_FAILED: CmsError  # value = <CmsError.API_CMS_ERROR_TEST_FAILED: -50>
API_CMS_ERROR_TX_EN_PIN_CTRL: CmsError  # value = <CmsError.API_CMS_ERROR_TX_EN_PIN_CTRL: -62>
API_CMS_ERROR_VCO_OUT_OF_RANGE: CmsError  # value = <CmsError.API_CMS_ERROR_VCO_OUT_OF_RANGE: -20>
COUPLING_AC: CmsSingalCoupling  # value = <CmsSingalCoupling.COUPLING_AC: 0>
COUPLING_DC: CmsSingalCoupling  # value = <CmsSingalCoupling.COUPLING_DC: 1>
COUPLING_UNKNOWN: CmsSingalCoupling  # value = <CmsSingalCoupling.COUPLING_UNKNOWN: 2>
DAC_0: DacSelect  # value = <DacSelect.DAC_0: 1>
DAC_1: DacSelect  # value = <DacSelect.DAC_1: 2>
DAC_2: DacSelect  # value = <DacSelect.DAC_2: 4>
DAC_3: DacSelect  # value = <DacSelect.DAC_3: 8>
DAC_ALL: DacSelect  # value = <DacSelect.DAC_ALL: 15>
DAC_CH_0: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_0: 1>
DAC_CH_1: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_1: 2>
DAC_CH_2: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_2: 4>
DAC_CH_3: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_3: 8>
DAC_CH_4: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_4: 16>
DAC_CH_5: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_5: 32>
DAC_CH_6: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_6: 64>
DAC_CH_7: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_7: 128>
DAC_CH_ALL: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_ALL: 255>
DAC_CH_NONE: DacChannelSelect  # value = <DacChannelSelect.DAC_CH_NONE: 0>
DAC_MUX_MODE_0: DacModMuxMode  # value = <DacModMuxMode.DAC_MUX_MODE_0: 0>
DAC_MUX_MODE_1: DacModMuxMode  # value = <DacModMuxMode.DAC_MUX_MODE_1: 1>
DAC_MUX_MODE_2: DacModMuxMode  # value = <DacModMuxMode.DAC_MUX_MODE_2: 2>
DAC_MUX_MODE_3: DacModMuxMode  # value = <DacModMuxMode.DAC_MUX_MODE_3: 3>
DAC_NONE: DacSelect  # value = <DacSelect.DAC_NONE: 0>
DAC_PAIR_0_1: DacPairSelect  # value = <DacPairSelect.DAC_PAIR_0_1: 1>
DAC_PAIR_2_3: DacPairSelect  # value = <DacPairSelect.DAC_PAIR_2_3: 2>
DAC_PAIR_ALL: DacPairSelect  # value = <DacPairSelect.DAC_PAIR_ALL: 3>
DAC_PAIR_NONE: DacPairSelect  # value = <DacPairSelect.DAC_PAIR_NONE: 0>
DES_CTLE_IL_0DB_6DB: DesCtleInsertionLoss  # value = <DesCtleInsertionLoss.DES_CTLE_IL_0DB_6DB: 2>
DES_CTLE_IL_4DB_10DB: DesCtleInsertionLoss  # value = <DesCtleInsertionLoss.DES_CTLE_IL_4DB_10DB: 1>
DES_CTLE_IL_FAR_LT_0DB: DesCtleInsertionLoss  # value = <DesCtleInsertionLoss.DES_CTLE_IL_FAR_LT_0DB: 4>
DES_CTLE_IL_LT_0DB: DesCtleInsertionLoss  # value = <DesCtleInsertionLoss.DES_CTLE_IL_LT_0DB: 3>
HARD_RESET: Reset  # value = <Reset.HARD_RESET: 1>
HARD_RESET_AND_INIT: Reset  # value = <Reset.HARD_RESET_AND_INIT: 3>
JESD_LINK_0: CmdJesdLink  # value = <CmdJesdLink.JESD_LINK_0: 1>
JESD_LINK_1: CmdJesdLink  # value = <CmdJesdLink.JESD_LINK_1: 2>
JESD_LINK_ALL: CmdJesdLink  # value = <CmdJesdLink.JESD_LINK_ALL: 3>
JESD_LINK_NONE: CmdJesdLink  # value = <CmdJesdLink.JESD_LINK_NONE: 0>
JESD_SUBCLASS_0: CmsJesdSubclass  # value = <CmsJesdSubclass.JESD_SUBCLASS_0: 0>
JESD_SUBCLASS_1: CmsJesdSubclass  # value = <CmsJesdSubclass.JESD_SUBCLASS_1: 1>
JESD_SUBCLASS_INVALID: CmsJesdSubclass  # value = <CmsJesdSubclass.JESD_SUBCLASS_INVALID: 2>
LINK_0: JesdLinkSelect  # value = <JesdLinkSelect.LINK_0: 1>
LINK_1: JesdLinkSelect  # value = <JesdLinkSelect.LINK_1: 2>
LINK_ALL: JesdLinkSelect  # value = <JesdLinkSelect.LINK_ALL: 3>
LINK_NONE: JesdLinkSelect  # value = <JesdLinkSelect.LINK_NONE: 0>
LINK_STATUS_EMB_ALIGNED: LinkStatus  # value = <LinkStatus.LINK_STATUS_EMB_ALIGNED: 4>
LINK_STATUS_EMB_SYNCED: LinkStatus  # value = <LinkStatus.LINK_STATUS_EMB_SYNCED: 3>
LINK_STATUS_LOCKED: LinkStatus  # value = <LinkStatus.LINK_STATUS_LOCKED: 6>
LINK_STATUS_LOCK_FAILURE: LinkStatus  # value = <LinkStatus.LINK_STATUS_LOCK_FAILURE: 5>
LINK_STATUS_RESET: LinkStatus  # value = <LinkStatus.LINK_STATUS_RESET: 0>
LINK_STATUS_SH_ALIGNED: LinkStatus  # value = <LinkStatus.LINK_STATUS_SH_ALIGNED: 2>
LINK_STATUS_SH_FAILURE: LinkStatus  # value = <LinkStatus.LINK_STATUS_SH_FAILURE: 1>
LINK_STATUS_UNKNOWN: LinkStatus  # value = <LinkStatus.LINK_STATUS_UNKNOWN: 255>
POST_EMP_SETTING: SerLaneSettingsField  # value = <SerLaneSettingsField.POST_EMP_SETTING: 2>
PRBS15: CmsJesdPrbsPattern  # value = <CmsJesdPrbsPattern.PRBS15: 3>
PRBS23: CmsJesdPrbsPattern  # value = <CmsJesdPrbsPattern.PRBS23: 4>
PRBS31: CmsJesdPrbsPattern  # value = <CmsJesdPrbsPattern.PRBS31: 5>
PRBS7: CmsJesdPrbsPattern  # value = <CmsJesdPrbsPattern.PRBS7: 1>
PRBS9: CmsJesdPrbsPattern  # value = <CmsJesdPrbsPattern.PRBS9: 2>
PRBS_MAX: CmsJesdPrbsPattern  # value = <CmsJesdPrbsPattern.PRBS_MAX: 6>
PRBS_NONE: CmsJesdPrbsPattern  # value = <CmsJesdPrbsPattern.PRBS_NONE: 0>
PRE_EMP_SETTING: SerLaneSettingsField  # value = <SerLaneSettingsField.PRE_EMP_SETTING: 1>
RX_ONLY: CmsChioOpMode  # value = <CmsChioOpMode.RX_ONLY: 2>
SER_POST_EMP_0DB: SerPostEmp  # value = <SerPostEmp.SER_POST_EMP_0DB: 0>
SER_POST_EMP_12DB: SerPostEmp  # value = <SerPostEmp.SER_POST_EMP_12DB: 4>
SER_POST_EMP_3DB: SerPostEmp  # value = <SerPostEmp.SER_POST_EMP_3DB: 1>
SER_POST_EMP_6DB: SerPostEmp  # value = <SerPostEmp.SER_POST_EMP_6DB: 2>
SER_POST_EMP_9DB: SerPostEmp  # value = <SerPostEmp.SER_POST_EMP_9DB: 3>
SER_PRE_EMP_0DB: SerPreEmp  # value = <SerPreEmp.SER_PRE_EMP_0DB: 0>
SER_PRE_EMP_3DB: SerPreEmp  # value = <SerPreEmp.SER_PRE_EMP_3DB: 1>
SER_PRE_EMP_6DB: SerPreEmp  # value = <SerPreEmp.SER_PRE_EMP_6DB: 2>
SER_SWING_1000: SerSwing  # value = <SerSwing.SER_SWING_1000: 0>
SER_SWING_500: SerSwing  # value = <SerSwing.SER_SWING_500: 3>
SER_SWING_750: SerSwing  # value = <SerSwing.SER_SWING_750: 2>
SER_SWING_850: SerSwing  # value = <SerSwing.SER_SWING_850: 1>
SIGNAL_CML: CmsSignalType  # value = <CmsSignalType.SIGNAL_CML: 2>
SIGNAL_CMOS: CmsSignalType  # value = <CmsSignalType.SIGNAL_CMOS: 0>
SIGNAL_LVDS: CmsSignalType  # value = <CmsSignalType.SIGNAL_LVDS: 1>
SIGNAL_LVPECL: CmsSignalType  # value = <CmsSignalType.SIGNAL_LVPECL: 3>
SIGNAL_UNKNOWN: CmsSignalType  # value = <CmsSignalType.SIGNAL_UNKNOWN: 4>
SOFT_RESET: Reset  # value = <Reset.SOFT_RESET: 0>
SOFT_RESET_AND_INIT: Reset  # value = <Reset.SOFT_RESET_AND_INIT: 2>
SPI_ADDR_DEC_AUTO: CmsSpiAddrInc  # value = <CmsSpiAddrInc.SPI_ADDR_DEC_AUTO: 0>
SPI_ADDR_INC_AUTO: CmsSpiAddrInc  # value = <CmsSpiAddrInc.SPI_ADDR_INC_AUTO: 1>
SPI_MSB_FIRST: CmsSpiMsbConfig  # value = <CmsSpiMsbConfig.SPI_MSB_FIRST: 1>
SPI_MSB_LAST: CmsSpiMsbConfig  # value = <CmsSpiMsbConfig.SPI_MSB_LAST: 0>
SPI_NONE: CmsSpiSdoConfig  # value = <CmsSpiSdoConfig.SPI_NONE: 0>
SPI_SDIO: CmsSpiSdoConfig  # value = <CmsSpiSdoConfig.SPI_SDIO: 2>
SPI_SDO: CmsSpiSdoConfig  # value = <CmsSpiSdoConfig.SPI_SDO: 1>
SWING_SETTING: SerLaneSettingsField  # value = <SerLaneSettingsField.SWING_SETTING: 0>
SYNCOUTB_0: CmsJesdSyncoutb  # value = <CmsJesdSyncoutb.SYNCOUTB_0: 0>
SYNCOUTB_1: CmsJesdSyncoutb  # value = <CmsJesdSyncoutb.SYNCOUTB_1: 1>
SYNCOUTB_ALL: CmsJesdSyncoutb  # value = <CmsJesdSyncoutb.SYNCOUTB_ALL: 255>
SYSREF_CONT: CmsJesdSysrefMode  # value = <CmsJesdSysrefMode.SYSREF_CONT: 2>
SYSREF_MODE_INVALID: CmsJesdSysrefMode  # value = <CmsJesdSysrefMode.SYSREF_MODE_INVALID: 4>
SYSREF_MON: CmsJesdSysrefMode  # value = <CmsJesdSysrefMode.SYSREF_MON: 3>
SYSREF_NONE: CmsJesdSysrefMode  # value = <CmsJesdSysrefMode.SYSREF_NONE: 0>
SYSREF_ONESHOT: CmsJesdSysrefMode  # value = <CmsJesdSysrefMode.SYSREF_ONESHOT: 1>
TX_ONLY: CmsChioOpMode  # value = <CmsChioOpMode.TX_ONLY: 1>
TX_RX_ONLY: CmsChioOpMode  # value = <CmsChioOpMode.TX_RX_ONLY: 3>
