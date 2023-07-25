from __future__ import annotations
import adi_ad9081_v106
import typing
import numpy

__all__ = [
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
    "API_CMS_ERROR_LOG_CLOSE",
    "API_CMS_ERROR_LOG_OPEN",
    "API_CMS_ERROR_LOG_WRITE",
    "API_CMS_ERROR_MODE_NOT_IN_TABLE",
    "API_CMS_ERROR_NCO_NOT_ENABLED",
    "API_CMS_ERROR_NOT_SUPPORTED",
    "API_CMS_ERROR_NULL_PARAM",
    "API_CMS_ERROR_OK",
    "API_CMS_ERROR_PLL_NOT_LOCKED",
    "API_CMS_ERROR_RESET_PIN_CTRL",
    "API_CMS_ERROR_SPI_SDO",
    "API_CMS_ERROR_SPI_XFER",
    "API_CMS_ERROR_TEST_FAILED",
    "API_CMS_ERROR_TX_EN_PIN_CTRL",
    "API_CMS_ERROR_VCO_OUT_OF_RANGE",
    "AdcCoarseDdcDcm",
    "AdcCoarseDdcSelect",
    "AdcFineDdcDcm",
    "AdcFineDdcSelect",
    "AdcNcoMode",
    "AddrData",
    "AdiCmsJesdParam",
    "ApiRevision",
    "CmsChipId",
    "CmsError",
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
    "DAC_MODE_0",
    "DAC_MODE_1",
    "DAC_MODE_2",
    "DAC_MODE_3",
    "DAC_MODE_SWITCH_GROUP_0",
    "DAC_MODE_SWITCH_GROUP_1",
    "DAC_MODE_SWITCH_GROUP_ALL",
    "DAC_MODE_SWITCH_GROUP_NONE",
    "DAC_NONE",
    "DacChannelSelect",
    "DacMode",
    "DacModeSwitchGroupSelect",
    "DacSelect",
    "DesSettings",
    "Device",
    "HARD_RESET",
    "HARD_RESET_AND_INIT",
    "Info",
    "JesdLinkSelect",
    "JtxConvSel",
    "LINK_0",
    "LINK_1",
    "LINK_ALL",
    "LINK_NONE",
    "NcoFtw",
    "POST_EMP_SETTING",
    "PRE_EMP_SETTING",
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
    "SerLaneSettings",
    "SerLaneSettingsField",
    "SerPostEmp",
    "SerPreEmp",
    "SerSettings",
    "SerSwing",
    "SerdesSettings",
    "adc_ddc_coarse_nco_ftw_get",
    "adc_ddc_coarse_nco_ftw_set",
    "adc_ddc_coarse_nco_mode_set",
    "adc_ddc_fine_nco_ftw_get",
    "adc_ddc_fine_nco_ftw_set",
    "adc_ddc_fine_nco_mode_set",
    "dac_duc_chan_skew_set",
    "dac_duc_nco_enable_set",
    "dac_duc_nco_ftw_set",
    "dac_duc_nco_phase_offset_set",
    "dac_duc_nco_reset_set",
    "dac_mode_set",
    "device_api_revision_get",
    "device_chip_id_get",
    "device_clk_config_set",
    "device_init",
    "device_reset",
    "device_startup_rx",
    "device_startup_tx",
    "hal_calc_nco_ftw",
    "hal_calc_nco_ftw_f",
    "hal_calc_rx_nco_ftw",
    "hal_calc_tx_nco_ftw",
    "hal_delay_us",
    "hal_reg_get",
    "hal_reg_set",
    "jesd_rx_lane_xbar_set",
    "jesd_rx_link_enable_set",
    "jesd_tx_lanes_xbar_set",
    "jesd_tx_lids_cfg_set",
    "jesd_tx_link_enable_set"
]


class AdcCoarseDdcDcm():
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
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    ADC_CDDC_DCM_1: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_1: 12>
    ADC_CDDC_DCM_12: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_12: 6>
    ADC_CDDC_DCM_16: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_16: 3>
    ADC_CDDC_DCM_18: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_18: 10>
    ADC_CDDC_DCM_2: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_2: 0>
    ADC_CDDC_DCM_24: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_24: 7>
    ADC_CDDC_DCM_3: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_3: 8>
    ADC_CDDC_DCM_36: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_36: 11>
    ADC_CDDC_DCM_4: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_4: 1>
    ADC_CDDC_DCM_6: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_6: 5>
    ADC_CDDC_DCM_8: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_8: 2>
    ADC_CDDC_DCM_9: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_9: 9>
    __members__: dict # value = {'ADC_CDDC_DCM_1': <AdcCoarseDdcDcm.ADC_CDDC_DCM_1: 12>, 'ADC_CDDC_DCM_2': <AdcCoarseDdcDcm.ADC_CDDC_DCM_2: 0>, 'ADC_CDDC_DCM_3': <AdcCoarseDdcDcm.ADC_CDDC_DCM_3: 8>, 'ADC_CDDC_DCM_4': <AdcCoarseDdcDcm.ADC_CDDC_DCM_4: 1>, 'ADC_CDDC_DCM_6': <AdcCoarseDdcDcm.ADC_CDDC_DCM_6: 5>, 'ADC_CDDC_DCM_8': <AdcCoarseDdcDcm.ADC_CDDC_DCM_8: 2>, 'ADC_CDDC_DCM_9': <AdcCoarseDdcDcm.ADC_CDDC_DCM_9: 9>, 'ADC_CDDC_DCM_12': <AdcCoarseDdcDcm.ADC_CDDC_DCM_12: 6>, 'ADC_CDDC_DCM_16': <AdcCoarseDdcDcm.ADC_CDDC_DCM_16: 3>, 'ADC_CDDC_DCM_18': <AdcCoarseDdcDcm.ADC_CDDC_DCM_18: 10>, 'ADC_CDDC_DCM_24': <AdcCoarseDdcDcm.ADC_CDDC_DCM_24: 7>, 'ADC_CDDC_DCM_36': <AdcCoarseDdcDcm.ADC_CDDC_DCM_36: 11>}
    pass
class AdcCoarseDdcSelect():
    """
    Members:

      ADC_CDDC_NONE

      ADC_CDDC_0

      ADC_CDDC_1

      ADC_CDDC_2

      ADC_CDDC_3

      ADC_CDDC_ALL
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    ADC_CDDC_0: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_0: 1>
    ADC_CDDC_1: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_1: 2>
    ADC_CDDC_2: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_2: 4>
    ADC_CDDC_3: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_3: 8>
    ADC_CDDC_ALL: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_ALL: 15>
    ADC_CDDC_NONE: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_NONE: 0>
    __members__: dict # value = {'ADC_CDDC_NONE': <AdcCoarseDdcSelect.ADC_CDDC_NONE: 0>, 'ADC_CDDC_0': <AdcCoarseDdcSelect.ADC_CDDC_0: 1>, 'ADC_CDDC_1': <AdcCoarseDdcSelect.ADC_CDDC_1: 2>, 'ADC_CDDC_2': <AdcCoarseDdcSelect.ADC_CDDC_2: 4>, 'ADC_CDDC_3': <AdcCoarseDdcSelect.ADC_CDDC_3: 8>, 'ADC_CDDC_ALL': <AdcCoarseDdcSelect.ADC_CDDC_ALL: 15>}
    pass
class AdcFineDdcDcm():
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
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    ADC_FDDC_DCM_1: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_1: 8>
    ADC_FDDC_DCM_12: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_12: 6>
    ADC_FDDC_DCM_16: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_16: 3>
    ADC_FDDC_DCM_2: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_2: 0>
    ADC_FDDC_DCM_24: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_24: 7>
    ADC_FDDC_DCM_3: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_3: 4>
    ADC_FDDC_DCM_4: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_4: 1>
    ADC_FDDC_DCM_6: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_6: 5>
    ADC_FDDC_DCM_8: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_8: 2>
    __members__: dict # value = {'ADC_FDDC_DCM_1': <AdcFineDdcDcm.ADC_FDDC_DCM_1: 8>, 'ADC_FDDC_DCM_2': <AdcFineDdcDcm.ADC_FDDC_DCM_2: 0>, 'ADC_FDDC_DCM_3': <AdcFineDdcDcm.ADC_FDDC_DCM_3: 4>, 'ADC_FDDC_DCM_4': <AdcFineDdcDcm.ADC_FDDC_DCM_4: 1>, 'ADC_FDDC_DCM_6': <AdcFineDdcDcm.ADC_FDDC_DCM_6: 5>, 'ADC_FDDC_DCM_8': <AdcFineDdcDcm.ADC_FDDC_DCM_8: 2>, 'ADC_FDDC_DCM_12': <AdcFineDdcDcm.ADC_FDDC_DCM_12: 6>, 'ADC_FDDC_DCM_16': <AdcFineDdcDcm.ADC_FDDC_DCM_16: 3>, 'ADC_FDDC_DCM_24': <AdcFineDdcDcm.ADC_FDDC_DCM_24: 7>}
    pass
class AdcFineDdcSelect():
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
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    ADC_FDDC_0: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_0: 1>
    ADC_FDDC_1: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_1: 2>
    ADC_FDDC_2: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_2: 4>
    ADC_FDDC_3: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_3: 8>
    ADC_FDDC_4: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_4: 16>
    ADC_FDDC_5: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_5: 32>
    ADC_FDDC_6: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_6: 64>
    ADC_FDDC_7: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_7: 128>
    ADC_FDDC_ALL: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_ALL: 255>
    ADC_FDDC_NONE: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_NONE: 0>
    __members__: dict # value = {'ADC_FDDC_NONE': <AdcFineDdcSelect.ADC_FDDC_NONE: 0>, 'ADC_FDDC_0': <AdcFineDdcSelect.ADC_FDDC_0: 1>, 'ADC_FDDC_1': <AdcFineDdcSelect.ADC_FDDC_1: 2>, 'ADC_FDDC_2': <AdcFineDdcSelect.ADC_FDDC_2: 4>, 'ADC_FDDC_3': <AdcFineDdcSelect.ADC_FDDC_3: 8>, 'ADC_FDDC_4': <AdcFineDdcSelect.ADC_FDDC_4: 16>, 'ADC_FDDC_5': <AdcFineDdcSelect.ADC_FDDC_5: 32>, 'ADC_FDDC_6': <AdcFineDdcSelect.ADC_FDDC_6: 64>, 'ADC_FDDC_7': <AdcFineDdcSelect.ADC_FDDC_7: 128>, 'ADC_FDDC_ALL': <AdcFineDdcSelect.ADC_FDDC_ALL: 255>}
    pass
class AdcNcoMode():
    """
    Members:

      ADC_NCO_VIF : Variable IF Mode

      ADC_NCO_ZIF : Zero IF Mode

      ADC_NCO_FS_4_IF : Fs/4 Hz IF Mode

      ADC_NCO_TEST : Test Mode
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    ADC_NCO_FS_4_IF: adi_ad9081_v106.AdcNcoMode # value = <AdcNcoMode.ADC_NCO_FS_4_IF: 2>
    ADC_NCO_TEST: adi_ad9081_v106.AdcNcoMode # value = <AdcNcoMode.ADC_NCO_TEST: 3>
    ADC_NCO_VIF: adi_ad9081_v106.AdcNcoMode # value = <AdcNcoMode.ADC_NCO_VIF: 0>
    ADC_NCO_ZIF: adi_ad9081_v106.AdcNcoMode # value = <AdcNcoMode.ADC_NCO_ZIF: 1>
    __members__: dict # value = {'ADC_NCO_VIF': <AdcNcoMode.ADC_NCO_VIF: 0>, 'ADC_NCO_ZIF': <AdcNcoMode.ADC_NCO_ZIF: 1>, 'ADC_NCO_FS_4_IF': <AdcNcoMode.ADC_NCO_FS_4_IF: 2>, 'ADC_NCO_TEST': <AdcNcoMode.ADC_NCO_TEST: 3>}
    pass
class AddrData():
    @typing.overload
    def __init__(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: int, arg1: int) -> None: ...
    @property
    def addr(self) -> int:
        """
        :type: int
        """
    @addr.setter
    def addr(self, arg0: int) -> None:
        pass
    @property
    def data(self) -> int:
        """
        :type: int
        """
    @data.setter
    def data(self, arg0: int) -> None:
        pass
    pass
class AdiCmsJesdParam():
    def __init__(self) -> None: ...
    @property
    def bid(self) -> int:
        """
        :type: int
        """
    @bid.setter
    def bid(self, arg0: int) -> None:
        pass
    @property
    def cf(self) -> int:
        """
        :type: int
        """
    @cf.setter
    def cf(self, arg0: int) -> None:
        pass
    @property
    def cs(self) -> int:
        """
        :type: int
        """
    @cs.setter
    def cs(self, arg0: int) -> None:
        pass
    @property
    def did(self) -> int:
        """
        :type: int
        """
    @did.setter
    def did(self, arg0: int) -> None:
        pass
    @property
    def duallink(self) -> int:
        """
        :type: int
        """
    @duallink.setter
    def duallink(self, arg0: int) -> None:
        pass
    @property
    def f(self) -> int:
        """
        :type: int
        """
    @f.setter
    def f(self, arg0: int) -> None:
        pass
    @property
    def hd(self) -> int:
        """
        :type: int
        """
    @hd.setter
    def hd(self, arg0: int) -> None:
        pass
    @property
    def jesdv(self) -> int:
        """
        :type: int
        """
    @jesdv.setter
    def jesdv(self, arg0: int) -> None:
        pass
    @property
    def k(self) -> int:
        """
        :type: int
        """
    @k.setter
    def k(self, arg0: int) -> None:
        pass
    @property
    def l(self) -> int:
        """
        :type: int
        """
    @l.setter
    def l(self, arg0: int) -> None:
        pass
    @property
    def lid0(self) -> int:
        """
        :type: int
        """
    @lid0.setter
    def lid0(self, arg0: int) -> None:
        pass
    @property
    def m(self) -> int:
        """
        :type: int
        """
    @m.setter
    def m(self, arg0: int) -> None:
        pass
    @property
    def mode_c2r_en(self) -> int:
        """
        :type: int
        """
    @mode_c2r_en.setter
    def mode_c2r_en(self, arg0: int) -> None:
        pass
    @property
    def mode_id(self) -> int:
        """
        :type: int
        """
    @mode_id.setter
    def mode_id(self, arg0: int) -> None:
        pass
    @property
    def mode_s_sel(self) -> int:
        """
        :type: int
        """
    @mode_s_sel.setter
    def mode_s_sel(self, arg0: int) -> None:
        pass
    @property
    def n(self) -> int:
        """
        :type: int
        """
    @n.setter
    def n(self, arg0: int) -> None:
        pass
    @property
    def np(self) -> int:
        """
        :type: int
        """
    @np.setter
    def np(self, arg0: int) -> None:
        pass
    @property
    def s(self) -> int:
        """
        :type: int
        """
    @s.setter
    def s(self, arg0: int) -> None:
        pass
    @property
    def scr(self) -> int:
        """
        :type: int
        """
    @scr.setter
    def scr(self, arg0: int) -> None:
        pass
    @property
    def subclass(self) -> int:
        """
        :type: int
        """
    @subclass.setter
    def subclass(self, arg0: int) -> None:
        pass
    pass
class ApiRevision():
    def __init__(self) -> None: ...
    @property
    def major(self) -> int:
        """
        :type: int
        """
    @property
    def minor(self) -> int:
        """
        :type: int
        """
    @property
    def rc(self) -> int:
        """
        :type: int
        """
    pass
class CmsChipId():
    def __init__(self) -> None: ...
    @property
    def chip_type(self) -> int:
        """
        :type: int
        """
    @property
    def dev_revision(self) -> int:
        """
        :type: int
        """
    @property
    def prod_grade(self) -> int:
        """
        :type: int
        """
    @property
    def prod_id(self) -> int:
        """
        :type: int
        """
    pass
class CmsError():
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
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    API_CMS_ERROR_DELAY_US: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_DELAY_US: -70>
    API_CMS_ERROR_DLL_NOT_LOCKED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_DLL_NOT_LOCKED: -22>
    API_CMS_ERROR_ERROR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_ERROR: -1>
    API_CMS_ERROR_EVENT_HNDL: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_EVENT_HNDL: -64>
    API_CMS_ERROR_FTW_LOAD_ACK: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_FTW_LOAD_ACK: -30>
    API_CMS_ERROR_HW_CLOSE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_HW_CLOSE: -66>
    API_CMS_ERROR_HW_OPEN: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_HW_OPEN: -65>
    API_CMS_ERROR_INIT_SEQ_FAIL: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INIT_SEQ_FAIL: -40>
    API_CMS_ERROR_INVALID_DELAYUS_PTR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_DELAYUS_PTR: -13>
    API_CMS_ERROR_INVALID_HANDLE_PTR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_HANDLE_PTR: -11>
    API_CMS_ERROR_INVALID_PARAM: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_PARAM: -14>
    API_CMS_ERROR_INVALID_RESET_CTRL_PTR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_RESET_CTRL_PTR: -15>
    API_CMS_ERROR_INVALID_XFER_PTR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_XFER_PTR: -12>
    API_CMS_ERROR_LOG_CLOSE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_LOG_CLOSE: -69>
    API_CMS_ERROR_LOG_OPEN: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_LOG_OPEN: -67>
    API_CMS_ERROR_LOG_WRITE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_LOG_WRITE: -68>
    API_CMS_ERROR_MODE_NOT_IN_TABLE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_MODE_NOT_IN_TABLE: -23>
    API_CMS_ERROR_NCO_NOT_ENABLED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_NCO_NOT_ENABLED: -31>
    API_CMS_ERROR_NOT_SUPPORTED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_NOT_SUPPORTED: -16>
    API_CMS_ERROR_NULL_PARAM: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_NULL_PARAM: -2>
    API_CMS_ERROR_OK: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_OK: 0>
    API_CMS_ERROR_PLL_NOT_LOCKED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_PLL_NOT_LOCKED: -21>
    API_CMS_ERROR_RESET_PIN_CTRL: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_RESET_PIN_CTRL: -63>
    API_CMS_ERROR_SPI_SDO: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_SPI_SDO: -10>
    API_CMS_ERROR_SPI_XFER: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_SPI_XFER: -60>
    API_CMS_ERROR_TEST_FAILED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_TEST_FAILED: -50>
    API_CMS_ERROR_TX_EN_PIN_CTRL: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_TX_EN_PIN_CTRL: -62>
    API_CMS_ERROR_VCO_OUT_OF_RANGE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_VCO_OUT_OF_RANGE: -20>
    __members__: dict # value = {'API_CMS_ERROR_OK': <CmsError.API_CMS_ERROR_OK: 0>, 'API_CMS_ERROR_ERROR': <CmsError.API_CMS_ERROR_ERROR: -1>, 'API_CMS_ERROR_NULL_PARAM': <CmsError.API_CMS_ERROR_NULL_PARAM: -2>, 'API_CMS_ERROR_SPI_SDO': <CmsError.API_CMS_ERROR_SPI_SDO: -10>, 'API_CMS_ERROR_INVALID_HANDLE_PTR': <CmsError.API_CMS_ERROR_INVALID_HANDLE_PTR: -11>, 'API_CMS_ERROR_INVALID_XFER_PTR': <CmsError.API_CMS_ERROR_INVALID_XFER_PTR: -12>, 'API_CMS_ERROR_INVALID_DELAYUS_PTR': <CmsError.API_CMS_ERROR_INVALID_DELAYUS_PTR: -13>, 'API_CMS_ERROR_INVALID_PARAM': <CmsError.API_CMS_ERROR_INVALID_PARAM: -14>, 'API_CMS_ERROR_INVALID_RESET_CTRL_PTR': <CmsError.API_CMS_ERROR_INVALID_RESET_CTRL_PTR: -15>, 'API_CMS_ERROR_NOT_SUPPORTED': <CmsError.API_CMS_ERROR_NOT_SUPPORTED: -16>, 'API_CMS_ERROR_VCO_OUT_OF_RANGE': <CmsError.API_CMS_ERROR_VCO_OUT_OF_RANGE: -20>, 'API_CMS_ERROR_PLL_NOT_LOCKED': <CmsError.API_CMS_ERROR_PLL_NOT_LOCKED: -21>, 'API_CMS_ERROR_DLL_NOT_LOCKED': <CmsError.API_CMS_ERROR_DLL_NOT_LOCKED: -22>, 'API_CMS_ERROR_MODE_NOT_IN_TABLE': <CmsError.API_CMS_ERROR_MODE_NOT_IN_TABLE: -23>, 'API_CMS_ERROR_FTW_LOAD_ACK': <CmsError.API_CMS_ERROR_FTW_LOAD_ACK: -30>, 'API_CMS_ERROR_NCO_NOT_ENABLED': <CmsError.API_CMS_ERROR_NCO_NOT_ENABLED: -31>, 'API_CMS_ERROR_INIT_SEQ_FAIL': <CmsError.API_CMS_ERROR_INIT_SEQ_FAIL: -40>, 'API_CMS_ERROR_TEST_FAILED': <CmsError.API_CMS_ERROR_TEST_FAILED: -50>, 'API_CMS_ERROR_SPI_XFER': <CmsError.API_CMS_ERROR_SPI_XFER: -60>, 'API_CMS_ERROR_TX_EN_PIN_CTRL': <CmsError.API_CMS_ERROR_TX_EN_PIN_CTRL: -62>, 'API_CMS_ERROR_RESET_PIN_CTRL': <CmsError.API_CMS_ERROR_RESET_PIN_CTRL: -63>, 'API_CMS_ERROR_EVENT_HNDL': <CmsError.API_CMS_ERROR_EVENT_HNDL: -64>, 'API_CMS_ERROR_HW_OPEN': <CmsError.API_CMS_ERROR_HW_OPEN: -65>, 'API_CMS_ERROR_HW_CLOSE': <CmsError.API_CMS_ERROR_HW_CLOSE: -66>, 'API_CMS_ERROR_LOG_OPEN': <CmsError.API_CMS_ERROR_LOG_OPEN: -67>, 'API_CMS_ERROR_LOG_WRITE': <CmsError.API_CMS_ERROR_LOG_WRITE: -68>, 'API_CMS_ERROR_LOG_CLOSE': <CmsError.API_CMS_ERROR_LOG_CLOSE: -69>, 'API_CMS_ERROR_DELAY_US': <CmsError.API_CMS_ERROR_DELAY_US: -70>}
    pass
class CmsSpiAddrInc():
    """
    Members:

      SPI_ADDR_DEC_AUTO : auto decremented

      SPI_ADDR_INC_AUTO : auto incremented
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    SPI_ADDR_DEC_AUTO: adi_ad9081_v106.CmsSpiAddrInc # value = <CmsSpiAddrInc.SPI_ADDR_DEC_AUTO: 0>
    SPI_ADDR_INC_AUTO: adi_ad9081_v106.CmsSpiAddrInc # value = <CmsSpiAddrInc.SPI_ADDR_INC_AUTO: 1>
    __members__: dict # value = {'SPI_ADDR_DEC_AUTO': <CmsSpiAddrInc.SPI_ADDR_DEC_AUTO: 0>, 'SPI_ADDR_INC_AUTO': <CmsSpiAddrInc.SPI_ADDR_INC_AUTO: 1>}
    pass
class CmsSpiMsbConfig():
    """
    Members:

      SPI_MSB_LAST : LSB first

      SPI_MSB_FIRST : MSB first
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    SPI_MSB_FIRST: adi_ad9081_v106.CmsSpiMsbConfig # value = <CmsSpiMsbConfig.SPI_MSB_FIRST: 1>
    SPI_MSB_LAST: adi_ad9081_v106.CmsSpiMsbConfig # value = <CmsSpiMsbConfig.SPI_MSB_LAST: 0>
    __members__: dict # value = {'SPI_MSB_LAST': <CmsSpiMsbConfig.SPI_MSB_LAST: 0>, 'SPI_MSB_FIRST': <CmsSpiMsbConfig.SPI_MSB_FIRST: 1>}
    pass
class CmsSpiSdoConfig():
    """
    Members:

      SPI_NONE : keep this for test

      SPI_SDO : SDO active, 4-wire only

      SPI_SDIO : SDIO active, 3-wire only
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    SPI_NONE: adi_ad9081_v106.CmsSpiSdoConfig # value = <CmsSpiSdoConfig.SPI_NONE: 0>
    SPI_SDIO: adi_ad9081_v106.CmsSpiSdoConfig # value = <CmsSpiSdoConfig.SPI_SDIO: 2>
    SPI_SDO: adi_ad9081_v106.CmsSpiSdoConfig # value = <CmsSpiSdoConfig.SPI_SDO: 1>
    __members__: dict # value = {'SPI_NONE': <CmsSpiSdoConfig.SPI_NONE: 0>, 'SPI_SDO': <CmsSpiSdoConfig.SPI_SDO: 1>, 'SPI_SDIO': <CmsSpiSdoConfig.SPI_SDIO: 2>}
    pass
class DacChannelSelect():
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
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    DAC_CH_0: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_0: 1>
    DAC_CH_1: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_1: 2>
    DAC_CH_2: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_2: 4>
    DAC_CH_3: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_3: 8>
    DAC_CH_4: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_4: 16>
    DAC_CH_5: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_5: 32>
    DAC_CH_6: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_6: 64>
    DAC_CH_7: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_7: 128>
    DAC_CH_ALL: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_ALL: 255>
    DAC_CH_NONE: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_NONE: 0>
    __members__: dict # value = {'DAC_CH_NONE': <DacChannelSelect.DAC_CH_NONE: 0>, 'DAC_CH_0': <DacChannelSelect.DAC_CH_0: 1>, 'DAC_CH_1': <DacChannelSelect.DAC_CH_1: 2>, 'DAC_CH_2': <DacChannelSelect.DAC_CH_2: 4>, 'DAC_CH_3': <DacChannelSelect.DAC_CH_3: 8>, 'DAC_CH_4': <DacChannelSelect.DAC_CH_4: 16>, 'DAC_CH_5': <DacChannelSelect.DAC_CH_5: 32>, 'DAC_CH_6': <DacChannelSelect.DAC_CH_6: 64>, 'DAC_CH_7': <DacChannelSelect.DAC_CH_7: 128>, 'DAC_CH_ALL': <DacChannelSelect.DAC_CH_ALL: 255>}
    pass
class DacMode():
    """
    Members:

      DAC_MODE_0 : I0.Q0 -> DAC0, I1.Q1 -> DAC1

      DAC_MODE_1 : (I0 + I1) / 2 -> DAC0, (Q0 + Q1) / 2 -> DAC1, Data Path NCOs Bypassed

      DAC_MODE_2 : I0 -> DAC0, Q0 -> DAC1, Datapath0 NCO Bypassed, Datapath1 Unused

      DAC_MODE_3 : (I0 + I1) / 2 -> DAC0, DAC1 Output Tied To Midscale
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    DAC_MODE_0: adi_ad9081_v106.DacMode # value = <DacMode.DAC_MODE_0: 0>
    DAC_MODE_1: adi_ad9081_v106.DacMode # value = <DacMode.DAC_MODE_1: 1>
    DAC_MODE_2: adi_ad9081_v106.DacMode # value = <DacMode.DAC_MODE_2: 2>
    DAC_MODE_3: adi_ad9081_v106.DacMode # value = <DacMode.DAC_MODE_3: 3>
    __members__: dict # value = {'DAC_MODE_0': <DacMode.DAC_MODE_0: 0>, 'DAC_MODE_1': <DacMode.DAC_MODE_1: 1>, 'DAC_MODE_2': <DacMode.DAC_MODE_2: 2>, 'DAC_MODE_3': <DacMode.DAC_MODE_3: 3>}
    pass
class DacModeSwitchGroupSelect():
    """
    Members:

      DAC_MODE_SWITCH_GROUP_NONE : No Group

      DAC_MODE_SWITCH_GROUP_0 : Group 0 (DAC0 & DAC1)

      DAC_MODE_SWITCH_GROUP_1 : Group 1 (DAC2 & DAC3)

      DAC_MODE_SWITCH_GROUP_ALL : All Groups
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    DAC_MODE_SWITCH_GROUP_0: adi_ad9081_v106.DacModeSwitchGroupSelect # value = <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_0: 1>
    DAC_MODE_SWITCH_GROUP_1: adi_ad9081_v106.DacModeSwitchGroupSelect # value = <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_1: 2>
    DAC_MODE_SWITCH_GROUP_ALL: adi_ad9081_v106.DacModeSwitchGroupSelect # value = <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_ALL: 3>
    DAC_MODE_SWITCH_GROUP_NONE: adi_ad9081_v106.DacModeSwitchGroupSelect # value = <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_NONE: 0>
    __members__: dict # value = {'DAC_MODE_SWITCH_GROUP_NONE': <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_NONE: 0>, 'DAC_MODE_SWITCH_GROUP_0': <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_0: 1>, 'DAC_MODE_SWITCH_GROUP_1': <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_1: 2>, 'DAC_MODE_SWITCH_GROUP_ALL': <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_ALL: 3>}
    pass
class DacSelect():
    """
    Members:

      DAC_NONE

      DAC_0

      DAC_1

      DAC_2

      DAC_3

      DAC_ALL
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    DAC_0: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_0: 1>
    DAC_1: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_1: 2>
    DAC_2: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_2: 4>
    DAC_3: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_3: 8>
    DAC_ALL: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_ALL: 15>
    DAC_NONE: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_NONE: 0>
    __members__: dict # value = {'DAC_NONE': <DacSelect.DAC_NONE: 0>, 'DAC_0': <DacSelect.DAC_0: 1>, 'DAC_1': <DacSelect.DAC_1: 2>, 'DAC_2': <DacSelect.DAC_2: 4>, 'DAC_3': <DacSelect.DAC_3: 8>, 'DAC_ALL': <DacSelect.DAC_ALL: 15>}
    pass
class DesSettings():
    def __init__(self) -> None: ...
    @property
    def boost_mask(self) -> int:
        """
        :type: int
        """
    @boost_mask.setter
    def boost_mask(self, arg0: int) -> None:
        pass
    @property
    def ctle_filter(self) -> numpy.ndarray:
        """
        :type: numpy.ndarray
        """
    @property
    def invert_mask(self) -> int:
        """
        :type: int
        """
    @invert_mask.setter
    def invert_mask(self, arg0: int) -> None:
        pass
    @property
    def lane_mapping0(self) -> numpy.ndarray:
        """
        :type: numpy.ndarray
        """
    @property
    def lane_mapping1(self) -> numpy.ndarray:
        """
        :type: numpy.ndarray
        """
    pass
class Device():
    def __init__(self) -> None: ...
    def callback_set(self, arg0: function, arg1: function, arg2: function, arg3: function) -> None: ...
    def callback_unset(self) -> None: ...
    def spi_conf_set(self, arg0: CmsSpiSdoConfig, arg1: CmsSpiMsbConfig, arg2: CmsSpiAddrInc) -> None: ...
    @property
    def dev_info(self) -> Info:
        """
        :type: Info
        """
    @dev_info.setter
    def dev_info(self, arg0: Info) -> None:
        pass
    @property
    def serdes_info(self) -> SerdesSettings:
        """
        :type: SerdesSettings
        """
    @serdes_info.setter
    def serdes_info(self, arg0: SerdesSettings) -> None:
        pass
    pass
class Info():
    def __init__(self) -> None: ...
    @property
    def adc_freq_hz(self) -> int:
        """
        :type: int
        """
    @adc_freq_hz.setter
    def adc_freq_hz(self, arg0: int) -> None:
        pass
    @property
    def dac_freq_hz(self) -> int:
        """
        :type: int
        """
    @dac_freq_hz.setter
    def dac_freq_hz(self, arg0: int) -> None:
        pass
    @property
    def dev_freq_hz(self) -> int:
        """
        :type: int
        """
    @dev_freq_hz.setter
    def dev_freq_hz(self, arg0: int) -> None:
        pass
    @property
    def dev_rev(self) -> int:
        """
        :type: int
        """
    @dev_rev.setter
    def dev_rev(self, arg0: int) -> None:
        pass
    pass
class JesdLinkSelect():
    """
    Members:

      LINK_NONE : No Link

      LINK_0 : Link 0

      LINK_1 : Link 1

      LINK_ALL : All Links
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    LINK_0: adi_ad9081_v106.JesdLinkSelect # value = <JesdLinkSelect.LINK_0: 1>
    LINK_1: adi_ad9081_v106.JesdLinkSelect # value = <JesdLinkSelect.LINK_1: 2>
    LINK_ALL: adi_ad9081_v106.JesdLinkSelect # value = <JesdLinkSelect.LINK_ALL: 3>
    LINK_NONE: adi_ad9081_v106.JesdLinkSelect # value = <JesdLinkSelect.LINK_NONE: 0>
    __members__: dict # value = {'LINK_NONE': <JesdLinkSelect.LINK_NONE: 0>, 'LINK_0': <JesdLinkSelect.LINK_0: 1>, 'LINK_1': <JesdLinkSelect.LINK_1: 2>, 'LINK_ALL': <JesdLinkSelect.LINK_ALL: 3>}
    pass
class JtxConvSel():
    def __init__(self) -> None: ...
    @property
    def virtual_converter0_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter0_index.setter
    def virtual_converter0_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter1_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter1_index.setter
    def virtual_converter1_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter2_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter2_index.setter
    def virtual_converter2_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter3_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter3_index.setter
    def virtual_converter3_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter4_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter4_index.setter
    def virtual_converter4_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter5_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter5_index.setter
    def virtual_converter5_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter6_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter6_index.setter
    def virtual_converter6_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter7_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter7_index.setter
    def virtual_converter7_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter8_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter8_index.setter
    def virtual_converter8_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converter9_index(self) -> int:
        """
        :type: int
        """
    @virtual_converter9_index.setter
    def virtual_converter9_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_convertera_index(self) -> int:
        """
        :type: int
        """
    @virtual_convertera_index.setter
    def virtual_convertera_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converterb_index(self) -> int:
        """
        :type: int
        """
    @virtual_converterb_index.setter
    def virtual_converterb_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converterc_index(self) -> int:
        """
        :type: int
        """
    @virtual_converterc_index.setter
    def virtual_converterc_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converterd_index(self) -> int:
        """
        :type: int
        """
    @virtual_converterd_index.setter
    def virtual_converterd_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_convertere_index(self) -> int:
        """
        :type: int
        """
    @virtual_convertere_index.setter
    def virtual_convertere_index(self, arg0: int) -> None:
        pass
    @property
    def virtual_converterf_index(self) -> int:
        """
        :type: int
        """
    @virtual_converterf_index.setter
    def virtual_converterf_index(self, arg0: int) -> None:
        pass
    pass
class NcoFtw():
    def __init__(self) -> None: ...
    @property
    def ftw(self) -> int:
        """
        :type: int
        """
    @ftw.setter
    def ftw(self, arg0: int) -> None:
        pass
    @property
    def modulus_a(self) -> int:
        """
        :type: int
        """
    @modulus_a.setter
    def modulus_a(self, arg0: int) -> None:
        pass
    @property
    def modulus_b(self) -> int:
        """
        :type: int
        """
    @modulus_b.setter
    def modulus_b(self, arg0: int) -> None:
        pass
    pass
class Reset():
    """
    Members:

      SOFT_RESET : Soft Reset

      HARD_RESET : Hard Reset

      SOFT_RESET_AND_INIT : Soft Reset Then Init

      HARD_RESET_AND_INIT : Hard Reset Then Init
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    HARD_RESET: adi_ad9081_v106.Reset # value = <Reset.HARD_RESET: 1>
    HARD_RESET_AND_INIT: adi_ad9081_v106.Reset # value = <Reset.HARD_RESET_AND_INIT: 3>
    SOFT_RESET: adi_ad9081_v106.Reset # value = <Reset.SOFT_RESET: 0>
    SOFT_RESET_AND_INIT: adi_ad9081_v106.Reset # value = <Reset.SOFT_RESET_AND_INIT: 2>
    __members__: dict # value = {'SOFT_RESET': <Reset.SOFT_RESET: 0>, 'HARD_RESET': <Reset.HARD_RESET: 1>, 'SOFT_RESET_AND_INIT': <Reset.SOFT_RESET_AND_INIT: 2>, 'HARD_RESET_AND_INIT': <Reset.HARD_RESET_AND_INIT: 3>}
    pass
class SerLaneSettings():
    def __init__(self) -> None: ...
    @property
    def post_emp_setting(self) -> SerPostEmp:
        """
        :type: SerPostEmp
        """
    @post_emp_setting.setter
    def post_emp_setting(self, arg0: SerPostEmp) -> None:
        pass
    @property
    def pre_emp_setting(self) -> SerPreEmp:
        """
        :type: SerPreEmp
        """
    @pre_emp_setting.setter
    def pre_emp_setting(self, arg0: SerPreEmp) -> None:
        pass
    @property
    def swing_setting(self) -> SerSwing:
        """
        :type: SerSwing
        """
    @swing_setting.setter
    def swing_setting(self, arg0: SerSwing) -> None:
        pass
    pass
class SerLaneSettingsField():
    """
    Members:

      SWING_SETTING

      PRE_EMP_SETTING

      POST_EMP_SETTING
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    POST_EMP_SETTING: adi_ad9081_v106.SerLaneSettingsField # value = <SerLaneSettingsField.POST_EMP_SETTING: 2>
    PRE_EMP_SETTING: adi_ad9081_v106.SerLaneSettingsField # value = <SerLaneSettingsField.PRE_EMP_SETTING: 1>
    SWING_SETTING: adi_ad9081_v106.SerLaneSettingsField # value = <SerLaneSettingsField.SWING_SETTING: 0>
    __members__: dict # value = {'SWING_SETTING': <SerLaneSettingsField.SWING_SETTING: 0>, 'PRE_EMP_SETTING': <SerLaneSettingsField.PRE_EMP_SETTING: 1>, 'POST_EMP_SETTING': <SerLaneSettingsField.POST_EMP_SETTING: 2>}
    pass
class SerPostEmp():
    """
    Members:

      SER_POST_EMP_0DB : 0 dB Post-Emphasis

      SER_POST_EMP_3DB : 3 dB Post-Emphasis

      SER_POST_EMP_6DB : 6 dB Post-Emphasis

      SER_POST_EMP_9DB : 9 dB Post-Emphasis

      SER_POST_EMP_12DB : 12 dB Post-Emphasis
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    SER_POST_EMP_0DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_0DB: 0>
    SER_POST_EMP_12DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_12DB: 4>
    SER_POST_EMP_3DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_3DB: 1>
    SER_POST_EMP_6DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_6DB: 2>
    SER_POST_EMP_9DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_9DB: 3>
    __members__: dict # value = {'SER_POST_EMP_0DB': <SerPostEmp.SER_POST_EMP_0DB: 0>, 'SER_POST_EMP_3DB': <SerPostEmp.SER_POST_EMP_3DB: 1>, 'SER_POST_EMP_6DB': <SerPostEmp.SER_POST_EMP_6DB: 2>, 'SER_POST_EMP_9DB': <SerPostEmp.SER_POST_EMP_9DB: 3>, 'SER_POST_EMP_12DB': <SerPostEmp.SER_POST_EMP_12DB: 4>}
    pass
class SerPreEmp():
    """
    Members:

      SER_PRE_EMP_0DB : 0 dB Pre-Emphasis

      SER_PRE_EMP_3DB : 3 dB Pre-Emphasis

      SER_PRE_EMP_6DB : 6 dB Pre-Emphasis
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    SER_PRE_EMP_0DB: adi_ad9081_v106.SerPreEmp # value = <SerPreEmp.SER_PRE_EMP_0DB: 0>
    SER_PRE_EMP_3DB: adi_ad9081_v106.SerPreEmp # value = <SerPreEmp.SER_PRE_EMP_3DB: 1>
    SER_PRE_EMP_6DB: adi_ad9081_v106.SerPreEmp # value = <SerPreEmp.SER_PRE_EMP_6DB: 2>
    __members__: dict # value = {'SER_PRE_EMP_0DB': <SerPreEmp.SER_PRE_EMP_0DB: 0>, 'SER_PRE_EMP_3DB': <SerPreEmp.SER_PRE_EMP_3DB: 1>, 'SER_PRE_EMP_6DB': <SerPreEmp.SER_PRE_EMP_6DB: 2>}
    pass
class SerSettings():
    def __init__(self) -> None: ...
    @property
    def invert_mask(self) -> int:
        """
        :type: int
        """
    @invert_mask.setter
    def invert_mask(self, arg0: int) -> None:
        pass
    @property
    def lane_mapping0(self) -> numpy.ndarray:
        """
        :type: numpy.ndarray
        """
    @property
    def lane_mapping1(self) -> numpy.ndarray:
        """
        :type: numpy.ndarray
        """
    @property
    def lane_settings(self) -> numpy.ndarray:
        """
        :type: numpy.ndarray
        """
    pass
class SerSwing():
    """
    Members:

      SER_SWING_1000 : 1000 mV Swing

      SER_SWING_850 : 850 mV Swing

      SER_SWING_750 : 750 mV Swing

      SER_SWING_500 : 500 mV Swing
    """
    def __and__(self, other: object) -> object: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __gt__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> object: ...
    def __le__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __or__(self, other: object) -> object: ...
    def __rand__(self, other: object) -> object: ...
    def __repr__(self) -> str: ...
    def __ror__(self, other: object) -> object: ...
    def __rxor__(self, other: object) -> object: ...
    def __setstate__(self, state: int) -> None: ...
    def __xor__(self, other: object) -> object: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    SER_SWING_1000: adi_ad9081_v106.SerSwing # value = <SerSwing.SER_SWING_1000: 0>
    SER_SWING_500: adi_ad9081_v106.SerSwing # value = <SerSwing.SER_SWING_500: 3>
    SER_SWING_750: adi_ad9081_v106.SerSwing # value = <SerSwing.SER_SWING_750: 2>
    SER_SWING_850: adi_ad9081_v106.SerSwing # value = <SerSwing.SER_SWING_850: 1>
    __members__: dict # value = {'SER_SWING_1000': <SerSwing.SER_SWING_1000: 0>, 'SER_SWING_850': <SerSwing.SER_SWING_850: 1>, 'SER_SWING_750': <SerSwing.SER_SWING_750: 2>, 'SER_SWING_500': <SerSwing.SER_SWING_500: 3>}
    pass
class SerdesSettings():
    def __init__(self) -> None: ...
    @property
    def des_settings(self) -> DesSettings:
        """
        :type: DesSettings
        """
    @des_settings.setter
    def des_settings(self, arg0: DesSettings) -> None:
        pass
    @property
    def ser_settings(self) -> SerSettings:
        """
        :type: SerSettings
        """
    @ser_settings.setter
    def ser_settings(self, arg0: SerSettings) -> None:
        pass
    pass
def adc_ddc_coarse_nco_ftw_get(arg0: Device, arg1: int, arg2: NcoFtw) -> int:
    pass
def adc_ddc_coarse_nco_ftw_set(arg0: Device, arg1: int, arg2: NcoFtw) -> int:
    pass
def adc_ddc_coarse_nco_mode_set(arg0: Device, arg1: int, arg2: AdcNcoMode) -> int:
    pass
def adc_ddc_fine_nco_ftw_get(arg0: Device, arg1: int, arg2: NcoFtw) -> int:
    pass
def adc_ddc_fine_nco_ftw_set(arg0: Device, arg1: int, arg2: NcoFtw) -> int:
    pass
def adc_ddc_fine_nco_mode_set(arg0: Device, arg1: int, arg2: AdcNcoMode) -> int:
    pass
def dac_duc_chan_skew_set(arg0: Device, arg1: int, arg2: int) -> int:
    pass
def dac_duc_nco_enable_set(arg0: Device, arg1: int, arg2: int, arg3: int) -> int:
    pass
def dac_duc_nco_ftw_set(arg0: Device, arg1: int, arg2: int, arg3: NcoFtw) -> int:
    pass
def dac_duc_nco_phase_offset_set(arg0: Device, arg1: int, arg2: int, arg3: int, arg4: int) -> int:
    pass
def dac_duc_nco_reset_set(arg0: Device, arg1: int, arg2: int) -> int:
    pass
def dac_mode_set(arg0: Device, arg1: DacModeSwitchGroupSelect, arg2: DacMode) -> int:
    pass
def device_api_revision_get(arg0: Device, arg1: ApiRevision) -> int:
    pass
def device_chip_id_get(arg0: Device, arg1: CmsChipId) -> int:
    pass
def device_clk_config_set(arg0: Device, arg1: int, arg2: int, arg3: int) -> int:
    pass
def device_init(arg0: Device) -> int:
    pass
def device_reset(arg0: Device, arg1: Reset) -> int:
    pass
def device_startup_rx(arg0: Device, arg1: AdcCoarseDdcSelect, arg2: AdcFineDdcSelect, arg3: typing.Tuple[int, int, int, int], arg4: typing.Tuple[int, int, int, int, int, int, int, int], arg5: typing.List[int], arg6: typing.List[int], arg7: typing.Tuple[int, int, int, int], arg8: typing.Tuple[int, int, int, int, int, int, int, int], arg9: typing.Tuple[AdiCmsJesdParam, ...], arg10: typing.Tuple[JtxConvSel, ...]) -> int:
    pass
def device_startup_tx(arg0: Device, arg1: int, arg2: int, arg3: typing.Tuple[int, int, int, int], arg4: typing.Tuple[int, int, int, int], arg5: typing.Tuple[int, int, int, int, int, int, int, int], arg6: AdiCmsJesdParam) -> int:
    pass
def hal_calc_nco_ftw(arg0: Device, arg1: int, arg2: int, arg3: NcoFtw) -> int:
    pass
def hal_calc_nco_ftw_f(arg0: Device, arg1: float, arg2: float, arg3: NcoFtw) -> int:
    pass
def hal_calc_rx_nco_ftw(arg0: Device, arg1: int, arg2: int, arg3: NcoFtw) -> int:
    pass
def hal_calc_tx_nco_ftw(arg0: Device, arg1: int, arg2: int, arg3: NcoFtw) -> int:
    pass
def hal_delay_us(arg0: Device, arg1: int) -> int:
    pass
def hal_reg_get(arg0: Device, arg1: AddrData) -> int:
    pass
def hal_reg_set(arg0: Device, arg1: AddrData) -> int:
    pass
def jesd_rx_lane_xbar_set(arg0: Device, arg1: JesdLinkSelect, arg2: int, arg3: int) -> int:
    pass
def jesd_rx_link_enable_set(arg0: Device, arg1: JesdLinkSelect, arg2: int) -> int:
    pass
def jesd_tx_lanes_xbar_set(arg0: Device, arg1: JesdLinkSelect, arg2: typing.Tuple[int, int, int, int, int, int, int, int]) -> int:
    pass
def jesd_tx_lids_cfg_set(arg0: Device, arg1: JesdLinkSelect, arg2: typing.List[int]) -> int:
    pass
def jesd_tx_link_enable_set(arg0: Device, arg1: JesdLinkSelect, arg2: int) -> int:
    pass
ADC_CDDC_0: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_0: 1>
ADC_CDDC_1: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_1: 2>
ADC_CDDC_2: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_2: 4>
ADC_CDDC_3: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_3: 8>
ADC_CDDC_ALL: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_ALL: 15>
ADC_CDDC_DCM_1: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_1: 12>
ADC_CDDC_DCM_12: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_12: 6>
ADC_CDDC_DCM_16: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_16: 3>
ADC_CDDC_DCM_18: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_18: 10>
ADC_CDDC_DCM_2: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_2: 0>
ADC_CDDC_DCM_24: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_24: 7>
ADC_CDDC_DCM_3: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_3: 8>
ADC_CDDC_DCM_36: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_36: 11>
ADC_CDDC_DCM_4: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_4: 1>
ADC_CDDC_DCM_6: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_6: 5>
ADC_CDDC_DCM_8: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_8: 2>
ADC_CDDC_DCM_9: adi_ad9081_v106.AdcCoarseDdcDcm # value = <AdcCoarseDdcDcm.ADC_CDDC_DCM_9: 9>
ADC_CDDC_NONE: adi_ad9081_v106.AdcCoarseDdcSelect # value = <AdcCoarseDdcSelect.ADC_CDDC_NONE: 0>
ADC_FDDC_0: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_0: 1>
ADC_FDDC_1: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_1: 2>
ADC_FDDC_2: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_2: 4>
ADC_FDDC_3: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_3: 8>
ADC_FDDC_4: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_4: 16>
ADC_FDDC_5: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_5: 32>
ADC_FDDC_6: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_6: 64>
ADC_FDDC_7: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_7: 128>
ADC_FDDC_ALL: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_ALL: 255>
ADC_FDDC_DCM_1: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_1: 8>
ADC_FDDC_DCM_12: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_12: 6>
ADC_FDDC_DCM_16: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_16: 3>
ADC_FDDC_DCM_2: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_2: 0>
ADC_FDDC_DCM_24: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_24: 7>
ADC_FDDC_DCM_3: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_3: 4>
ADC_FDDC_DCM_4: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_4: 1>
ADC_FDDC_DCM_6: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_6: 5>
ADC_FDDC_DCM_8: adi_ad9081_v106.AdcFineDdcDcm # value = <AdcFineDdcDcm.ADC_FDDC_DCM_8: 2>
ADC_FDDC_NONE: adi_ad9081_v106.AdcFineDdcSelect # value = <AdcFineDdcSelect.ADC_FDDC_NONE: 0>
ADC_NCO_FS_4_IF: adi_ad9081_v106.AdcNcoMode # value = <AdcNcoMode.ADC_NCO_FS_4_IF: 2>
ADC_NCO_TEST: adi_ad9081_v106.AdcNcoMode # value = <AdcNcoMode.ADC_NCO_TEST: 3>
ADC_NCO_VIF: adi_ad9081_v106.AdcNcoMode # value = <AdcNcoMode.ADC_NCO_VIF: 0>
ADC_NCO_ZIF: adi_ad9081_v106.AdcNcoMode # value = <AdcNcoMode.ADC_NCO_ZIF: 1>
API_CMS_ERROR_DELAY_US: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_DELAY_US: -70>
API_CMS_ERROR_DLL_NOT_LOCKED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_DLL_NOT_LOCKED: -22>
API_CMS_ERROR_ERROR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_ERROR: -1>
API_CMS_ERROR_EVENT_HNDL: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_EVENT_HNDL: -64>
API_CMS_ERROR_FTW_LOAD_ACK: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_FTW_LOAD_ACK: -30>
API_CMS_ERROR_HW_CLOSE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_HW_CLOSE: -66>
API_CMS_ERROR_HW_OPEN: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_HW_OPEN: -65>
API_CMS_ERROR_INIT_SEQ_FAIL: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INIT_SEQ_FAIL: -40>
API_CMS_ERROR_INVALID_DELAYUS_PTR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_DELAYUS_PTR: -13>
API_CMS_ERROR_INVALID_HANDLE_PTR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_HANDLE_PTR: -11>
API_CMS_ERROR_INVALID_PARAM: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_PARAM: -14>
API_CMS_ERROR_INVALID_RESET_CTRL_PTR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_RESET_CTRL_PTR: -15>
API_CMS_ERROR_INVALID_XFER_PTR: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_INVALID_XFER_PTR: -12>
API_CMS_ERROR_LOG_CLOSE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_LOG_CLOSE: -69>
API_CMS_ERROR_LOG_OPEN: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_LOG_OPEN: -67>
API_CMS_ERROR_LOG_WRITE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_LOG_WRITE: -68>
API_CMS_ERROR_MODE_NOT_IN_TABLE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_MODE_NOT_IN_TABLE: -23>
API_CMS_ERROR_NCO_NOT_ENABLED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_NCO_NOT_ENABLED: -31>
API_CMS_ERROR_NOT_SUPPORTED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_NOT_SUPPORTED: -16>
API_CMS_ERROR_NULL_PARAM: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_NULL_PARAM: -2>
API_CMS_ERROR_OK: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_OK: 0>
API_CMS_ERROR_PLL_NOT_LOCKED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_PLL_NOT_LOCKED: -21>
API_CMS_ERROR_RESET_PIN_CTRL: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_RESET_PIN_CTRL: -63>
API_CMS_ERROR_SPI_SDO: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_SPI_SDO: -10>
API_CMS_ERROR_SPI_XFER: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_SPI_XFER: -60>
API_CMS_ERROR_TEST_FAILED: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_TEST_FAILED: -50>
API_CMS_ERROR_TX_EN_PIN_CTRL: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_TX_EN_PIN_CTRL: -62>
API_CMS_ERROR_VCO_OUT_OF_RANGE: adi_ad9081_v106.CmsError # value = <CmsError.API_CMS_ERROR_VCO_OUT_OF_RANGE: -20>
DAC_0: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_0: 1>
DAC_1: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_1: 2>
DAC_2: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_2: 4>
DAC_3: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_3: 8>
DAC_ALL: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_ALL: 15>
DAC_CH_0: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_0: 1>
DAC_CH_1: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_1: 2>
DAC_CH_2: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_2: 4>
DAC_CH_3: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_3: 8>
DAC_CH_4: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_4: 16>
DAC_CH_5: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_5: 32>
DAC_CH_6: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_6: 64>
DAC_CH_7: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_7: 128>
DAC_CH_ALL: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_ALL: 255>
DAC_CH_NONE: adi_ad9081_v106.DacChannelSelect # value = <DacChannelSelect.DAC_CH_NONE: 0>
DAC_MODE_0: adi_ad9081_v106.DacMode # value = <DacMode.DAC_MODE_0: 0>
DAC_MODE_1: adi_ad9081_v106.DacMode # value = <DacMode.DAC_MODE_1: 1>
DAC_MODE_2: adi_ad9081_v106.DacMode # value = <DacMode.DAC_MODE_2: 2>
DAC_MODE_3: adi_ad9081_v106.DacMode # value = <DacMode.DAC_MODE_3: 3>
DAC_MODE_SWITCH_GROUP_0: adi_ad9081_v106.DacModeSwitchGroupSelect # value = <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_0: 1>
DAC_MODE_SWITCH_GROUP_1: adi_ad9081_v106.DacModeSwitchGroupSelect # value = <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_1: 2>
DAC_MODE_SWITCH_GROUP_ALL: adi_ad9081_v106.DacModeSwitchGroupSelect # value = <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_ALL: 3>
DAC_MODE_SWITCH_GROUP_NONE: adi_ad9081_v106.DacModeSwitchGroupSelect # value = <DacModeSwitchGroupSelect.DAC_MODE_SWITCH_GROUP_NONE: 0>
DAC_NONE: adi_ad9081_v106.DacSelect # value = <DacSelect.DAC_NONE: 0>
HARD_RESET: adi_ad9081_v106.Reset # value = <Reset.HARD_RESET: 1>
HARD_RESET_AND_INIT: adi_ad9081_v106.Reset # value = <Reset.HARD_RESET_AND_INIT: 3>
LINK_0: adi_ad9081_v106.JesdLinkSelect # value = <JesdLinkSelect.LINK_0: 1>
LINK_1: adi_ad9081_v106.JesdLinkSelect # value = <JesdLinkSelect.LINK_1: 2>
LINK_ALL: adi_ad9081_v106.JesdLinkSelect # value = <JesdLinkSelect.LINK_ALL: 3>
LINK_NONE: adi_ad9081_v106.JesdLinkSelect # value = <JesdLinkSelect.LINK_NONE: 0>
POST_EMP_SETTING: adi_ad9081_v106.SerLaneSettingsField # value = <SerLaneSettingsField.POST_EMP_SETTING: 2>
PRE_EMP_SETTING: adi_ad9081_v106.SerLaneSettingsField # value = <SerLaneSettingsField.PRE_EMP_SETTING: 1>
SER_POST_EMP_0DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_0DB: 0>
SER_POST_EMP_12DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_12DB: 4>
SER_POST_EMP_3DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_3DB: 1>
SER_POST_EMP_6DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_6DB: 2>
SER_POST_EMP_9DB: adi_ad9081_v106.SerPostEmp # value = <SerPostEmp.SER_POST_EMP_9DB: 3>
SER_PRE_EMP_0DB: adi_ad9081_v106.SerPreEmp # value = <SerPreEmp.SER_PRE_EMP_0DB: 0>
SER_PRE_EMP_3DB: adi_ad9081_v106.SerPreEmp # value = <SerPreEmp.SER_PRE_EMP_3DB: 1>
SER_PRE_EMP_6DB: adi_ad9081_v106.SerPreEmp # value = <SerPreEmp.SER_PRE_EMP_6DB: 2>
SER_SWING_1000: adi_ad9081_v106.SerSwing # value = <SerSwing.SER_SWING_1000: 0>
SER_SWING_500: adi_ad9081_v106.SerSwing # value = <SerSwing.SER_SWING_500: 3>
SER_SWING_750: adi_ad9081_v106.SerSwing # value = <SerSwing.SER_SWING_750: 2>
SER_SWING_850: adi_ad9081_v106.SerSwing # value = <SerSwing.SER_SWING_850: 1>
SOFT_RESET: adi_ad9081_v106.Reset # value = <Reset.SOFT_RESET: 0>
SOFT_RESET_AND_INIT: adi_ad9081_v106.Reset # value = <Reset.SOFT_RESET_AND_INIT: 2>
SPI_ADDR_DEC_AUTO: adi_ad9081_v106.CmsSpiAddrInc # value = <CmsSpiAddrInc.SPI_ADDR_DEC_AUTO: 0>
SPI_ADDR_INC_AUTO: adi_ad9081_v106.CmsSpiAddrInc # value = <CmsSpiAddrInc.SPI_ADDR_INC_AUTO: 1>
SPI_MSB_FIRST: adi_ad9081_v106.CmsSpiMsbConfig # value = <CmsSpiMsbConfig.SPI_MSB_FIRST: 1>
SPI_MSB_LAST: adi_ad9081_v106.CmsSpiMsbConfig # value = <CmsSpiMsbConfig.SPI_MSB_LAST: 0>
SPI_NONE: adi_ad9081_v106.CmsSpiSdoConfig # value = <CmsSpiSdoConfig.SPI_NONE: 0>
SPI_SDIO: adi_ad9081_v106.CmsSpiSdoConfig # value = <CmsSpiSdoConfig.SPI_SDIO: 2>
SPI_SDO: adi_ad9081_v106.CmsSpiSdoConfig # value = <CmsSpiSdoConfig.SPI_SDO: 1>
SWING_SETTING: adi_ad9081_v106.SerLaneSettingsField # value = <SerLaneSettingsField.SWING_SETTING: 0>
