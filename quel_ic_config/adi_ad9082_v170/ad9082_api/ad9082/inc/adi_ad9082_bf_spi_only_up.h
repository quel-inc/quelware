/*!
 * @brief     SPI Register Definition Header File, automatically generated file at 1/20/2020 6:24:30 AM.
 * 
 * @copyright copyright(c) 2018 - Analog Devices Inc.All Rights Reserved.
 *            This software is proprietary to Analog Devices, Inc. and its
 *            licensor. By using this software you agree to the terms of the
 *            associated analog devices software license agreement.
 */

/*! 
 * @addtogroup AD9082_BF
 * @{
 */
#ifndef __ADI_AD9082_BF_SPI_ONLY_UP_H__
#define __ADI_AD9082_BF_SPI_ONLY_UP_H__

/*============= I N C L U D E S ============*/
#include "adi_ad9082_config.h"

/*============= D E F I N E S ==============*/
#define REG_CRP_CDIVH_OVRRD_ADDR           0x00003D12
#define BF_CDIVH_OVRD_SPI_INFO             0x00000500
#define BF_CDIVH_OVRD_SPI(val)             (val & 0x0000001F)
#define BF_CDIVH_OVRD_SPI_GET(val)         (val & 0x0000001F)

#define REG_CRP_SDIVPDIV_OVRRD_ADDR        0x00003D13
#define BF_PDIV_OVRD_SPI_INFO              0x00000300
#define BF_PDIV_OVRD_SPI(val)              (val & 0x00000007)
#define BF_PDIV_OVRD_SPI_GET(val)          (val & 0x00000007)
#define BF_SDIV_OVRD_SPI_INFO              0x00000304
#define BF_SDIV_OVRD_SPI(val)              ((val & 0x00000007) << 0x00000004)
#define BF_SDIV_OVRD_SPI_GET(val)          ((val >> 0x00000004) & 0x00000007)

#define REG_CRP_CDIVL_OVRRD_ADDR           0x00003D14
#define BF_CDIVL_OVRD_SPI_INFO             0x00000500
#define BF_CDIVL_OVRD_SPI(val)             (val & 0x0000001F)
#define BF_CDIVL_OVRD_SPI_GET(val)         (val & 0x0000001F)

#define REG_SPI_BASE_ADDR3_ADDR            0x00003D23
#define BF_SPI_BASE_ADDR_0_INFO            0x00000800
#define BF_SPI_BASE_ADDR_0(val)            (val & 0x000000FF)
#define BF_SPI_BASE_ADDR_0_GET(val)        (val & 0x000000FF)

#define REG_SPI_BASE_ADDR2_ADDR            0x00003D22
#define BF_SPI_BASE_ADDR_1_INFO            0x00000800
#define BF_SPI_BASE_ADDR_1(val)            (val & 0x000000FF)
#define BF_SPI_BASE_ADDR_1_GET(val)        (val & 0x000000FF)

#define REG_SPI_BASE_ADDR1_ADDR            0x00003D21
#define BF_SPI_BASE_ADDR_2_INFO            0x00000800
#define BF_SPI_BASE_ADDR_2(val)            (val & 0x000000FF)
#define BF_SPI_BASE_ADDR_2_GET(val)        (val & 0x000000FF)

#define REG_SPI_BASE_ADDR0_ADDR            0x00003D20
#define BF_SPI_BASE_ADDR_3_INFO            0x00000800
#define BF_SPI_BASE_ADDR_3(val)            (val & 0x000000FF)
#define BF_SPI_BASE_ADDR_3_GET(val)        (val & 0x000000FF)

#define REG_OTP_DIV_ADDR                   0x00003D24
#define BF_MCLKDIV_SPI_INFO                0x00000800
#define BF_MCLKDIV_SPI(val)                (val & 0x000000FF)
#define BF_MCLKDIV_SPI_GET(val)            (val & 0x000000FF)

#define REG_UP_STALL_ADDR                  0x00003D25
#define BF_UP_STALL_INFO                   0x00000100
#define BF_UP_STALL(val)                   (val & 0x00000001)
#define BF_UP_STALL_GET(val)               (val & 0x00000001)

#define REG_UP_CTRL_ADDR                   0x00003D26
#define BF_UP_BRESET_INFO                  0x00000100
#define BF_UP_BRESET(val)                  (val & 0x00000001)
#define BF_UP_BRESET_GET(val)              (val & 0x00000001)
#define BF_UP_STATVECTORSEL_INFO           0x00000101
#define BF_UP_STATVECTORSEL(val)           ((val & 0x00000001) << 0x00000001)
#define BF_UP_STATVECTORSEL_GET(val)       ((val >> 0x00000001) & 0x00000001)
#define BF_UP_SPI_EDGE_INTERRUPT_INFO      0x00000103
#define BF_UP_SPI_EDGE_INTERRUPT(val)      ((val & 0x00000001) << 0x00000003)
#define BF_UP_SPI_EDGE_INTERRUPT_GET(val)  ((val >> 0x00000003) & 0x00000001)

#define REG_UP_STATUS_ADDR                 0x00003D27
#define BF_UP_STATUS_INFO                  0x00000100
#define BF_UP_STATUS(val)                  (val & 0x00000001)
#define BF_UP_STATUS_GET(val)              (val & 0x00000001)
#define BF_UP_PWAITMODE_INFO               0x00000101
#define BF_UP_PWAITMODE(val)               ((val & 0x00000001) << 0x00000001)
#define BF_UP_PWAITMODE_GET(val)           ((val >> 0x00000001) & 0x00000001)

#define REG_BLOCKOUT_UP_ADDR               0x00003D28
#define BF_BLOCKOUT_UP_EN_INFO             0x00000100
#define BF_BLOCKOUT_UP_EN(val)             (val & 0x00000001)
#define BF_BLOCKOUT_UP_EN_GET(val)         (val & 0x00000001)
#define BF_BLOCKOUT_WINDOW_INFO            0x00000301
#define BF_BLOCKOUT_WINDOW(val)            ((val & 0x00000007) << 0x00000001)
#define BF_BLOCKOUT_WINDOW_GET(val)        ((val >> 0x00000001) & 0x00000007)

#define REG_REG8_BYP_ADDR                  0x00003D29
#define BF_REG8_ACC_MOD_INFO               0x00000103
#define BF_REG8_ACC_MOD(val)               ((val & 0x00000001) << 0x00000003)
#define BF_REG8_ACC_MOD_GET(val)           ((val >> 0x00000003) & 0x00000001)
#define BF_SPI2_SELECT_INFO                0x00000104
#define BF_SPI2_SELECT(val)                ((val & 0x00000001) << 0x00000004)
#define BF_SPI2_SELECT_GET(val)            ((val >> 0x00000004) & 0x00000001)
#define BF_SPI3_SELECT_INFO                0x00000105
#define BF_SPI3_SELECT(val)                ((val & 0x00000001) << 0x00000005)
#define BF_SPI3_SELECT_GET(val)            ((val >> 0x00000005) & 0x00000001)

#define REG_REG8_SCRATCH3_ADDR             0x00003D2D
#define BF_REG8_SCRATCH_0_INFO             0x00000800
#define BF_REG8_SCRATCH_0(val)             (val & 0x000000FF)
#define BF_REG8_SCRATCH_0_GET(val)         (val & 0x000000FF)

#define REG_REG8_SCRATCH2_ADDR             0x00003D2C
#define BF_REG8_SCRATCH_1_INFO             0x00000800
#define BF_REG8_SCRATCH_1(val)             (val & 0x000000FF)
#define BF_REG8_SCRATCH_1_GET(val)         (val & 0x000000FF)

#define REG_REG8_SCRATCH1_ADDR             0x00003D2B
#define BF_REG8_SCRATCH_2_INFO             0x00000800
#define BF_REG8_SCRATCH_2(val)             (val & 0x000000FF)
#define BF_REG8_SCRATCH_2_GET(val)         (val & 0x000000FF)

#define REG_REG8_SCRATCH0_ADDR             0x00003D2A
#define BF_REG8_SCRATCH_3_INFO             0x00000800
#define BF_REG8_SCRATCH_3(val)             (val & 0x000000FF)
#define BF_REG8_SCRATCH_3_GET(val)         (val & 0x000000FF)

#define REG_SCAN_CTL_ADDR                  0x00003D30
#define BF_SCAN_MODE_INFO                  0x00000100
#define BF_SCAN_MODE(val)                  (val & 0x00000001)
#define BF_SCAN_MODE_GET(val)              (val & 0x00000001)
#define BF_COMPRESSION_INFO                0x00000101
#define BF_COMPRESSION(val)                ((val & 0x00000001) << 0x00000001)
#define BF_COMPRESSION_GET(val)            ((val >> 0x00000001) & 0x00000001)
#define BF_OPCGENABLE_CCLK_INFO            0x00000102
#define BF_OPCGENABLE_CCLK(val)            ((val & 0x00000001) << 0x00000002)
#define BF_OPCGENABLE_CCLK_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_OPCGENABLE_SYSCLK_INFO          0x00000103
#define BF_OPCGENABLE_SYSCLK(val)          ((val & 0x00000001) << 0x00000003)
#define BF_OPCGENABLE_SYSCLK_GET(val)      ((val >> 0x00000003) & 0x00000001)
#define BF_OPCGENABLE_NVMCLK_INFO          0x00000104
#define BF_OPCGENABLE_NVMCLK(val)          ((val & 0x00000001) << 0x00000004)
#define BF_OPCGENABLE_NVMCLK_GET(val)      ((val >> 0x00000004) & 0x00000001)
#define BF_SCAN_SPREADEN_INFO              0x00000105
#define BF_SCAN_SPREADEN(val)              ((val & 0x00000001) << 0x00000005)
#define BF_SCAN_SPREADEN_GET(val)          ((val >> 0x00000005) & 0x00000001)
#define BF_SCAN_MEM_NOBYPASS_INFO          0x00000106
#define BF_SCAN_MEM_NOBYPASS(val)          ((val & 0x00000001) << 0x00000006)
#define BF_SCAN_MEM_NOBYPASS_GET(val)      ((val >> 0x00000006) & 0x00000001)
#define BF_RST_MODE_INFO                   0x00000107
#define BF_RST_MODE(val)                   ((val & 0x00000001) << 0x00000007)
#define BF_RST_MODE_GET(val)               ((val >> 0x00000007) & 0x00000001)

#define REG_OSC_TRIM_ADDR                  0x00003D31
#define BF_TRIM_COARSE_INFO                0x00000400
#define BF_TRIM_COARSE(val)                (val & 0x0000000F)
#define BF_TRIM_COARSE_GET(val)            (val & 0x0000000F)
#define BF_TRIM_FINE_INFO                  0x00000404
#define BF_TRIM_FINE(val)                  ((val & 0x0000000F) << 0x00000004)
#define BF_TRIM_FINE_GET(val)              ((val >> 0x00000004) & 0x0000000F)

#define REG_OSC_CLKSEL_ADDR                0x00003D32
#define BF_ADC_CLK_AVAIL_INFO              0x00000100
#define BF_ADC_CLK_AVAIL(val)              (val & 0x00000001)
#define BF_ADC_CLK_AVAIL_GET(val)          (val & 0x00000001)
#define BF_OSCCLK_DBG_FORCE_SEL_INFO       0x00000106
#define BF_OSCCLK_DBG_FORCE_SEL(val)       ((val & 0x00000001) << 0x00000006)
#define BF_OSCCLK_DBG_FORCE_SEL_GET(val)   ((val >> 0x00000006) & 0x00000001)
#define BF_OSCCLK_DBG_SEL_INFO             0x00000107
#define BF_OSCCLK_DBG_SEL(val)             ((val & 0x00000001) << 0x00000007)
#define BF_OSCCLK_DBG_SEL_GET(val)         ((val >> 0x00000007) & 0x00000001)

#define REG_OSC_CLKDIV_ADDR                0x00003D33
#define BF_ADC_CLK_AVAIL_OVERRIDE_INFO     0x00000100
#define BF_ADC_CLK_AVAIL_OVERRIDE(val)     (val & 0x00000001)
#define BF_ADC_CLK_AVAIL_OVERRIDE_GET(val) (val & 0x00000001)
#define BF_CRP_CLKDIV_OVERRIDE_INFO        0x00000101
#define BF_CRP_CLKDIV_OVERRIDE(val)        ((val & 0x00000001) << 0x00000001)
#define BF_CRP_CLKDIV_OVERRIDE_GET(val)    ((val >> 0x00000001) & 0x00000001)
#define BF_OSC_MONITOR_EN_INFO             0x00000102
#define BF_OSC_MONITOR_EN(val)             ((val & 0x00000001) << 0x00000002)
#define BF_OSC_MONITOR_EN_GET(val)         ((val >> 0x00000002) & 0x00000001)
#define BF_MC_OSC_CLKDIV_RATIO_INFO        0x00000204
#define BF_MC_OSC_CLKDIV_RATIO(val)        ((val & 0x00000003) << 0x00000004)
#define BF_MC_OSC_CLKDIV_RATIO_GET(val)    ((val >> 0x00000004) & 0x00000003)

#define REG_UP_MSGREG_STATUS3_ADDR         0x00003D37
#define BF_UP_MSGBIT_STAT_0_INFO           0x00000800
#define BF_UP_MSGBIT_STAT_0(val)           (val & 0x000000FF)
#define BF_UP_MSGBIT_STAT_0_GET(val)       (val & 0x000000FF)

#define REG_UP_MSGREG_STATUS2_ADDR         0x00003D36
#define BF_UP_MSGBIT_STAT_1_INFO           0x00000800
#define BF_UP_MSGBIT_STAT_1(val)           (val & 0x000000FF)
#define BF_UP_MSGBIT_STAT_1_GET(val)       (val & 0x000000FF)

#define REG_UP_MSGREG_STATUS1_ADDR         0x00003D35
#define BF_UP_MSGBIT_STAT_2_INFO           0x00000800
#define BF_UP_MSGBIT_STAT_2(val)           (val & 0x000000FF)
#define BF_UP_MSGBIT_STAT_2_GET(val)       (val & 0x000000FF)

#define REG_UP_MSGREG_STATUS0_ADDR         0x00003D34
#define BF_UP_MSGBIT_STAT_3_INFO           0x00000800
#define BF_UP_MSGBIT_STAT_3(val)           (val & 0x000000FF)
#define BF_UP_MSGBIT_STAT_3_GET(val)       (val & 0x000000FF)

#define REG_MBIST_MODE_REG_ADDR            0x00003D39
#define BF_MBIST_MODE_INFO                 0x00000100
#define BF_MBIST_MODE(val)                 (val & 0x00000001)
#define BF_MBIST_MODE_GET(val)             (val & 0x00000001)

#define REG_UP_CLOCKS_OFF_ADDR             0x00003D3A
#define BF_UP_CLOCKS_OFF_INFO              0x00000100
#define BF_UP_CLOCKS_OFF(val)              (val & 0x00000001)
#define BF_UP_CLOCKS_OFF_GET(val)          (val & 0x00000001)

#define REG_ENH_REG_ADDR                   0x00003D3B
#define BF_SPI_ENHANCEMENT_EN_INFO         0x00000100
#define BF_SPI_ENHANCEMENT_EN(val)         (val & 0x00000001)
#define BF_SPI_ENHANCEMENT_EN_GET(val)     (val & 0x00000001)

#define REG_SYNC_REG_ADDR                  0x00003D3C
#define BF_SYNC_EN_INFO                    0x00000100
#define BF_SYNC_EN(val)                    (val & 0x00000001)
#define BF_SYNC_EN_GET(val)                (val & 0x00000001)



#endif /* __ADI_AD9082_BF_SPI_ONLY_UP_H__ */
/*! @} */