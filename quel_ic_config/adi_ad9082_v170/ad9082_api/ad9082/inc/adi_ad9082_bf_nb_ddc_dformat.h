/*!
 * @brief     SPI Register Definition Header File, automatically generated file at 1/20/2020 6:24:26 AM.
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
#ifndef __ADI_AD9082_BF_NB_DDC_DFORMAT_H__
#define __ADI_AD9082_BF_NB_DDC_DFORMAT_H__

/*============= I N C L U D E S ============*/
#include "adi_ad9082_config.h"

/*============= D E F I N E S ==============*/
#define REG_ADC_COARSE_CB_ADDR                 0x00000280
#define BF_ADC_COARSE_CB_INFO                  0x00000400
#define BF_ADC_COARSE_CB(val)                  (val & 0x0000000F)
#define BF_ADC_COARSE_CB_GET(val)              (val & 0x0000000F)
#define BF_C_MXR_IQ_SFL_INFO                   0x00000404
#define BF_C_MXR_IQ_SFL(val)                   ((val & 0x0000000F) << 0x00000004)
#define BF_C_MXR_IQ_SFL_GET(val)               ((val >> 0x00000004) & 0x0000000F)

#define REG_COARSE_FINE_CB_ADDR                0x00000281
#define BF_COARSE_FINE_CB_INFO                 0x00000800
#define BF_COARSE_FINE_CB(val)                 (val & 0x000000FF)
#define BF_COARSE_FINE_CB_GET(val)             (val & 0x000000FF)

#define REG_COARSE_DEC_CTRL_ADDR               0x00000282
#define BF_COARSE_DEC_SEL_INFO                 0x00000400
#define BF_COARSE_DEC_SEL(val)                 (val & 0x0000000F)
#define BF_COARSE_DEC_SEL_GET(val)             (val & 0x0000000F)
#define BF_COARSE_C2R_EN_INFO                  0x00000104
#define BF_COARSE_C2R_EN(val)                  ((val & 0x00000001) << 0x00000004)
#define BF_COARSE_C2R_EN_GET(val)              ((val >> 0x00000004) & 0x00000001)
#define BF_COARSE_GAIN_INFO                    0x00000105
#define BF_COARSE_GAIN(val)                    ((val & 0x00000001) << 0x00000005)
#define BF_COARSE_GAIN_GET(val)                ((val >> 0x00000005) & 0x00000001)
#define BF_COARSE_MXR_IF_INFO                  0x00000206
#define BF_COARSE_MXR_IF(val)                  ((val & 0x00000003) << 0x00000006)
#define BF_COARSE_MXR_IF_GET(val)              ((val >> 0x00000006) & 0x00000003)

#define REG_FINE_DEC_CTRL_ADDR                 0x00000283
#define BF_FINE_DEC_SEL_INFO                   0x00000300
#define BF_FINE_DEC_SEL(val)                   (val & 0x00000007)
#define BF_FINE_DEC_SEL_GET(val)               (val & 0x00000007)
#define BF_FINE_C2R_EN_INFO                    0x00000104
#define BF_FINE_C2R_EN(val)                    ((val & 0x00000001) << 0x00000004)
#define BF_FINE_C2R_EN_GET(val)                ((val >> 0x00000004) & 0x00000001)
#define BF_FINE_GAIN_INFO                      0x00000105
#define BF_FINE_GAIN(val)                      ((val & 0x00000001) << 0x00000005)
#define BF_FINE_GAIN_GET(val)                  ((val >> 0x00000005) & 0x00000001)
#define BF_FINE_MXR_IF_INFO                    0x00000206
#define BF_FINE_MXR_IF(val)                    ((val & 0x00000003) << 0x00000006)
#define BF_FINE_MXR_IF_GET(val)                ((val >> 0x00000006) & 0x00000003)

#define REG_DDC_OVERALL_DECIM_ADDR             0x00000284
#define BF_DDC_OVERALL_DECIM_INFO              0x00000800
#define BF_DDC_OVERALL_DECIM(val)              (val & 0x000000FF)
#define BF_DDC_OVERALL_DECIM_GET(val)          (val & 0x000000FF)

#define REG_COARSE_DDC_EN_ADDR                 0x00000285
#define BF_COARSE_DDC_EN_INFO                  0x00000400
#define BF_COARSE_DDC_EN(val)                  (val & 0x0000000F)
#define BF_COARSE_DDC_EN_GET(val)              (val & 0x0000000F)

#define REG_FINE_DDC_EN_ADDR                   0x00000286
#define BF_FINE_DDC_EN_INFO                    0x00000800
#define BF_FINE_DDC_EN(val)                    (val & 0x000000FF)
#define BF_FINE_DDC_EN_GET(val)                (val & 0x000000FF)

#define REG_FINE_BYPASS_ADDR                   0x00000287
#define BF_FINE_BYPASS_INFO                    0x00000800
#define BF_FINE_BYPASS(val)                    (val & 0x000000FF)
#define BF_FINE_BYPASS_GET(val)                (val & 0x000000FF)

#define REG_CHIP_DECIMATION_RATIO_ADDR         0x00000289
#define BF_CHIP_DECIMATION_RATIO_INFO          0x00000800
#define BF_CHIP_DECIMATION_RATIO(val)          (val & 0x000000FF)
#define BF_CHIP_DECIMATION_RATIO_GET(val)      (val & 0x000000FF)

#define REG_COMMON_HOP_EN_ADDR                 0x0000028A
#define BF_COMMON_HOP_EN_INFO                  0x00000100
#define BF_COMMON_HOP_EN(val)                  (val & 0x00000001)
#define BF_COMMON_HOP_EN_GET(val)              (val & 0x00000001)

#define REG_CTRL_0_1_SEL_ADDR                  0x000002A1
#define BF_DFORMAT_CTRL_BIT_0_SEL_INFO         0x00000400
#define BF_DFORMAT_CTRL_BIT_0_SEL(val)         (val & 0x0000000F)
#define BF_DFORMAT_CTRL_BIT_0_SEL_GET(val)     (val & 0x0000000F)
#define BF_DFORMAT_CTRL_BIT_1_SEL_INFO         0x00000404
#define BF_DFORMAT_CTRL_BIT_1_SEL(val)         ((val & 0x0000000F) << 0x00000004)
#define BF_DFORMAT_CTRL_BIT_1_SEL_GET(val)     ((val >> 0x00000004) & 0x0000000F)

#define REG_CTRL_2_SEL_ADDR                    0x000002A2
#define BF_DFORMAT_CTRL_BIT_2_SEL_INFO         0x00000400
#define BF_DFORMAT_CTRL_BIT_2_SEL(val)         (val & 0x0000000F)
#define BF_DFORMAT_CTRL_BIT_2_SEL_GET(val)     (val & 0x0000000F)

#define REG_OUT_FORMAT_SEL_ADDR                0x000002A3
#define BF_DFORMAT_SEL_INFO                    0x00000200
#define BF_DFORMAT_SEL(val)                    (val & 0x00000003)
#define BF_DFORMAT_SEL_GET(val)                (val & 0x00000003)
#define BF_DFORMAT_INV_INFO                    0x00000102
#define BF_DFORMAT_INV(val)                    ((val & 0x00000001) << 0x00000002)
#define BF_DFORMAT_INV_GET(val)                ((val >> 0x00000002) & 0x00000001)

#define REG_OVR_CLR_0_ADDR                     0x000002A4
#define BF_DFORMAT_OVR_CLR_INFO                0x00001000
#define BF_DFORMAT_OVR_CLR(val)                (val & 0x0000FFFF)
#define BF_DFORMAT_OVR_CLR_GET(val)            (val & 0x0000FFFF)

#define REG_OVR_CLR_1_ADDR                     0x000002A5

#define REG_OVR_STATUS_0_ADDR                  0x000002A6
#define BF_DFORMAT_OVR_STATUS_INFO             0x00001000
#define BF_DFORMAT_OVR_STATUS(val)             (val & 0x0000FFFF)
#define BF_DFORMAT_OVR_STATUS_GET(val)         (val & 0x0000FFFF)

#define REG_OVR_STATUS_1_ADDR                  0x000002A7

#define REG_OUT_RES_ADDR                       0x000002A8
#define BF_DFORMAT_RES_INFO                    0x00000400
#define BF_DFORMAT_RES(val)                    (val & 0x0000000F)
#define BF_DFORMAT_RES_GET(val)                (val & 0x0000000F)
#define BF_DFORMAT_FBW_DITHER_EN_INFO          0x00000104
#define BF_DFORMAT_FBW_DITHER_EN(val)          ((val & 0x00000001) << 0x00000004)
#define BF_DFORMAT_FBW_DITHER_EN_GET(val)      ((val >> 0x00000004) & 0x00000001)
#define BF_DFORMAT_DDC_DITHER_EN_INFO          0x00000105
#define BF_DFORMAT_DDC_DITHER_EN(val)          ((val & 0x00000001) << 0x00000005)
#define BF_DFORMAT_DDC_DITHER_EN_GET(val)      ((val >> 0x00000005) & 0x00000001)

#define REG_FD_SEL_0_ADDR                      0x000002A9
#define BF_DFORMAT_FD_SEL_INFO                 0x00001000
#define BF_DFORMAT_FD_SEL(val)                 (val & 0x0000FFFF)
#define BF_DFORMAT_FD_SEL_GET(val)             (val & 0x0000FFFF)

#define REG_FD_SEL_1_ADDR                      0x000002AA

#define REG_FBW_SEL_0_ADDR                     0x000002AB
#define BF_DFORMAT_FBW_SEL_INFO                0x00001000
#define BF_DFORMAT_FBW_SEL(val)                (val & 0x0000FFFF)
#define BF_DFORMAT_FBW_SEL_GET(val)            (val & 0x0000FFFF)

#define REG_FBW_SEL_1_ADDR                     0x000002AC

#define REG_TMODE_SEL_0_ADDR                   0x000002AD
#define BF_DFORMAT_TMODE_SEL_INFO              0x00001000
#define BF_DFORMAT_TMODE_SEL(val)              (val & 0x0000FFFF)
#define BF_DFORMAT_TMODE_SEL_GET(val)          (val & 0x0000FFFF)

#define REG_TMODE_SEL_1_ADDR                   0x000002AE

#define REG_TMODE_I_CTRL1_ADDR                 0x000002B0
#define BF_TMODE_I_PN_SEL_INFO                 0x00000400
#define BF_TMODE_I_PN_SEL(val)                 (val & 0x0000000F)
#define BF_TMODE_I_PN_SEL_GET(val)             (val & 0x0000000F)
#define BF_TMODE_I_TYPE_SEL_INFO               0x00000404
#define BF_TMODE_I_TYPE_SEL(val)               ((val & 0x0000000F) << 0x00000004)
#define BF_TMODE_I_TYPE_SEL_GET(val)           ((val >> 0x00000004) & 0x0000000F)

#define REG_TMODE_I_CTRL2_ADDR                 0x000002B1
#define BF_TMODE_I_RES_INFO                    0x00000400
#define BF_TMODE_I_RES(val)                    (val & 0x0000000F)
#define BF_TMODE_I_RES_GET(val)                (val & 0x0000000F)
#define BF_TMODE_I_USR_PAT_SEL_INFO            0x00000104
#define BF_TMODE_I_USR_PAT_SEL(val)            ((val & 0x00000001) << 0x00000004)
#define BF_TMODE_I_USR_PAT_SEL_GET(val)        ((val >> 0x00000004) & 0x00000001)
#define BF_TMODE_I_FORCE_RST_INFO              0x00000105
#define BF_TMODE_I_FORCE_RST(val)              ((val & 0x00000001) << 0x00000005)
#define BF_TMODE_I_FORCE_RST_GET(val)          ((val >> 0x00000005) & 0x00000001)
#define BF_TMODE_I_PN_FORCE_RST_INFO           0x00000106
#define BF_TMODE_I_PN_FORCE_RST(val)           ((val & 0x00000001) << 0x00000006)
#define BF_TMODE_I_PN_FORCE_RST_GET(val)       ((val >> 0x00000006) & 0x00000001)
#define BF_TMODE_I_FLUSH_INFO                  0x00000107
#define BF_TMODE_I_FLUSH(val)                  ((val & 0x00000001) << 0x00000007)
#define BF_TMODE_I_FLUSH_GET(val)              ((val >> 0x00000007) & 0x00000001)

#define REG_TMODE_I_USR_PAT0_LSB_ADDR          0x000002B2
#define BF_TMODE_I_USR_PAT0_INFO               0x00001000
#define BF_TMODE_I_USR_PAT0(val)               (val & 0x0000FFFF)
#define BF_TMODE_I_USR_PAT0_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_I_USR_PAT0_MSB_ADDR          0x000002B3

#define REG_TMODE_I_USR_PAT1_LSB_ADDR          0x000002B4
#define BF_TMODE_I_USR_PAT1_INFO               0x00001000
#define BF_TMODE_I_USR_PAT1(val)               (val & 0x0000FFFF)
#define BF_TMODE_I_USR_PAT1_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_I_USR_PAT1_MSB_ADDR          0x000002B5

#define REG_TMODE_I_USR_PAT2_LSB_ADDR          0x000002B6
#define BF_TMODE_I_USR_PAT2_INFO               0x00001000
#define BF_TMODE_I_USR_PAT2(val)               (val & 0x0000FFFF)
#define BF_TMODE_I_USR_PAT2_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_I_USR_PAT2_MSB_ADDR          0x000002B7

#define REG_TMODE_I_USR_PAT3_LSB_ADDR          0x000002B8
#define BF_TMODE_I_USR_PAT3_INFO               0x00001000
#define BF_TMODE_I_USR_PAT3(val)               (val & 0x0000FFFF)
#define BF_TMODE_I_USR_PAT3_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_I_USR_PAT3_MSB_ADDR          0x000002B9

#define REG_SYNC_CTRL1_ADDR                    0x000002BA
#define BF_DP_CLK_FORCEN_INFO                  0x00000100
#define BF_DP_CLK_FORCEN(val)                  (val & 0x00000001)
#define BF_DP_CLK_FORCEN_GET(val)              (val & 0x00000001)
#define BF_RISEDGE_SYSREF_INFO                 0x00000101
#define BF_RISEDGE_SYSREF(val)                 ((val & 0x00000001) << 0x00000001)
#define BF_RISEDGE_SYSREF_GET(val)             ((val >> 0x00000001) & 0x00000001)
#define BF_SYSREF_RESYNC_MODE_INFO             0x00000102
#define BF_SYSREF_RESYNC_MODE(val)             ((val & 0x00000001) << 0x00000002)
#define BF_SYSREF_RESYNC_MODE_GET(val)         ((val >> 0x00000002) & 0x00000001)
#define BF_NCORESET_ALL_SYSREF_INFO            0x00000103
#define BF_NCORESET_ALL_SYSREF(val)            ((val & 0x00000001) << 0x00000003)
#define BF_NCORESET_ALL_SYSREF_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_ALLOW_SYSREFMASK_INFO               0x00000104
#define BF_ALLOW_SYSREFMASK(val)               ((val & 0x00000001) << 0x00000004)
#define BF_ALLOW_SYSREFMASK_GET(val)           ((val >> 0x00000004) & 0x00000001)

#define REG_TRIG_PROG_DELAY_ADDR               0x000002BB
#define BF_TRIG_PROG_DELAY_INFO                0x00000800
#define BF_TRIG_PROG_DELAY(val)                (val & 0x000000FF)
#define BF_TRIG_PROG_DELAY_GET(val)            (val & 0x000000FF)

#define REG_SYSREF_PROG_DELAY_ADDR             0x000002BC
#define BF_SYSREF_PROG_DELAY_INFO              0x00000800
#define BF_SYSREF_PROG_DELAY(val)              (val & 0x000000FF)
#define BF_SYSREF_PROG_DELAY_GET(val)          (val & 0x000000FF)

#define REG_TRIG_CTRL_ADDR                     0x000002BD
#define BF_GPIO_TRIG_EN_INFO                   0x00000100
#define BF_GPIO_TRIG_EN(val)                   (val & 0x00000001)
#define BF_GPIO_TRIG_EN_GET(val)               (val & 0x00000001)
#define BF_MASTERTRIG_EN_INFO                  0x00000101
#define BF_MASTERTRIG_EN(val)                  ((val & 0x00000001) << 0x00000001)
#define BF_MASTERTRIG_EN_GET(val)              ((val >> 0x00000001) & 0x00000001)
#define BF_LOOPBACK_MASTERTRIG_INFO            0x00000102
#define BF_LOOPBACK_MASTERTRIG(val)            ((val & 0x00000001) << 0x00000002)
#define BF_LOOPBACK_MASTERTRIG_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_RESYNC_AFTER_TRIG_INFO              0x00000103
#define BF_RESYNC_AFTER_TRIG(val)              ((val & 0x00000001) << 0x00000003)
#define BF_RESYNC_AFTER_TRIG_GET(val)          ((val >> 0x00000003) & 0x00000001)
#define BF_GPIOTRIG_SYNCEN_INFO                0x00000104
#define BF_GPIOTRIG_SYNCEN(val)                ((val & 0x00000001) << 0x00000004)
#define BF_GPIOTRIG_SYNCEN_GET(val)            ((val >> 0x00000004) & 0x00000001)
#define BF_GPIOTRIG_DEGLITCH_INFO              0x00000105
#define BF_GPIOTRIG_DEGLITCH(val)              ((val & 0x00000001) << 0x00000005)
#define BF_GPIOTRIG_DEGLITCH_GET(val)          ((val >> 0x00000005) & 0x00000001)
#define BF_RISEDGE_TRIG_INFO                   0x00000106
#define BF_RISEDGE_TRIG(val)                   ((val & 0x00000001) << 0x00000006)
#define BF_RISEDGE_TRIG_GET(val)               ((val >> 0x00000006) & 0x00000001)

#define REG_I_PHASEMAX_POSTSRC_STATUS_LSB_ADDR 0x000002C2
#define BF_I_PHASEMAX_POSTSRC_STATUS_INFO      0x00000C00
#define BF_I_PHASEMAX_POSTSRC_STATUS(val)      (val & 0x00000FFF)
#define BF_I_PHASEMAX_POSTSRC_STATUS_GET(val)  (val & 0x00000FFF)

#define REG_I_PHASEMAX_POSTSRC_STATUS_MSB_ADDR 0x000002C3

#define REG_I_PHASEMAX_PRESRC_STATUS_LSB_ADDR  0x000002C4
#define BF_I_PHASEMAX_PRESRC_STATUS_INFO       0x00000C00
#define BF_I_PHASEMAX_PRESRC_STATUS(val)       (val & 0x00000FFF)
#define BF_I_PHASEMAX_PRESRC_STATUS_GET(val)   (val & 0x00000FFF)

#define REG_I_PHASEMAX_PRESRC_STATUS_MSB_ADDR  0x000002C5

#define REG_RXEN0_SEL0_ADDR                    0x000002C6
#define BF_RXEN0_FDDC_SEL_INFO                 0x00000800
#define BF_RXEN0_FDDC_SEL(val)                 (val & 0x000000FF)
#define BF_RXEN0_FDDC_SEL_GET(val)             (val & 0x000000FF)

#define REG_RXEN0_SEL1_ADDR                    0x000002C7
#define BF_RXEN0_CDDC_SEL_INFO                 0x00000400
#define BF_RXEN0_CDDC_SEL(val)                 (val & 0x0000000F)
#define BF_RXEN0_CDDC_SEL_GET(val)             (val & 0x0000000F)
#define BF_RXEN0_JTXL_SEL_INFO                 0x00000204
#define BF_RXEN0_JTXL_SEL(val)                 ((val & 0x00000003) << 0x00000004)
#define BF_RXEN0_JTXL_SEL_GET(val)             ((val >> 0x00000004) & 0x00000003)
#define BF_RXEN0_ADC_SEL_INFO                  0x00000206
#define BF_RXEN0_ADC_SEL(val)                  ((val & 0x00000003) << 0x00000006)
#define BF_RXEN0_ADC_SEL_GET(val)              ((val >> 0x00000006) & 0x00000003)

#define REG_RXEN0_SEL2_ADDR                    0x000002C8
#define BF_RXEN0_JTXPHY_SEL_INFO               0x00000800
#define BF_RXEN0_JTXPHY_SEL(val)               (val & 0x000000FF)
#define BF_RXEN0_JTXPHY_SEL_GET(val)           (val & 0x000000FF)

#define REG_RXEN1_SEL0_ADDR                    0x000002C9
#define BF_RXEN1_FDDC_SEL_INFO                 0x00000800
#define BF_RXEN1_FDDC_SEL(val)                 (val & 0x000000FF)
#define BF_RXEN1_FDDC_SEL_GET(val)             (val & 0x000000FF)

#define REG_RXEN1_SEL1_ADDR                    0x000002CA
#define BF_RXEN1_CDDC_SEL_INFO                 0x00000400
#define BF_RXEN1_CDDC_SEL(val)                 (val & 0x0000000F)
#define BF_RXEN1_CDDC_SEL_GET(val)             (val & 0x0000000F)
#define BF_RXEN1_JTXL_SEL_INFO                 0x00000204
#define BF_RXEN1_JTXL_SEL(val)                 ((val & 0x00000003) << 0x00000004)
#define BF_RXEN1_JTXL_SEL_GET(val)             ((val >> 0x00000004) & 0x00000003)
#define BF_RXEN1_ADC_SEL_INFO                  0x00000206
#define BF_RXEN1_ADC_SEL(val)                  ((val & 0x00000003) << 0x00000006)
#define BF_RXEN1_ADC_SEL_GET(val)              ((val >> 0x00000006) & 0x00000003)

#define REG_RXEN1_SEL2_ADDR                    0x000002CB
#define BF_RXEN1_JTXPHY_SEL_INFO               0x00000800
#define BF_RXEN1_JTXPHY_SEL(val)               (val & 0x000000FF)
#define BF_RXEN1_JTXPHY_SEL_GET(val)           (val & 0x000000FF)

#define REG_FINE_DDC_STATUS_SEL_ADDR           0x000002CC
#define BF_FINE_DDC_I_STATUS_SEL_INFO          0x00000200
#define BF_FINE_DDC_I_STATUS_SEL(val)          (val & 0x00000003)
#define BF_FINE_DDC_I_STATUS_SEL_GET(val)      (val & 0x00000003)
#define BF_FINE_DDC_Q_STATUS_SEL_INFO          0x00000202
#define BF_FINE_DDC_Q_STATUS_SEL(val)          ((val & 0x00000003) << 0x00000002)
#define BF_FINE_DDC_Q_STATUS_SEL_GET(val)      ((val >> 0x00000002) & 0x00000003)

#define REG_FD_EQ_STATUS_SEL_ADDR              0x000002CD
#define BF_FD_EQ_I_STATUS_SEL_INFO             0x00000200
#define BF_FD_EQ_I_STATUS_SEL(val)             (val & 0x00000003)
#define BF_FD_EQ_I_STATUS_SEL_GET(val)         (val & 0x00000003)
#define BF_FD_EQ_Q_STATUS_SEL_INFO             0x00000202
#define BF_FD_EQ_Q_STATUS_SEL(val)             ((val & 0x00000003) << 0x00000002)
#define BF_FD_EQ_Q_STATUS_SEL_GET(val)         ((val >> 0x00000002) & 0x00000003)

#define REG_RXENGP0_SEL0_ADDR                  0x000002CE
#define BF_RXENGP0_FDDC_SEL_INFO               0x00000800
#define BF_RXENGP0_FDDC_SEL(val)               (val & 0x000000FF)
#define BF_RXENGP0_FDDC_SEL_GET(val)           (val & 0x000000FF)

#define REG_RXENGP0_SEL1_ADDR                  0x000002CF
#define BF_RXENGP0_CDDC_SEL_INFO               0x00000400
#define BF_RXENGP0_CDDC_SEL(val)               (val & 0x0000000F)
#define BF_RXENGP0_CDDC_SEL_GET(val)           (val & 0x0000000F)
#define BF_RXENGP0_JTXL_SEL_INFO               0x00000204
#define BF_RXENGP0_JTXL_SEL(val)               ((val & 0x00000003) << 0x00000004)
#define BF_RXENGP0_JTXL_SEL_GET(val)           ((val >> 0x00000004) & 0x00000003)
#define BF_RXENGP0_ADC_SEL_INFO                0x00000206
#define BF_RXENGP0_ADC_SEL(val)                ((val & 0x00000003) << 0x00000006)
#define BF_RXENGP0_ADC_SEL_GET(val)            ((val >> 0x00000006) & 0x00000003)

#define REG_RXENGP0_SEL2_ADDR                  0x000002D0
#define BF_RXENGP0_JTXPHY_SEL_INFO             0x00000800
#define BF_RXENGP0_JTXPHY_SEL(val)             (val & 0x000000FF)
#define BF_RXENGP0_JTXPHY_SEL_GET(val)         (val & 0x000000FF)

#define REG_RXENGP1_SEL0_ADDR                  0x000002D1
#define BF_RXENGP1_FDDC_SEL_INFO               0x00000800
#define BF_RXENGP1_FDDC_SEL(val)               (val & 0x000000FF)
#define BF_RXENGP1_FDDC_SEL_GET(val)           (val & 0x000000FF)

#define REG_RXENGP1_SEL1_ADDR                  0x000002D2
#define BF_RXENGP1_CDDC_SEL_INFO               0x00000400
#define BF_RXENGP1_CDDC_SEL(val)               (val & 0x0000000F)
#define BF_RXENGP1_CDDC_SEL_GET(val)           (val & 0x0000000F)
#define BF_RXENGP1_JTXL_SEL_INFO               0x00000204
#define BF_RXENGP1_JTXL_SEL(val)               ((val & 0x00000003) << 0x00000004)
#define BF_RXENGP1_JTXL_SEL_GET(val)           ((val >> 0x00000004) & 0x00000003)
#define BF_RXENGP1_ADC_SEL_INFO                0x00000206
#define BF_RXENGP1_ADC_SEL(val)                ((val & 0x00000003) << 0x00000006)
#define BF_RXENGP1_ADC_SEL_GET(val)            ((val >> 0x00000006) & 0x00000003)

#define REG_RXENGP1_SEL2_ADDR                  0x000002D3
#define BF_RXENGP1_JTXPHY_SEL_INFO             0x00000800
#define BF_RXENGP1_JTXPHY_SEL(val)             (val & 0x000000FF)
#define BF_RXENGP1_JTXPHY_SEL_GET(val)         (val & 0x000000FF)

#define REG_TMODE_Q_CTRL1_ADDR                 0x000002D4
#define BF_TMODE_Q_PN_SEL_INFO                 0x00000400
#define BF_TMODE_Q_PN_SEL(val)                 (val & 0x0000000F)
#define BF_TMODE_Q_PN_SEL_GET(val)             (val & 0x0000000F)
#define BF_TMODE_Q_TYPE_SEL_INFO               0x00000404
#define BF_TMODE_Q_TYPE_SEL(val)               ((val & 0x0000000F) << 0x00000004)
#define BF_TMODE_Q_TYPE_SEL_GET(val)           ((val >> 0x00000004) & 0x0000000F)

#define REG_TMODE_Q_CTRL2_ADDR                 0x000002D5
#define BF_TMODE_Q_RES_INFO                    0x00000400
#define BF_TMODE_Q_RES(val)                    (val & 0x0000000F)
#define BF_TMODE_Q_RES_GET(val)                (val & 0x0000000F)
#define BF_TMODE_Q_USR_PAT_SEL_INFO            0x00000104
#define BF_TMODE_Q_USR_PAT_SEL(val)            ((val & 0x00000001) << 0x00000004)
#define BF_TMODE_Q_USR_PAT_SEL_GET(val)        ((val >> 0x00000004) & 0x00000001)
#define BF_TMODE_Q_FORCE_RST_INFO              0x00000105
#define BF_TMODE_Q_FORCE_RST(val)              ((val & 0x00000001) << 0x00000005)
#define BF_TMODE_Q_FORCE_RST_GET(val)          ((val >> 0x00000005) & 0x00000001)
#define BF_TMODE_Q_PN_FORCE_RST_INFO           0x00000106
#define BF_TMODE_Q_PN_FORCE_RST(val)           ((val & 0x00000001) << 0x00000006)
#define BF_TMODE_Q_PN_FORCE_RST_GET(val)       ((val >> 0x00000006) & 0x00000001)
#define BF_TMODE_Q_FLUSH_INFO                  0x00000107
#define BF_TMODE_Q_FLUSH(val)                  ((val & 0x00000001) << 0x00000007)
#define BF_TMODE_Q_FLUSH_GET(val)              ((val >> 0x00000007) & 0x00000001)

#define REG_TMODE_Q_USR_PAT0_LSB_ADDR          0x000002D6
#define BF_TMODE_Q_USR_PAT0_INFO               0x00001000
#define BF_TMODE_Q_USR_PAT0(val)               (val & 0x0000FFFF)
#define BF_TMODE_Q_USR_PAT0_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_Q_USR_PAT0_MSB_ADDR          0x000002D7

#define REG_TMODE_Q_USR_PAT1_LSB_ADDR          0x000002D8
#define BF_TMODE_Q_USR_PAT1_INFO               0x00001000
#define BF_TMODE_Q_USR_PAT1(val)               (val & 0x0000FFFF)
#define BF_TMODE_Q_USR_PAT1_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_Q_USR_PAT1_MSB_ADDR          0x000002D9

#define REG_TMODE_Q_USR_PAT2_LSB_ADDR          0x000002DA
#define BF_TMODE_Q_USR_PAT2_INFO               0x00001000
#define BF_TMODE_Q_USR_PAT2(val)               (val & 0x0000FFFF)
#define BF_TMODE_Q_USR_PAT2_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_Q_USR_PAT2_MSB_ADDR          0x000002DB

#define REG_TMODE_Q_USR_PAT3_LSB_ADDR          0x000002DC
#define BF_TMODE_Q_USR_PAT3_INFO               0x00001000
#define BF_TMODE_Q_USR_PAT3(val)               (val & 0x0000FFFF)
#define BF_TMODE_Q_USR_PAT3_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_Q_USR_PAT3_MSB_ADDR          0x000002DD

#define REG_ADC_MODES_ADDR                     0x000002E4
#define BF_ADC0_ADC1_MODES_INFO                0x00000300
#define BF_ADC0_ADC1_MODES(val)                (val & 0x00000007)
#define BF_ADC0_ADC1_MODES_GET(val)            (val & 0x00000007)
#define BF_ADC2_ADC3_MODES_INFO                0x00000303
#define BF_ADC2_ADC3_MODES(val)                ((val & 0x00000007) << 0x00000003)
#define BF_ADC2_ADC3_MODES_GET(val)            ((val >> 0x00000003) & 0x00000007)
#define BF_QUAD_MODES_INFO                     0x00000206
#define BF_QUAD_MODES(val)                     ((val & 0x00000003) << 0x00000006)
#define BF_QUAD_MODES_GET(val)                 ((val >> 0x00000006) & 0x00000003)

#define REG_HEAD_ROOM_GROWTH_ADDR              0x000002E8
#define BF_HEAD_ROOM_INFO                      0x00000400
#define BF_HEAD_ROOM(val)                      (val & 0x0000000F)
#define BF_HEAD_ROOM_GET(val)                  (val & 0x0000000F)

#define REG_COARSE_FSRC_EN_ADDR                0x000002E9
#define BF_COARSE_FSRC_EN_INFO                 0x00000400
#define BF_COARSE_FSRC_EN(val)                 (val & 0x0000000F)
#define BF_COARSE_FSRC_EN_GET(val)             (val & 0x0000000F)

#define REG_TMODE_I_USR_PAT4_LSB_ADDR          0x000002EA
#define BF_TMODE_I_USR_PAT4_INFO               0x00001000
#define BF_TMODE_I_USR_PAT4(val)               (val & 0x0000FFFF)
#define BF_TMODE_I_USR_PAT4_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_I_USR_PAT4_MSB_ADDR          0x000002EB

#define REG_TMODE_I_USR_PAT5_LSB_ADDR          0x000002EC
#define BF_TMODE_I_USR_PAT5_INFO               0x00001000
#define BF_TMODE_I_USR_PAT5(val)               (val & 0x0000FFFF)
#define BF_TMODE_I_USR_PAT5_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_I_USR_PAT5_MSB_ADDR          0x000002ED

#define REG_TMODE_I_USR_PAT6_LSB_ADDR          0x000002EE
#define BF_TMODE_I_USR_PAT6_INFO               0x00001000
#define BF_TMODE_I_USR_PAT6(val)               (val & 0x0000FFFF)
#define BF_TMODE_I_USR_PAT6_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_I_USR_PAT6_MSB_ADDR          0x000002EF

#define REG_TMODE_I_USR_PAT7_LSB_ADDR          0x000002F0
#define BF_TMODE_I_USR_PAT7_INFO               0x00001000
#define BF_TMODE_I_USR_PAT7(val)               (val & 0x0000FFFF)
#define BF_TMODE_I_USR_PAT7_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_I_USR_PAT7_MSB_ADDR          0x000002F1

#define REG_TMODE_Q_USR_PAT4_LSB_ADDR          0x000002F2
#define BF_TMODE_Q_USR_PAT4_INFO               0x00001000
#define BF_TMODE_Q_USR_PAT4(val)               (val & 0x0000FFFF)
#define BF_TMODE_Q_USR_PAT4_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_Q_USR_PAT4_MSB_ADDR          0x000002F3

#define REG_TMODE_Q_USR_PAT5_LSB_ADDR          0x000002F4
#define BF_TMODE_Q_USR_PAT5_INFO               0x00001000
#define BF_TMODE_Q_USR_PAT5(val)               (val & 0x0000FFFF)
#define BF_TMODE_Q_USR_PAT5_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_Q_USR_PAT5_MSB_ADDR          0x000002F5

#define REG_TMODE_Q_USR_PAT6_LSB_ADDR          0x000002F6
#define BF_TMODE_Q_USR_PAT6_INFO               0x00001000
#define BF_TMODE_Q_USR_PAT6(val)               (val & 0x0000FFFF)
#define BF_TMODE_Q_USR_PAT6_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_Q_USR_PAT6_MSB_ADDR          0x000002F7

#define REG_TMODE_Q_USR_PAT7_LSB_ADDR          0x000002F8
#define BF_TMODE_Q_USR_PAT7_INFO               0x00001000
#define BF_TMODE_Q_USR_PAT7(val)               (val & 0x0000FFFF)
#define BF_TMODE_Q_USR_PAT7_GET(val)           (val & 0x0000FFFF)

#define REG_TMODE_Q_USR_PAT7_MSB_ADDR          0x000002F9

#define REG_RXEN_CTRL_ADDR                     0x000002FA
#define BF_RXEN0_USETXEN_INFO                  0x00000100
#define BF_RXEN0_USETXEN(val)                  (val & 0x00000001)
#define BF_RXEN0_USETXEN_GET(val)              (val & 0x00000001)
#define BF_RXEN1_USETXEN_INFO                  0x00000101
#define BF_RXEN1_USETXEN(val)                  ((val & 0x00000001) << 0x00000001)
#define BF_RXEN1_USETXEN_GET(val)              ((val >> 0x00000001) & 0x00000001)
#define BF_RXEN0_POL_INFO                      0x00000102
#define BF_RXEN0_POL(val)                      ((val & 0x00000001) << 0x00000002)
#define BF_RXEN0_POL_GET(val)                  ((val >> 0x00000002) & 0x00000001)
#define BF_RXEN1_POL_INFO                      0x00000103
#define BF_RXEN1_POL(val)                      ((val & 0x00000001) << 0x00000003)
#define BF_RXEN1_POL_GET(val)                  ((val >> 0x00000003) & 0x00000001)
#define BF_RXENGP0_POL_INFO                    0x00000104
#define BF_RXENGP0_POL(val)                    ((val & 0x00000001) << 0x00000004)
#define BF_RXENGP0_POL_GET(val)                ((val >> 0x00000004) & 0x00000001)
#define BF_RXENGP1_POL_INFO                    0x00000105
#define BF_RXENGP1_POL(val)                    ((val & 0x00000001) << 0x00000005)
#define BF_RXENGP1_POL_GET(val)                ((val >> 0x00000005) & 0x00000001)

#define REG_RXEN_SPI_CTRL_ADDR                 0x000002FB
#define BF_RXEN0_SPIEN_INFO                    0x00000100
#define BF_RXEN0_SPIEN(val)                    (val & 0x00000001)
#define BF_RXEN0_SPIEN_GET(val)                (val & 0x00000001)
#define BF_RXEN1_SPIEN_INFO                    0x00000101
#define BF_RXEN1_SPIEN(val)                    ((val & 0x00000001) << 0x00000001)
#define BF_RXEN1_SPIEN_GET(val)                ((val >> 0x00000001) & 0x00000001)
#define BF_RXENGP0_SPIEN_INFO                  0x00000102
#define BF_RXENGP0_SPIEN(val)                  ((val & 0x00000001) << 0x00000002)
#define BF_RXENGP0_SPIEN_GET(val)              ((val >> 0x00000002) & 0x00000001)
#define BF_RXENGP1_SPIEN_INFO                  0x00000103
#define BF_RXENGP1_SPIEN(val)                  ((val & 0x00000001) << 0x00000003)
#define BF_RXENGP1_SPIEN_GET(val)              ((val >> 0x00000003) & 0x00000001)
#define BF_RXEN0_SPI_INFO                      0x00000104
#define BF_RXEN0_SPI(val)                      ((val & 0x00000001) << 0x00000004)
#define BF_RXEN0_SPI_GET(val)                  ((val >> 0x00000004) & 0x00000001)
#define BF_RXEN1_SPI_INFO                      0x00000105
#define BF_RXEN1_SPI(val)                      ((val & 0x00000001) << 0x00000005)
#define BF_RXEN1_SPI_GET(val)                  ((val >> 0x00000005) & 0x00000001)
#define BF_RXENGP0_SPI_INFO                    0x00000106
#define BF_RXENGP0_SPI(val)                    ((val & 0x00000001) << 0x00000006)
#define BF_RXENGP0_SPI_GET(val)                ((val >> 0x00000006) & 0x00000001)
#define BF_RXENGP1_SPI_INFO                    0x00000107
#define BF_RXENGP1_SPI(val)                    ((val & 0x00000001) << 0x00000007)
#define BF_RXENGP1_SPI_GET(val)                ((val >> 0x00000007) & 0x00000001)

#define REG_RXEN_NOVALP_CTRL1_ADDR             0x000002FC
#define BF_RXEN0_0S_CTRL_INFO                  0x00000100
#define BF_RXEN0_0S_CTRL(val)                  (val & 0x00000001)
#define BF_RXEN0_0S_CTRL_GET(val)              (val & 0x00000001)
#define BF_RXENGP0_0S_CTRL_INFO                0x00000101
#define BF_RXENGP0_0S_CTRL(val)                ((val & 0x00000001) << 0x00000001)
#define BF_RXENGP0_0S_CTRL_GET(val)            ((val >> 0x00000001) & 0x00000001)
#define BF_RXEN1_1S_CTRL_INFO                  0x00000102
#define BF_RXEN1_1S_CTRL(val)                  ((val & 0x00000001) << 0x00000002)
#define BF_RXEN1_1S_CTRL_GET(val)              ((val >> 0x00000002) & 0x00000001)
#define BF_RXENGP1_1S_CTRL_INFO                0x00000103
#define BF_RXENGP1_1S_CTRL(val)                ((val & 0x00000001) << 0x00000003)
#define BF_RXENGP1_1S_CTRL_GET(val)            ((val >> 0x00000003) & 0x00000001)
#define BF_RXEN0_0F_CTRL_INFO                  0x00000104
#define BF_RXEN0_0F_CTRL(val)                  ((val & 0x00000001) << 0x00000004)
#define BF_RXEN0_0F_CTRL_GET(val)              ((val >> 0x00000004) & 0x00000001)
#define BF_RXENGP0_0F_CTRL_INFO                0x00000105
#define BF_RXENGP0_0F_CTRL(val)                ((val & 0x00000001) << 0x00000005)
#define BF_RXENGP0_0F_CTRL_GET(val)            ((val >> 0x00000005) & 0x00000001)
#define BF_RXEN1_1F_CTRL_INFO                  0x00000106
#define BF_RXEN1_1F_CTRL(val)                  ((val & 0x00000001) << 0x00000006)
#define BF_RXEN1_1F_CTRL_GET(val)              ((val >> 0x00000006) & 0x00000001)
#define BF_RXENGP1_1F_CTRL_INFO                0x00000107
#define BF_RXENGP1_1F_CTRL(val)                ((val & 0x00000001) << 0x00000007)
#define BF_RXENGP1_1F_CTRL_GET(val)            ((val >> 0x00000007) & 0x00000001)

#define REG_RXEN_NOVALP_CTRL2_ADDR             0x000002FD
#define BF_RXEN0_2S_CTRL_INFO                  0x00000100
#define BF_RXEN0_2S_CTRL(val)                  (val & 0x00000001)
#define BF_RXEN0_2S_CTRL_GET(val)              (val & 0x00000001)
#define BF_RXENGP0_2S_CTRL_INFO                0x00000101
#define BF_RXENGP0_2S_CTRL(val)                ((val & 0x00000001) << 0x00000001)
#define BF_RXENGP0_2S_CTRL_GET(val)            ((val >> 0x00000001) & 0x00000001)
#define BF_RXEN1_3S_CTRL_INFO                  0x00000102
#define BF_RXEN1_3S_CTRL(val)                  ((val & 0x00000001) << 0x00000002)
#define BF_RXEN1_3S_CTRL_GET(val)              ((val >> 0x00000002) & 0x00000001)
#define BF_RXENGP1_3S_CTRL_INFO                0x00000103
#define BF_RXENGP1_3S_CTRL(val)                ((val & 0x00000001) << 0x00000003)
#define BF_RXENGP1_3S_CTRL_GET(val)            ((val >> 0x00000003) & 0x00000001)
#define BF_RXEN0_2F_CTRL_INFO                  0x00000104
#define BF_RXEN0_2F_CTRL(val)                  ((val & 0x00000001) << 0x00000004)
#define BF_RXEN0_2F_CTRL_GET(val)              ((val >> 0x00000004) & 0x00000001)
#define BF_RXENGP0_2F_CTRL_INFO                0x00000105
#define BF_RXENGP0_2F_CTRL(val)                ((val & 0x00000001) << 0x00000005)
#define BF_RXENGP0_2F_CTRL_GET(val)            ((val >> 0x00000005) & 0x00000001)
#define BF_RXEN1_3F_CTRL_INFO                  0x00000106
#define BF_RXEN1_3F_CTRL(val)                  ((val & 0x00000001) << 0x00000006)
#define BF_RXEN1_3F_CTRL_GET(val)              ((val >> 0x00000006) & 0x00000001)
#define BF_RXENGP1_3F_CTRL_INFO                0x00000107
#define BF_RXENGP1_3F_CTRL(val)                ((val & 0x00000001) << 0x00000007)
#define BF_RXENGP1_3F_CTRL_GET(val)            ((val >> 0x00000007) & 0x00000001)



#endif /* __ADI_AD9082_BF_NB_DDC_DFORMAT_H__ */
/*! @} */