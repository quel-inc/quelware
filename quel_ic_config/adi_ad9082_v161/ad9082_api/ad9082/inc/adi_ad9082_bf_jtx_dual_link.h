/*!
 * @brief     SPI Register Definition Header File, automatically generated file at 1/20/2020 6:24:25 AM.
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
#ifndef __ADI_AD9082_BF_JTX_DUAL_LINK_H__
#define __ADI_AD9082_BF_JTX_DUAL_LINK_H__

/*============= I N C L U D E S ============*/
#include "adi_ad9082_config.h"

/*============= D E F I N E S ==============*/
#define REG_JTX_CORE_0_CONV15_ADDR                        0x0000060F
#define BF_JTX_CONV_SEL_0_INFO                            0x00000700
#define BF_JTX_CONV_SEL_0(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_0_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_0_INFO                           0x00000107
#define BF_JTX_CONV_MASK_0(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_0_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV14_ADDR                        0x0000060E
#define BF_JTX_CONV_SEL_1_INFO                            0x00000700
#define BF_JTX_CONV_SEL_1(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_1_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_1_INFO                           0x00000107
#define BF_JTX_CONV_MASK_1(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_1_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV13_ADDR                        0x0000060D
#define BF_JTX_CONV_SEL_2_INFO                            0x00000700
#define BF_JTX_CONV_SEL_2(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_2_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_2_INFO                           0x00000107
#define BF_JTX_CONV_MASK_2(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_2_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV12_ADDR                        0x0000060C
#define BF_JTX_CONV_SEL_3_INFO                            0x00000700
#define BF_JTX_CONV_SEL_3(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_3_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_3_INFO                           0x00000107
#define BF_JTX_CONV_MASK_3(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_3_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV11_ADDR                        0x0000060B
#define BF_JTX_CONV_SEL_4_INFO                            0x00000700
#define BF_JTX_CONV_SEL_4(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_4_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_4_INFO                           0x00000107
#define BF_JTX_CONV_MASK_4(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_4_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV10_ADDR                        0x0000060A
#define BF_JTX_CONV_SEL_5_INFO                            0x00000700
#define BF_JTX_CONV_SEL_5(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_5_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_5_INFO                           0x00000107
#define BF_JTX_CONV_MASK_5(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_5_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV9_ADDR                         0x00000609
#define BF_JTX_CONV_SEL_6_INFO                            0x00000700
#define BF_JTX_CONV_SEL_6(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_6_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_6_INFO                           0x00000107
#define BF_JTX_CONV_MASK_6(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_6_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV8_ADDR                         0x00000608
#define BF_JTX_CONV_SEL_7_INFO                            0x00000700
#define BF_JTX_CONV_SEL_7(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_7_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_7_INFO                           0x00000107
#define BF_JTX_CONV_MASK_7(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_7_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV7_ADDR                         0x00000607
#define BF_JTX_CONV_SEL_8_INFO                            0x00000700
#define BF_JTX_CONV_SEL_8(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_8_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_8_INFO                           0x00000107
#define BF_JTX_CONV_MASK_8(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_8_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV6_ADDR                         0x00000606
#define BF_JTX_CONV_SEL_9_INFO                            0x00000700
#define BF_JTX_CONV_SEL_9(val)                            (val & 0x0000007F)
#define BF_JTX_CONV_SEL_9_GET(val)                        (val & 0x0000007F)
#define BF_JTX_CONV_MASK_9_INFO                           0x00000107
#define BF_JTX_CONV_MASK_9(val)                           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_9_GET(val)                       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV5_ADDR                         0x00000605
#define BF_JTX_CONV_SEL_10_INFO                           0x00000700
#define BF_JTX_CONV_SEL_10(val)                           (val & 0x0000007F)
#define BF_JTX_CONV_SEL_10_GET(val)                       (val & 0x0000007F)
#define BF_JTX_CONV_MASK_10_INFO                          0x00000107
#define BF_JTX_CONV_MASK_10(val)                          ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_10_GET(val)                      ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV4_ADDR                         0x00000604
#define BF_JTX_CONV_SEL_11_INFO                           0x00000700
#define BF_JTX_CONV_SEL_11(val)                           (val & 0x0000007F)
#define BF_JTX_CONV_SEL_11_GET(val)                       (val & 0x0000007F)
#define BF_JTX_CONV_MASK_11_INFO                          0x00000107
#define BF_JTX_CONV_MASK_11(val)                          ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_11_GET(val)                      ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV3_ADDR                         0x00000603
#define BF_JTX_CONV_SEL_12_INFO                           0x00000700
#define BF_JTX_CONV_SEL_12(val)                           (val & 0x0000007F)
#define BF_JTX_CONV_SEL_12_GET(val)                       (val & 0x0000007F)
#define BF_JTX_CONV_MASK_12_INFO                          0x00000107
#define BF_JTX_CONV_MASK_12(val)                          ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_12_GET(val)                      ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV2_ADDR                         0x00000602
#define BF_JTX_CONV_SEL_13_INFO                           0x00000700
#define BF_JTX_CONV_SEL_13(val)                           (val & 0x0000007F)
#define BF_JTX_CONV_SEL_13_GET(val)                       (val & 0x0000007F)
#define BF_JTX_CONV_MASK_13_INFO                          0x00000107
#define BF_JTX_CONV_MASK_13(val)                          ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_13_GET(val)                      ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV1_ADDR                         0x00000601
#define BF_JTX_CONV_SEL_14_INFO                           0x00000700
#define BF_JTX_CONV_SEL_14(val)                           (val & 0x0000007F)
#define BF_JTX_CONV_SEL_14_GET(val)                       (val & 0x0000007F)
#define BF_JTX_CONV_MASK_14_INFO                          0x00000107
#define BF_JTX_CONV_MASK_14(val)                          ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_14_GET(val)                      ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_0_CONV0_ADDR                         0x00000600
#define BF_JTX_CONV_SEL_15_INFO                           0x00000700
#define BF_JTX_CONV_SEL_15(val)                           (val & 0x0000007F)
#define BF_JTX_CONV_SEL_15_GET(val)                       (val & 0x0000007F)
#define BF_JTX_CONV_MASK_15_INFO                          0x00000107
#define BF_JTX_CONV_MASK_15(val)                          ((val & 0x00000001) << 0x00000007)
#define BF_JTX_CONV_MASK_15_GET(val)                      ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_1_ADDR                               0x00000611
#define BF_JTX_CHKSUM_LSB_ALG_INFO                        0x00000100
#define BF_JTX_CHKSUM_LSB_ALG(val)                        (val & 0x00000001)
#define BF_JTX_CHKSUM_LSB_ALG_GET(val)                    (val & 0x00000001)
#define BF_JTX_CHKSUM_DISABLE_INFO                        0x00000101
#define BF_JTX_CHKSUM_DISABLE(val)                        ((val & 0x00000001) << 0x00000001)
#define BF_JTX_CHKSUM_DISABLE_GET(val)                    ((val >> 0x00000001) & 0x00000001)
#define BF_JTX_LINK_204C_SEL_INFO                         0x00000204
#define BF_JTX_LINK_204C_SEL(val)                         ((val & 0x00000003) << 0x00000004)
#define BF_JTX_LINK_204C_SEL_GET(val)                     ((val >> 0x00000004) & 0x00000003)
#define BF_JTX_SYSREF_FOR_STARTUP_INFO                    0x00000106
#define BF_JTX_SYSREF_FOR_STARTUP(val)                    ((val & 0x00000001) << 0x00000006)
#define BF_JTX_SYSREF_FOR_STARTUP_GET(val)                ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_SYSREF_FOR_RELINK_INFO                     0x00000107
#define BF_JTX_SYSREF_FOR_RELINK(val)                     ((val & 0x00000001) << 0x00000007)
#define BF_JTX_SYSREF_FOR_RELINK_GET(val)                 ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_2_LANE7_ADDR                         0x00000622
#define BF_JTX_LANE_ASSIGN_0_INFO                         0x00000500
#define BF_JTX_LANE_ASSIGN_0(val)                         (val & 0x0000001F)
#define BF_JTX_LANE_ASSIGN_0_GET(val)                     (val & 0x0000001F)
#define BF_JTX_LANE_INV_0_INFO                            0x00000105
#define BF_JTX_LANE_INV_0(val)                            ((val & 0x00000001) << 0x00000005)
#define BF_JTX_LANE_INV_0_GET(val)                        ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_FORCE_LANE_PD_0_INFO                       0x00000106
#define BF_JTX_FORCE_LANE_PD_0(val)                       ((val & 0x00000001) << 0x00000006)
#define BF_JTX_FORCE_LANE_PD_0_GET(val)                   ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_LANE_PD_0_INFO                             0x00000107
#define BF_JTX_LANE_PD_0(val)                             ((val & 0x00000001) << 0x00000007)
#define BF_JTX_LANE_PD_0_GET(val)                         ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_2_LANE6_ADDR                         0x00000621
#define BF_JTX_LANE_ASSIGN_1_INFO                         0x00000500
#define BF_JTX_LANE_ASSIGN_1(val)                         (val & 0x0000001F)
#define BF_JTX_LANE_ASSIGN_1_GET(val)                     (val & 0x0000001F)
#define BF_JTX_LANE_INV_1_INFO                            0x00000105
#define BF_JTX_LANE_INV_1(val)                            ((val & 0x00000001) << 0x00000005)
#define BF_JTX_LANE_INV_1_GET(val)                        ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_FORCE_LANE_PD_1_INFO                       0x00000106
#define BF_JTX_FORCE_LANE_PD_1(val)                       ((val & 0x00000001) << 0x00000006)
#define BF_JTX_FORCE_LANE_PD_1_GET(val)                   ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_LANE_PD_1_INFO                             0x00000107
#define BF_JTX_LANE_PD_1(val)                             ((val & 0x00000001) << 0x00000007)
#define BF_JTX_LANE_PD_1_GET(val)                         ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_2_LANE5_ADDR                         0x00000620
#define BF_JTX_LANE_ASSIGN_2_INFO                         0x00000500
#define BF_JTX_LANE_ASSIGN_2(val)                         (val & 0x0000001F)
#define BF_JTX_LANE_ASSIGN_2_GET(val)                     (val & 0x0000001F)
#define BF_JTX_LANE_INV_2_INFO                            0x00000105
#define BF_JTX_LANE_INV_2(val)                            ((val & 0x00000001) << 0x00000005)
#define BF_JTX_LANE_INV_2_GET(val)                        ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_FORCE_LANE_PD_2_INFO                       0x00000106
#define BF_JTX_FORCE_LANE_PD_2(val)                       ((val & 0x00000001) << 0x00000006)
#define BF_JTX_FORCE_LANE_PD_2_GET(val)                   ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_LANE_PD_2_INFO                             0x00000107
#define BF_JTX_LANE_PD_2(val)                             ((val & 0x00000001) << 0x00000007)
#define BF_JTX_LANE_PD_2_GET(val)                         ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_2_LANE4_ADDR                         0x0000061F
#define BF_JTX_LANE_ASSIGN_3_INFO                         0x00000500
#define BF_JTX_LANE_ASSIGN_3(val)                         (val & 0x0000001F)
#define BF_JTX_LANE_ASSIGN_3_GET(val)                     (val & 0x0000001F)
#define BF_JTX_LANE_INV_3_INFO                            0x00000105
#define BF_JTX_LANE_INV_3(val)                            ((val & 0x00000001) << 0x00000005)
#define BF_JTX_LANE_INV_3_GET(val)                        ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_FORCE_LANE_PD_3_INFO                       0x00000106
#define BF_JTX_FORCE_LANE_PD_3(val)                       ((val & 0x00000001) << 0x00000006)
#define BF_JTX_FORCE_LANE_PD_3_GET(val)                   ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_LANE_PD_3_INFO                             0x00000107
#define BF_JTX_LANE_PD_3(val)                             ((val & 0x00000001) << 0x00000007)
#define BF_JTX_LANE_PD_3_GET(val)                         ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_2_LANE3_ADDR                         0x0000061E
#define BF_JTX_LANE_ASSIGN_4_INFO                         0x00000500
#define BF_JTX_LANE_ASSIGN_4(val)                         (val & 0x0000001F)
#define BF_JTX_LANE_ASSIGN_4_GET(val)                     (val & 0x0000001F)
#define BF_JTX_LANE_INV_4_INFO                            0x00000105
#define BF_JTX_LANE_INV_4(val)                            ((val & 0x00000001) << 0x00000005)
#define BF_JTX_LANE_INV_4_GET(val)                        ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_FORCE_LANE_PD_4_INFO                       0x00000106
#define BF_JTX_FORCE_LANE_PD_4(val)                       ((val & 0x00000001) << 0x00000006)
#define BF_JTX_FORCE_LANE_PD_4_GET(val)                   ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_LANE_PD_4_INFO                             0x00000107
#define BF_JTX_LANE_PD_4(val)                             ((val & 0x00000001) << 0x00000007)
#define BF_JTX_LANE_PD_4_GET(val)                         ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_2_LANE2_ADDR                         0x0000061D
#define BF_JTX_LANE_ASSIGN_5_INFO                         0x00000500
#define BF_JTX_LANE_ASSIGN_5(val)                         (val & 0x0000001F)
#define BF_JTX_LANE_ASSIGN_5_GET(val)                     (val & 0x0000001F)
#define BF_JTX_LANE_INV_5_INFO                            0x00000105
#define BF_JTX_LANE_INV_5(val)                            ((val & 0x00000001) << 0x00000005)
#define BF_JTX_LANE_INV_5_GET(val)                        ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_FORCE_LANE_PD_5_INFO                       0x00000106
#define BF_JTX_FORCE_LANE_PD_5(val)                       ((val & 0x00000001) << 0x00000006)
#define BF_JTX_FORCE_LANE_PD_5_GET(val)                   ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_LANE_PD_5_INFO                             0x00000107
#define BF_JTX_LANE_PD_5(val)                             ((val & 0x00000001) << 0x00000007)
#define BF_JTX_LANE_PD_5_GET(val)                         ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_2_LANE1_ADDR                         0x0000061C
#define BF_JTX_LANE_ASSIGN_6_INFO                         0x00000500
#define BF_JTX_LANE_ASSIGN_6(val)                         (val & 0x0000001F)
#define BF_JTX_LANE_ASSIGN_6_GET(val)                     (val & 0x0000001F)
#define BF_JTX_LANE_INV_6_INFO                            0x00000105
#define BF_JTX_LANE_INV_6(val)                            ((val & 0x00000001) << 0x00000005)
#define BF_JTX_LANE_INV_6_GET(val)                        ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_FORCE_LANE_PD_6_INFO                       0x00000106
#define BF_JTX_FORCE_LANE_PD_6(val)                       ((val & 0x00000001) << 0x00000006)
#define BF_JTX_FORCE_LANE_PD_6_GET(val)                   ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_LANE_PD_6_INFO                             0x00000107
#define BF_JTX_LANE_PD_6(val)                             ((val & 0x00000001) << 0x00000007)
#define BF_JTX_LANE_PD_6_GET(val)                         ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_2_LANE0_ADDR                         0x0000061B
#define BF_JTX_LANE_ASSIGN_7_INFO                         0x00000500
#define BF_JTX_LANE_ASSIGN_7(val)                         (val & 0x0000001F)
#define BF_JTX_LANE_ASSIGN_7_GET(val)                     (val & 0x0000001F)
#define BF_JTX_LANE_INV_7_INFO                            0x00000105
#define BF_JTX_LANE_INV_7(val)                            ((val & 0x00000001) << 0x00000005)
#define BF_JTX_LANE_INV_7_GET(val)                        ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_FORCE_LANE_PD_7_INFO                       0x00000106
#define BF_JTX_FORCE_LANE_PD_7(val)                       ((val & 0x00000001) << 0x00000006)
#define BF_JTX_FORCE_LANE_PD_7_GET(val)                   ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_LANE_PD_7_INFO                             0x00000107
#define BF_JTX_LANE_PD_7(val)                             ((val & 0x00000001) << 0x00000007)
#define BF_JTX_LANE_PD_7_GET(val)                         ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_3_ADDR                               0x00000624
#define BF_JTX_TEST_GEN_MODE_INFO                         0x00000400
#define BF_JTX_TEST_GEN_MODE(val)                         (val & 0x0000000F)
#define BF_JTX_TEST_GEN_MODE_GET(val)                     (val & 0x0000000F)
#define BF_JTX_TEST_GEN_SEL_INFO                          0x00000204
#define BF_JTX_TEST_GEN_SEL(val)                          ((val & 0x00000003) << 0x00000004)
#define BF_JTX_TEST_GEN_SEL_GET(val)                      ((val >> 0x00000004) & 0x00000003)
#define BF_JTX_TEST_MIRROR_INFO                           0x00000106
#define BF_JTX_TEST_MIRROR(val)                           ((val & 0x00000001) << 0x00000006)
#define BF_JTX_TEST_MIRROR_GET(val)                       ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_TEST_USER_GO_INFO                          0x00000107
#define BF_JTX_TEST_USER_GO(val)                          ((val & 0x00000001) << 0x00000007)
#define BF_JTX_TEST_USER_GO_GET(val)                      ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_CORE_4_ADDR                               0x00000625

#define REG_JTX_CORE_5_ADDR                               0x00000626

#define REG_JTX_CORE_6_ADDR                               0x00000627

#define REG_JTX_CORE_7_ADDR                               0x00000628

#define REG_JTX_CORE_8_ADDR                               0x00000629

#define REG_JTX_CORE_9_ADDR                               0x0000062A

#define REG_JTX_CORE_10_ADDR                              0x0000062B

#define REG_JTX_CORE_11_ADDR                              0x0000062C

#define REG_JTX_CORE_12_ADDR                              0x0000062D
#define BF_JTX_SYNC_N_SEL_INFO                            0x00000305
#define BF_JTX_SYNC_N_SEL(val)                            ((val & 0x00000007) << 0x00000005)
#define BF_JTX_SYNC_N_SEL_GET(val)                        ((val >> 0x00000005) & 0x00000007)

#define REG_JTX_CORE_13_ADDR                              0x0000062E
#define BF_JTX_LINK_EN_INFO                               0x00000100
#define BF_JTX_LINK_EN(val)                               (val & 0x00000001)
#define BF_JTX_LINK_EN_GET(val)                           (val & 0x00000001)

#define REG_JTX_TPL_0_ADDR                                0x00000630
#define BF_JTX_TPL_ADAPTIVE_LATENCY_INFO                  0x00000100
#define BF_JTX_TPL_ADAPTIVE_LATENCY(val)                  (val & 0x00000001)
#define BF_JTX_TPL_ADAPTIVE_LATENCY_GET(val)              (val & 0x00000001)
#define BF_JTX_TPL_TEST_ENABLE_INFO                       0x00000101
#define BF_JTX_TPL_TEST_ENABLE(val)                       ((val & 0x00000001) << 0x00000001)
#define BF_JTX_TPL_TEST_ENABLE_GET(val)                   ((val >> 0x00000001) & 0x00000001)
#define BF_JTX_CONV_ASYNCHRONOUS_INFO                     0x00000102
#define BF_JTX_CONV_ASYNCHRONOUS(val)                     ((val & 0x00000001) << 0x00000002)
#define BF_JTX_CONV_ASYNCHRONOUS_GET(val)                 ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_NS_CFG_INFO                                0x00000503
#define BF_JTX_NS_CFG(val)                                ((val & 0x0000001F) << 0x00000003)
#define BF_JTX_NS_CFG_GET(val)                            ((val >> 0x00000003) & 0x0000001F)

#define REG_JTX_TPL_1_ADDR                                0x00000631
#define BF_JTX_TPL_LATENCY_ADJUST_INFO                    0x00000800
#define BF_JTX_TPL_LATENCY_ADJUST(val)                    (val & 0x000000FF)
#define BF_JTX_TPL_LATENCY_ADJUST_GET(val)                (val & 0x000000FF)

#define REG_JTX_TPL_2_ADDR                                0x00000632
#define BF_JTX_TPL_PHASE_ADJUST_INFO                      0x00001000
#define BF_JTX_TPL_PHASE_ADJUST(val)                      (val & 0x0000FFFF)
#define BF_JTX_TPL_PHASE_ADJUST_GET(val)                  (val & 0x0000FFFF)

#define REG_JTX_TPL_3_ADDR                                0x00000633

#define REG_JTX_TPL_4_ADDR                                0x00000634
#define BF_JTX_TPL_TEST_NUM_FRAMES_M1_INFO                0x00001000
#define BF_JTX_TPL_TEST_NUM_FRAMES_M1(val)                (val & 0x0000FFFF)
#define BF_JTX_TPL_TEST_NUM_FRAMES_M1_GET(val)            (val & 0x0000FFFF)

#define REG_JTX_TPL_5_ADDR                                0x00000635

#define REG_JTX_TPL_6_ADDR                                0x00000636
#define BF_JTX_TPL_INVALID_CFG_INFO                       0x00000100
#define BF_JTX_TPL_INVALID_CFG(val)                       (val & 0x00000001)
#define BF_JTX_TPL_INVALID_CFG_GET(val)                   (val & 0x00000001)
#define BF_JTX_TPL_SYSREF_RCVD_INFO                       0x00000101
#define BF_JTX_TPL_SYSREF_RCVD(val)                       ((val & 0x00000001) << 0x00000001)
#define BF_JTX_TPL_SYSREF_RCVD_GET(val)                   ((val >> 0x00000001) & 0x00000001)
#define BF_JTX_TPL_SYSREF_PHASE_ERR_INFO                  0x00000102
#define BF_JTX_TPL_SYSREF_PHASE_ERR(val)                  ((val & 0x00000001) << 0x00000002)
#define BF_JTX_TPL_SYSREF_PHASE_ERR_GET(val)              ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_TPL_SYSREF_MASK_INFO                       0x00000105
#define BF_JTX_TPL_SYSREF_MASK(val)                       ((val & 0x00000001) << 0x00000005)
#define BF_JTX_TPL_SYSREF_MASK_GET(val)                   ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_TPL_SYSREF_CLR_PHASE_ERR_INFO              0x00000106
#define BF_JTX_TPL_SYSREF_CLR_PHASE_ERR(val)              ((val & 0x00000001) << 0x00000006)
#define BF_JTX_TPL_SYSREF_CLR_PHASE_ERR_GET(val)          ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_TPL_SYSREF_IGNORE_WHEN_LINKED_INFO         0x00000107
#define BF_JTX_TPL_SYSREF_IGNORE_WHEN_LINKED(val)         ((val & 0x00000001) << 0x00000007)
#define BF_JTX_TPL_SYSREF_IGNORE_WHEN_LINKED_GET(val)     ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_TPL_7_ADDR                                0x00000637
#define BF_JTX_TPL_SYSREF_N_SHOT_COUNT_INFO               0x00000400
#define BF_JTX_TPL_SYSREF_N_SHOT_COUNT(val)               (val & 0x0000000F)
#define BF_JTX_TPL_SYSREF_N_SHOT_COUNT_GET(val)           (val & 0x0000000F)
#define BF_JTX_TPL_SYSREF_N_SHOT_ENABLE_INFO              0x00000104
#define BF_JTX_TPL_SYSREF_N_SHOT_ENABLE(val)              ((val & 0x00000001) << 0x00000004)
#define BF_JTX_TPL_SYSREF_N_SHOT_ENABLE_GET(val)          ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_TPL_8_ADDR                                0x00000638
#define BF_JTX_TPL_LATENCY_ADDED_INFO                     0x00000800
#define BF_JTX_TPL_LATENCY_ADDED(val)                     (val & 0x000000FF)
#define BF_JTX_TPL_LATENCY_ADDED_GET(val)                 (val & 0x000000FF)

#define REG_JTX_TPL_9_ADDR                                0x00000639
#define BF_JTX_TPL_BUF_FRAMES_INFO                        0x00000800
#define BF_JTX_TPL_BUF_FRAMES(val)                        (val & 0x000000FF)
#define BF_JTX_TPL_BUF_FRAMES_GET(val)                    (val & 0x000000FF)

#define REG_JTX_L0_0_ADDR                                 0x0000063A
#define BF_JTX_DID_CFG_INFO                               0x00000800
#define BF_JTX_DID_CFG(val)                               (val & 0x000000FF)
#define BF_JTX_DID_CFG_GET(val)                           (val & 0x000000FF)

#define REG_JTX_L0_1_ADDR                                 0x0000063B
#define BF_JTX_BID_CFG_INFO                               0x00000400
#define BF_JTX_BID_CFG(val)                               (val & 0x0000000F)
#define BF_JTX_BID_CFG_GET(val)                           (val & 0x0000000F)
#define BF_JTX_ADJCNT_CFG_INFO                            0x00000404
#define BF_JTX_ADJCNT_CFG(val)                            ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_ADJCNT_CFG_GET(val)                        ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_L0_2_ADDR                                 0x0000063C
#define BF_JTX_PHADJ_CFG_INFO                             0x00000105
#define BF_JTX_PHADJ_CFG(val)                             ((val & 0x00000001) << 0x00000005)
#define BF_JTX_PHADJ_CFG_GET(val)                         ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_ADJDIR_CFG_INFO                            0x00000106
#define BF_JTX_ADJDIR_CFG(val)                            ((val & 0x00000001) << 0x00000006)
#define BF_JTX_ADJDIR_CFG_GET(val)                        ((val >> 0x00000006) & 0x00000001)

#define REG_JTX_L0_3_ADDR                                 0x0000063D
#define BF_JTX_L_CFG_INFO                                 0x00000500
#define BF_JTX_L_CFG(val)                                 (val & 0x0000001F)
#define BF_JTX_L_CFG_GET(val)                             (val & 0x0000001F)
#define BF_JTX_SCR_CFG_INFO                               0x00000107
#define BF_JTX_SCR_CFG(val)                               ((val & 0x00000001) << 0x00000007)
#define BF_JTX_SCR_CFG_GET(val)                           ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_L0_4_ADDR                                 0x0000063E
#define BF_JTX_F_CFG_INFO                                 0x00000800
#define BF_JTX_F_CFG(val)                                 (val & 0x000000FF)
#define BF_JTX_F_CFG_GET(val)                             (val & 0x000000FF)

#define REG_JTX_L0_5_ADDR                                 0x0000063F
#define BF_JTX_K_CFG_INFO                                 0x00000800
#define BF_JTX_K_CFG(val)                                 (val & 0x000000FF)
#define BF_JTX_K_CFG_GET(val)                             (val & 0x000000FF)

#define REG_JTX_L0_6_ADDR                                 0x00000640
#define BF_JTX_M_CFG_INFO                                 0x00000800
#define BF_JTX_M_CFG(val)                                 (val & 0x000000FF)
#define BF_JTX_M_CFG_GET(val)                             (val & 0x000000FF)

#define REG_JTX_L0_7_ADDR                                 0x00000641
#define BF_JTX_N_CFG_INFO                                 0x00000500
#define BF_JTX_N_CFG(val)                                 (val & 0x0000001F)
#define BF_JTX_N_CFG_GET(val)                             (val & 0x0000001F)
#define BF_JTX_CS_CFG_INFO                                0x00000206
#define BF_JTX_CS_CFG(val)                                ((val & 0x00000003) << 0x00000006)
#define BF_JTX_CS_CFG_GET(val)                            ((val >> 0x00000006) & 0x00000003)

#define REG_JTX_L0_8_ADDR                                 0x00000642
#define BF_JTX_NP_CFG_INFO                                0x00000500
#define BF_JTX_NP_CFG(val)                                (val & 0x0000001F)
#define BF_JTX_NP_CFG_GET(val)                            (val & 0x0000001F)
#define BF_JTX_SUBCLASSV_CFG_INFO                         0x00000305
#define BF_JTX_SUBCLASSV_CFG(val)                         ((val & 0x00000007) << 0x00000005)
#define BF_JTX_SUBCLASSV_CFG_GET(val)                     ((val >> 0x00000005) & 0x00000007)

#define REG_JTX_L0_9_ADDR                                 0x00000643
#define BF_JTX_S_CFG_INFO                                 0x00000500
#define BF_JTX_S_CFG(val)                                 (val & 0x0000001F)
#define BF_JTX_S_CFG_GET(val)                             (val & 0x0000001F)
#define BF_JTX_JESDV_CFG_INFO                             0x00000305
#define BF_JTX_JESDV_CFG(val)                             ((val & 0x00000007) << 0x00000005)
#define BF_JTX_JESDV_CFG_GET(val)                         ((val >> 0x00000005) & 0x00000007)

#define REG_JTX_L0_10_ADDR                                0x00000644
#define BF_JTX_HD_CFG_INFO                                0x00000107
#define BF_JTX_HD_CFG(val)                                ((val & 0x00000001) << 0x00000007)
#define BF_JTX_HD_CFG_GET(val)                            ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_L0_13_LANE7_ADDR                          0x0000064E
#define BF_JTX_CHKSUM_CFG_0_INFO                          0x00000800
#define BF_JTX_CHKSUM_CFG_0(val)                          (val & 0x000000FF)
#define BF_JTX_CHKSUM_CFG_0_GET(val)                      (val & 0x000000FF)

#define REG_JTX_L0_13_LANE6_ADDR                          0x0000064D
#define BF_JTX_CHKSUM_CFG_1_INFO                          0x00000800
#define BF_JTX_CHKSUM_CFG_1(val)                          (val & 0x000000FF)
#define BF_JTX_CHKSUM_CFG_1_GET(val)                      (val & 0x000000FF)

#define REG_JTX_L0_13_LANE5_ADDR                          0x0000064C
#define BF_JTX_CHKSUM_CFG_2_INFO                          0x00000800
#define BF_JTX_CHKSUM_CFG_2(val)                          (val & 0x000000FF)
#define BF_JTX_CHKSUM_CFG_2_GET(val)                      (val & 0x000000FF)

#define REG_JTX_L0_13_LANE4_ADDR                          0x0000064B
#define BF_JTX_CHKSUM_CFG_3_INFO                          0x00000800
#define BF_JTX_CHKSUM_CFG_3(val)                          (val & 0x000000FF)
#define BF_JTX_CHKSUM_CFG_3_GET(val)                      (val & 0x000000FF)

#define REG_JTX_L0_13_LANE3_ADDR                          0x0000064A
#define BF_JTX_CHKSUM_CFG_4_INFO                          0x00000800
#define BF_JTX_CHKSUM_CFG_4(val)                          (val & 0x000000FF)
#define BF_JTX_CHKSUM_CFG_4_GET(val)                      (val & 0x000000FF)

#define REG_JTX_L0_13_LANE2_ADDR                          0x00000649
#define BF_JTX_CHKSUM_CFG_5_INFO                          0x00000800
#define BF_JTX_CHKSUM_CFG_5(val)                          (val & 0x000000FF)
#define BF_JTX_CHKSUM_CFG_5_GET(val)                      (val & 0x000000FF)

#define REG_JTX_L0_13_LANE1_ADDR                          0x00000648
#define BF_JTX_CHKSUM_CFG_6_INFO                          0x00000800
#define BF_JTX_CHKSUM_CFG_6(val)                          (val & 0x000000FF)
#define BF_JTX_CHKSUM_CFG_6_GET(val)                      (val & 0x000000FF)

#define REG_JTX_L0_13_LANE0_ADDR                          0x00000647
#define BF_JTX_CHKSUM_CFG_7_INFO                          0x00000800
#define BF_JTX_CHKSUM_CFG_7(val)                          (val & 0x000000FF)
#define BF_JTX_CHKSUM_CFG_7_GET(val)                      (val & 0x000000FF)

#define REG_JTX_L0_14_LANE7_ADDR                          0x00000657
#define BF_JTX_LID_CFG_0_INFO                             0x00000500
#define BF_JTX_LID_CFG_0(val)                             (val & 0x0000001F)
#define BF_JTX_LID_CFG_0_GET(val)                         (val & 0x0000001F)

#define REG_JTX_L0_14_LANE6_ADDR                          0x00000656
#define BF_JTX_LID_CFG_1_INFO                             0x00000500
#define BF_JTX_LID_CFG_1(val)                             (val & 0x0000001F)
#define BF_JTX_LID_CFG_1_GET(val)                         (val & 0x0000001F)

#define REG_JTX_L0_14_LANE5_ADDR                          0x00000655
#define BF_JTX_LID_CFG_2_INFO                             0x00000500
#define BF_JTX_LID_CFG_2(val)                             (val & 0x0000001F)
#define BF_JTX_LID_CFG_2_GET(val)                         (val & 0x0000001F)

#define REG_JTX_L0_14_LANE4_ADDR                          0x00000654
#define BF_JTX_LID_CFG_3_INFO                             0x00000500
#define BF_JTX_LID_CFG_3(val)                             (val & 0x0000001F)
#define BF_JTX_LID_CFG_3_GET(val)                         (val & 0x0000001F)

#define REG_JTX_L0_14_LANE3_ADDR                          0x00000653
#define BF_JTX_LID_CFG_4_INFO                             0x00000500
#define BF_JTX_LID_CFG_4(val)                             (val & 0x0000001F)
#define BF_JTX_LID_CFG_4_GET(val)                         (val & 0x0000001F)

#define REG_JTX_L0_14_LANE2_ADDR                          0x00000652
#define BF_JTX_LID_CFG_5_INFO                             0x00000500
#define BF_JTX_LID_CFG_5(val)                             (val & 0x0000001F)
#define BF_JTX_LID_CFG_5_GET(val)                         (val & 0x0000001F)

#define REG_JTX_L0_14_LANE1_ADDR                          0x00000651
#define BF_JTX_LID_CFG_6_INFO                             0x00000500
#define BF_JTX_LID_CFG_6(val)                             (val & 0x0000001F)
#define BF_JTX_LID_CFG_6_GET(val)                         (val & 0x0000001F)

#define REG_JTX_L0_14_LANE0_ADDR                          0x00000650
#define BF_JTX_LID_CFG_7_INFO                             0x00000500
#define BF_JTX_LID_CFG_7(val)                             (val & 0x0000001F)
#define BF_JTX_LID_CFG_7_GET(val)                         (val & 0x0000001F)

#define REG_JTX_DL_204B_0_ADDR                            0x00000659
#define BF_JTX_DL_204B_BYP_ACG_CFG_INFO                   0x00000100
#define BF_JTX_DL_204B_BYP_ACG_CFG(val)                   (val & 0x00000001)
#define BF_JTX_DL_204B_BYP_ACG_CFG_GET(val)               (val & 0x00000001)
#define BF_JTX_DL_204B_BYP_8B10B_CFG_INFO                 0x00000101
#define BF_JTX_DL_204B_BYP_8B10B_CFG(val)                 ((val & 0x00000001) << 0x00000001)
#define BF_JTX_DL_204B_BYP_8B10B_CFG_GET(val)             ((val >> 0x00000001) & 0x00000001)
#define BF_JTX_DL_204B_ILAS_TEST_EN_CFG_INFO              0x00000102
#define BF_JTX_DL_204B_ILAS_TEST_EN_CFG(val)              ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_ILAS_TEST_EN_CFG_GET(val)          ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_BYP_ILAS_CFG_INFO                  0x00000103
#define BF_JTX_DL_204B_BYP_ILAS_CFG(val)                  ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_BYP_ILAS_CFG_GET(val)              ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_ILAS_DELAY_CFG_INFO                0x00000404
#define BF_JTX_DL_204B_ILAS_DELAY_CFG(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_DL_204B_ILAS_DELAY_CFG_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_DL_204B_1_ADDR                            0x0000065A
#define BF_JTX_DL_204B_10B_MIRROR_INFO                    0x00000100
#define BF_JTX_DL_204B_10B_MIRROR(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_10B_MIRROR_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_DEL_SCR_CFG_INFO                   0x00000101
#define BF_JTX_DL_204B_DEL_SCR_CFG(val)                   ((val & 0x00000001) << 0x00000001)
#define BF_JTX_DL_204B_DEL_SCR_CFG_GET(val)               ((val >> 0x00000001) & 0x00000001)
#define BF_JTX_DL_204B_LSYNC_EN_CFG_INFO                  0x00000102
#define BF_JTX_DL_204B_LSYNC_EN_CFG(val)                  ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_LSYNC_EN_CFG_GET(val)              ((val >> 0x00000002) & 0x00000001)

#define REG_JTX_DL_204B_2_ADDR                            0x0000065B
#define BF_JTX_DL_204B_KF_ILAS_CFG_INFO                   0x00000800
#define BF_JTX_DL_204B_KF_ILAS_CFG(val)                   (val & 0x000000FF)
#define BF_JTX_DL_204B_KF_ILAS_CFG_GET(val)               (val & 0x000000FF)

#define REG_JTX_DL_204B_3_ADDR                            0x0000065C
#define BF_JTX_DL_204B_RJSPAT_EN_CFG_INFO                 0x00000100
#define BF_JTX_DL_204B_RJSPAT_EN_CFG(val)                 (val & 0x00000001)
#define BF_JTX_DL_204B_RJSPAT_EN_CFG_GET(val)             (val & 0x00000001)
#define BF_JTX_DL_204B_RJSPAT_SEL_CFG_INFO                0x00000201
#define BF_JTX_DL_204B_RJSPAT_SEL_CFG(val)                ((val & 0x00000003) << 0x00000001)
#define BF_JTX_DL_204B_RJSPAT_SEL_CFG_GET(val)            ((val >> 0x00000001) & 0x00000003)
#define BF_JTX_DL_204B_TPL_TEST_EN_CFG_INFO               0x00000104
#define BF_JTX_DL_204B_TPL_TEST_EN_CFG(val)               ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_TPL_TEST_EN_CFG_GET(val)           ((val >> 0x00000004) & 0x00000001)
#define BF_JTX_DL_204B_SYNC_N_INFO                        0x00000105
#define BF_JTX_DL_204B_SYNC_N(val)                        ((val & 0x00000001) << 0x00000005)
#define BF_JTX_DL_204B_SYNC_N_GET(val)                    ((val >> 0x00000005) & 0x00000001)
#define BF_JTX_DL_204B_TESTMODE_IGNORE_SYNCN_CFG_INFO     0x00000106
#define BF_JTX_DL_204B_TESTMODE_IGNORE_SYNCN_CFG(val)     ((val & 0x00000001) << 0x00000006)
#define BF_JTX_DL_204B_TESTMODE_IGNORE_SYNCN_CFG_GET(val) ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_DL_204B_CLEAR_SYNC_NE_COUNT_INFO           0x00000107
#define BF_JTX_DL_204B_CLEAR_SYNC_NE_COUNT(val)           ((val & 0x00000001) << 0x00000007)
#define BF_JTX_DL_204B_CLEAR_SYNC_NE_COUNT_GET(val)       ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_DL_204B_4_ADDR                            0x0000065D
#define BF_JTX_DL_204B_STATE_INFO                         0x00000400
#define BF_JTX_DL_204B_STATE(val)                         (val & 0x0000000F)
#define BF_JTX_DL_204B_STATE_GET(val)                     (val & 0x0000000F)
#define BF_JTX_DL_204B_SYNC_N_FORCE_VAL_INFO              0x00000106
#define BF_JTX_DL_204B_SYNC_N_FORCE_VAL(val)              ((val & 0x00000001) << 0x00000006)
#define BF_JTX_DL_204B_SYNC_N_FORCE_VAL_GET(val)          ((val >> 0x00000006) & 0x00000001)
#define BF_JTX_DL_204B_SYNC_N_FORCE_EN_INFO               0x00000107
#define BF_JTX_DL_204B_SYNC_N_FORCE_EN(val)               ((val & 0x00000001) << 0x00000007)
#define BF_JTX_DL_204B_SYNC_N_FORCE_EN_GET(val)           ((val >> 0x00000007) & 0x00000001)

#define REG_JTX_DL_204B_5_ADDR                            0x0000065E
#define BF_JTX_DL_204B_SYNC_NE_COUNT_INFO                 0x00000800
#define BF_JTX_DL_204B_SYNC_NE_COUNT(val)                 (val & 0x000000FF)
#define BF_JTX_DL_204B_SYNC_NE_COUNT_GET(val)             (val & 0x000000FF)

#define REG_JTX_DL_204B_6_LANE7_ADDR                      0x00000666
#define BF_JTX_DL_204B_L_EN_CFG_0_INFO                    0x00000100
#define BF_JTX_DL_204B_L_EN_CFG_0(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_L_EN_CFG_0_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_0_INFO            0x00000102
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_0(val)            ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_0_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_0_INFO            0x00000103
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_0(val)            ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_0_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_0_INFO             0x00000104
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_0(val)             ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_0_GET(val)         ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_DL_204B_6_LANE6_ADDR                      0x00000665
#define BF_JTX_DL_204B_L_EN_CFG_1_INFO                    0x00000100
#define BF_JTX_DL_204B_L_EN_CFG_1(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_L_EN_CFG_1_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_1_INFO            0x00000102
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_1(val)            ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_1_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_1_INFO            0x00000103
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_1(val)            ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_1_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_1_INFO             0x00000104
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_1(val)             ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_1_GET(val)         ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_DL_204B_6_LANE5_ADDR                      0x00000664
#define BF_JTX_DL_204B_L_EN_CFG_2_INFO                    0x00000100
#define BF_JTX_DL_204B_L_EN_CFG_2(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_L_EN_CFG_2_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_2_INFO            0x00000102
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_2(val)            ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_2_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_2_INFO            0x00000103
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_2(val)            ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_2_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_2_INFO             0x00000104
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_2(val)             ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_2_GET(val)         ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_DL_204B_6_LANE4_ADDR                      0x00000663
#define BF_JTX_DL_204B_L_EN_CFG_3_INFO                    0x00000100
#define BF_JTX_DL_204B_L_EN_CFG_3(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_L_EN_CFG_3_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_3_INFO            0x00000102
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_3(val)            ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_3_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_3_INFO            0x00000103
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_3(val)            ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_3_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_3_INFO             0x00000104
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_3(val)             ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_3_GET(val)         ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_DL_204B_6_LANE3_ADDR                      0x00000662
#define BF_JTX_DL_204B_L_EN_CFG_4_INFO                    0x00000100
#define BF_JTX_DL_204B_L_EN_CFG_4(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_L_EN_CFG_4_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_4_INFO            0x00000102
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_4(val)            ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_4_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_4_INFO            0x00000103
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_4(val)            ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_4_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_4_INFO             0x00000104
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_4(val)             ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_4_GET(val)         ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_DL_204B_6_LANE2_ADDR                      0x00000661
#define BF_JTX_DL_204B_L_EN_CFG_5_INFO                    0x00000100
#define BF_JTX_DL_204B_L_EN_CFG_5(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_L_EN_CFG_5_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_5_INFO            0x00000102
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_5(val)            ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_5_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_5_INFO            0x00000103
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_5(val)            ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_5_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_5_INFO             0x00000104
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_5(val)             ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_5_GET(val)         ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_DL_204B_6_LANE1_ADDR                      0x00000660
#define BF_JTX_DL_204B_L_EN_CFG_6_INFO                    0x00000100
#define BF_JTX_DL_204B_L_EN_CFG_6(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_L_EN_CFG_6_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_6_INFO            0x00000102
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_6(val)            ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_6_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_6_INFO            0x00000103
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_6(val)            ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_6_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_6_INFO             0x00000104
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_6(val)             ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_6_GET(val)         ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_DL_204B_6_LANE0_ADDR                      0x0000065F
#define BF_JTX_DL_204B_L_EN_CFG_7_INFO                    0x00000100
#define BF_JTX_DL_204B_L_EN_CFG_7(val)                    (val & 0x00000001)
#define BF_JTX_DL_204B_L_EN_CFG_7_GET(val)                (val & 0x00000001)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_7_INFO            0x00000102
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_7(val)            ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204B_PHY_DATA_SEL_CFG_7_GET(val)        ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_7_INFO            0x00000103
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_7(val)            ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204B_SCR_DATA_SEL_CFG_7_GET(val)        ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_7_INFO             0x00000104
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_7(val)             ((val & 0x00000001) << 0x00000004)
#define BF_JTX_DL_204B_SCR_IN_CTRL_CFG_7_GET(val)         ((val >> 0x00000004) & 0x00000001)

#define REG_JTX_DL_204C_0_ADDR                            0x00000667
#define BF_JTX_CRC_FEC_REVERSE_CFG_INFO                   0x00000100
#define BF_JTX_CRC_FEC_REVERSE_CFG(val)                   (val & 0x00000001)
#define BF_JTX_CRC_FEC_REVERSE_CFG_GET(val)               (val & 0x00000001)
#define BF_JTX_LINK_FEC_ENABLE_INFO                       0x00000101
#define BF_JTX_LINK_FEC_ENABLE(val)                       ((val & 0x00000001) << 0x00000001)
#define BF_JTX_LINK_FEC_ENABLE_GET(val)                   ((val >> 0x00000001) & 0x00000001)
#define BF_JTX_FORCE_METABITS_INFO                        0x00000102
#define BF_JTX_FORCE_METABITS(val)                        ((val & 0x00000001) << 0x00000002)
#define BF_JTX_FORCE_METABITS_GET(val)                    ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204C_SYSREF_RCVD_INFO                   0x00000103
#define BF_JTX_DL_204C_SYSREF_RCVD(val)                   ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204C_SYSREF_RCVD_GET(val)               ((val >> 0x00000003) & 0x00000001)

#define REG_JTX_DL_204C_1_ADDR                            0x00000668
#define BF_JTX_E_CFG_INFO                                 0x00000800
#define BF_JTX_E_CFG(val)                                 (val & 0x000000FF)
#define BF_JTX_E_CFG_GET(val)                             (val & 0x000000FF)

#define REG_JTX_DL_204C_2_ADDR                            0x00000669
#define BF_JTX_BURST_ERROR_INJECT_INFO                    0x00000100
#define BF_JTX_BURST_ERROR_INJECT(val)                    (val & 0x00000001)
#define BF_JTX_BURST_ERROR_INJECT_GET(val)                (val & 0x00000001)
#define BF_JTX_BURST_ERROR_LENGTH_INFO                    0x00000404
#define BF_JTX_BURST_ERROR_LENGTH(val)                    ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_BURST_ERROR_LENGTH_GET(val)                ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_DL_204C_3_ADDR                            0x0000066A
#define BF_JTX_BURST_ERROR_LOCATION_INFO                  0x00000B00
#define BF_JTX_BURST_ERROR_LOCATION(val)                  (val & 0x000007FF)
#define BF_JTX_BURST_ERROR_LOCATION_GET(val)              (val & 0x000007FF)

#define REG_JTX_DL_204H_0_ADDR                            0x0000066B
#define BF_JTX_DL_204H_ACG_BYP_INFO                       0x00000100
#define BF_JTX_DL_204H_ACG_BYP(val)                       (val & 0x00000001)
#define BF_JTX_DL_204H_ACG_BYP_GET(val)                   (val & 0x00000001)
#define BF_JTX_DL_204H_BYP_ILAS_CFG_INFO                  0x00000101
#define BF_JTX_DL_204H_BYP_ILAS_CFG(val)                  ((val & 0x00000001) << 0x00000001)
#define BF_JTX_DL_204H_BYP_ILAS_CFG_GET(val)              ((val >> 0x00000001) & 0x00000001)
#define BF_JTX_DL_204H_CLEAR_SYNC_NE_COUNT_INFO           0x00000102
#define BF_JTX_DL_204H_CLEAR_SYNC_NE_COUNT(val)           ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204H_CLEAR_SYNC_NE_COUNT_GET(val)       ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204H_LANE_SYNC_2SIDES_INFO              0x00000103
#define BF_JTX_DL_204H_LANE_SYNC_2SIDES(val)              ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204H_LANE_SYNC_2SIDES_GET(val)          ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204H_ILAS_DELAY_CFG_INFO                0x00000404
#define BF_JTX_DL_204H_ILAS_DELAY_CFG(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_DL_204H_ILAS_DELAY_CFG_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_DL_204H_1_ADDR                            0x0000066C
#define BF_JTX_DL_204H_INTERLEAVE_MODE_INFO               0x00000200
#define BF_JTX_DL_204H_INTERLEAVE_MODE(val)               (val & 0x00000003)
#define BF_JTX_DL_204H_INTERLEAVE_MODE_GET(val)           (val & 0x00000003)
#define BF_JTX_DL_204H_PARITY_BYPASS_INFO                 0x00000102
#define BF_JTX_DL_204H_PARITY_BYPASS(val)                 ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204H_PARITY_BYPASS_GET(val)             ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204H_PARITY_MODE_INFO                   0x00000103
#define BF_JTX_DL_204H_PARITY_MODE(val)                   ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204H_PARITY_MODE_GET(val)               ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204H_STATE_INFO                         0x00000404
#define BF_JTX_DL_204H_STATE(val)                         ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_DL_204H_STATE_GET(val)                     ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_DL_204H_2_ADDR                            0x0000066D
#define BF_JTX_DL_204H_KF_ILAS_CFG_INFO                   0x00000800
#define BF_JTX_DL_204H_KF_ILAS_CFG(val)                   (val & 0x000000FF)
#define BF_JTX_DL_204H_KF_ILAS_CFG_GET(val)               (val & 0x000000FF)

#define REG_JTX_DL_204H_3_ADDR                            0x0000066E
#define BF_JTX_DL_204H_PARITY_ODD_ENABLE_INFO             0x00000100
#define BF_JTX_DL_204H_PARITY_ODD_ENABLE(val)             (val & 0x00000001)
#define BF_JTX_DL_204H_PARITY_ODD_ENABLE_GET(val)         (val & 0x00000001)
#define BF_JTX_DL_204H_SCR_CFG_INFO                       0x00000101
#define BF_JTX_DL_204H_SCR_CFG(val)                       ((val & 0x00000001) << 0x00000001)
#define BF_JTX_DL_204H_SCR_CFG_GET(val)                   ((val >> 0x00000001) & 0x00000001)
#define BF_JTX_DL_204H_SYNC_N_FORCE_EN_INFO               0x00000102
#define BF_JTX_DL_204H_SYNC_N_FORCE_EN(val)               ((val & 0x00000001) << 0x00000002)
#define BF_JTX_DL_204H_SYNC_N_FORCE_EN_GET(val)           ((val >> 0x00000002) & 0x00000001)
#define BF_JTX_DL_204H_SYNC_N_FORCE_VAL_INFO              0x00000103
#define BF_JTX_DL_204H_SYNC_N_FORCE_VAL(val)              ((val & 0x00000001) << 0x00000003)
#define BF_JTX_DL_204H_SYNC_N_FORCE_VAL_GET(val)          ((val >> 0x00000003) & 0x00000001)
#define BF_JTX_DL_204H_TEST_MODE_INFO                     0x00000204
#define BF_JTX_DL_204H_TEST_MODE(val)                     ((val & 0x00000003) << 0x00000004)
#define BF_JTX_DL_204H_TEST_MODE_GET(val)                 ((val >> 0x00000004) & 0x00000003)

#define REG_JTX_DL_204H_4_ADDR                            0x0000066F
#define BF_JTX_DL_204H_SYNC_NE_COUNT_INFO                 0x00000800
#define BF_JTX_DL_204H_SYNC_NE_COUNT(val)                 (val & 0x000000FF)
#define BF_JTX_DL_204H_SYNC_NE_COUNT_GET(val)             (val & 0x000000FF)

#define REG_JTX_PHY_IFX_0_LANE7_ADDR                      0x00000677
#define BF_JTX_BR_LOG2_RATIO_0_INFO                       0x00000400
#define BF_JTX_BR_LOG2_RATIO_0(val)                       (val & 0x0000000F)
#define BF_JTX_BR_LOG2_RATIO_0_GET(val)                   (val & 0x0000000F)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_0_INFO                0x00000404
#define BF_JTX_LANE_FIFO_WR_ENTRIES_0(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_0_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_PHY_IFX_0_LANE6_ADDR                      0x00000676
#define BF_JTX_BR_LOG2_RATIO_1_INFO                       0x00000400
#define BF_JTX_BR_LOG2_RATIO_1(val)                       (val & 0x0000000F)
#define BF_JTX_BR_LOG2_RATIO_1_GET(val)                   (val & 0x0000000F)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_1_INFO                0x00000404
#define BF_JTX_LANE_FIFO_WR_ENTRIES_1(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_1_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_PHY_IFX_0_LANE5_ADDR                      0x00000675
#define BF_JTX_BR_LOG2_RATIO_2_INFO                       0x00000400
#define BF_JTX_BR_LOG2_RATIO_2(val)                       (val & 0x0000000F)
#define BF_JTX_BR_LOG2_RATIO_2_GET(val)                   (val & 0x0000000F)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_2_INFO                0x00000404
#define BF_JTX_LANE_FIFO_WR_ENTRIES_2(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_2_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_PHY_IFX_0_LANE4_ADDR                      0x00000674
#define BF_JTX_BR_LOG2_RATIO_3_INFO                       0x00000400
#define BF_JTX_BR_LOG2_RATIO_3(val)                       (val & 0x0000000F)
#define BF_JTX_BR_LOG2_RATIO_3_GET(val)                   (val & 0x0000000F)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_3_INFO                0x00000404
#define BF_JTX_LANE_FIFO_WR_ENTRIES_3(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_3_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_PHY_IFX_0_LANE3_ADDR                      0x00000673
#define BF_JTX_BR_LOG2_RATIO_4_INFO                       0x00000400
#define BF_JTX_BR_LOG2_RATIO_4(val)                       (val & 0x0000000F)
#define BF_JTX_BR_LOG2_RATIO_4_GET(val)                   (val & 0x0000000F)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_4_INFO                0x00000404
#define BF_JTX_LANE_FIFO_WR_ENTRIES_4(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_4_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_PHY_IFX_0_LANE2_ADDR                      0x00000672
#define BF_JTX_BR_LOG2_RATIO_5_INFO                       0x00000400
#define BF_JTX_BR_LOG2_RATIO_5(val)                       (val & 0x0000000F)
#define BF_JTX_BR_LOG2_RATIO_5_GET(val)                   (val & 0x0000000F)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_5_INFO                0x00000404
#define BF_JTX_LANE_FIFO_WR_ENTRIES_5(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_5_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_PHY_IFX_0_LANE1_ADDR                      0x00000671
#define BF_JTX_BR_LOG2_RATIO_6_INFO                       0x00000400
#define BF_JTX_BR_LOG2_RATIO_6(val)                       (val & 0x0000000F)
#define BF_JTX_BR_LOG2_RATIO_6_GET(val)                   (val & 0x0000000F)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_6_INFO                0x00000404
#define BF_JTX_LANE_FIFO_WR_ENTRIES_6(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_6_GET(val)            ((val >> 0x00000004) & 0x0000000F)

#define REG_JTX_PHY_IFX_0_LANE0_ADDR                      0x00000670
#define BF_JTX_BR_LOG2_RATIO_7_INFO                       0x00000400
#define BF_JTX_BR_LOG2_RATIO_7(val)                       (val & 0x0000000F)
#define BF_JTX_BR_LOG2_RATIO_7_GET(val)                   (val & 0x0000000F)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_7_INFO                0x00000404
#define BF_JTX_LANE_FIFO_WR_ENTRIES_7(val)                ((val & 0x0000000F) << 0x00000004)
#define BF_JTX_LANE_FIFO_WR_ENTRIES_7_GET(val)            ((val >> 0x00000004) & 0x0000000F)



#endif /* __ADI_AD9082_BF_JTX_DUAL_LINK_H__ */
/*! @} */