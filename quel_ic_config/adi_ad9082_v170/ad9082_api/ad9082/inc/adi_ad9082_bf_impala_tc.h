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
#ifndef __ADI_AD9082_BF_IMPALA_TC_H__
#define __ADI_AD9082_BF_IMPALA_TC_H__

/*============= I N C L U D E S ============*/
#include "adi_ad9082_config.h"

/*============= D E F I N E S ==============*/
#define REG_POWERDOWN_REG_0_ADDR          0x000000E0
#define BF_D_PD_DIV8_INFO                 0x00000100
#define BF_D_PD_DIV8(val)                 (val & 0x00000001)
#define BF_D_PD_DIV8_GET(val)             (val & 0x00000001)
#define BF_D_PD_VCO_DIV_INFO              0x00000101
#define BF_D_PD_VCO_DIV(val)              ((val & 0x00000001) << 0x00000001)
#define BF_D_PD_VCO_DIV_GET(val)          ((val >> 0x00000001) & 0x00000001)
#define BF_D_PD_VCO_DRIVER_INFO           0x00000102
#define BF_D_PD_VCO_DRIVER(val)           ((val & 0x00000001) << 0x00000002)
#define BF_D_PD_VCO_DRIVER_GET(val)       ((val >> 0x00000002) & 0x00000001)
#define BF_D_PD_VCO_BUF_INFO              0x00000103
#define BF_D_PD_VCO_BUF(val)              ((val & 0x00000001) << 0x00000003)
#define BF_D_PD_VCO_BUF_GET(val)          ((val >> 0x00000003) & 0x00000001)
#define BF_D_PD_CURR_INFO                 0x00000304
#define BF_D_PD_CURR(val)                 ((val & 0x00000007) << 0x00000004)
#define BF_D_PD_CURR_GET(val)             ((val >> 0x00000004) & 0x00000007)
#define BF_D_PD_REG_INFO                  0x00000107
#define BF_D_PD_REG(val)                  ((val & 0x00000001) << 0x00000007)
#define BF_D_PD_REG_GET(val)              ((val >> 0x00000007) & 0x00000001)

#define REG_POWERDOWN_REG_1_ADDR          0x000000E1
#define BF_D_PD_REFCLK_DIV_INFO           0x00000100
#define BF_D_PD_REFCLK_DIV(val)           (val & 0x00000001)
#define BF_D_PD_REFCLK_DIV_GET(val)       (val & 0x00000001)
#define BF_D_PD_VCM_C_INFO                0x00000101
#define BF_D_PD_VCM_C(val)                ((val & 0x00000001) << 0x00000001)
#define BF_D_PD_VCM_C_GET(val)            ((val >> 0x00000001) & 0x00000001)
#define BF_D_PD_VCM_F_INFO                0x00000102
#define BF_D_PD_VCM_F(val)                ((val & 0x00000001) << 0x00000002)
#define BF_D_PD_VCM_F_GET(val)            ((val >> 0x00000002) & 0x00000001)
#define BF_D_PD_COARSE_BUFF_INFO          0x00000103
#define BF_D_PD_COARSE_BUFF(val)          ((val & 0x00000001) << 0x00000003)
#define BF_D_PD_COARSE_BUFF_GET(val)      ((val >> 0x00000003) & 0x00000001)
#define BF_D_PD_CP_INFO                   0x00000104
#define BF_D_PD_CP(val)                   ((val & 0x00000001) << 0x00000004)
#define BF_D_PD_CP_GET(val)               ((val >> 0x00000004) & 0x00000001)

#define REG_RESET_REG_ADDR                0x000000E2
#define BF_D_RESET_VCO_DIV_INFO           0x00000100
#define BF_D_RESET_VCO_DIV(val)           (val & 0x00000001)
#define BF_D_RESET_VCO_DIV_GET(val)       (val & 0x00000001)
#define BF_D_CAL_RESET_INFO               0x00000101
#define BF_D_CAL_RESET(val)               ((val & 0x00000001) << 0x00000001)
#define BF_D_CAL_RESET_GET(val)           ((val >> 0x00000001) & 0x00000001)
#define BF_D_PFD_RESET_INFO               0x00000102
#define BF_D_PFD_RESET(val)               ((val & 0x00000001) << 0x00000002)
#define BF_D_PFD_RESET_GET(val)           ((val >> 0x00000002) & 0x00000001)
#define BF_D_RESET_REF_DIV_INFO           0x00000103
#define BF_D_RESET_REF_DIV(val)           ((val & 0x00000001) << 0x00000003)
#define BF_D_RESET_REF_DIV_GET(val)       ((val >> 0x00000003) & 0x00000001)
#define BF_D_RESET_FEEDBACK_DIV_INFO      0x00000104
#define BF_D_RESET_FEEDBACK_DIV(val)      ((val & 0x00000001) << 0x00000004)
#define BF_D_RESET_FEEDBACK_DIV_GET(val)  ((val >> 0x00000004) & 0x00000001)

#define REG_INPUT_MISC_REG_ADDR           0x000000E3
#define BF_D_REFIN_DIV_INFO               0x00000200
#define BF_D_REFIN_DIV(val)               (val & 0x00000003)
#define BF_D_REFIN_DIV_GET(val)           (val & 0x00000003)
#define BF_D_CLK_EDGE_INFO                0x00000102
#define BF_D_CLK_EDGE(val)                ((val & 0x00000001) << 0x00000002)
#define BF_D_CLK_EDGE_GET(val)            ((val >> 0x00000002) & 0x00000001)
#define BF_D_PFD_DELAY_INFO               0x00000303
#define BF_D_PFD_DELAY(val)               ((val & 0x00000007) << 0x00000003)
#define BF_D_PFD_DELAY_GET(val)           ((val >> 0x00000003) & 0x00000007)
#define BF_D_COARSE_CONTROL_INFO          0x00000106
#define BF_D_COARSE_CONTROL(val)          ((val & 0x00000001) << 0x00000006)
#define BF_D_COARSE_CONTROL_GET(val)      ((val >> 0x00000006) & 0x00000001)

#define REG_CHARGEPUMP_REG_0_ADDR         0x000000E4
#define BF_D_CP_CURRENT_INFO              0x00000600
#define BF_D_CP_CURRENT(val)              (val & 0x0000003F)
#define BF_D_CP_CURRENT_GET(val)          (val & 0x0000003F)
#define BF_D_CP_CAL_EN_INFO               0x00000106
#define BF_D_CP_CAL_EN(val)               ((val & 0x00000001) << 0x00000006)
#define BF_D_CP_CAL_EN_GET(val)           ((val >> 0x00000006) & 0x00000001)
#define BF_D_CP_CALIBRATE_INFO            0x00000107
#define BF_D_CP_CALIBRATE(val)            ((val & 0x00000001) << 0x00000007)
#define BF_D_CP_CALIBRATE_GET(val)        ((val >> 0x00000007) & 0x00000001)

#define REG_CHARGEPUMP_REG_1_ADDR         0x000000E5
#define BF_D_CP_CALBITS_INFO              0x00000400
#define BF_D_CP_CALBITS(val)              (val & 0x0000000F)
#define BF_D_CP_CALBITS_GET(val)          (val & 0x0000000F)
#define BF_D_CP_OFFSET_DNB_INFO           0x00000104
#define BF_D_CP_OFFSET_DNB(val)           ((val & 0x00000001) << 0x00000004)
#define BF_D_CP_OFFSET_DNB_GET(val)       ((val >> 0x00000004) & 0x00000001)
#define BF_D_CP_RECONF_INFO               0x00000105
#define BF_D_CP_RECONF(val)               ((val & 0x00000001) << 0x00000005)
#define BF_D_CP_RECONF_GET(val)           ((val >> 0x00000005) & 0x00000001)
#define BF_D_CP_TEST_INFO                 0x00000206
#define BF_D_CP_TEST(val)                 ((val & 0x00000003) << 0x00000006)
#define BF_D_CP_TEST_GET(val)             ((val >> 0x00000006) & 0x00000003)

#define REG_VCM_CONTROL_REG_ADDR          0x000000E6
#define BF_D_VCM_C_CONTROL_INFO           0x00000400
#define BF_D_VCM_C_CONTROL(val)           (val & 0x0000000F)
#define BF_D_VCM_C_CONTROL_GET(val)       (val & 0x0000000F)
#define BF_D_VCM_F_CONTROL_INFO           0x00000404
#define BF_D_VCM_F_CONTROL(val)           ((val & 0x0000000F) << 0x00000004)
#define BF_D_VCM_F_CONTROL_GET(val)       ((val >> 0x00000004) & 0x0000000F)

#define REG_BIAS_REG_0_ADDR               0x000000E7
#define BF_D_BIAS_FIXED_TRIM_INFO         0x00000600
#define BF_D_BIAS_FIXED_TRIM(val)         (val & 0x0000003F)
#define BF_D_BIAS_FIXED_TRIM_GET(val)     (val & 0x0000003F)
#define BF_D_REG_SLICE_SEL_INFO           0x00000206
#define BF_D_REG_SLICE_SEL(val)           ((val & 0x00000003) << 0x00000006)
#define BF_D_REG_SLICE_SEL_GET(val)       ((val >> 0x00000006) & 0x00000003)

#define REG_BIAS_REG_1_ADDR               0x000000E8
#define BF_D_BIAS_POLY_TRIM_INFO          0x00000600
#define BF_D_BIAS_POLY_TRIM(val)          (val & 0x0000003F)
#define BF_D_BIAS_POLY_TRIM_GET(val)      (val & 0x0000003F)
#define BF_D_REG_BYPASS_FIT_INFO          0x00000106
#define BF_D_REG_BYPASS_FIT(val)          ((val & 0x00000001) << 0x00000006)
#define BF_D_REG_BYPASS_FIT_GET(val)      ((val >> 0x00000006) & 0x00000001)

#define REG_DIVIDER_REG_ADDR              0x000000E9
#define BF_D_DIVIDE_CONTROL_INFO          0x00000600
#define BF_D_DIVIDE_CONTROL(val)          (val & 0x0000003F)
#define BF_D_DIVIDE_CONTROL_GET(val)      (val & 0x0000003F)

#define REG_VCO_CAL_CONTROL_REG_0_ADDR    0x000000EA
#define BF_D_IMPALA_CAL_CONTROL_INFO      0x00001000
#define BF_D_IMPALA_CAL_CONTROL(val)      (val & 0x0000FFFF)
#define BF_D_IMPALA_CAL_CONTROL_GET(val)  (val & 0x0000FFFF)

#define REG_VCO_CAL_CONTROL_REG_1_ADDR    0x000000EB

#define REG_VCO_CAL_LOCK_REG_ADDR         0x000000EC
#define BF_D_CAL_OVERRIDE_INFO            0x00000100
#define BF_D_CAL_OVERRIDE(val)            (val & 0x00000001)
#define BF_D_CAL_OVERRIDE_GET(val)        (val & 0x00000001)
#define BF_D_PLL_LOCK_CONTROL_INFO        0x00000201
#define BF_D_PLL_LOCK_CONTROL(val)        ((val & 0x00000003) << 0x00000001)
#define BF_D_PLL_LOCK_CONTROL_GET(val)    ((val >> 0x00000001) & 0x00000003)
#define BF_D_FREQUENCY_LOCK_OOR_INFO      0x00000103
#define BF_D_FREQUENCY_LOCK_OOR(val)      ((val & 0x00000001) << 0x00000003)
#define BF_D_FREQUENCY_LOCK_OOR_GET(val)  ((val >> 0x00000003) & 0x00000001)
#define BF_D_CONTROL_HS_FB_DIV_INFO       0x00000204
#define BF_D_CONTROL_HS_FB_DIV(val)       ((val & 0x00000003) << 0x00000004)
#define BF_D_CONTROL_HS_FB_DIV_GET(val)   ((val >> 0x00000004) & 0x00000003)

#define REG_VCO_CAL_MOMCAP_REG_0_ADDR     0x000000ED

#define REG_VCO_CAL_MOMCAP_REG_1_ADDR     0x000000EE
#define BF_D_VCO_MOMCAP_INFO              0x00000B00
#define BF_D_VCO_MOMCAP(val)              (val & 0x000007FF)
#define BF_D_VCO_MOMCAP_GET(val)          (val & 0x000007FF)
#define BF_D_IMPALA_TEMP_INFO             0x00000203
#define BF_D_IMPALA_TEMP(val)             ((val & 0x00000003) << 0x00000003)
#define BF_D_IMPALA_TEMP_GET(val)         ((val >> 0x00000003) & 0x00000003)
#define BF_D_VCO_CAL_TYPE_INFO            0x00000105
#define BF_D_VCO_CAL_TYPE(val)            ((val & 0x00000001) << 0x00000005)
#define BF_D_VCO_CAL_TYPE_GET(val)        ((val >> 0x00000005) & 0x00000001)
#define BF_D_VCO_FINE_CAP_PRE_INFO        0x00000206
#define BF_D_VCO_FINE_CAP_PRE(val)        ((val & 0x00000003) << 0x00000006)
#define BF_D_VCO_FINE_CAP_PRE_GET(val)    ((val >> 0x00000006) & 0x00000003)

#define REG_VCO_CAL_MOMCAP_PRE_REG_0_ADDR 0x000000EF

#define REG_VCO_CAL_MOMCAP_PRE_REG_1_ADDR 0x000000F0
#define BF_D_VCO_MOMCAP_PRE_INFO          0x00000B00
#define BF_D_VCO_MOMCAP_PRE(val)          (val & 0x000007FF)
#define BF_D_VCO_MOMCAP_PRE_GET(val)      (val & 0x000007FF)
#define BF_D_EN_VAR_COARSE_PRE_INFO       0x00000103
#define BF_D_EN_VAR_COARSE_PRE(val)       ((val & 0x00000001) << 0x00000003)
#define BF_D_EN_VAR_COARSE_PRE_GET(val)   ((val >> 0x00000003) & 0x00000001)
#define BF_D_EN_VAR_FINE_PRE_INFO         0x00000104
#define BF_D_EN_VAR_FINE_PRE(val)         ((val & 0x00000001) << 0x00000004)
#define BF_D_EN_VAR_FINE_PRE_GET(val)     ((val >> 0x00000004) & 0x00000001)
#define BF_D_VCO_COARSE_CAP_PRE_INFO      0x00000205
#define BF_D_VCO_COARSE_CAP_PRE(val)      ((val & 0x00000003) << 0x00000005)
#define BF_D_VCO_COARSE_CAP_PRE_GET(val)  ((val >> 0x00000005) & 0x00000003)

#define REG_VCO_CAL_STATE_REG_ADDR        0x000000F1
#define BF_D_VCO_CAL_INCREMENT_INFO       0x00000100
#define BF_D_VCO_CAL_INCREMENT(val)       (val & 0x00000001)
#define BF_D_VCO_CAL_INCREMENT_GET(val)   (val & 0x00000001)
#define BF_D_IMPALA_CAL_STATE_INFO        0x00000401
#define BF_D_IMPALA_CAL_STATE(val)        ((val & 0x0000000F) << 0x00000001)
#define BF_D_IMPALA_CAL_STATE_GET(val)    ((val >> 0x00000001) & 0x0000000F)
#define BF_D_REGULATOR_CAL_WAIT_INFO      0x00000205
#define BF_D_REGULATOR_CAL_WAIT(val)      ((val & 0x00000003) << 0x00000005)
#define BF_D_REGULATOR_CAL_WAIT_GET(val)  ((val >> 0x00000005) & 0x00000003)
#define BF_D_VCO_SEL_INFO                 0x00000107
#define BF_D_VCO_SEL(val)                 ((val & 0x00000001) << 0x00000007)
#define BF_D_VCO_SEL_GET(val)             ((val >> 0x00000007) & 0x00000001)

#define REG_VCO_CAL_CYCLES_REG_ADDR       0x000000F2
#define BF_D_VCO_PULLH_INFO               0x00000100
#define BF_D_VCO_PULLH(val)               (val & 0x00000001)
#define BF_D_VCO_PULLH_GET(val)           (val & 0x00000001)
#define BF_D_VCO_CAL_CYCLES_INFO          0x00000201
#define BF_D_VCO_CAL_CYCLES(val)          ((val & 0x00000003) << 0x00000001)
#define BF_D_VCO_CAL_CYCLES_GET(val)      ((val >> 0x00000001) & 0x00000003)
#define BF_D_VCO_CAL_WAIT_INFO            0x00000203
#define BF_D_VCO_CAL_WAIT(val)            ((val & 0x00000003) << 0x00000003)
#define BF_D_VCO_CAL_WAIT_GET(val)        ((val >> 0x00000003) & 0x00000003)
#define BF_D_MOMCAP_DUAL_START_INFO       0x00000305
#define BF_D_MOMCAP_DUAL_START(val)       ((val & 0x00000007) << 0x00000005)
#define BF_D_MOMCAP_DUAL_START_GET(val)   ((val >> 0x00000005) & 0x00000007)

#define REG_VCO_COUNT_DIFF_REG_0_ADDR     0x000000F3
#define BF_D_VCO_COUNT_DIFF_INFO          0x00001000
#define BF_D_VCO_COUNT_DIFF(val)          (val & 0x0000FFFF)
#define BF_D_VCO_COUNT_DIFF_GET(val)      (val & 0x0000FFFF)

#define REG_VCO_COUNT_DIFF_REG_1_ADDR     0x000000F4

#define REG_VCM_CONTROL_C_REG_ADDR        0x000000F5
#define BF_D_VCM_C_CONTROL_C_INFO         0x00000400
#define BF_D_VCM_C_CONTROL_C(val)         (val & 0x0000000F)
#define BF_D_VCM_C_CONTROL_C_GET(val)     (val & 0x0000000F)
#define BF_D_VCM_F_CONTROL_C_INFO         0x00000404
#define BF_D_VCM_F_CONTROL_C(val)         ((val & 0x0000000F) << 0x00000004)
#define BF_D_VCM_F_CONTROL_C_GET(val)     ((val >> 0x00000004) & 0x0000000F)

#define REG_VCM_CONTROL_H_REG_ADDR        0x000000F6
#define BF_D_VCM_C_CONTROL_H_INFO         0x00000400
#define BF_D_VCM_C_CONTROL_H(val)         (val & 0x0000000F)
#define BF_D_VCM_C_CONTROL_H_GET(val)     (val & 0x0000000F)
#define BF_D_VCM_F_CONTROL_H_INFO         0x00000404
#define BF_D_VCM_F_CONTROL_H(val)         ((val & 0x0000000F) << 0x00000004)
#define BF_D_VCM_F_CONTROL_H_GET(val)     ((val >> 0x00000004) & 0x0000000F)

#define REG_CHARGEPUMP_REG_2_ADDR         0x000000F7
#define BF_D_CP_BLEED_INFO                0x00000600
#define BF_D_CP_BLEED(val)                (val & 0x0000003F)
#define BF_D_CP_BLEED_GET(val)            (val & 0x0000003F)

#define REG_FASTV_COMP_LOWL_REG_0_ADDR    0x000000F8
#define BF_D_FASTV_COMP_LOWL_INFO         0x00000B00
#define BF_D_FASTV_COMP_LOWL(val)         (val & 0x000007FF)
#define BF_D_FASTV_COMP_LOWL_GET(val)     (val & 0x000007FF)

#define REG_FASTV_COMP_LOWL_REG_1_ADDR    0x000000F9

#define REG_FASTV_COMP_HIGHL_REG_0_ADDR   0x000000FA
#define BF_D_FASTV_COMP_HIGHL_INFO        0x00000B00
#define BF_D_FASTV_COMP_HIGHL(val)        (val & 0x000007FF)
#define BF_D_FASTV_COMP_HIGHL_GET(val)    (val & 0x000007FF)

#define REG_FASTV_COMP_HIGHL_REG_1_ADDR   0x000000FB

#define REG_SLOWV_COMP_LOWL_REG_0_ADDR    0x000000FC
#define BF_D_SLOWV_COMP_LOWL_INFO         0x00000B00
#define BF_D_SLOWV_COMP_LOWL(val)         (val & 0x000007FF)
#define BF_D_SLOWV_COMP_LOWL_GET(val)     (val & 0x000007FF)

#define REG_SLOWV_COMP_LOWL_REG_1_ADDR    0x000000FD

#define REG_SLOWV_COMP_HIGHL_REG_0_ADDR   0x000000FE
#define BF_D_SLOWV_COMP_HIGHL_INFO        0x00000B00
#define BF_D_SLOWV_COMP_HIGHL(val)        (val & 0x000007FF)
#define BF_D_SLOWV_COMP_HIGHL_GET(val)    (val & 0x000007FF)

#define REG_SLOWV_COMP_HIGHL_REG_1_ADDR   0x000000FF

#define REG_IMPALA_REV_ID_ADDR            0x00000100
#define BF_D_IMPALA_REV_ID_INFO           0x00000800
#define BF_D_IMPALA_REV_ID(val)           (val & 0x000000FF)
#define BF_D_IMPALA_REV_ID_GET(val)       (val & 0x000000FF)



#endif /* __ADI_AD9082_BF_IMPALA_TC_H__ */
/*! @} */