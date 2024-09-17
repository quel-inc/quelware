/*!
 * @brief     ADI utility functions header file.
 *
 * @version   0.1.x
 *
 * @copyright copyright(c) 2018 analog devices, inc. all rights reserved.
 *            This software is proprietary to Analog Devices, Inc. and its
 *            licensor. By using this software you agree to the terms of the
 *            associated analog devices software license agreement.
 */

/*! 
 * @addtogroup __ADI_UTILS__
 * @{
 */
#ifndef __ADI_UTILS_H__
#define __ADI_UTILS_H__

/*============= I N C L U D E S ============*/
#include "adi_cms_api_common.h"

/*============= D E F I N E S ==============*/
#define ADI_UTILS_POW2_32         ((uint64_t)1 << 32)
#define ADI_UTILS_POW2_48         ((uint64_t)1 << 48)
#define ADI_UTILS_MAXUINT24       (0xffffff)
#define ADI_UTILS_MAXUINT32       (ADI_UTILS_POW2_32 - 1)
#define ADI_UTILS_MAXUINT48       (ADI_UTILS_POW2_48 - 1)

#define ADI_UTILS_GET_BYTE(w, p)  (uint8_t)(((w) >> (p)) & 0xff)
#define ADI_UTILS_DIV_U64(x, y)   ((x) / (y))
#define ADI_UTILS_BIT(x)          ((1) << (x))
#define ADI_UTILS_ALL             (-1)
#define ADI_UTILS_ARRAY_SIZE(a)   (sizeof(a) / sizeof((a)[0]))

/*============= E X P O R T S ==============*/
#ifdef __cplusplus
extern "C" {
#endif

int32_t adi_api_utils_gcd(int32_t u, int32_t v);

int32_t adi_api_utils_is_power_of_two(uint64_t x);

void adi_api_utils_mult_64(uint32_t a, uint32_t b, uint32_t *hi, uint32_t *lo);

void adi_api_utils_lshift_128(uint64_t *hi, uint64_t *lo);

void adi_api_utils_rshift_128(uint64_t *hi, uint64_t *lo);

void adi_api_utils_mult_128(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo);

void adi_api_utils_div_128(uint64_t a_hi, uint64_t a_lo, uint64_t b_hi, 
    uint64_t b_lo, uint64_t *hi, uint64_t *lo);

void adi_api_utils_mod_128(uint64_t ah, uint64_t al, uint64_t div, uint64_t *mod);

void adi_api_utils_add_128(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl,
    uint64_t *hi, uint64_t *lo);

void adi_api_utils_subt_128(uint64_t ah, uint64_t al, uint64_t bh,uint64_t bl, 
    uint64_t *hi,uint64_t *lo);

uint32_t adi_api_utils_log2(uint32_t a);

#ifdef __cplusplus
}
#endif

#endif /*__ADI_UTILS_H__*/

/*! @} */