// SPDX-License-Identifier: GPL-2.0
/*!
 * @brief     APIs to call HAL functions
 *
 * @copyright copyright(c) 2018 analog devices, inc. all rights reserved.
 *            This software is proprietary to Analog Devices, Inc. and its
 *            licensor. By using this software you agree to the terms of the
 *            associated analog devices software license agreement.
 */

/*!
 * @addtogroup AD9081_HAL_API
 * @{
 */

/*============= I N C L U D E S ============*/
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "adi_ad9081_hal.h"

// defined in ad9081_wrapper.cpp
// handle is an opaque pointer to a C++ object holding callbacks.
int ad9081_callback_regread(void *handle, uint32_t address, uint8_t* value);
int ad9081_callback_regwrite(void *handle, uint32_t address, uint8_t value);
int ad9081_callback_delay_us(void *handle, uint32_t us);
int ad9081_callback_log_write(void *handle, adi_cms_log_type_e log_type, const char *ptr);

/*============= C O D E ====================*/
int32_t adi_ad9081_hal_hw_open(adi_ad9081_device_t *device)
{
	AD9081_NULL_POINTER_RETURN(device);
	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_hw_close(adi_ad9081_device_t *device)
{
	AD9081_NULL_POINTER_RETURN(device);
	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_delay_us(adi_ad9081_device_t *device, uint32_t us)
{
	AD9081_NULL_POINTER_RETURN(device);
    AD9081_NULL_POINTER_RETURN(device->hal_info.callback_handle);
    if (ad9081_callback_delay_us(device->hal_info.callback_handle, us) == 0) {
        return API_CMS_ERROR_OK;
    } else {
        return API_CMS_ERROR_ERROR;
    }
}

int32_t adi_ad9081_hal_reset_pin_ctrl(adi_ad9081_device_t *device,
				      uint8_t enable)
{
	AD9081_NULL_POINTER_RETURN(device);
    return API_CMS_ERROR_NOT_SUPPORTED;
}

int32_t adi_ad9081_hal_log_write(adi_ad9081_device_t *device,
                                 adi_cms_log_type_e log_type,
                                 const char *comment, ...)
{
    va_list argp;
    char logMessage[512];

    va_start(argp, comment);
    vsnprintf(logMessage, sizeof(logMessage), comment, argp);
    va_end(argp);

    int32_t retcode = API_CMS_ERROR_OK;
    if (device == NULL || device->hal_info.callback_handle == NULL) {
        fprintf(stderr, "%s\n", logMessage);  // for emergency.
    } else {
        if (ad9081_callback_log_write(device->hal_info.callback_handle, log_type, logMessage) != 0) {
            retcode = API_CMS_ERROR_ERROR;
        }
    }

    return retcode;
}

int32_t adi_ad9081_hal_bf_get(adi_ad9081_device_t *device, uint32_t reg,
			      uint32_t info, uint8_t *value,
			      uint8_t value_size_bytes)
{
	int32_t err;
	uint8_t reg_offset = 0, data8 = 0;
	uint8_t offset = (uint8_t)(info >> 0), width = (uint8_t)(info >> 8);
	uint32_t data32 = 0, mask = 0, endian_test_val = 0x11223344;
	uint64_t bf_val = 0;
	uint8_t reg_bytes =
		((width + offset) >> 3) + (((width + offset) & 7) == 0 ? 0 : 1);
	uint8_t i = 0, j = 0, filled_bits = 0;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_NULL_POINTER_RETURN(value);
	AD9081_INVALID_PARAM_RETURN(width > 64);
	AD9081_INVALID_PARAM_RETURN(width < 1);
	AD9081_INVALID_PARAM_RETURN(value_size_bytes > 8);

	if (reg < 0x4000) {
		for (reg_offset = 0; reg_offset < reg_bytes; reg_offset++) {
			err = adi_ad9081_hal_reg_get(device, reg + reg_offset,
						     &data8);
			AD9081_ERROR_RETURN(err);
			if ((offset + width) <= 8) { /* last 8bits */
				mask = (1 << width) - 1;
				data8 = (data8 >> offset) & mask;
				bf_val = bf_val +
					 ((uint64_t)data8 << filled_bits);
				filled_bits = filled_bits + width;
			} else {
				mask = (1 << (8 - offset)) - 1;
				data8 = (data8 >> offset) & mask;
				bf_val = bf_val +
					 ((uint64_t)data8 << filled_bits);
				width = offset + width - 8;
				filled_bits = filled_bits + (8 - offset);
				offset = 0;
			}
		}
	} else { /* access extended space */
		for (reg_offset = 0; reg_offset < reg_bytes; reg_offset += 4) {
			err = adi_ad9081_hal_reg_get(device, reg + reg_offset,
						     (uint8_t *)&data32);
			AD9081_ERROR_RETURN(err);
			if ((offset + width) <= 32) { /* last 32bits */
				mask = ((uint64_t)1 << width) - 1;
				data32 = (data32 >> offset) & mask;
				bf_val = bf_val +
					 ((uint64_t)data32 << filled_bits);
				filled_bits = filled_bits + width;
			} else {
				mask = ((uint64_t)1 << (32 - offset)) - 1;
				data32 = (data32 >> offset) & mask;
				bf_val = bf_val +
					 ((uint64_t)data32 << filled_bits);
				width = offset + width - 32;
				filled_bits = filled_bits + (32 - offset);
				offset = 0;
			}
		}
	}

	/* save bitfield value to buffer */
	for (i = 0; i < value_size_bytes; i++) {
		j = (*(uint8_t *)&endian_test_val == 0x44) ?
			    (i) :
			    (value_size_bytes - 1 - i);
		value[j] = (uint8_t)(bf_val >> (i << 3));
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_bf_set(adi_ad9081_device_t *device, uint32_t reg,
			      uint32_t info, uint64_t value)
{
	int32_t err;
	uint8_t reg_offset = 0, data8 = 0;
	uint8_t offset = (uint8_t)(info >> 0), width = (uint8_t)(info >> 8);
	uint32_t data32 = 0, mask = 0;
	uint8_t reg_bytes =
		((width + offset) >> 3) + (((width + offset) & 7) == 0 ? 0 : 1);
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_INVALID_PARAM_RETURN(width > 64);
	AD9081_INVALID_PARAM_RETURN(width < 1);

	if (reg < 0x4000) {
		for (reg_offset = 0; reg_offset < reg_bytes; reg_offset++) {
			if ((offset + width) <= 8) { /* last 8bits */
				if ((offset > 0) || ((offset + width) < 8)) {
					err = adi_ad9081_hal_reg_get(
						device, reg + reg_offset,
						&data8);
					AD9081_ERROR_RETURN(err);
				}
				mask = (1 << width) - 1;
				data8 = data8 & (~(mask << offset));
				data8 = data8 | ((value & mask) << offset);
			} else {
				if (offset > 0) {
					err = adi_ad9081_hal_reg_get(
						device, reg + reg_offset,
						&data8);
					AD9081_ERROR_RETURN(err);
				}
				mask = (1 << (8 - offset)) - 1;
				data8 = data8 & (~(mask << offset));
				data8 = data8 | ((value & mask) << offset);
				value = value >> (8 - offset);
				width = offset + width - 8;
				offset = 0;
			}
			err = adi_ad9081_hal_reg_set(device, reg + reg_offset,
						     data8);
			AD9081_ERROR_RETURN(err);
		}
	} else { /* access extended space */
		for (reg_offset = 0; reg_offset < reg_bytes; reg_offset += 4) {
			if ((offset + width) <= 32) { /* last 32bits */
				if ((offset > 0) || ((offset + width) < 32)) {
					err = adi_ad9081_hal_reg_get(
						device, reg + reg_offset,
						(uint8_t *)&data32);
					AD9081_ERROR_RETURN(err);
				}
				mask = ((uint64_t)1 << width) - 1;
				data32 = data32 & (~(mask << offset));
				data32 = data32 | ((value & mask) << offset);
			} else {
				if (offset > 0) {
					err = adi_ad9081_hal_reg_get(
						device, reg + reg_offset,
						&data8);
					AD9081_ERROR_RETURN(err);
				}
				mask = ((uint64_t)1 << (32 - offset)) - 1;
				data32 = data32 & (~(mask << offset));
				data32 = data32 | ((value & mask) << offset);
				value = value >> (32 - offset);
				width = offset + width - 32;
				offset = 0;
			}
			err = adi_ad9081_hal_reg_set(device, reg + reg_offset,
						     data32);
			AD9081_ERROR_RETURN(err);
		}
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_reg_get(adi_ad9081_device_t *device, uint32_t reg,
			       uint8_t *data)
{
	AD9081_NULL_POINTER_RETURN(device);
    AD9081_NULL_POINTER_RETURN(device->hal_info.callback_handle);
	AD9081_NULL_POINTER_RETURN(data);

    int32_t retcode = API_CMS_ERROR_OK;
	if (reg < 0x4000) {
        if (ad9081_callback_regread(device->hal_info.callback_handle, reg, data) != 0) {
            retcode = API_CMS_ERROR_ERROR;
        }
	} else {
        retcode = API_CMS_ERROR_NOT_SUPPORTED;
	}
	return retcode;
}

int32_t adi_ad9081_hal_reg_set(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t data)
{
    AD9081_NULL_POINTER_RETURN(device);
    AD9081_NULL_POINTER_RETURN(device->hal_info.callback_handle);

    int32_t retcode = API_CMS_ERROR_OK;
	if (reg < 0x4000) {
        if (ad9081_callback_regwrite(device->hal_info.callback_handle, reg, data) != 0) {
            retcode = API_CMS_ERROR_ERROR;
        }
	} else { /* access extended 32-bit data space */
        retcode = API_CMS_ERROR_NOT_SUPPORTED;
	}
	return retcode;
}

int32_t adi_ad9081_hal_cbusjrx_reg_get(adi_ad9081_device_t *device,
				       uint32_t reg, uint8_t *data,
				       uint8_t lane)
{
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_NULL_POINTER_RETURN(data);
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x406, reg))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x409, 0x00))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK !=
	    adi_ad9081_hal_reg_set(device, 0x409, 1 << lane))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_delay_us(device, 500))
		return API_CMS_ERROR_DELAY_US;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_get(device, 0x40a, data))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x409, 0x00))
		return API_CMS_ERROR_SPI_XFER;

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_cbusjrx_reg_set(adi_ad9081_device_t *device,
				       uint32_t reg, uint8_t data, uint8_t lane)
{
	AD9081_NULL_POINTER_RETURN(device);
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x406, reg))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x408, data))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_delay_us(device, 500))
		return API_CMS_ERROR_DELAY_US;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x407, 0x00))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x407, lane))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_delay_us(device, 500))
		return API_CMS_ERROR_DELAY_US;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x407, 0x00))
		return API_CMS_ERROR_SPI_XFER;

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_cbusjtx_reg_get(adi_ad9081_device_t *device,
				       uint32_t reg, uint8_t *data,
				       uint8_t lane)
{
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_NULL_POINTER_RETURN(data);
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x790, reg))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x794, 0x00))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK !=
	    adi_ad9081_hal_reg_set(device, 0x794, 1 << lane))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_delay_us(device, 500))
		return API_CMS_ERROR_DELAY_US;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_get(device, 0x796, data))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x794, 0x00))
		return API_CMS_ERROR_SPI_XFER;

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_cbusjtx_reg_set(adi_ad9081_device_t *device,
				       uint32_t reg, uint8_t data, uint8_t lane)
{
	AD9081_NULL_POINTER_RETURN(device);
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x790, reg))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x793, data))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_delay_us(device, 500))
		return API_CMS_ERROR_DELAY_US;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x791, 0x00))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x791, lane))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_delay_us(device, 500))
		return API_CMS_ERROR_DELAY_US;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x791, 0x00))
		return API_CMS_ERROR_SPI_XFER;

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_cbuspll_reg_get(adi_ad9081_device_t *device,
				       uint32_t reg, uint8_t *data)
{
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_NULL_POINTER_RETURN(data);
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x740, reg))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x72E, 0x01))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_delay_us(device, 500))
		return API_CMS_ERROR_DELAY_US;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_get(device, 0x742, data))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x72E, 0x00))
		return API_CMS_ERROR_SPI_XFER;

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_cbuspll_reg_set(adi_ad9081_device_t *device,
				       uint32_t reg, uint8_t data)
{
	AD9081_NULL_POINTER_RETURN(device);
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x740, reg))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x741, data))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_delay_us(device, 500))
		return API_CMS_ERROR_DELAY_US;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x72F, 0x00))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x72F, 0x01))
		return API_CMS_ERROR_SPI_XFER;
	if (API_CMS_ERROR_OK != adi_ad9081_hal_reg_set(device, 0x72F, 0x00))
		return API_CMS_ERROR_SPI_XFER;

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_bf_wait_to_clear(adi_ad9081_device_t *device,
					uint32_t reg, uint32_t info)
{
	int32_t err;
	uint8_t i = 0, bf_value = 0;
	AD9081_NULL_POINTER_RETURN(device);
	for (i = 0; i < 200; i++) {
		err = adi_ad9081_hal_delay_us(device, 20);
		AD9081_ERROR_RETURN(err);
		err = adi_ad9081_hal_bf_get(device, reg, info, &bf_value, 1);
		AD9081_ERROR_RETURN(err);
		if (bf_value == 0) {
			break;
		}
		if (i == 199) {
			return API_CMS_ERROR_ERROR;
		}
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_bf_wait_to_set(adi_ad9081_device_t *device, uint32_t reg,
				      uint32_t info)
{
	int32_t err;
	uint8_t i = 0, bf_value = 0;
	AD9081_NULL_POINTER_RETURN(device);
	for (i = 0; i < 200; i++) {
		err = adi_ad9081_hal_delay_us(device, 20);
		AD9081_ERROR_RETURN(err);
		err = adi_ad9081_hal_bf_get(device, reg, info, &bf_value, 1);
		AD9081_ERROR_RETURN(err);
		if (bf_value == 1) {
			break;
		}
		if (i == 199) {
			return API_CMS_ERROR_ERROR;
		}
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_error_report(adi_ad9081_device_t *device,
				    adi_cms_log_type_e log_type, int32_t error,
				    const char *file_name,
				    const char *func_name, uint32_t line_num,
				    const char *var_name, const char *comment)
{
	if (device == NULL)
		return API_CMS_ERROR_NULL_PARAM;

	if (API_CMS_ERROR_OK !=
	    adi_ad9081_hal_log_write(
		    device, log_type, "%s, \"%s\" in %s(...), line%d in %s",
		    comment, var_name, func_name, line_num, file_name))
		return API_CMS_ERROR_LOG_WRITE;

	return API_CMS_ERROR_OK;
}

void adi_ad9081_hal_add_128(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl,
			    uint64_t *hi, uint64_t *lo)
{
	uint64_t rl = al + bl, rh = ah + bh;
	if (rl < al)
		rh++;
	*lo = rl;
	*hi = rh;
}

void adi_ad9081_hal_sub_128(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl,
			    uint64_t *hi, uint64_t *lo)
{
	uint64_t rl, rh;
	if (bl <= al) {
		rl = al - bl;
		rh = ah - bh;
	} else {
		rl = bl - al - 1;
		rl = 0xffffffffffffffffull - rl;
		ah--;
		rh = ah - bh;
	}
	*lo = rl;
	*hi = rh;
}

void adi_ad9081_hal_mult_128(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo)
{
	uint64_t ah = a >> 32, al = a & 0xffffffff, bh = b >> 32,
		 bl = b & 0xffffffff, rh = ah * bh, rl = al * bl, rm1 = ah * bl,
		 rm2 = al * bh, rm1h = rm1 >> 32, rm2h = rm2 >> 32,
		 rm1l = rm1 & 0xffffffff, rm2l = rm2 & 0xffffffff,
		 rmh = rm1h + rm2h, rml = rm1l + rm2l,
		 c = ((rl >> 32) + rml) >> 32;
	rl = rl + (rml << 32);
	rh = rh + rmh + c;
	*lo = rl;
	*hi = rh;
}

void adi_ad9081_hal_lshift_128(uint64_t *hi, uint64_t *lo)
{
	*hi <<= 1;
	if (*lo & 0x8000000000000000ull)
		*hi |= 1ull;
	*lo <<= 1;
}

void adi_ad9081_hal_rshift_128(uint64_t *hi, uint64_t *lo)
{
	*lo >>= 1;
	if (*hi & 1ull)
		*lo |= 0x8000000000000000ull;
	*hi >>= 1;
}

void adi_ad9081_hal_div_128(uint64_t a_hi, uint64_t a_lo, uint64_t b_hi,
			    uint64_t b_lo, uint64_t *hi, uint64_t *lo)
{
	uint64_t remain_lo = a_lo, remain_hi = a_hi, part1_lo = b_lo,
		 part1_hi = b_hi;
	uint64_t result_lo = 0, result_hi = 0, mask_lo = 1, mask_hi = 0;

	while (!(part1_hi & 0x8000000000000000ull)) {
		adi_ad9081_hal_lshift_128(&part1_hi, &part1_lo);
		adi_ad9081_hal_lshift_128(&mask_hi, &mask_lo);
	}

	do {
		if ((remain_hi > part1_hi) ||
		    ((remain_hi == part1_hi) && (remain_lo >= part1_lo))) {
			adi_ad9081_hal_sub_128(remain_hi, remain_lo, part1_hi,
					       part1_lo, &remain_hi,
					       &remain_lo);
			adi_ad9081_hal_add_128(result_hi, result_lo, mask_hi,
					       mask_lo, &result_hi, &result_lo);
		}
		adi_ad9081_hal_rshift_128(&part1_hi, &part1_lo);
		adi_ad9081_hal_rshift_128(&mask_hi, &mask_lo);
	} while ((mask_hi != 0) || (mask_lo != 0));
	*lo = result_lo;
	*hi = result_hi;
}

int32_t adi_ad9081_hal_calc_nco_ftw(adi_ad9081_device_t *device, uint64_t freq,
				    int64_t nco_shift, uint64_t *ftw,
				    uint64_t *a, uint64_t *b)
{
	uint64_t hi, lo, hi1, hi2, lo2, hi3, lo3, hi4, lo4;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_LOG_FUNC();
	AD9081_INVALID_PARAM_RETURN(freq == 0);

	/* ftw + a/b   nco_shift */
	/* --------- = --------- */
	/*    2^48        freq   */
	if (nco_shift >= 0) {
		adi_ad9081_hal_mult_128(281474976710656ull, nco_shift, &hi,
					&lo);
		adi_ad9081_hal_div_128(hi, lo, 0, freq, &hi1, ftw);
		adi_ad9081_hal_mult_128(*ftw, freq, &hi2, &lo2);
		adi_ad9081_hal_sub_128(hi, lo, hi2, lo2, &hi3, &lo3);
		adi_ad9081_hal_mult_128(lo3, 281474976710655ull, &hi4, &lo4);
		adi_ad9081_hal_div_128(hi4, lo4, 0, freq, &hi1, a);
		*b = 281474976710655ull;
	} else {
		adi_ad9081_hal_mult_128(281474976710656ull, -nco_shift, &hi,
					&lo);
		adi_ad9081_hal_div_128(hi, lo, 0, freq, &hi, ftw);
		adi_ad9081_hal_mult_128(*ftw, freq, &hi2, &lo2);
		adi_ad9081_hal_sub_128(hi, lo, hi2, lo2, &hi3, &lo3);
		adi_ad9081_hal_mult_128(lo3, 281474976710655ull, &hi4, &lo4);
		adi_ad9081_hal_div_128(hi4, lo4, 0, freq, &hi1, a);
		*b = 281474976710655ull;
		*a = (*a > 0) ?
			     (281474976710656ull - *a) :
			     *a; /* assume register a/b is unsigned 48bit value */
		*ftw = 281474976710656ull - *ftw - (*a > 0 ? 1 : 0);
	}

	return API_CMS_ERROR_OK;
}

#if AD9081_USE_FLOATING_TYPE > 0
int32_t adi_ad9081_hal_calc_nco_ftw_f(adi_ad9081_device_t *device, double freq,
				      double nco_shift, uint64_t *ftw,
				      uint64_t *a, uint64_t *b)
{
	double set_shift, rem_shift;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_LOG_FUNC();
	AD9081_INVALID_PARAM_RETURN(freq == 0);

	/* ftw + a/b   nco_shift */
	/* --------- = --------- */
	/*    2^48        freq   */
	if (nco_shift >= 0) {
		*ftw = (uint64_t)(281474976710656ull * nco_shift / freq);
		set_shift = (*ftw) * freq / 281474976710656ull;
		rem_shift = nco_shift - set_shift;
		*b = 281474976710655ull;
		*a = (uint64_t)((rem_shift * 281474976710656ull / freq) * (*b));
	} else {
		*ftw = (uint64_t)(281474976710656ull * (-nco_shift) / freq);
		set_shift = (*ftw) * freq / 281474976710656ull;
		rem_shift = -nco_shift - set_shift;
		*b = 281474976710655ull;
		*a = (uint64_t)((rem_shift * 281474976710656ull / freq) * (*b));
		*a = (*a > 0) ?
			     (281474976710656ull - *a) :
			     *a; /* assume register a/b is unsigned 48bit value */
		*ftw = 281474976710656ull - *ftw - (*a > 0 ? 1 : 0);
	}

	return API_CMS_ERROR_OK;
}
#endif

int32_t adi_ad9081_hal_calc_rx_nco_ftw(adi_ad9081_device_t *device,
				       uint64_t adc_freq, int64_t nco_shift,
				       uint64_t *ftw)
{
	uint64_t hi, lo;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_LOG_FUNC();
	AD9081_INVALID_PARAM_RETURN(adc_freq == 0);

	if (nco_shift >= 0) {
		adi_ad9081_hal_mult_128(281474976710656ull, nco_shift, &hi,
					&lo);
		adi_ad9081_hal_div_128(hi, lo, 0, adc_freq, &hi, ftw);
	} else {
		adi_ad9081_hal_mult_128(281474976710656ull, -nco_shift, &hi,
					&lo);
		adi_ad9081_hal_div_128(hi, lo, 0, adc_freq, &hi, ftw);
		*ftw = 281474976710656ull - *ftw;
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_calc_rx_nco_ftw32(adi_ad9081_device_t *device,
					 uint64_t adc_freq, int64_t nco_shift,
					 uint64_t *ftw)
{
	uint64_t hi, lo;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_LOG_FUNC();
	AD9081_INVALID_PARAM_RETURN(adc_freq == 0);

	if (nco_shift >= 0) {
		adi_ad9081_hal_mult_128(4294967296ull, nco_shift, &hi, &lo);
		adi_ad9081_hal_div_128(hi, lo, 0, adc_freq, &hi, ftw);
	} else {
		adi_ad9081_hal_mult_128(4294967296ull, -nco_shift, &hi, &lo);
		adi_ad9081_hal_div_128(hi, lo, 0, adc_freq, &hi, ftw);
		*ftw = 4294967296ull - *ftw;
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_calc_tx_nco_ftw(adi_ad9081_device_t *device,
				       uint64_t dac_freq, int64_t nco_shift,
				       uint64_t *ftw)
{
	uint64_t hi, lo;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_LOG_FUNC();
	AD9081_INVALID_PARAM_RETURN(dac_freq == 0);

	if (nco_shift >= 0) {
		adi_ad9081_hal_mult_128(281474976710656ull, nco_shift, &hi,
					&lo);
		adi_ad9081_hal_div_128(hi, lo, 0, dac_freq, &hi, ftw);
	} else {
		adi_ad9081_hal_mult_128(281474976710656ull, -nco_shift, &hi,
					&lo);
		adi_ad9081_hal_div_128(hi, lo, 0, dac_freq, &hi, ftw);
		*ftw = 281474976710656ull - *ftw;
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_calc_tx_nco_ftw32(adi_ad9081_device_t *device,
					 uint64_t dac_freq, int64_t nco_shift,
					 uint64_t *ftw)
{
	uint64_t hi, lo;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_LOG_FUNC();
	AD9081_INVALID_PARAM_RETURN(dac_freq == 0);

	if (nco_shift >= 0) {
		adi_ad9081_hal_mult_128(4294967296ull, nco_shift, &hi, &lo);
		adi_ad9081_hal_div_128(hi, lo, 0, dac_freq, &hi, ftw);
	} else {
		adi_ad9081_hal_mult_128(4294967296ull, -nco_shift, &hi, &lo);
		adi_ad9081_hal_div_128(hi, lo, 0, dac_freq, &hi, ftw);
		*ftw = 4294967296ull - *ftw;
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_2bf_get(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint8_t *value0, uint32_t info1,
			       uint8_t *value1, uint8_t value_size_bytes)
{
	uint32_t info[2] = { info0, info1 };
	uint8_t *value[2] = { value0, value1 };
	return adi_ad9081_hal_multi_bf_get(device, reg, info, value,
					   value_size_bytes, 2);
}

int32_t adi_ad9081_hal_3bf_get(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint8_t *value0, uint32_t info1,
			       uint8_t *value1, uint32_t info2, uint8_t *value2,
			       uint8_t value_size_bytes)
{
	uint32_t info[3] = { info0, info1, info2 };
	uint8_t *value[3] = { value0, value1, value2 };
	return adi_ad9081_hal_multi_bf_get(device, reg, info, value,
					   value_size_bytes, 3);
}

int32_t adi_ad9081_hal_4bf_get(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint8_t *value0, uint32_t info1,
			       uint8_t *value1, uint32_t info2, uint8_t *value2,
			       uint32_t info3, uint8_t *value3,
			       uint8_t value_size_bytes)
{
	uint32_t info[4] = { info0, info1, info2, info3 };
	uint8_t *value[4] = { value0, value1, value2, value3 };
	return adi_ad9081_hal_multi_bf_get(device, reg, info, value,
					   value_size_bytes, 4);
}

int32_t adi_ad9081_hal_5bf_get(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint8_t *value0, uint32_t info1,
			       uint8_t *value1, uint32_t info2, uint8_t *value2,
			       uint32_t info3, uint8_t *value3, uint32_t info4,
			       uint8_t *value4, uint8_t value_size_bytes)
{
	uint32_t info[5] = { info0, info1, info2, info3, info4 };
	uint8_t *value[5] = { value0, value1, value2, value3, value4 };
	return adi_ad9081_hal_multi_bf_get(device, reg, info, value,
					   value_size_bytes, 5);
}

int32_t adi_ad9081_hal_6bf_get(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint8_t *value0, uint32_t info1,
			       uint8_t *value1, uint32_t info2, uint8_t *value2,
			       uint32_t info3, uint8_t *value3, uint32_t info4,
			       uint8_t *value4, uint32_t info5, uint8_t *value5,
			       uint8_t value_size_bytes)
{
	uint32_t info[6] = { info0, info1, info2, info3, info4, info5 };
	uint8_t *value[6] = { value0, value1, value2, value3, value4, value5 };
	return adi_ad9081_hal_multi_bf_get(device, reg, info, value,
					   value_size_bytes, 6);
}

int32_t adi_ad9081_hal_7bf_get(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint8_t *value0, uint32_t info1,
			       uint8_t *value1, uint32_t info2, uint8_t *value2,
			       uint32_t info3, uint8_t *value3, uint32_t info4,
			       uint8_t *value4, uint32_t info5, uint8_t *value5,
			       uint32_t info6, uint8_t *value6,
			       uint8_t value_size_bytes)
{
	uint32_t info[7] = { info0, info1, info2, info3, info4, info5, info6 };
	uint8_t *value[7] = { value0, value1, value2, value3,
			      value4, value5, value6 };
	return adi_ad9081_hal_multi_bf_get(device, reg, info, value,
					   value_size_bytes, 7);
}

int32_t adi_ad9081_hal_8bf_get(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint8_t *value0, uint32_t info1,
			       uint8_t *value1, uint32_t info2, uint8_t *value2,
			       uint32_t info3, uint8_t *value3, uint32_t info4,
			       uint8_t *value4, uint32_t info5, uint8_t *value5,
			       uint32_t info6, uint8_t *value6, uint32_t info7,
			       uint8_t *value7, uint8_t value_size_bytes)
{
	uint32_t info[8] = { info0, info1, info2, info3,
			     info4, info5, info6, info7 };
	uint8_t *value[8] = { value0, value1, value2, value3,
			      value4, value5, value6, value7 };
	return adi_ad9081_hal_multi_bf_get(device, reg, info, value,
					   value_size_bytes, 8);
}

int32_t adi_ad9081_hal_2bf_set(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint64_t value0, uint32_t info1,
			       uint64_t value1)
{
	uint32_t info[2] = { info0, info1 };
	uint64_t value[2] = { value0, value1 };
	return adi_ad9081_hal_multi_bf_set(device, reg, info, value, 2);
}

int32_t adi_ad9081_hal_3bf_set(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint64_t value0, uint32_t info1,
			       uint64_t value1, uint32_t info2, uint64_t value2)
{
	uint32_t info[3] = { info0, info1, info2 };
	uint64_t value[3] = { value0, value1, value2 };
	return adi_ad9081_hal_multi_bf_set(device, reg, info, value, 3);
}

int32_t adi_ad9081_hal_4bf_set(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint64_t value0, uint32_t info1,
			       uint64_t value1, uint32_t info2, uint64_t value2,
			       uint32_t info3, uint64_t value3)
{
	uint32_t info[4] = { info0, info1, info2, info3 };
	uint64_t value[4] = { value0, value1, value2, value3 };
	return adi_ad9081_hal_multi_bf_set(device, reg, info, value, 4);
}

int32_t adi_ad9081_hal_5bf_set(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint64_t value0, uint32_t info1,
			       uint64_t value1, uint32_t info2, uint64_t value2,
			       uint32_t info3, uint64_t value3, uint32_t info4,
			       uint64_t value4)
{
	uint32_t info[5] = { info0, info1, info2, info3, info4 };
	uint64_t value[5] = { value0, value1, value2, value3, value4 };
	return adi_ad9081_hal_multi_bf_set(device, reg, info, value, 5);
}

int32_t adi_ad9081_hal_6bf_set(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint64_t value0, uint32_t info1,
			       uint64_t value1, uint32_t info2, uint64_t value2,
			       uint32_t info3, uint64_t value3, uint32_t info4,
			       uint64_t value4, uint32_t info5, uint64_t value5)
{
	uint32_t info[6] = { info0, info1, info2, info3, info4, info5 };
	uint64_t value[6] = { value0, value1, value2, value3, value4, value5 };
	return adi_ad9081_hal_multi_bf_set(device, reg, info, value, 6);
}

int32_t adi_ad9081_hal_7bf_set(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint64_t value0, uint32_t info1,
			       uint64_t value1, uint32_t info2, uint64_t value2,
			       uint32_t info3, uint64_t value3, uint32_t info4,
			       uint64_t value4, uint32_t info5, uint64_t value5,
			       uint32_t info6, uint64_t value6)
{
	uint32_t info[7] = { info0, info1, info2, info3, info4, info5, info6 };
	uint64_t value[7] = { value0, value1, value2, value3,
			      value4, value5, value6 };
	return adi_ad9081_hal_multi_bf_set(device, reg, info, value, 7);
}

int32_t adi_ad9081_hal_8bf_set(adi_ad9081_device_t *device, uint32_t reg,
			       uint32_t info0, uint64_t value0, uint32_t info1,
			       uint64_t value1, uint32_t info2, uint64_t value2,
			       uint32_t info3, uint64_t value3, uint32_t info4,
			       uint64_t value4, uint32_t info5, uint64_t value5,
			       uint32_t info6, uint64_t value6, uint32_t info7,
			       uint64_t value7)
{
	uint32_t info[8] = { info0, info1, info2, info3,
			     info4, info5, info6, info7 };
	uint64_t value[8] = { value0, value1, value2, value3,
			      value4, value5, value6, value7 };
	return adi_ad9081_hal_multi_bf_set(device, reg, info, value, 8);
}

int32_t adi_ad9081_hal_multi_bf_get(adi_ad9081_device_t *device, uint32_t reg,
				    uint32_t *info, uint8_t **value,
				    uint8_t value_size_bytes, uint8_t num_bfs)
{
	int32_t err;
	uint32_t mask = 0;
	uint8_t data8 = 0, offset = 0, width = 0;
	uint8_t i = 0, reg_bytes = 0, reg_read_reqd = 1;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_NULL_POINTER_RETURN(info);
	AD9081_NULL_POINTER_RETURN(value);
	AD9081_INVALID_PARAM_RETURN(reg >= 0x4000);

	if (num_bfs == 1) {
		/* Use the standard non multi bit field */
		return adi_ad9081_hal_bf_get(device, reg, *info, *value,
					     value_size_bytes);
	}

	/* Extract the mulit bit-fields from a single reg, or use standard method for more complex situations. */
	for (i = 0; i < num_bfs; i++) {
		offset = (uint8_t)(*(info + i) >> 0);
		width = (uint8_t)(*(info + i) >> 8);
		reg_bytes = ((width + offset) >> 3) +
			    (((width + offset) & 7) == 0 ? 0 : 1);

		if ((reg_bytes == 1) && (value_size_bytes == 1)) {
			if (reg_read_reqd == 1) {
				reg_read_reqd = 0;
				err = adi_ad9081_hal_reg_get(device, reg,
							     &data8);
				AD9081_ERROR_RETURN(err);
			}
			mask = (1 << width) - 1;
			**(value + i) = (data8 >> offset) & mask;
		} else {
			/* Use non-multi bf get */
			AD9081_LOG_WARN(
				"Multi bit-field get doesn't support cross register access. Will use standard, but incurs extra SPI reads.");
			err = adi_ad9081_hal_bf_get(device, reg, *(info + i),
						    *(value + i),
						    value_size_bytes);
			AD9081_ERROR_RETURN(err);
		}
	}

	return API_CMS_ERROR_OK;
}

int32_t adi_ad9081_hal_multi_bf_set(adi_ad9081_device_t *device, uint32_t reg,
				    uint32_t *info, uint64_t *value,
				    uint8_t num_bfs)
{
	int32_t err;
	uint32_t mask = 0;
	uint8_t data8 = 0, offset = 0, width = 0;
	uint8_t i = 0, reg_bytes = 0, reg_read_reqd = 1;
	AD9081_NULL_POINTER_RETURN(device);
	AD9081_NULL_POINTER_RETURN(info);
	AD9081_NULL_POINTER_RETURN(value);
	AD9081_INVALID_PARAM_RETURN(reg >= 0x4000);

	if (num_bfs == 1) {
		/* Use the standard non multi bit field */
		return adi_ad9081_hal_bf_set(device, reg, *info, *value);
	}

	/* Write the bit fields */
	for (i = 0; i < num_bfs; i++) {
		offset = (uint8_t)(*(info + i) >> 0);
		width = (uint8_t)(*(info + i) >> 8);
		reg_bytes = ((width + offset) >> 3) +
			    (((width + offset) & 7) == 0 ? 0 : 1);

		if (reg_bytes == 1) {
			if ((reg_read_reqd == 1) &&
			    ((offset > 0) || ((offset + width) < 8))) {
				reg_read_reqd = 0;
				err = adi_ad9081_hal_reg_get(device, reg,
							     &data8);
				AD9081_ERROR_RETURN(err);
			}
			mask = (1 << width) - 1;
			data8 = data8 & (~(mask << offset));
			data8 = data8 | ((*(value + i) & mask) << offset);
		} else {
			/* Use non-multi bf set */
			AD9081_LOG_WARN(
				"multi bit-field set doesn't support cross register bit-fields.Will use standard, but incurs extra SPI reads.");
			err = adi_ad9081_hal_bf_set(device, reg, *(info + i),
						    *(value + i));
			AD9081_ERROR_RETURN(err);
		}
	}

	if (reg_read_reqd == 0) {
		err = adi_ad9081_hal_reg_set(device, reg, data8);
		AD9081_ERROR_RETURN(err);
	}

	return API_CMS_ERROR_OK;
}

/*! @} */
