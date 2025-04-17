from quel_ic_config.ad9082_nco import AbstractNcoFtw


# Notes: this reference function is taken from adi_ad9082_hal_calc_nco_ftw_f().
#        this is not as precise as our implementation based on rational number, but our function can be validated
#        in terms of equivalence of integer part.
# Notes: a and b is flipped between this function (that is based on the ADI's code) and ours due to historical reasons.
def ref_calc(nco_shift: float, freq: float) -> tuple[int, int, int]:
    """
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
        *a = (*a > 0) ? (281474976710656ull - *a) : *a; /* assume register a/b is unsigned 48bit value */
        *ftw = 281474976710656ull - *ftw - (*a > 0 ? 1 : 0);
    }
    """
    x48: int = 1 << 48

    if nco_shift >= 0:
        ftw: int = int((x48 * nco_shift) / freq)
        set_shift: float = ftw * freq / x48
        rem_shift: float = nco_shift - set_shift
        b: int = x48 - 1
        a: int = round(rem_shift * x48 * b / freq)
    else:
        ftw = int((x48 * (-nco_shift)) / freq)
        set_shift = ftw * freq / x48
        rem_shift = -nco_shift - set_shift
        b = x48 - 1
        a = round(rem_shift * x48 * b / freq)
        if a > 0:
            a = x48 - a
        ftw = x48 - ftw - (1 if a > 0 else 0)

    return ftw, a, b


def test_abstract_nco_ftw():
    # accuracy DAC positive
    dac_ftw1g = AbstractNcoFtw.from_frequency(1.0e9, 12e9)
    dac_ftw1g1 = AbstractNcoFtw.from_frequency(1.0e9 + 1, 12e9)
    assert dac_ftw1g1.to_frequency(12e9) - dac_ftw1g.to_frequency(12e9) == 1.0
    assert dac_ftw1g.modulus_a >= 0
    assert dac_ftw1g.delta_b > 0
    assert dac_ftw1g1.modulus_a >= 0
    assert dac_ftw1g1.delta_b > 0

    x0, _, _ = ref_calc(1.0e9, 12e9)
    assert AbstractNcoFtw._encode_s48_as_u48(dac_ftw1g.ftw) == x0

    x1, _, _ = ref_calc(1.0e9 + 1, 12e9)
    assert AbstractNcoFtw._encode_s48_as_u48(dac_ftw1g1.ftw) == x1

    # accuracy DAC negative
    dac_ftwn1g = AbstractNcoFtw.from_frequency(-1.0e9, 12e9)
    dac_ftwn1g1 = AbstractNcoFtw.from_frequency(-1.0e9 + 1, 12e9)
    assert dac_ftwn1g1.to_frequency(12e9) - dac_ftwn1g.to_frequency(12e9) == 1.0
    assert dac_ftwn1g.modulus_a >= 0
    assert dac_ftwn1g.delta_b > 0
    assert dac_ftwn1g1.modulus_a >= 0
    assert dac_ftwn1g1.delta_b > 0

    x2, _, _ = ref_calc(-1.0e9, 12e9)
    assert AbstractNcoFtw._encode_s48_as_u48(dac_ftwn1g.ftw) == x2

    x3, _, _ = ref_calc(-1.0e9 + 1, 12e9)
    assert AbstractNcoFtw._encode_s48_as_u48(dac_ftwn1g1.ftw) == x3

    # accuracy ADC positive
    adc_ftw1g = AbstractNcoFtw.from_frequency(1.0e9, 6e9)
    adc_ftw1g1 = AbstractNcoFtw.from_frequency(1.0e9 + 1, 6e9)
    assert adc_ftw1g1.to_frequency(6e9) - adc_ftw1g.to_frequency(6e9) == 1.0
    assert adc_ftw1g.modulus_a >= 0
    assert adc_ftw1g.delta_b > 0
    assert adc_ftw1g1.modulus_a >= 0
    assert adc_ftw1g1.delta_b > 0

    x4, _, _ = ref_calc(1.0e9, 6e9)
    assert adc_ftw1g.ftw == x4

    x5, _, _ = ref_calc(1.0e9 + 1, 6e9)
    assert adc_ftw1g1.ftw == x5

    # accuracy ADC negative
    adc_ftwn1g = AbstractNcoFtw.from_frequency(-1.0e9, 6e9)
    adc_ftwn1g1 = AbstractNcoFtw.from_frequency(-1.0e9 + 1, 6e9)
    assert adc_ftwn1g1.to_frequency(6e9) - adc_ftwn1g.to_frequency(6e9) == 1.0
    assert adc_ftwn1g.modulus_a >= 0
    assert adc_ftwn1g.delta_b > 0
    assert adc_ftwn1g1.modulus_a >= 0
    assert adc_ftwn1g1.delta_b > 0

    x6, _, _ = ref_calc(-1.0e9, 6e9)
    assert AbstractNcoFtw._encode_s48_as_u48(adc_ftwn1g.ftw) == x6

    x7, _, _ = ref_calc(-1.0e9 + 1, 6e9)
    assert AbstractNcoFtw._encode_s48_as_u48(adc_ftwn1g1.ftw) == x7

    # between DAC and ADC, no error!
    assert dac_ftw1g.to_frequency(12e9) == adc_ftw1g.to_frequency(6e9)
    assert dac_ftwn1g.to_frequency(12e9) == adc_ftwn1g.to_frequency(6e9)

    # rounding positive
    dac_ftw1g.round()
    adc_ftw1g.round()
    assert 0.0 <= abs(dac_ftw1g.to_frequency(12e9) - adc_ftw1g.to_frequency(6e9)) < (6e9 / (1 << 48)) * 1.5

    # multiply positive
    adc_ftw1g_matched = dac_ftw1g.multiply(2)
    assert dac_ftw1g.to_frequency(12e9) == adc_ftw1g_matched.to_frequency(6e9)

    # rounding negative
    dac_ftwn1g.round()
    adc_ftwn1g.round()
    assert 0.0 <= abs(dac_ftwn1g.to_frequency(12e9) - adc_ftwn1g.to_frequency(6e9)) < (6e9 / (1 << 48)) * 1.5

    # multiply negative
    adc_ftwn1g_matched = dac_ftwn1g.multiply(2)
    assert dac_ftwn1g.to_frequency(12e9) == adc_ftwn1g_matched.to_frequency(6e9)


def test_abstract_nco_ftw_integer():
    f0 = 375000000  # Notes: 375e6 is one of special values because 375e6 * (1 << 48) // int(12e9) == 0x800_0000_0000

    for i in range(16):
        for pm in {1, -1}:
            f1 = f0 / (1 << i) * pm
            ftw1 = AbstractNcoFtw.from_frequency(f1, 12e9)
            ftw2 = AbstractNcoFtw.from_frequency(f1, 6e9)
            assert ftw1.to_frequency(12e9) == f1
            assert ftw2.to_frequency(6e9) == f1
