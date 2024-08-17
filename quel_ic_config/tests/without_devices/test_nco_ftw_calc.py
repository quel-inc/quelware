from quel_ic_config.ad9082_nco import AbstractNcoFtw


def test_abstract_nco_ftw():
    # accuracy DAC positive
    dac_ftw1g = AbstractNcoFtw.from_frequency(1.0e9, 12e9)
    dac_ftw1g1 = AbstractNcoFtw.from_frequency(1.0e9 + 1, 12e9)
    assert dac_ftw1g1.to_frequency(12e9) - dac_ftw1g.to_frequency(12e9) == 1.0

    # accuracy DAC negative
    dac_ftwn1g = AbstractNcoFtw.from_frequency(-1.0e9, 12e9)
    dac_ftwn1g1 = AbstractNcoFtw.from_frequency(-1.0e9 + 1, 12e9)
    assert dac_ftwn1g1.to_frequency(12e9) - dac_ftwn1g.to_frequency(12e9) == 1.0

    # accuracy ADC positive
    adc_ftw1g = AbstractNcoFtw.from_frequency(1.0e9, 6e9)
    adc_ftw1g1 = AbstractNcoFtw.from_frequency(1.0e9 + 1, 6e9)
    assert adc_ftw1g1.to_frequency(6e9) - adc_ftw1g.to_frequency(6e9) == 1.0

    # accuracy ADC negative
    adc_ftwn1g = AbstractNcoFtw.from_frequency(-1.0e9, 6e9)
    adc_ftwn1g1 = AbstractNcoFtw.from_frequency(-1.0e9 + 1, 6e9)
    assert adc_ftwn1g1.to_frequency(6e9) - adc_ftwn1g.to_frequency(6e9) == 1.0

    # between DAC and ADC
    assert dac_ftw1g.to_frequency(12e9) == adc_ftw1g.to_frequency(6e9)
    assert dac_ftwn1g.to_frequency(12e9) == adc_ftwn1g.to_frequency(6e9)

    # rounding positive
    dac_ftw1g.round()
    adc_ftw1g.round()
    assert 0.0 < abs(dac_ftw1g.to_frequency(12e9) - adc_ftw1g.to_frequency(6e9)) < (6e9 / (1 << 48)) * 1.5

    # multiply positive
    adc_ftw1g_matched = dac_ftw1g.multiply(2)
    assert dac_ftw1g.to_frequency(12e9) == adc_ftw1g_matched.to_frequency(6e9)

    # rounding negative
    dac_ftwn1g.round()
    adc_ftwn1g.round()
    assert 0.0 < abs(dac_ftwn1g.to_frequency(12e9) - adc_ftwn1g.to_frequency(6e9)) < (6e9 / (1 << 48)) * 1.5

    # multiply negative
    adc_ftwn1g_matched = dac_ftwn1g.multiply(2)
    assert dac_ftwn1g.to_frequency(12e9) == adc_ftwn1g_matched.to_frequency(6e9)
