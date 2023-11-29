cmake_minimum_required(VERSION 3.24)
project(v106)

find_package(pybind11)
set(CMAKE_CXX_STANDARD 17)
pybind11_add_module(adi_ad9081_v106 ad9081_wrapper.cpp adi_ad9081_adc.c adi_ad9081_dac.c adi_ad9081_device.c adi_ad9081_hal.c adi_ad9081_jesd.c)
