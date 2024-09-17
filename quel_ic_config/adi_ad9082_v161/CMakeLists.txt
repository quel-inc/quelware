cmake_minimum_required(VERSION 3.24)
project(v161)

find_package(pybind11)
set(CMAKE_CXX_STANDARD 17)
pybind11_add_module(
        adi_ad9082_v161
        ad9082_wrapper.cpp
        ad9082_api/ad9082/src/adi_ad9082_adc.c
        ad9082_api/ad9082/src/adi_ad9082_dac.c
        ad9082_api/ad9082/src/adi_ad9082_device.c
        ad9082_api/ad9082/src/adi_ad9082_hal.c
        ad9082_api/ad9082/src/adi_ad9082_jesd.c
        ad9082_api/ad9082/src/adi_ad9082_sync.c
        ad9082_api/adi_utils/src/adi_utils.c
)
target_include_directories(
        adi_ad9082_v161
        PUBLIC ad9082_api/adi_inc ad9082_api/ad9082/inc ad9082_api/adi_utils/inc
)
