#
# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ===============================================================================
# Generated file for building library with user generated CPU extensions
#
# Contains implementation of the basic layer methods
#
# Refer to the section "Adding Your Own Kernels to the Inference Engine" in
# OpenVINO* documentation (either online or offline in
# <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
# to the corresponding section).
# ===============================================================================

cmake_minimum_required(VERSION 3.14)

set(CPU_EXTENSIONS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

set(CMAKE_CXX_STANDARD 11)

if(NOT IIE_DIR)
    set(IIE_DIR "/opt/dldt")
endif()


file(GLOB SRC ${CPU_EXTENSIONS_BASE_DIR}/*.cpp)
set(TARGET_NAME custom_cpu_extensions)
add_library(${TARGET_NAME} SHARED ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC ${IIE_DIR}/inference-engine/include ${IIE_DIR}/ngraph/core/include/ ${IIE_DIR}/inference-engine/temp/tbb/include)
target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE
        ${IIE_DIR}/bin/intel64/Release/lib/libinference_engine_s.a
        ${IIE_DIR}/bin/intel64/Release/lib/libngraph_s.a
        ${IIE_DIR}/inference-engine/temp/tbb/lib/libtbb.so
        ${IIE_DIR}/bin/intel64/Release/lib/libitt.a)

