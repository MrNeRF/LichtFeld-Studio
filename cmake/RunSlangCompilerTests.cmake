# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Runs the shader compiler tests, then spirv-val on every module they emitted
# when the build has a validator (validation builds and CI only).
#
# cmake -DTEST_EXECUTABLE=<exe> -DSPIRV_DIR=<dir> [-DSPIRV_VAL=<spirv-val>] -P RunSlangCompilerTests.cmake

file(REMOVE_RECURSE "${SPIRV_DIR}")
execute_process(COMMAND "${TEST_EXECUTABLE}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Shader compiler tests failed: ${result}")
endif()
if(NOT SPIRV_VAL)
    return()
endif()
file(GLOB modules "${SPIRV_DIR}/*.spv")
if(NOT modules)
    message(FATAL_ERROR "The shader compiler tests emitted no SPIR-V to validate")
endif()
foreach(module IN LISTS modules)
    execute_process(COMMAND "${SPIRV_VAL}" "${module}" RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "spirv-val rejected ${module}")
    endif()
endforeach()
