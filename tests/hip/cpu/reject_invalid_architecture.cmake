# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
execute_process(COMMAND "${CMAKE_COMMAND}" -DTEST_UNSUPPORTED=ON
    -P "${CMAKE_CURRENT_LIST_DIR}/../../cmake/hip_architectures.cmake"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "Unsupported HIP target")
    message(FATAL_ERROR "Invalid architecture did not fail at the target validator: ${output}${error}")
endif()
