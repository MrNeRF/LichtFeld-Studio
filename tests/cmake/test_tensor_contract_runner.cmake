# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

if(NOT EXISTS "${RUNNER}")
    message(FATAL_ERROR "Tensor contract runner does not exist: ${RUNNER}")
endif()

function(expect_runner label expected_result expected_message)
    execute_process(COMMAND "${RUNNER}" ${ARGN}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 20)
    if(NOT "${result}" STREQUAL "${expected_result}")
        message(FATAL_ERROR "${label}: expected exit ${expected_result}, got ${result}\n${output}\n${error}")
    endif()
    string(FIND "${output}\n${error}" "${expected_message}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "${label}: missing '${expected_message}'\n${output}\n${error}")
    endif()
endfunction()

if(CASE STREQUAL "required-backend")
    # This entry is GPU-labelled: availability must not hide a coverage failure.
    expect_runner(required_backend_coverage 2 "No tensor contracts executed for required backend: ${BACKEND}"
        "--require-tensor-backend=${BACKEND}" "--gtest_filter=Backends/TensorOperationContracts.*/CPU")
    return()
endif()

if(NOT EXISTS "${EXIT_RUNNER}")
    message(FATAL_ERROR "Backend exit contract runner does not exist: ${EXIT_RUNNER}")
endif()
execute_process(COMMAND "${EXIT_RUNNER}"
    RESULT_VARIABLE exit_result OUTPUT_VARIABLE exit_output ERROR_VARIABLE exit_error TIMEOUT 20)
if(NOT "${exit_result}" STREQUAL "0" OR NOT exit_output MATCHES "Backend driver exit contracts passed")
    message(FATAL_ERROR "Backend exit contract failed: ${exit_result}\n${exit_output}\n${exit_error}")
endif()

expect_runner(cpu_selection 0 "[  PASSED  ]"
    "--gtest_filter=Backends/TensorOperationContracts.*/CPU")
expect_runner(empty_selection 2 "No tensor contracts executed"
    "--gtest_filter=NoSuchContract.*")
expect_runner(zero_repeats 2 "No tensor contracts executed"
    "--gtest_filter=Backends/TensorOperationContracts.*/CPU" "--gtest_repeat=0")
expect_runner(skipped_only 2 "No tensor contracts executed"
    "--gtest_filter=Backends/TensorBackendSemantics.InBatchUploadOrdersAfterPendingReaders/CPU")
expect_runner(unknown_argument 2 "Unknown argument: --typo" "--typo")
expect_runner(unknown_backend 2 "Required tensor backend not compiled: NoSuchBackend"
    "--require-tensor-backend=NoSuchBackend")
expect_runner(listing 0 "SoftmaxMatchesAnIndependentStableReference"
    "--gtest_list_tests" "--gtest_filter=Backends/TensorOperationContracts.*")
if(NOT "${BACKEND}" STREQUAL "")
    # Listing a compiled backend must work even on a CI host without its GPU.
    expect_runner(required_backend_listing 0 "PointwiseBroadcastUsesAnalyticValuesAtOddSizes/${BACKEND}"
        "--gtest_list_tests" "--require-tensor-backend=${BACKEND}"
        "--gtest_filter=Backends/TensorOperationContracts.*/${BACKEND}")
endif()
message(STATUS "Tensor contract runner regressions passed")
