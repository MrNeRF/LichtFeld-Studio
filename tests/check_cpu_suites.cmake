# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Verifies that every suite named in LFS_CPU_TEST_SUITES still exists in the
# test binary.
#
# An allowlist --gtest_filter degrades silently: GoogleTest warns when a pattern
# matches nothing but still exits 0, so renaming or deleting a suite shrinks the
# CPU gate without turning CI red. Checking for zero matches is not enough
# either — losing 5 of 116 suites leaves the run green and much weaker.
#
# Usage (invoked by ctest):
#   cmake -DTEST_BINARY=<path> -DEXPECTED_SUITES=<a;b;c> -P check_cpu_suites.cmake

if(NOT TEST_BINARY OR NOT EXPECTED_SUITES)
    message(FATAL_ERROR "check_cpu_suites.cmake requires -DTEST_BINARY and -DEXPECTED_SUITES")
endif()

execute_process(
    COMMAND "${TEST_BINARY}" --gtest_list_tests
    OUTPUT_VARIABLE listing
    ERROR_VARIABLE listing_err
    RESULT_VARIABLE listing_result
    TIMEOUT 120
)

if(NOT listing_result EQUAL 0)
    message(FATAL_ERROR "could not list tests (exit ${listing_result}):\n${listing_err}")
endif()

# --gtest_list_tests prints "SuiteName." at column 0 and indents each test.
# MATCHALL rather than splitting on newlines: string(REPLACE "\n" ";") does not
# reliably produce a traversable list from this output.
string(REGEX MATCHALL "(^|\n)[A-Za-z0-9_/]+\\.\n" suite_hits "${listing}")
set(present_suites "")
foreach(hit IN LISTS suite_hits)
    string(STRIP "${hit}" hit)
    string(REGEX REPLACE "\\.$" "" hit "${hit}")
    if(hit)
        list(APPEND present_suites "${hit}")
    endif()
endforeach()

list(LENGTH present_suites present_count)
if(present_count EQUAL 0)
    message(FATAL_ERROR "parsed zero suites from --gtest_list_tests; the checker itself is broken")
endif()

set(missing "")
foreach(suite IN LISTS EXPECTED_SUITES)
    # Value-parameterised suites are listed with an "Instantiation/" prefix.
    set(found FALSE)
    foreach(present IN LISTS present_suites)
        if(present STREQUAL suite OR present MATCHES "/${suite}$")
            set(found TRUE)
            break()
        endif()
    endforeach()
    if(NOT found)
        list(APPEND missing "${suite}")
    endif()
endforeach()

list(LENGTH EXPECTED_SUITES expected_count)
list(LENGTH missing missing_count)

if(missing_count GREATER 0)
    string(REPLACE ";" "\n  " missing_text "${missing}")
    message(FATAL_ERROR
        "${missing_count} of ${expected_count} suites in LFS_CPU_TEST_SUITES no longer exist:\n"
        "  ${missing_text}\n"
        "They were renamed or removed. Update LFS_CPU_TEST_SUITES in tests/CMakeLists.txt, "
        "re-verifying each replacement with:\n"
        "  CUDA_VISIBLE_DEVICES=\"\" ./build/tests/tests/lichtfeld_tests --gtest_filter='Suite.*'")
endif()

message(STATUS "CPU gate manifest: all ${expected_count} suites present")
